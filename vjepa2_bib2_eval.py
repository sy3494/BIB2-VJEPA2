#!/usr/bin/env python3
"""
Zero-shot BabyView V-JEPA2 evaluation on BIB2.

Protocol
--------
Each BIB2 item contains:
    - 8 familiarization episodes
    - 1 expected test video
    - 1 unexpected test video

The BIB2 JSON contains the two video paths and 9 episode-start indices.
The first 8 intervals are familiarization episodes; the ninth start is the
beginning of the test phase.

This script does NOT train or fine-tune on BIB2.

Because the released checkpoint is a 16-frame V-JEPA2 model, it cannot keep
all eight long familiarization movies in persistent recurrent memory. The
zero-shot adaptation used here is:

1. Sample each familiarization and test phase at 4 fps.
2. Make 12-frame windows within each familiarization.
3. For each test transition:
       12 fam frames + 2 current-test frames = V-JEPA context (14 frames)
       next 2 test frames                  = prediction target
4. Use the pretrained V-JEPA2 predictor to predict the target latent tokens.
5. Compare predicted target latents to encoder latents for the true target
   using mean absolute error (the V-JEPA-style latent prediction loss).
6. Average windows within each familiarization episode, then average the
   eight familiarization episodes equally.
7. Define test surprise as the maximum latent prediction error over the
   test phase.
8. Compare expected and unexpected:
       voe_gap = unexpected_surprise - expected_surprise
   Positive voe_gap means the unexpected outcome was more surprising.

Important implementation detail
-------------------------------
We intentionally do NOT call the high-level model forward on the complete
16-frame clip with a context mask, because the Hugging Face implementation
first encodes the entire supplied clip. Instead, the context encoder receives
only the 14 visible frames, then the native V-JEPA2 predictor predicts token
positions corresponding to the final two-frame tubelet. The full 16-frame
clip is encoded separately only to obtain the target representation.

Tested against the Hugging Face VJEPA2Model API shape/config used by:
    awwkl/vjepa2-vitl-fpc16-256-babyview-bs3072-e140
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from transformers import AutoModel, AutoVideoProcessor


DEFAULT_MODEL_ID = "awwkl/vjepa2-vitl-fpc16-256-babyview-bs3072-e140"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(
    model_id: str = DEFAULT_MODEL_ID,
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    processor = AutoVideoProcessor.from_pretrained(model_id)

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device).eval()

    for p in model.parameters():
        p.requires_grad = False

    if model.config.frames_per_clip != 16:
        raise ValueError(
            f"This evaluator assumes a 16-frame checkpoint, but "
            f"model.config.frames_per_clip={model.config.frames_per_clip}."
        )

    if model.config.tubelet_size != 2:
        raise ValueError(
            f"This evaluator assumes tubelet_size=2, but got "
            f"{model.config.tubelet_size}."
        )

    return processor, model


# ---------------------------------------------------------------------------
# JSON / path handling
# ---------------------------------------------------------------------------

def _video_label(path: str) -> str | None:
    stem = Path(path).stem.lower()

    expected_suffixes = ("_e", "_expected", "-e", "-expected")
    unexpected_suffixes = ("_u", "_unexpected", "-u", "-unexpected")

    if stem.endswith(expected_suffixes):
        return "expected"
    if stem.endswith(unexpected_suffixes):
        return "unexpected"
    return None


def parse_bib2_json(json_path: str | Path):
    json_path = Path(json_path)

    with open(json_path, "r") as f:
        meta = json.load(f)

    tuples = meta["data_tuples"]

    if len(tuples) < 2:
        raise ValueError(f"{json_path} has fewer than two data_tuples.")

    labelled = {}
    unlabelled = []

    for item in tuples:
        label = _video_label(item[0])
        if label is None:
            unlabelled.append(item)
        else:
            labelled[label] = item

    if "expected" in labelled and "unexpected" in labelled:
        expected_info = labelled["expected"]
        unexpected_info = labelled["unexpected"]
    else:
        # Fallback to the ordering used in the supplied BIB2 JSON.
        expected_info = tuples[0]
        unexpected_info = tuples[1]

    starts_e = list(expected_info[1])
    starts_u = list(unexpected_info[1])

    if starts_e != starts_u:
        raise ValueError(
            f"Expected/unexpected episode boundaries differ in {json_path}."
        )

    if len(starts_e) < 9:
        raise ValueError(
            f"Expected >=9 episode starts (8 familiarizations + test), "
            f"got {len(starts_e)} in {json_path}."
        )

    return {
        "json_path": json_path,
        "expected_raw_path": expected_info[0],
        "unexpected_raw_path": unexpected_info[0],
        "episode_starts": starts_e[:9],
    }


def resolve_video_path(
    raw_path: str,
    json_path: Path,
    project_root: str | Path | None = None,
) -> Path:
    raw = Path(raw_path)

    candidates = []

    if raw.is_absolute():
        candidates.append(raw)

    if project_root is not None:
        candidates.append(Path(project_root) / raw)

    candidates.extend([
        Path.cwd() / raw,
        json_path.parent / raw,
        json_path.parent.parent / raw,
    ])

    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate

    attempted = "\n".join(f"  - {p.resolve()}" for p in candidates)
    raise FileNotFoundError(
        f"Could not resolve video path '{raw_path}'. Tried:\n{attempted}\n"
        f"Pass --project-root if the JSON paths are relative to your project root."
    )


# ---------------------------------------------------------------------------
# Video loading / sampling
# ---------------------------------------------------------------------------

def sample_segment_at_fps(
    vr: VideoReader,
    start_frame: int,
    end_frame: int,
    target_fps: float = 4.0,
) -> torch.Tensor:
    """
    Returns a uint8 tensor [T, C, H, W].

    start_frame/end_frame use source-video frame indices from the BIB2 JSON.
    """
    source_fps = float(vr.get_avg_fps())

    start_frame = max(0, int(start_frame))
    end_frame = min(int(end_frame), len(vr))

    if end_frame <= start_frame:
        raise ValueError(
            f"Invalid segment [{start_frame}, {end_frame}) for video with "
            f"{len(vr)} frames."
        )

    step = source_fps / float(target_fps)

    indices = np.arange(start_frame, end_frame, step)
    indices = np.rint(indices).astype(np.int64)
    indices = np.clip(indices, start_frame, end_frame - 1)
    indices = np.unique(indices)

    frames = vr.get_batch(indices).asnumpy()     # [T,H,W,C]
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [T,C,H,W]

    return frames


def load_bib2_item(
    json_path: str | Path,
    project_root: str | Path | None = None,
    target_fps: float = 4.0,
):
    info = parse_bib2_json(json_path)
    json_path = info["json_path"]

    expected_path = resolve_video_path(
        info["expected_raw_path"], json_path, project_root
    )
    unexpected_path = resolve_video_path(
        info["unexpected_raw_path"], json_path, project_root
    )

    vr_e = VideoReader(str(expected_path), ctx=cpu(0))
    vr_u = VideoReader(str(unexpected_path), ctx=cpu(0))

    starts = info["episode_starts"]

    # Canonical familiarization comes from the expected movie.
    fams = []
    for i in range(8):
        fams.append(
            sample_segment_at_fps(
                vr_e,
                starts[i],
                starts[i + 1],
                target_fps=target_fps,
            )
        )

    test_start = starts[8]

    expected = sample_segment_at_fps(
        vr_e,
        test_start,
        len(vr_e),
        target_fps=target_fps,
    )

    unexpected = sample_segment_at_fps(
        vr_u,
        test_start,
        len(vr_u),
        target_fps=target_fps,
    )

    return {
        "fams": fams,
        "expected": expected,
        "unexpected": unexpected,
        "expected_path": expected_path,
        "unexpected_path": unexpected_path,
        "episode_starts": starts,
        "source_fps_expected": float(vr_e.get_avg_fps()),
        "source_fps_unexpected": float(vr_u.get_avg_fps()),
    }


def make_windows(
    frames: torch.Tensor,
    length: int = 12,
    stride: int = 6,
) -> List[torch.Tensor]:
    """
    Make overlapping [length,C,H,W] windows and always include the final window.
    """
    T = len(frames)

    if T < length:
        # For unusually short familiarizations, repeat the final frame.
        pad = frames[-1:].repeat(length - T, 1, 1, 1)
        frames = torch.cat([frames, pad], dim=0)
        T = len(frames)

    starts = list(range(0, T - length + 1, stride))
    last_start = T - length

    if not starts:
        starts = [0]
    elif starts[-1] != last_start:
        starts.append(last_start)

    return [frames[s:s + length] for s in starts]


# ---------------------------------------------------------------------------
# Native V-JEPA2 latent prediction
# ---------------------------------------------------------------------------

def _pixel_values(processor, videos, device, dtype):
    inputs = processor(videos, return_tensors="pt")
    return inputs["pixel_values_videos"].to(device=device, dtype=dtype)


@torch.inference_mode()
def prediction_error_batch(
    fam_windows: Sequence[torch.Tensor],
    current_test: torch.Tensor,
    target_test: torch.Tensor,
    processor,
    model,
) -> torch.Tensor:
    """
    One latent prediction error per familiarization window.

    Context is:
        12 familiarization frames + 2 current-test frames = 14 frames

    Target is:
        next 2 test frames = one temporal tubelet

    The context encoder only sees the 14 context frames.
    A separate full-clip encoder pass supplies the target latent.
    """
    if len(fam_windows) == 0:
        raise ValueError("fam_windows is empty.")

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    context_videos = [
        torch.cat([fam, current_test], dim=0)
        for fam in fam_windows
    ]
    full_videos = [
        torch.cat([fam, current_test, target_test], dim=0)
        for fam in fam_windows
    ]

    context_pixels = _pixel_values(
        processor, context_videos, device=device, dtype=dtype
    )
    full_pixels = _pixel_values(
        processor, full_videos, device=device, dtype=dtype
    )

    # IMPORTANT: encode the 14-frame context only, avoiding target leakage.
    context_hidden = model.encoder(
        pixel_values_videos=context_pixels
    ).last_hidden_state

    B, n_context_tokens, _ = context_hidden.shape

    spatial_tokens = (
        model.config.crop_size // model.config.patch_size
    ) ** 2

    # One two-frame tubelet is the target.
    n_target_tokens = spatial_tokens

    context_ids = torch.arange(
        n_context_tokens, device=device, dtype=torch.long
    ).unsqueeze(0).repeat(B, 1)

    target_ids = torch.arange(
        n_context_tokens,
        n_context_tokens + n_target_tokens,
        device=device,
        dtype=torch.long,
    ).unsqueeze(0).repeat(B, 1)

    pred = model.predictor(
        encoder_hidden_states=context_hidden,
        context_mask=[context_ids],
        target_mask=[target_ids],
    ).last_hidden_state

    # Encode the full clip ONLY to define the target representation.
    full_hidden = model.encoder(
        pixel_values_videos=full_pixels
    ).last_hidden_state

    actual = full_hidden[
        :,
        n_context_tokens:n_context_tokens + n_target_tokens,
        :
    ]

    actual = F.layer_norm(
        actual.float(),
        (actual.shape[-1],),
    )

    # Native V-JEPA training code uses |prediction-target|^loss_exp/loss_exp;
    # for the standard loss_exp=1 this is mean absolute latent error.
    errors = (
        pred.float() - actual
    ).abs().mean(dim=(1, 2))

    return errors.cpu()


def prediction_error(
    fam_windows: Sequence[torch.Tensor],
    current_test: torch.Tensor,
    target_test: torch.Tensor,
    processor,
    model,
    batch_size: int = 4,
) -> torch.Tensor:
    chunks = []

    for start in range(0, len(fam_windows), batch_size):
        batch = fam_windows[start:start + batch_size]

        chunks.append(
            prediction_error_batch(
                batch,
                current_test,
                target_test,
                processor,
                model,
            )
        )

    return torch.cat(chunks)


@torch.inference_mode()
def evaluate_test_video(
    fam_windows_by_trial: Sequence[Sequence[torch.Tensor]],
    test_frames: torch.Tensor,
    processor,
    model,
    batch_size: int = 4,
):
    """
    Return per-test-step latent error and max test surprise.

    Each test step uses one 2-frame tubelet as current test context and predicts
    the next 2-frame tubelet.
    """
    if len(fam_windows_by_trial) != 8:
        raise ValueError(
            f"Expected 8 familiarization episodes, got "
            f"{len(fam_windows_by_trial)}."
        )

    if len(test_frames) < 4:
        raise ValueError(
            f"Need at least 4 test frames at target fps, got {len(test_frames)}."
        )

    step_errors = []
    per_fam_errors = []

    for start in range(0, len(test_frames) - 3, 2):
        current_test = test_frames[start:start + 2]
        target_test = test_frames[start + 2:start + 4]

        fam_trial_errors = []

        for fam_windows in fam_windows_by_trial:
            window_errors = prediction_error(
                fam_windows,
                current_test,
                target_test,
                processor,
                model,
                batch_size=batch_size,
            )

            # Equal weight per familiarization episode.
            fam_trial_errors.append(float(window_errors.mean()))

        step_errors.append(float(np.mean(fam_trial_errors)))
        per_fam_errors.append(fam_trial_errors)

    step_errors = np.asarray(step_errors, dtype=np.float64)
    per_fam_errors = np.asarray(per_fam_errors, dtype=np.float64)

    surprise = float(step_errors.max())
    max_step = int(step_errors.argmax())

    return {
        "step_errors": step_errors,
        "per_fam_errors": per_fam_errors,
        "surprise": surprise,
        "max_step": max_step,
    }


def task_name_from_json(json_path: str | Path) -> str:
    stem = Path(json_path).stem
    for prefix in ("index_dict_eval_", "index_dict_"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


def evaluate_bib2_json(
    json_path: str | Path,
    processor,
    model,
    project_root: str | Path | None = None,
    target_fps: float = 4.0,
    fam_window_frames: int = 12,
    fam_window_stride: int = 6,
    batch_size: int = 4,
):
    item = load_bib2_item(
        json_path,
        project_root=project_root,
        target_fps=target_fps,
    )

    fam_windows_by_trial = [
        make_windows(
            fam,
            length=fam_window_frames,
            stride=fam_window_stride,
        )
        for fam in item["fams"]
    ]

    expected_eval = evaluate_test_video(
        fam_windows_by_trial,
        item["expected"],
        processor,
        model,
        batch_size=batch_size,
    )

    unexpected_eval = evaluate_test_video(
        fam_windows_by_trial,
        item["unexpected"],
        processor,
        model,
        batch_size=batch_size,
    )

    expected_surprise = expected_eval["surprise"]
    unexpected_surprise = unexpected_eval["surprise"]

    voe_gap = unexpected_surprise - expected_surprise

    result = {
        "task": task_name_from_json(json_path),
        "json_path": str(Path(json_path).resolve()),
        "expected_video": str(item["expected_path"]),
        "unexpected_video": str(item["unexpected_path"]),
        "expected_surprise": expected_surprise,
        "unexpected_surprise": unexpected_surprise,
        "expected_score": -expected_surprise,
        "unexpected_score": -unexpected_surprise,
        "voe_gap": voe_gap,
        "prefers_expected": bool(expected_surprise < unexpected_surprise),
        "expected_max_step": expected_eval["max_step"],
        "unexpected_max_step": unexpected_eval["max_step"],
        "n_expected_steps": len(expected_eval["step_errors"]),
        "n_unexpected_steps": len(unexpected_eval["step_errors"]),
        "fam_window_counts": [len(x) for x in fam_windows_by_trial],
        "expected_step_errors": expected_eval["step_errors"],
        "unexpected_step_errors": unexpected_eval["step_errors"],
    }

    return result


# ---------------------------------------------------------------------------
# CSV / command line
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "task",
    "json_path",
    "expected_video",
    "unexpected_video",
    "expected_surprise",
    "unexpected_surprise",
    "expected_score",
    "unexpected_score",
    "voe_gap",
    "prefers_expected",
    "expected_max_step",
    "unexpected_max_step",
    "n_expected_steps",
    "n_unexpected_steps",
    "fam_window_counts",
]


def csv_safe_result(result):
    row = {}
    for field in CSV_FIELDS:
        value = result[field]
        if isinstance(value, (list, tuple, np.ndarray)):
            value = json.dumps(
                value.tolist() if isinstance(value, np.ndarray) else value
            )
        row[field] = value
    return row


def collect_jsons(
    json_path: str | None,
    json_dir: str | None,
    pattern: str,
):
    if bool(json_path) == bool(json_dir):
        raise ValueError("Specify exactly one of --json or --json-dir.")

    if json_path:
        return [Path(json_path)]

    return sorted(Path(json_dir).glob(pattern))


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot BabyView V-JEPA2 evaluation on BIB2."
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--json", type=str, help="One BIB2 index JSON.")
    source.add_argument(
        "--json-dir",
        type=str,
        help="Directory containing BIB2 index JSONs.",
    )

    parser.add_argument(
        "--pattern",
        default="index_dict_eval_*.json",
        help="Glob pattern used with --json-dir.",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help=(
            "Root against which video paths stored in JSONs are resolved. "
            "Useful when JSON entries look like ../data/..."
        ),
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
    )
    parser.add_argument(
        "--device",
        default=None,
        help="e.g. cuda, cuda:0, or cpu. Defaults to CUDA when available.",
    )
    parser.add_argument("--target-fps", type=float, default=4.0)
    parser.add_argument("--fam-window-frames", type=int, default=12)
    parser.add_argument("--fam-window-stride", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--output",
        type=str,
        default="vjepa2_bib2_results.csv",
    )

    args = parser.parse_args()

    jsons = collect_jsons(args.json, args.json_dir, args.pattern)

    if not jsons:
        raise FileNotFoundError("No JSON files matched.")

    print(f"Loading model: {args.model_id}")
    processor, model = load_model(
        model_id=args.model_id,
        device=args.device,
    )

    print(model.config)
    print(f"Evaluating {len(jsons)} JSON file(s).")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for idx, json_path in enumerate(jsons, 1):
        print(f"\n[{idx}/{len(jsons)}] {json_path}")

        try:
            result = evaluate_bib2_json(
                json_path,
                processor,
                model,
                project_root=args.project_root,
                target_fps=args.target_fps,
                fam_window_frames=args.fam_window_frames,
                fam_window_stride=args.fam_window_stride,
                batch_size=args.batch_size,
            )

            print(
                f"  expected surprise   = "
                f"{result['expected_surprise']:.6f}"
            )
            print(
                f"  unexpected surprise = "
                f"{result['unexpected_surprise']:.6f}"
            )
            print(f"  VOE gap             = {result['voe_gap']:.6f}")
            print(
                f"  prefers expected    = "
                f"{result['prefers_expected']}"
            )

            rows.append(csv_safe_result(result))

        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            rows.append({
                field: "" for field in CSV_FIELDS
            })
            rows[-1]["task"] = task_name_from_json(json_path)
            rows[-1]["json_path"] = str(Path(json_path).resolve())
            rows[-1]["prefers_expected"] = f"ERROR: {exc}"

        # Write after every item so partial progress survives interruptions.
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nSaved: {output_path.resolve()}")


if __name__ == "__main__":
    main()
