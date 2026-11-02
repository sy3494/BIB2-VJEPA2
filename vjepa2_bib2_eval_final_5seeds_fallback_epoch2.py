#!/usr/bin/env python3
"""
Evaluate the FINAL fine-tuned BabyView V-JEPA2 model for seeds 42-46 on BIB2.

This is based on the user's existing evaluator. The BIB2 evaluation procedure
(prediction target, cumulative test context, surprise score, trimming, etc.) is
unchanged. The main change is that the script loops over the final_hf_model
for each requested seed and writes one combined CSV with seed/model columns.

By default it looks for:
  /scratch/sy3494/saved_models/vjepa2_bib2_2gpu_seed{seed}/final_hf_model
and, for compatibility with the existing seed-42 run, also falls back to:
  /scratch/sy3494/saved_models/vjepa2_bib2_2gpu_seed{seed}_test/final_hf_model
"""

from __future__ import print_function

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from transformers import AutoModel, AutoVideoProcessor


DEFAULT_MODEL_ID = "awwkl/vjepa2-vitl-fpc16-256-babyview-bs3072-e140"
DEFAULT_N_FAM_TRIALS = 8
DEFAULT_FRAMES_PER_FAM = 90
DEFAULT_TEST_FPS = 25.0
DEFAULT_TEST_STRIDE = 1
DEFAULT_FAM_END_TRIM_SECONDS = 2.0
DEFAULT_TEST_END_TRIM_SECONDS = 2.0


def load_model(model_id=DEFAULT_MODEL_ID, device=None):
    """Load either a Hugging Face model directory/repo or a raw .pt checkpoint.

    If model_id points to epoch_002.pt, instantiate the original BabyView model
    architecture and then load checkpoint["model_state_dict"].
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    source = Path(model_id).expanduser()

    if source.is_file():
        # Raw training checkpoint such as epoch_002.pt.
        print("  Loading raw checkpoint:", source)
        processor = AutoVideoProcessor.from_pretrained(DEFAULT_MODEL_ID)
        model = AutoModel.from_pretrained(
            DEFAULT_MODEL_ID,
            dtype=dtype,
            attn_implementation="sdpa",
        )
        checkpoint = torch.load(source, map_location="cpu")
        if "model_state_dict" not in checkpoint:
            raise KeyError(
                "Checkpoint %s does not contain model_state_dict" % source
            )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model = model.to(device).eval()
    else:
        # Hugging Face model directory (final_hf_model) or repo ID.
        processor = AutoVideoProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(
            model_id,
            dtype=dtype,
            attn_implementation="sdpa",
        ).to(device).eval()

    for p in model.parameters():
        p.requires_grad = False

    if int(model.config.tubelet_size) != 2:
        raise ValueError(
            "This evaluator assumes tubelet_size=2; got %s"
            % model.config.tubelet_size
        )

    return processor, model


def _video_label(path):
    stem = Path(path).stem.lower()
    if stem.endswith(("_e", "_expected", "-e", "-expected")):
        return "expected"
    if stem.endswith(("_u", "_unexpected", "-u", "-unexpected")):
        return "unexpected"
    return None


def parse_bib2_json(json_path):
    json_path = Path(json_path)

    with open(json_path, "r") as f:
        meta = json.load(f)

    tuples = meta["data_tuples"]
    if len(tuples) < 2:
        raise ValueError("%s contains fewer than 2 data_tuples." % json_path)

    labelled = {}
    for item in tuples:
        label = _video_label(item[0])
        if label is not None:
            labelled[label] = item

    if "expected" in labelled and "unexpected" in labelled:
        expected_info = labelled["expected"]
        unexpected_info = labelled["unexpected"]
    else:
        expected_info = tuples[0]
        unexpected_info = tuples[1]

    starts_e = list(expected_info[1])
    starts_u = list(unexpected_info[1])

    if starts_e != starts_u:
        raise ValueError("Expected/unexpected episode boundaries differ.")

    if len(starts_e) < 9:
        raise ValueError(
            "Need >=9 episode starts (8 familiarizations + test); got %d."
            % len(starts_e)
        )

    return {
        "json_path": json_path,
        "expected_raw_path": expected_info[0],
        "unexpected_raw_path": unexpected_info[0],
        "episode_starts": starts_e[:9],
    }


def resolve_video_path(raw_path, json_path, project_root=None):
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

    attempted = "\n".join("  - %s" % p.resolve() for p in candidates)
    raise FileNotFoundError(
        "Could not resolve video path '%s'. Tried:\n%s\n"
        "Pass --project-root if needed." % (raw_path, attempted)
    )


def sample_uniform_frames(vr, start_frame, end_frame, n_frames):
    """Sample exactly n_frames across the full familiarization trial."""
    start_frame = max(0, int(start_frame))
    end_frame = min(int(end_frame), len(vr))

    if end_frame <= start_frame:
        raise ValueError("Invalid interval [%d, %d)." % (start_frame, end_frame))

    indices = np.linspace(start_frame, end_frame - 1, n_frames)
    indices = np.rint(indices).astype(np.int64)
    indices = np.clip(indices, start_frame, end_frame - 1)

    frames = vr.get_batch(indices).asnumpy()  # [T,H,W,C]
    return torch.from_numpy(frames).permute(0, 3, 1, 2)  # [T,C,H,W]


def sample_segment_at_fps(vr, start_frame, end_frame, target_fps=4.0):
    """Sample the test segment at a target FPS."""
    source_fps = float(vr.get_avg_fps())
    start_frame = max(0, int(start_frame))
    end_frame = min(int(end_frame), len(vr))

    if end_frame <= start_frame:
        raise ValueError("Invalid interval [%d, %d)." % (start_frame, end_frame))

    step = source_fps / float(target_fps)
    indices = np.arange(start_frame, end_frame, step, dtype=np.float64)
    indices = np.rint(indices).astype(np.int64)
    indices = np.clip(indices, start_frame, end_frame - 1)
    indices = np.unique(indices)

    frames = vr.get_batch(indices).asnumpy()
    return torch.from_numpy(frames).permute(0, 3, 1, 2)


def load_bib2_item(
    json_path,
    project_root=None,
    n_fam_trials=DEFAULT_N_FAM_TRIALS,
    frames_per_fam=DEFAULT_FRAMES_PER_FAM,
    test_fps=DEFAULT_TEST_FPS,
    fam_end_trim_seconds=DEFAULT_FAM_END_TRIM_SECONDS,
    test_end_trim_seconds=DEFAULT_TEST_END_TRIM_SECONDS,
):
    """
    Load one BIB2 expected/unexpected pair.

    Familiarization:
      - Use the first `n_fam_trials` trials.
      - Remove the final `fam_end_trim_seconds` seconds from EACH trial.
      - Then uniformly sample `frames_per_fam` frames from the remaining
        behavioral portion of the trial.

    Test:
      - The test phase still begins at episode_starts[8], i.e. after all
        8 familiarization trials in the source movie.
      - Sample the test at `test_fps`.
      - Remove the final `test_end_trim_seconds` seconds so the fade-to-black
        does not contribute to the BIB max-error score.
    """
    if not (1 <= n_fam_trials <= 8):
        raise ValueError("n_fam_trials must be 1..8.")

    if frames_per_fam <= 0:
        raise ValueError("frames_per_fam must be positive.")

    info = parse_bib2_json(json_path)
    json_path = info["json_path"]

    expected_path = resolve_video_path(
        info["expected_raw_path"],
        json_path,
        project_root,
    )

    unexpected_path = resolve_video_path(
        info["unexpected_raw_path"],
        json_path,
        project_root,
    )

    vr_e = VideoReader(str(expected_path), ctx=cpu(0))
    vr_u = VideoReader(str(unexpected_path), ctx=cpu(0))

    starts = info["episode_starts"]

    source_fps_e = float(vr_e.get_avg_fps())
    source_fps_u = float(vr_u.get_avg_fps())

    # ------------------------------------------------------------------
    # Familiarization
    # ------------------------------------------------------------------
    # Familiarization sampling happens from source frames, so convert the
    # trim duration to source-video frames before sampling.
    fam_trim_source_frames = int(
        round(float(fam_end_trim_seconds) * source_fps_e)
    )

    fams = []

    for i in range(n_fam_trials):
        fam_start = int(starts[i])
        fam_end = int(starts[i + 1])
        trimmed_fam_end = fam_end - fam_trim_source_frames

        if trimmed_fam_end <= fam_start:
            raise ValueError(
                "Fam %d is too short to trim %.2f seconds."
                % (i + 1, fam_end_trim_seconds)
            )

        fam = sample_uniform_frames(
            vr_e,
            start_frame=fam_start,
            end_frame=trimmed_fam_end,
            n_frames=frames_per_fam,
        )
        fams.append(fam)

        print(
            "  Fam %d: source %d:%d -> trimmed %d:%d -> sampled %d frames"
            % (
                i + 1,
                fam_start,
                fam_end,
                fam_start,
                trimmed_fam_end,
                len(fam),
            )
        )

    fam_context = torch.cat(fams, dim=0)

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------
    test_start = int(starts[8])

    expected_test = sample_segment_at_fps(
        vr_e,
        start_frame=test_start,
        end_frame=len(vr_e),
        target_fps=test_fps,
    )

    unexpected_test = sample_segment_at_fps(
        vr_u,
        start_frame=test_start,
        end_frame=len(vr_u),
        target_fps=test_fps,
    )

    # Test frames have already been resampled to test_fps, so trim using
    # the sampled-frame rate.
    test_trim_frames = int(
        round(float(test_end_trim_seconds) * float(test_fps))
    )

    if test_trim_frames > 0:
        if len(expected_test) <= test_trim_frames:
            raise ValueError(
                "Expected test is too short to trim %.2f seconds."
                % test_end_trim_seconds
            )
        if len(unexpected_test) <= test_trim_frames:
            raise ValueError(
                "Unexpected test is too short to trim %.2f seconds."
                % test_end_trim_seconds
            )

        expected_test = expected_test[:-test_trim_frames]
        unexpected_test = unexpected_test[:-test_trim_frames]

    print(
        "  Trimmed %.2f s from each fam trial and %.2f s from test."
        % (fam_end_trim_seconds, test_end_trim_seconds)
    )
    print("  Fam context:", tuple(fam_context.shape))
    print("  Expected test:", tuple(expected_test.shape))
    print("  Unexpected test:", tuple(unexpected_test.shape))

    return {
        "fams": fams,
        "fam_context": fam_context,
        "expected": expected_test,
        "unexpected": unexpected_test,
        "expected_path": expected_path,
        "unexpected_path": unexpected_path,
        "episode_starts": starts,
        "source_fps_expected": source_fps_e,
        "source_fps_unexpected": source_fps_u,
        "fam_end_trim_seconds": fam_end_trim_seconds,
        "test_end_trim_seconds": test_end_trim_seconds,
    }


def _processor_pixels(processor, videos, device, dtype):
    inputs = processor(videos, return_tensors="pt")
    return inputs["pixel_values_videos"].to(device=device, dtype=dtype)


@torch.inference_mode()
def predict_next_tubelet_error(
    fam_context,
    current_test,
    target_test,
    processor,
    model,
):

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    tubelet = int(model.config.tubelet_size)

    # ---------------------------------------------------------
    # Validate familiarization context
    # ---------------------------------------------------------

    if len(fam_context) % tubelet != 0:
        raise ValueError(
            "fam_context length must be divisible by tubelet_size."
        )

    # ---------------------------------------------------------
    # Validate cumulative test history
    # ---------------------------------------------------------

    if len(current_test) < tubelet:
        raise ValueError(
            f"current_test must contain at least {tubelet} frames; "
            f"got {len(current_test)}."
        )

    if len(current_test) % tubelet != 0:
        raise ValueError(
            f"current_test must contain a multiple of {tubelet} frames; "
            f"got {len(current_test)}."
        )

    # Target is always the NEXT tubelet
    if len(target_test) != tubelet:
        raise ValueError(
            f"target_test must contain exactly {tubelet} frames; "
            f"got {len(target_test)}."
        )

    # ---------------------------------------------------------
    # Build context
    # ---------------------------------------------------------

    context_video = torch.cat(
        [
            fam_context,
            current_test,
        ],
        dim=0,
    )

    full_video = torch.cat(
        [
            fam_context,
            current_test,
            target_test,
        ],
        dim=0,
    )

    context_pixels = _processor_pixels(
        processor,
        [context_video],
        device,
        dtype,
    )

    full_pixels = _processor_pixels(
        processor,
        [full_video],
        device,
        dtype,
    )

    # ---------------------------------------------------------
    # Encode context
    # ---------------------------------------------------------

    context_hidden = model.encoder(
        pixel_values_videos=context_pixels
    ).last_hidden_state

    batch_size, n_context_tokens, _ = (
        context_hidden.shape
    )

    grid = int(
        model.config.crop_size
        // model.config.patch_size
    )

    tokens_per_tubelet = grid * grid

    context_ids = torch.arange(
        0,
        n_context_tokens,
        device=device,
        dtype=torch.long,
    ).unsqueeze(0).repeat(
        batch_size,
        1,
    )

    target_ids = torch.arange(
        n_context_tokens,
        n_context_tokens + tokens_per_tubelet,
        device=device,
        dtype=torch.long,
    ).unsqueeze(0).repeat(
        batch_size,
        1,
    )

    # ---------------------------------------------------------
    # Predict next tubelet
    # ---------------------------------------------------------

    predicted = model.predictor(
        encoder_hidden_states=context_hidden,
        context_mask=[context_ids],
        target_mask=[target_ids],
    ).last_hidden_state

    # ---------------------------------------------------------
    # Actual next-tubelet representation
    # ---------------------------------------------------------

    full_hidden = model.encoder(
        pixel_values_videos=full_pixels
    ).last_hidden_state

    actual = full_hidden[
        :,
        n_context_tokens:
        n_context_tokens + tokens_per_tubelet,
        :
    ]

    actual = F.layer_norm(
        actual.float(),
        (actual.shape[-1],),
    )

    error = (
        predicted.float()
        - actual
    ).abs().mean()

    return float(error.cpu())


@torch.inference_mode()
def evaluate_test_video_bib_style(
    fam_context,
    test_frames,
    processor,
    model,
):
    """
    Cumulative test evaluation.

    Step 0:
        fam + test[0:2]
        -> predict test[2:4]

    Step 1:
        fam + test[0:4]
        -> predict test[4:6]

    Step 2:
        fam + test[0:6]
        -> predict test[6:8]

    etc.
    """

    tubelet = int(model.config.tubelet_size)

    step_errors = []
    step_starts = []

    # Move forward one COMPLETE tubelet at a time.
    for target_start in range(
        tubelet,
        len(test_frames) - tubelet + 1,
        tubelet,
    ):

        # All test frames observed so far
        current_test = test_frames[
            :target_start
        ]

        # Next unseen tubelet
        target_test = test_frames[
            target_start:
            target_start + tubelet
        ]

        error = predict_next_tubelet_error(
            fam_context=fam_context,
            current_test=current_test,
            target_test=target_test,
            processor=processor,
            model=model,
        )

        step_errors.append(error)
        step_starts.append(target_start)

    step_errors = np.asarray(
        step_errors,
        dtype=np.float64,
    )

    step_starts = np.asarray(
        step_starts,
        dtype=np.int64,
    )

    max_index = int(
        step_errors.argmax()
    )

    return {
        "step_errors": step_errors,

        "step_starts": step_starts,

        "surprise": float(
            step_errors[max_index]
        ),

        "max_step_index": max_index,

        "max_step_start_frame": int(
            step_starts[max_index]
        ),

        "mean_error": float(
            step_errors.mean()
        ),
    }


def task_name_from_json(json_path):
    stem = Path(json_path).stem
    for prefix in ("index_dict_eval_", "index_dict_"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


def evaluate_bib2_json(
    json_path,
    processor,
    model,
    project_root=None,
    n_fam_trials=DEFAULT_N_FAM_TRIALS,
    frames_per_fam=DEFAULT_FRAMES_PER_FAM,
    test_fps=DEFAULT_TEST_FPS,
    test_stride=DEFAULT_TEST_STRIDE,
    fam_end_trim_seconds=DEFAULT_FAM_END_TRIM_SECONDS,
    test_end_trim_seconds=DEFAULT_TEST_END_TRIM_SECONDS,
):
    item = load_bib2_item(
        json_path=json_path,
        project_root=project_root,
        n_fam_trials=n_fam_trials,
        frames_per_fam=frames_per_fam,
        test_fps=test_fps,
        fam_end_trim_seconds=fam_end_trim_seconds,
        test_end_trim_seconds=test_end_trim_seconds,
    )

    expected_eval = evaluate_test_video_bib_style(
    item["fam_context"],
    item["expected"],
    processor,
    model,
    )
    
    unexpected_eval = evaluate_test_video_bib_style(
        item["fam_context"],
        item["unexpected"],
        processor,
        model,
    )

    expected_surprise = expected_eval["surprise"]
    unexpected_surprise = unexpected_eval["surprise"]
    voe_gap = unexpected_surprise - expected_surprise

    return {
        "task": task_name_from_json(json_path),
        "json_path": str(Path(json_path).resolve()),
        "expected_video": str(item["expected_path"]),
        "unexpected_video": str(item["unexpected_path"]),
        "n_fam_trials": n_fam_trials,
        "fam_trials_used": list(range(1, n_fam_trials + 1)),
        "frames_per_fam": frames_per_fam,
        "total_fam_frames": int(len(item["fam_context"])),
        "fam_end_trim_seconds": fam_end_trim_seconds,
        "test_end_trim_seconds": test_end_trim_seconds,
        "test_fps": test_fps,
        "test_stride": test_stride,
        "expected_surprise": expected_surprise,
        "unexpected_surprise": unexpected_surprise,
        "expected_score": -expected_surprise,
        "unexpected_score": -unexpected_surprise,
        "voe_gap": voe_gap,
        "prefers_expected": bool(expected_surprise < unexpected_surprise),
        "expected_max_step_index": expected_eval["max_step_index"],
        "unexpected_max_step_index": unexpected_eval["max_step_index"],
        "expected_max_step_start_frame": expected_eval["max_step_start_frame"],
        "unexpected_max_step_start_frame": unexpected_eval["max_step_start_frame"],
        "expected_mean_error": expected_eval["mean_error"],
        "unexpected_mean_error": unexpected_eval["mean_error"],
        "expected_step_errors": expected_eval["step_errors"],
        "unexpected_step_errors": unexpected_eval["step_errors"],
    }


CSV_FIELDS = [
    "seed", "model_dir",
    "task", "json_path", "expected_video", "unexpected_video",
    "n_fam_trials", "fam_trials_used", "frames_per_fam", "total_fam_frames",
    "fam_end_trim_seconds", "test_end_trim_seconds",
    "test_fps", "test_stride",
    "expected_surprise", "unexpected_surprise",
    "expected_score", "unexpected_score", "voe_gap", "prefers_expected",
    "expected_max_step_index", "unexpected_max_step_index",
    "expected_max_step_start_frame", "unexpected_max_step_start_frame",
    "expected_mean_error", "unexpected_mean_error",
    "expected_step_errors", "unexpected_step_errors",
]


def _csv_value(value):
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist())
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value))
    return value


def result_to_csv_row(result):
    return {
        field: _csv_value(result.get(field, ""))
        for field in CSV_FIELDS
    }


def write_results_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)




def resolve_seed_model_dir(model_root, seed, model_dir_template=None):
    """Resolve model source for a seed.

    Priority:
      1. final_hf_model
      2. epoch_002.pt if final_hf_model does not yet exist
    """
    root = Path(model_root)
    run_dirs = []

    if model_dir_template:
        run_dirs.append(root / model_dir_template.format(seed=seed))

    run_dirs.append(root / ("vjepa2_bib2_2gpu_seed%d" % seed))
    run_dirs.append(root / ("vjepa2_bib2_2gpu_seed%d_test" % seed))

    # De-duplicate while preserving order.
    unique_dirs = []
    seen = set()
    for d in run_dirs:
        d = d.expanduser()
        key = str(d)
        if key not in seen:
            unique_dirs.append(d)
            seen.add(key)

    # Prefer a completed final Hugging Face model from any recognized run dir.
    for d in unique_dirs:
        candidate = d / "final_hf_model"
        if candidate.is_dir():
            return candidate.resolve()

    # Otherwise fall back to the completed epoch-2 checkpoint.
    for d in unique_dirs:
        candidate = d / "epoch_002.pt"
        if candidate.is_file():
            print(
                "WARNING: final_hf_model not found for seed %d; "
                "falling back to epoch_002.pt" % seed
            )
            return candidate.resolve()

    attempted = []
    for d in unique_dirs:
        attempted.append(d / "final_hf_model")
        attempted.append(d / "epoch_002.pt")
    attempted_text = "\n".join("  - %s" % c for c in attempted)
    raise FileNotFoundError(
        "Could not find final_hf_model or epoch_002.pt for seed %d. Tried:\n%s"
        % (seed, attempted_text)
    )

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the final fine-tuned BabyView V-JEPA2 models for multiple "
            "seeds on BIB2 and save one combined CSV."
        )
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--json", type=str, help="One BIB2 index JSON.")
    source.add_argument("--json-dir", type=str, help="Directory of BIB2 JSONs.")

    parser.add_argument("--pattern", default="index_dict_eval_*.json")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--device", default=None)

    parser.add_argument(
        "--model-root",
        default="/scratch/sy3494/saved_models",
        help="Directory containing the per-seed fine-tuning output directories.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
        help="Seeds to evaluate. Default: 42 43 44 45 46.",
    )
    parser.add_argument(
        "--model-dir-template",
        default="vjepa2_bib2_2gpu_seed{seed}",
        help=(
            "Per-seed directory name under --model-root. Must contain {seed}. "
            "The script also falls back to a *_test directory if needed."
        ),
    )

    # Keep the user's current evaluation defaults exactly as in the original script.
    parser.add_argument("--n-fam-trials", type=int, default=8)
    parser.add_argument("--frames-per-fam", type=int, default=90)
    parser.add_argument("--test-fps", type=float, default=25.0)
    parser.add_argument(
        "--fam-end-trim-seconds",
        type=float,
        default=DEFAULT_FAM_END_TRIM_SECONDS,
        help=(
            "Seconds removed from the end of EACH familiarization trial "
            "before frame sampling. Default 2.0."
        ),
    )
    parser.add_argument(
        "--test-end-trim-seconds",
        type=float,
        default=DEFAULT_TEST_END_TRIM_SECONDS,
        help=(
            "Seconds removed from the end of each test video before "
            "evaluation. Default 2.0."
        ),
    )
    parser.add_argument(
        "--test-stride",
        type=int,
        default=1,
        help="Preserved from the original evaluator.",
    )
    parser.add_argument(
        "--output",
        default="vjepa2_bib2_final_5seeds_results.csv",
    )

    args = parser.parse_args()

    if args.frames_per_fam % 2 != 0:
        raise ValueError("--frames-per-fam must be even for tubelet_size=2.")

    if "{seed}" not in args.model_dir_template:
        raise ValueError("--model-dir-template must contain {seed}.")

    if args.json is not None:
        jsons = [Path(args.json)]
    else:
        jsons = sorted(Path(args.json_dir).glob(args.pattern))

    if not jsons:
        if args.json_dir is not None:
            print(Path(args.json_dir))
            print(Path(args.json_dir).glob(args.pattern))
        raise FileNotFoundError("No JSON files matched.")

    print(
        "Protocol: first %d familiarizations x %d frames = %d fam frames"
        % (
            args.n_fam_trials,
            args.frames_per_fam,
            args.n_fam_trials * args.frames_per_fam,
        )
    )
    print(
        "Test: %.2f fps, stride=%d, score=max prediction error"
        % (args.test_fps, args.test_stride)
    )
    print(
        "Trimming: %.2f s from each fam trial, %.2f s from test"
        % (args.fam_end_trim_seconds, args.test_end_trim_seconds)
    )
    print("Seeds:", args.seeds)

    rows = []
    per_seed_summary = []

    for seed_index, seed in enumerate(args.seeds, 1):
        model_dir = resolve_seed_model_dir(
            args.model_root,
            seed,
            args.model_dir_template,
        )

        print("\n" + "=" * 72)
        print(
            "[%d/%d seeds] Seed %d"
            % (seed_index, len(args.seeds), seed)
        )
        print("Loading model source:", model_dir)
        print("=" * 72)

        processor, model = load_model(str(model_dir), args.device)

        n_success = 0
        n_prefers_expected = 0

        for i, json_path in enumerate(jsons, 1):
            print(
                "\n[seed %d | %d/%d] %s"
                % (seed, i, len(jsons), json_path)
            )

            try:
                result = evaluate_bib2_json(
                    json_path,
                    processor,
                    model,
                    project_root=args.project_root,
                    n_fam_trials=args.n_fam_trials,
                    frames_per_fam=args.frames_per_fam,
                    test_fps=args.test_fps,
                    test_stride=args.test_stride,
                    fam_end_trim_seconds=args.fam_end_trim_seconds,
                    test_end_trim_seconds=args.test_end_trim_seconds,
                )

                result["seed"] = seed
                result["model_dir"] = str(model_dir)

                print("  expected max error   = %.6f" % result["expected_surprise"])
                print("  unexpected max error = %.6f" % result["unexpected_surprise"])
                print("  VOE gap              = %.6f" % result["voe_gap"])
                print("  prefers expected     = %s" % result["prefers_expected"])

                rows.append(result_to_csv_row(result))
                n_success += 1
                n_prefers_expected += int(result["prefers_expected"])

            except Exception as exc:
                print("  ERROR: %s: %s" % (type(exc).__name__, exc))
                row = {field: "" for field in CSV_FIELDS}
                row["seed"] = seed
                row["model_dir"] = str(model_dir)
                row["task"] = task_name_from_json(json_path)
                row["json_path"] = str(Path(json_path).resolve())
                row["prefers_expected"] = "ERROR: %s" % exc
                rows.append(row)

            # Save incrementally so completed seeds/tasks survive interruption.
            write_results_csv(rows, args.output)

        if n_success:
            acc = n_prefers_expected / float(n_success)
            per_seed_summary.append((seed, n_prefers_expected, n_success, acc))
            print(
                "\nSeed %d pairwise relative accuracy: %d/%d = %.2f%%"
                % (seed, n_prefers_expected, n_success, 100.0 * acc)
            )

        # Free GPU memory before loading the next seed.
        del model
        del processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nSaved combined results:", Path(args.output).resolve())

    if per_seed_summary:
        print("\nPer-seed summary:")
        for seed, n_pref, n_total, acc in per_seed_summary:
            print(
                "  seed %d: %d/%d = %.2f%%"
                % (seed, n_pref, n_total, 100.0 * acc)
            )


if __name__ == "__main__":
    main()
