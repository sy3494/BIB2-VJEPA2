#!/usr/bin/env python3
"""Two-GPU distributed fine-tuning of BabyView V-JEPA2 on BIB2 with two self-supervised objectives.

Modes
-----
masked:
  Native-style JEPA domain adaptation on continuous BIB2 clips. The online
  encoder sees only visible patch tokens; the predictor predicts masked latent
  targets produced by an EMA target encoder.

fam_future:
  BIB2 task-structured adaptation. Sampled familiarization trials + cumulative
  observed test history predict the NEXT test tubelet in latent space.

mixed:
  One-stage multi-objective training in which EVERY training data_tuple is
  used for BOTH objectives before the optimizer update:
    A) masked JEPA on a continuous clip from that tuple
    B) familiarization -> future prediction from that same tuple
  The two gradients are combined with configurable loss weights.

FULL-DATASET EPOCHS
-------------------
One epoch = one shuffled visit to every training data_tuple exactly once.
There is no random-with-replacement steps_per_epoch.

In mixed mode, both objectives therefore see the ENTIRE training set every
epoch. For example, with 32,000 training tuples:
  masked JEPA sees 32,000 tuples
  fam->future sees 32,000 tuples
  examples are split across GPU ranks and grouped into per-GPU mini-batches.
  With 2 GPUs and batch_size=8 PER GPU, the effective global batch is 16,
  so 32,000 tuples produce 2,000 synchronized optimizer updates.

The tuple order is reshuffled each epoch. Stochastic clip locations, masks,
and test-history lengths also change across training epochs.

Training and validation may live in the same root directory. By default the
script uses index_dict_train_*.json for gradient updates and
index_dict_val_*.json for validation. index_dict_test_*.json is excluded.

Expected JSON shape:
  {"data_tuples": [[video_path, [fam1_start,...,fam8_start,test_start], ...], ...]}

Method A is a compact Hugging Face implementation of the core JEPA pattern. It
uses practical spatial block masks spanning time; it does not reproduce Meta's
full distributed multi-mask collator exactly.
"""
from __future__ import annotations

import argparse, copy, csv, json, math, os, random, sys, time
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# Make Slurm logs update immediately even when stdout is redirected to a file.
try:
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
    sys.stderr.reconfigure(line_buffering=True, write_through=True)
except Exception:
    pass

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from decord import VideoReader, cpu
from transformers import AutoModel, AutoVideoProcessor

DEFAULT_MODEL_ID = "awwkl/vjepa2-vitl-fpc16-256-babyview-bs3072-e140"
DEFAULT_CLIP_FRAMES = 16
DEFAULT_CLIP_FPS = 4.0
DEFAULT_MASK_RATIO = 0.60
DEFAULT_N_FAM_TRIALS = 4
DEFAULT_FRAMES_PER_FAM = 16
DEFAULT_TEST_FPS = 4.0
DEFAULT_FAM_END_TRIM_SECONDS = 2.0
DEFAULT_TEST_END_TRIM_SECONDS = 2.0
DEFAULT_PREDICTOR_LR = 1e-4
DEFAULT_ENCODER_LR = 1e-5
DEFAULT_WEIGHT_DECAY = 0.04
DEFAULT_EMA = 0.99925

@dataclass
class TrainConfig:
    train_json_dir: str
    project_root: Optional[str] = None
    val_json_dir: Optional[str] = None
    train_pattern: str = "index_dict_train_*.json"
    val_pattern: str = "index_dict_val_*.json"
    model_id: str = DEFAULT_MODEL_ID
    output_dir: str = "vjepa2_bib2_finetune"
    mode: str = "mixed"
    device: Optional[str] = None
    epochs: int = 1
    # None/0 means use the ENTIRE split. Positive values are debugging caps.
    max_train_examples: Optional[int] = None
    max_val_examples: Optional[int] = None
    grad_accum_steps: int = 1
    # Number of BIB2 tuples grouped into one training batch PER GPU.
    batch_size: int = 8
    # Fam->future uses the same tuples but may be split into smaller
    # microbatches because the 8-familiarization context is memory-heavy.
    fam_microbatch_size: int = 2
    # Print every N BATCHES, not every N tuples.
    print_every: int = 10
    seed: int = 42
    predictor_lr: float = DEFAULT_PREDICTOR_LR
    encoder_lr: float = DEFAULT_ENCODER_LR
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    warmup_fraction: float = 0.05
    ema: float = DEFAULT_EMA
    max_grad_norm: float = 1.0
    # 0 = predictor only; N = last N encoder blocks too; -1 = full encoder
    unfreeze_last_n: int = 0
    gradient_checkpointing: bool = False
    # Method A
    clip_frames: int = DEFAULT_CLIP_FRAMES
    clip_fps: float = DEFAULT_CLIP_FPS
    mask_ratio: float = DEFAULT_MASK_RATIO
    # Method B
    n_fam_trials: int = DEFAULT_N_FAM_TRIALS
    frames_per_fam: int = DEFAULT_FRAMES_PER_FAM
    test_fps: float = DEFAULT_TEST_FPS
    min_history_tubelets: int = 1
    max_history_tubelets: Optional[int] = None
    fam_end_trim_seconds: float = DEFAULT_FAM_END_TRIM_SECONDS
    test_end_trim_seconds: float = DEFAULT_TEST_END_TRIM_SECONDS
    # mixed mode: every tuple receives BOTH objectives.
    # The effective fam_future weight is 1 - mixed_masked_weight.
    mixed_masked_weight: float = 0.5
    save_every_epoch: int = 1
    resume: Optional[str] = None


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def resolve_video_path(raw_path, json_path, project_root=None):
    raw = Path(raw_path); candidates = []
    if raw.is_absolute(): candidates.append(raw)
    if project_root is not None: candidates.append(Path(project_root) / raw)
    candidates += [Path.cwd()/raw, Path(json_path).parent/raw, Path(json_path).parent.parent/raw]
    for c in candidates:
        c = c.resolve()
        if c.exists(): return c
    raise FileNotFoundError("Could not resolve %s. Tried:\n%s" % (raw_path, "\n".join(str(x.resolve()) for x in candidates)))


def read_bib_json(json_path):
    json_path = Path(json_path)
    with open(json_path) as f: meta = json.load(f)
    examples = []
    for ti, item in enumerate(meta.get("data_tuples", [])):
        if len(item) < 2 or len(item[1]) < 9: continue
        examples.append({
            "json_path": json_path,
            "source_file": json_path.name,
            "task": json_path.stem,
            "tuple_idx": ti,
            "raw_video_path": item[0],
            "starts": [int(x) for x in item[1][:9]],
        })
    return examples


class VideoReaderCache:
    def __init__(self, max_items=8): self.max_items=max_items; self.cache=OrderedDict()
    def get(self, path):
        k=str(path)
        if k in self.cache:
            vr=self.cache.pop(k); self.cache[k]=vr; return vr
        vr=VideoReader(k, ctx=cpu(0)); self.cache[k]=vr
        while len(self.cache)>self.max_items: self.cache.popitem(last=False)
        return vr


class BIB2Sampler:
    """BIB2 split with explicit per-example access.

    Training code shuffles indices externally, so each tuple is visited exactly
    once per epoch. Stochasticity is only inside the transformation applied to
    that tuple (clip location, mask, or test-history length).
    """
    def __init__(self, json_dir, pattern="*.json", project_root=None, seed=42, cache_size=8):
        self.json_dir = Path(json_dir)
        self.project_root = project_root
        self.seed = int(seed)
        self.rng = random.Random(seed)
        self.cache = VideoReaderCache(cache_size)

        files = sorted(self.json_dir.rglob(pattern))
        if not files:
            raise FileNotFoundError(
                f"No JSON files matched {pattern!r} under {self.json_dir}"
            )

        self.examples = []
        self.file_counts = {}
        for p in files:
            file_examples = read_bib_json(p)
            self.file_counts[p.name] = len(file_examples)
            self.examples.extend(file_examples)

        if not self.examples:
            raise ValueError("No usable data_tuples with >=9 episode starts.")

        if int(os.environ.get("RANK", "0")) == 0:
            print(f"Loaded {len(self.examples)} data tuples from {len(files)} JSON files", flush=True)
            for name, count in self.file_counts.items():
                print(f"  {name}: {count}", flush=True)

    def __len__(self):
        return len(self.examples)

    def reseed(self, seed):
        self.rng.seed(int(seed))

    def _open_index(self, example_index):
        ex = self.examples[int(example_index)]
        path = resolve_video_path(
            ex["raw_video_path"],
            ex["json_path"],
            self.project_root,
        )
        return ex, path, self.cache.get(path)

    def _bounds(self, ex, vr, ep, fam_trim, test_trim):
        fps = float(vr.get_avg_fps())
        s = ex["starts"]
        if ep < 8:
            start = s[ep]
            end = s[ep + 1] - int(round(fam_trim * fps))
        else:
            start = s[8]
            end = len(vr) - int(round(test_trim * fps))

        start = max(0, int(start))
        end = min(int(end), len(vr))
        if end <= start:
            raise ValueError(f"Invalid trimmed segment {start}:{end}")
        return start, end, fps

    def sample_continuous_clip(
        self,
        example_index,
        clip_frames=16,
        clip_fps=4.0,
        fam_trim_seconds=2.0,
        test_trim_seconds=2.0,
    ):
        """Method A: make one stochastic continuous clip from THIS tuple."""
        ex, path, vr = self._open_index(example_index)

        # Randomize episode order, but never substitute another tuple.
        episodes = list(range(9))
        self.rng.shuffle(episodes)

        for ep in episodes:
            start, end, fps = self._bounds(
                ex, vr, ep, fam_trim_seconds, test_trim_seconds
            )
            step = fps / float(clip_fps)
            span = (clip_frames - 1) * step
            max_first = (end - 1) - span
            if max_first < start:
                continue

            first = self.rng.uniform(start, max_first)
            ids = np.rint(
                first + np.arange(clip_frames) * step
            ).astype(np.int64)
            ids = np.clip(ids, start, end - 1)

            arr = vr.get_batch(ids).asnumpy()
            return {
                "frames": torch.from_numpy(arr).permute(0, 3, 1, 2),
                "video_path": str(path),
                "episode_idx": ep,
                "example_index": int(example_index),
            }

        raise RuntimeError(
            f"Tuple {example_index} has no episode long enough for "
            f"{clip_frames} frames at {clip_fps} fps."
        )

    def sample_fam_future(
        self,
        example_index,
        n_fam_trials=4,
        frames_per_fam=16,
        test_fps=4.0,
        fam_trim_seconds=2.0,
        test_trim_seconds=2.0,
        tubelet_size=2,
        min_history_tubelets=1,
        max_history_tubelets=None,
        history_tubelets=None,
    ):
        """Method B: fam context + sampled cumulative history for THIS tuple."""
        if not 1 <= n_fam_trials <= 8:
            raise ValueError("n_fam_trials must be 1..8")
        if frames_per_fam % tubelet_size:
            raise ValueError("frames_per_fam must be divisible by tubelet_size")

        ex, path, vr = self._open_index(example_index)
        fps = float(vr.get_avg_fps())
        starts = ex["starts"]

        fams = []
        trim = int(round(fam_trim_seconds * fps))
        for i in range(n_fam_trials):
            a = int(starts[i])
            b = int(starts[i + 1]) - trim
            if b <= a:
                raise ValueError(f"Fam {i + 1} too short after trim")
            ids = np.rint(
                np.linspace(a, b - 1, frames_per_fam)
            ).astype(np.int64)
            ids = np.clip(ids, a, b - 1)
            fams.append(
                torch.from_numpy(vr.get_batch(ids).asnumpy()).permute(0, 3, 1, 2)
            )

        fam_context = torch.cat(fams, 0)

        a = int(starts[8])
        b = len(vr) - int(round(test_trim_seconds * fps))
        if b <= a:
            raise ValueError("Test too short after trim")

        ids = np.rint(
            np.arange(a, b, fps / float(test_fps))
        ).astype(np.int64)
        ids = np.unique(np.clip(ids, a, b - 1))

        if len(ids) < 2 * tubelet_size:
            raise ValueError("Too few test frames")

        test = torch.from_numpy(vr.get_batch(ids).asnumpy()).permute(0, 3, 1, 2)

        max_hist = len(test) // tubelet_size - 1
        if max_history_tubelets is not None:
            max_hist = min(max_hist, int(max_history_tubelets))
        min_hist = max(1, int(min_history_tubelets))
        if max_hist < min_hist:
            raise ValueError("Not enough test frames for requested history")

        if history_tubelets is None:
            h = self.rng.randint(min_hist, max_hist)
        else:
            h = int(history_tubelets)
            if h < min_hist or h > max_hist:
                raise ValueError(
                    f"Requested history_tubelets={h}, but valid range is "
                    f"{min_hist}..{max_hist} for tuple {example_index}."
                )
        hf = h * tubelet_size

        return {
            "fam_context": fam_context,
            "current_test": test[:hf],
            "target_test": test[hf:hf + tubelet_size],
            "history_tubelets": h,
            "video_path": str(path),
            "example_index": int(example_index),
        }

    def max_valid_history_tubelets(
        self,
        example_index,
        test_fps=4.0,
        test_trim_seconds=2.0,
        tubelet_size=2,
        max_history_tubelets=None,
    ):
        """Largest valid cumulative test-history length for one tuple."""
        ex, _, vr = self._open_index(example_index)
        fps = float(vr.get_avg_fps())
        a = int(ex["starts"][8])
        b = len(vr) - int(round(test_trim_seconds * fps))
        if b <= a:
            raise ValueError("Test too short after trim")
        ids = np.rint(
            np.arange(a, b, fps / float(test_fps))
        ).astype(np.int64)
        ids = np.unique(np.clip(ids, a, b - 1))
        max_hist = len(ids) // int(tubelet_size) - 1
        if max_history_tubelets is not None:
            max_hist = min(max_hist, int(max_history_tubelets))
        return int(max_hist)

def load_model_and_processor(model_id, device=None):
    if device is None: device="cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32
    processor=AutoVideoProcessor.from_pretrained(model_id)
    model=AutoModel.from_pretrained(model_id, torch_dtype=dtype, attn_implementation="sdpa").to(device)
    if not hasattr(model,"encoder") or not hasattr(model,"predictor"): raise AttributeError("Model needs encoder and predictor")
    return processor,model


def configure_trainable_parameters(model, unfreeze_last_n=0):
    for p in model.parameters(): p.requires_grad=False
    for p in model.predictor.parameters(): p.requires_grad=True
    if unfreeze_last_n==-1:
        for p in model.encoder.parameters(): p.requires_grad=True
    elif unfreeze_last_n>0:
        n=min(unfreeze_last_n,len(model.encoder.layer))
        for layer in model.encoder.layer[-n:]:
            for p in layer.parameters(): p.requires_grad=True
        for p in model.encoder.layernorm.parameters(): p.requires_grad=True
    elif unfreeze_last_n!=0: raise ValueError("unfreeze_last_n must be -1 or >=0")
    npred=sum(p.numel() for p in model.predictor.parameters() if p.requires_grad)
    nenc=sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"Trainable predictor parameters: {npred:,}", flush=True)
        print(f"Trainable encoder parameters:   {nenc:,}", flush=True)


def build_optimizer(model,cfg):
    groups=[]
    pp=[p for p in model.predictor.parameters() if p.requires_grad]
    ep=[p for p in model.encoder.parameters() if p.requires_grad]
    if pp: groups.append({"params":pp,"lr":cfg.predictor_lr,"name":"predictor"})
    if ep: groups.append({"params":ep,"lr":cfg.encoder_lr,"name":"encoder"})
    return torch.optim.AdamW(groups,weight_decay=cfg.weight_decay,betas=(0.9,0.95))


def build_scheduler(opt,total_updates,warmup_fraction):
    warm=int(round(total_updates*warmup_fraction)); warm=min(warm,max(0,total_updates-1))
    def f(step):
        if total_updates<=1:return 1.0
        if warm and step<warm:return float(step+1)/warm
        prog=min(max((step-warm)/float(max(1,total_updates-warm)),0.0),1.0)
        return 0.5*(1+math.cos(math.pi*prog))
    return torch.optim.lr_scheduler.LambdaLR(opt,f)


@torch.no_grad()
def ema_update(target,online,tau):
    for pt,po in zip(target.parameters(),online.parameters()): pt.data.mul_(tau).add_(po.data,alpha=1-tau)


def processor_pixels(processor,videos,model):
    inp=processor(videos,return_tensors="pt")
    return inp["pixel_values_videos"].to(device=next(model.parameters()).device,dtype=next(model.parameters()).dtype)


def gather_tokens(x,ids):
    ids=ids.to(x.device); return torch.gather(x,1,ids.unsqueeze(-1).expand(-1,-1,x.size(-1)))


def encode_visible_tokens(encoder,pixels,visible_ids):
    # Critical for Method A: gather BEFORE transformer layers so masked target
    # content is never exposed to the online encoder.
    h=encoder.embeddings(pixels); h=gather_tokens(h,visible_ids)
    for layer in encoder.layer: h=layer(h,position_mask=visible_ids)[0]
    return encoder.layernorm(h)


def predict_from_visible_tokens(
    predictor,
    visible_hidden,
    visible_ids,
    target_ids,
    mask_index=1,
):
    """Predict target tokens without relying on HF internal sort_tokens APIs."""
    B = visible_hidden.size(0)
    Nc = visible_hidden.size(1)
    Nt = target_ids.size(1)

    h = predictor.embeddings.predictor_embeddings(visible_hidden)

    idx = mask_index % predictor.embeddings.num_mask_tokens
    target = predictor.embeddings.mask_tokens[idx].repeat(B, Nt, 1)

    # Original concatenation order: [context tokens | target tokens]
    h = torch.cat([h, target], dim=1)
    pos = torch.cat([visible_ids, target_ids], dim=1)

    # Sort into original spatiotemporal patch order.
    order = torch.argsort(pos, dim=1)
    pos = torch.gather(pos, dim=1, index=order)
    gather_idx = order.unsqueeze(-1).expand(-1, -1, h.size(-1))
    h = torch.gather(h, dim=1, index=gather_idx)

    for layer in predictor.layer:
        h = layer(h, position_mask=pos)[0]

    h = predictor.layernorm(h)

    # Undo the token sort.
    reverse_order = torch.argsort(order, dim=1)
    reverse_idx = reverse_order.unsqueeze(-1).expand(-1, -1, h.size(-1))
    h = torch.gather(h, dim=1, index=reverse_idx)

    # Keep only target predictions and project back to encoder dimension.
    h = h[:, Nc:]
    return predictor.proj(h)

def make_block_mask_ids(config,batch_size,num_frames,target_ratio=0.60,device="cpu",generator=None):
    tubelet=int(config.tubelet_size); grid=int(config.crop_size//config.patch_size); T=num_frames//tubelet
    if num_frames%tubelet: raise ValueError("num_frames must be divisible by tubelet_size")
    spatial=grid*grid; total=T*spatial
    desired=max(1,min(spatial-1,int(round(spatial*target_ratio))))
    vis_all=[]; tgt_all=[]
    for _ in range(batch_size):
        sm=torch.zeros((grid,grid),dtype=torch.bool,device=device); attempts=0
        while int(sm.sum())<desired and attempts<200:
            attempts+=1; frac=0.25 if attempts%4==0 else 0.08; area=max(1,int(round(spatial*frac)))
            aspect=float(torch.empty(1,device=device).uniform_(0.75,1.5,generator=generator).item())
            h=max(1,min(grid,int(round(math.sqrt(area/aspect))))); w=max(1,min(grid,int(round(math.sqrt(area*aspect)))))
            y=int(torch.randint(0,max(1,grid-h+1),(1,),device=device,generator=generator).item())
            x=int(torch.randint(0,max(1,grid-w+1),(1,),device=device,generator=generator).item())
            sm[y:y+h,x:x+w]=True
        # make exact target spatial count for consistent batch shapes
        flat=sm.flatten(); ids=torch.nonzero(flat,as_tuple=False).flatten()
        if len(ids)>desired:
            ids=ids[torch.randperm(len(ids),device=device,generator=generator)[:desired]]
        elif len(ids)<desired:
            avail=torch.nonzero(~flat,as_tuple=False).flatten(); need=desired-len(ids)
            ids=torch.cat([ids,avail[torch.randperm(len(avail),device=device,generator=generator)[:need]]])
        spatial_bool=torch.zeros(spatial,dtype=torch.bool,device=device); spatial_bool[ids]=True
        target_bool=spatial_bool.repeat(T); tgt=torch.nonzero(target_bool,as_tuple=False).flatten(); vis=torch.nonzero(~target_bool,as_tuple=False).flatten()
        tgt_all.append(tgt); vis_all.append(vis)
    return torch.stack(vis_all),torch.stack(tgt_all)


def masked_jepa_loss(model,target_encoder,processor,frames,mask_ratio=0.60,torch_generator=None):
    # `frames` may be one [T,C,H,W] tensor or a list of such tensors.
    videos = frames if isinstance(frames, (list, tuple)) else [frames]
    pixels=processor_pixels(processor,videos,model); B=pixels.size(0)
    vis,tgt=make_block_mask_ids(model.config,B,pixels.size(1),mask_ratio,pixels.device,torch_generator)
    ctx=encode_visible_tokens(model.encoder,pixels,vis)
    pred=predict_from_visible_tokens(model.predictor,ctx,vis,tgt)
    with torch.no_grad():
        full=target_encoder(pixel_values_videos=pixels).last_hidden_state
        target=gather_tokens(full,tgt); target=F.layer_norm(target.float(),(target.shape[-1],))
    return (pred.float()-target).abs().mean()


def fam_future_loss(model,target_encoder,processor,fam_context,current_test,target_test):
    """Fam->future loss for one example or a same-length mini-batch.

    Each argument may be a single [T,C,H,W] tensor or a list of tensors.
    For a batch, every example must use the same history length so all
    contexts contain the same number of frames.
    """
    tubelet=int(model.config.tubelet_size)

    fams = list(fam_context) if isinstance(fam_context, (list, tuple)) else [fam_context]
    currents = list(current_test) if isinstance(current_test, (list, tuple)) else [current_test]
    targets = list(target_test) if isinstance(target_test, (list, tuple)) else [target_test]
    if not (len(fams) == len(currents) == len(targets)):
        raise ValueError("fam_context/current_test/target_test batch sizes differ")

    contexts=[]; fulls=[]
    expected_context_len=None
    for fam,cur,tgt in zip(fams,currents,targets):
        if len(fam)%tubelet or len(cur)%tubelet:
            raise ValueError("Context lengths must contain complete tubelets")
        if len(tgt)!=tubelet:
            raise ValueError("target_test must be exactly one tubelet")
        context=torch.cat([fam,cur],0)
        if expected_context_len is None:
            expected_context_len=len(context)
        elif len(context)!=expected_context_len:
            raise ValueError(
                "Batched fam->future examples need the same context length. "
                "Use a shared history_tubelets value within each microbatch."
            )
        contexts.append(context)
        fulls.append(torch.cat([context,tgt],0))

    cp=processor_pixels(processor,contexts,model)
    fp=processor_pixels(processor,fulls,model)
    ch=model.encoder(pixel_values_videos=cp).last_hidden_state
    B,N,_=ch.shape
    grid=int(model.config.crop_size//model.config.patch_size); tpt=grid*grid
    cids=torch.arange(N,device=ch.device).unsqueeze(0).repeat(B,1)
    tids=torch.arange(N,N+tpt,device=ch.device).unsqueeze(0).repeat(B,1)
    pred=model.predictor(
        encoder_hidden_states=ch,
        context_mask=[cids],
        target_mask=[tids],
    ).last_hidden_state
    with torch.no_grad():
        th=target_encoder(pixel_values_videos=fp).last_hidden_state[:,N:N+tpt,:]
        th=F.layer_norm(th.float(),(th.shape[-1],))
    return (pred.float()-th).abs().mean()


def autocast_context(model):
    dev=next(model.parameters()).device
    return torch.autocast(device_type="cuda",dtype=torch.bfloat16) if dev.type=="cuda" else nullcontext()



def compute_one_loss(
    mode,
    sampler,
    example_index,
    model,
    target_encoder,
    processor,
    cfg,
    generator,
):
    """Compute one objective for one specific BIB2 tuple."""
    tubelet = int(model.config.tubelet_size)

    if mode == "masked":
        x = sampler.sample_continuous_clip(
            example_index=example_index,
            clip_frames=cfg.clip_frames,
            clip_fps=cfg.clip_fps,
            fam_trim_seconds=cfg.fam_end_trim_seconds,
            test_trim_seconds=cfg.test_end_trim_seconds,
        )
        loss = masked_jepa_loss(
            model,
            target_encoder,
            processor,
            x["frames"],
            cfg.mask_ratio,
            generator,
        )
        return loss, x

    if mode == "fam_future":
        x = sampler.sample_fam_future(
            example_index=example_index,
            n_fam_trials=cfg.n_fam_trials,
            frames_per_fam=cfg.frames_per_fam,
            test_fps=cfg.test_fps,
            fam_trim_seconds=cfg.fam_end_trim_seconds,
            test_trim_seconds=cfg.test_end_trim_seconds,
            tubelet_size=tubelet,
            min_history_tubelets=cfg.min_history_tubelets,
            max_history_tubelets=cfg.max_history_tubelets,
        )
        loss = fam_future_loss(
            model,
            target_encoder,
            processor,
            x["fam_context"],
            x["current_test"],
            x["target_test"],
        )
        return loss, x

    raise ValueError("mode must be masked or fam_future")


def mixed_weights(cfg):
    """Return normalized masked/fam_future weights for mixed mode."""
    w_masked = float(cfg.mixed_masked_weight)
    if not 0.0 <= w_masked <= 1.0:
        raise ValueError("--mixed-masked-weight must be between 0 and 1.")
    return w_masked, 1.0 - w_masked



def sample_masked_batch(sampler, example_indices, cfg):
    samples=[]
    for example_index in example_indices:
        samples.append(
            sampler.sample_continuous_clip(
                example_index=example_index,
                clip_frames=cfg.clip_frames,
                clip_fps=cfg.clip_fps,
                fam_trim_seconds=cfg.fam_end_trim_seconds,
                test_trim_seconds=cfg.test_end_trim_seconds,
            )
        )
    return samples


def sample_fam_microbatch(sampler, example_indices, model, cfg):
    """Sample a fam->future microbatch with one shared test-history length."""
    tubelet=int(model.config.tubelet_size)
    min_hist=max(1,int(cfg.min_history_tubelets))
    max_hist=min(
        sampler.max_valid_history_tubelets(
            example_index=i,
            test_fps=cfg.test_fps,
            test_trim_seconds=cfg.test_end_trim_seconds,
            tubelet_size=tubelet,
            max_history_tubelets=cfg.max_history_tubelets,
        )
        for i in example_indices
    )
    if max_hist < min_hist:
        raise ValueError(
            f"No common valid history length for batch: min={min_hist}, max={max_hist}"
        )
    history_tubelets=sampler.rng.randint(min_hist,max_hist)

    samples=[]
    for example_index in example_indices:
        samples.append(
            sampler.sample_fam_future(
                example_index=example_index,
                n_fam_trials=cfg.n_fam_trials,
                frames_per_fam=cfg.frames_per_fam,
                test_fps=cfg.test_fps,
                fam_trim_seconds=cfg.fam_end_trim_seconds,
                test_trim_seconds=cfg.test_end_trim_seconds,
                tubelet_size=tubelet,
                min_history_tubelets=cfg.min_history_tubelets,
                max_history_tubelets=cfg.max_history_tubelets,
                history_tubelets=history_tubelets,
            )
        )
    return samples


def masked_batch_loss(model,target_encoder,processor,sampler,example_indices,cfg,generator):
    samples=sample_masked_batch(sampler,example_indices,cfg)
    loss=masked_jepa_loss(
        model,target_encoder,processor,
        [x["frames"] for x in samples],
        cfg.mask_ratio,generator,
    )
    return loss,samples


def fam_microbatch_loss(model,target_encoder,processor,sampler,example_indices,cfg):
    samples=sample_fam_microbatch(sampler,example_indices,model,cfg)
    loss=fam_future_loss(
        model,target_encoder,processor,
        [x["fam_context"] for x in samples],
        [x["current_test"] for x in samples],
        [x["target_test"] for x in samples],
    )
    return loss,samples


def iter_chunks(items, chunk_size):
    for i in range(0,len(items),chunk_size):
        yield items[i:i+chunk_size]




def setup_distributed():
    """Initialize one process per GPU under torchrun."""
    if "RANK" not in os.environ or "LOCAL_RANK" not in os.environ:
        raise RuntimeError(
            "This script must be launched with torchrun, e.g. "
            "torchrun --standalone --nproc_per_node=2 ..."
        )
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 2:
        raise RuntimeError(f"Expected WORLD_SIZE=2, got {world_size}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return rank, local_rank, world_size, torch.device(f"cuda:{local_rank}")


def rank0_print(rank, *args, **kwargs):
    if rank == 0:
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)


def cuda_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reduce_sum_tensor(values, device):
    t = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def reduce_max_tensor(values, device):
    t = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t


def sync_gradients(model, world_size, bucket_mb=128):
    """Average gradients across ranks using flattened dtype/device buckets.

    This is synchronous data-parallel gradient averaging, equivalent in update
    semantics to DDP, while keeping the current custom encoder/predictor calls
    unchanged.
    """
    grad_items = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not grad_items:
        return

    max_bytes = int(bucket_mb * 1024 * 1024)
    bucket = []
    bucket_bytes = 0
    bucket_dtype = None
    bucket_device = None

    def flush(items):
        if not items:
            return
        flat = torch._utils._flatten_dense_tensors(items)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(world_size)
        synced = torch._utils._unflatten_dense_tensors(flat, items)
        for dst, src in zip(items, synced):
            dst.copy_(src)

    for g in grad_items:
        nbytes = g.numel() * g.element_size()
        if (
            bucket
            and (
                bucket_bytes + nbytes > max_bytes
                or g.dtype != bucket_dtype
                or g.device != bucket_device
            )
        ):
            flush(bucket)
            bucket = []
            bucket_bytes = 0
        if not bucket:
            bucket_dtype = g.dtype
            bucket_device = g.device
        bucket.append(g)
        bucket_bytes += nbytes
    flush(bucket)


@torch.no_grad()
def validate_distributed(model, target_encoder, processor, sampler, cfg, rank, world_size, device):
    if sampler is None:
        return None, None

    model.eval()
    target_encoder.eval()

    n_val = len(sampler)
    if cfg.max_val_examples is not None and int(cfg.max_val_examples) > 0:
        n_val = min(n_val, int(cfg.max_val_examples))

    # Equal work on both ranks. The full 2,000-example validation split is even.
    usable = (n_val // world_size) * world_size
    if usable != n_val and rank == 0:
        print(f"WARNING: trimming validation from {n_val} to {usable} for equal rank sizes", flush=True)
    global_indices = list(range(usable))
    rank_indices = global_indices[rank::world_size]

    sampler.reseed(cfg.seed + 600_000 + rank * 10_000)
    gen = torch.Generator(device=device.type)
    gen.manual_seed(cfg.seed + 700_000 + rank * 10_000)

    if cfg.mode == "mixed":
        w_masked, w_future = mixed_weights(cfg)
    elif cfg.mode == "masked":
        w_masked, w_future = 1.0, 0.0
    elif cfg.mode == "fam_future":
        w_masked, w_future = 0.0, 1.0
    else:
        raise ValueError("mode must be masked, fam_future, or mixed")

    local_combined_sum = 0.0
    local_masked_sum = 0.0
    local_future_sum = 0.0
    local_n = local_masked_n = local_future_n = 0

    for batch_indices in iter_chunks(rank_indices, cfg.batch_size):
        bs = len(batch_indices)
        masked_value = None
        future_value = None

        if cfg.mode in ("masked", "mixed"):
            with autocast_context(model):
                loss_masked, _ = masked_batch_loss(
                    model, target_encoder, processor, sampler,
                    batch_indices, cfg, gen,
                )
            masked_value = float(loss_masked.detach().cpu())
            local_masked_sum += masked_value * bs
            local_masked_n += bs

        if cfg.mode in ("fam_future", "mixed"):
            fam_weighted_sum = 0.0
            for micro_indices in iter_chunks(batch_indices, cfg.fam_microbatch_size):
                with autocast_context(model):
                    loss_future, _ = fam_microbatch_loss(
                        model, target_encoder, processor, sampler,
                        micro_indices, cfg,
                    )
                fv = float(loss_future.detach().cpu())
                fam_weighted_sum += fv * len(micro_indices)
                local_future_sum += fv * len(micro_indices)
                local_future_n += len(micro_indices)
            future_value = fam_weighted_sum / bs

        if cfg.mode == "mixed":
            combined_value = w_masked * masked_value + w_future * future_value
        elif cfg.mode == "masked":
            combined_value = masked_value
        else:
            combined_value = future_value

        local_combined_sum += combined_value * bs
        local_n += bs

    totals = reduce_sum_tensor(
        [
            local_combined_sum, local_masked_sum, local_future_sum,
            local_n, local_masked_n, local_future_n,
        ],
        device,
    ).tolist()

    combined_sum, masked_sum, future_sum, n, masked_n, future_n = totals
    n = int(round(n)); masked_n = int(round(masked_n)); future_n = int(round(future_n))

    model.train()
    details = {
        "overall": combined_sum / max(1, n),
        "masked": masked_sum / masked_n if masked_n else None,
        "fam_future": future_sum / future_n if future_n else None,
        "n": n,
        "masked_n": masked_n,
        "fam_future_n": future_n,
    }
    return details["overall"], details

def save_checkpoint(out,model,target,opt,sched,cfg,epoch,update,best,name):
    out=Path(out); out.mkdir(parents=True,exist_ok=True); p=out/name
    torch.save({"epoch":epoch,"global_update":update,"model_state_dict":model.state_dict(),
                "target_encoder_state_dict":target.state_dict(),"optimizer_state_dict":opt.state_dict(),
                "scheduler_state_dict":sched.state_dict(),"best_val":best,"config":asdict(cfg)},p)
    return p


def load_checkpoint(path,model,target,opt=None,sched=None):
    ck=torch.load(path,map_location="cpu"); model.load_state_dict(ck["model_state_dict"]); target.load_state_dict(ck["target_encoder_state_dict"])
    if opt is not None: opt.load_state_dict(ck["optimizer_state_dict"])
    if sched is not None: sched.load_state_dict(ck["scheduler_state_dict"])
    return ck




def train(cfg: TrainConfig):
    rank, local_rank, world_size, device = setup_distributed()

    try:
        # Different stochastic transforms on each rank; pretrained weights are
        # loaded identically from the same checkpoint.
        seed_everything(cfg.seed + rank)

        if cfg.mode not in ("masked", "fam_future", "mixed"):
            raise ValueError("mode must be masked, fam_future, or mixed")
        if cfg.clip_frames % 2 or cfg.frames_per_fam % 2:
            raise ValueError("clip_frames and frames_per_fam must be even")
        if cfg.batch_size < 1 or cfg.fam_microbatch_size < 1:
            raise ValueError("batch sizes must be >= 1")
        if cfg.fam_microbatch_size > cfg.batch_size:
            raise ValueError("fam_microbatch_size cannot exceed batch_size")

        if cfg.mode == "mixed":
            w_masked, w_future = mixed_weights(cfg)
        elif cfg.mode == "masked":
            w_masked, w_future = 1.0, 0.0
        else:
            w_masked, w_future = 0.0, 1.0

        out = Path(cfg.output_dir)
        if rank == 0:
            out.mkdir(parents=True, exist_ok=True)
            with open(out / "train_config.json", "w") as f:
                cfg_dict = asdict(cfg)
                cfg_dict["world_size"] = world_size
                cfg_dict["effective_global_batch_size"] = cfg.batch_size * world_size
                json.dump(cfg_dict, f, indent=2)
        dist.barrier()

        rank0_print(rank, f"Distributed training: {world_size} GPUs")
        rank0_print(rank, f"Per-GPU batch size: {cfg.batch_size}")
        rank0_print(rank, f"Global batch size:  {cfg.batch_size * world_size}")
        rank0_print(rank, f"Per-GPU fam microbatch: {cfg.fam_microbatch_size}")
        rank0_print(rank, "Loading:", cfg.model_id)

        processor, model = load_model_and_processor(cfg.model_id, str(device))
        configure_trainable_parameters(model, cfg.unfreeze_last_n)

        target = copy.deepcopy(model.encoder).to(device).eval()
        for p in target.parameters():
            p.requires_grad = False

        if cfg.gradient_checkpointing:
            try:
                model.gradient_checkpointing_enable()
                rank0_print(rank, "Gradient checkpointing enabled")
            except Exception as e:
                rank0_print(rank, "WARNING: gradient checkpointing failed:", e)

        train_sampler = BIB2Sampler(
            cfg.train_json_dir,
            pattern=cfg.train_pattern,
            project_root=cfg.project_root,
            seed=cfg.seed + rank * 10_000,
        )
        val_sampler = (
            BIB2Sampler(
                cfg.val_json_dir,
                pattern=cfg.val_pattern,
                project_root=cfg.project_root,
                seed=cfg.seed + 12345 + rank * 10_000,
            )
            if cfg.val_json_dir else None
        )

        n_train = len(train_sampler)
        if cfg.max_train_examples is not None and int(cfg.max_train_examples) > 0:
            n_train = min(n_train, int(cfg.max_train_examples))
        if n_train <= 0:
            raise ValueError("Training split is empty")

        usable_train = (n_train // world_size) * world_size
        if usable_train != n_train:
            rank0_print(
                rank,
                f"WARNING: trimming training set from {n_train} to {usable_train} "
                "so both GPU ranks have equal numbers of examples.",
            )
        n_train = usable_train
        local_n_train = n_train // world_size
        local_batches = math.ceil(local_n_train / cfg.batch_size)

        rank0_print(rank, f"Training tuples per epoch: {n_train} / {len(train_sampler)}")
        rank0_print(
            rank,
            f"Each GPU receives {local_n_train} tuples in {local_batches} batches/epoch",
        )
        if cfg.mode == "mixed":
            rank0_print(
                rank,
                "Mixed objective: EVERY selected tuple is used for BOTH objectives "
                f"(masked weight={w_masked:.3f}, fam_future weight={w_future:.3f}).",
            )
            rank0_print(rank, f"  masked JEPA examples/epoch: {n_train}")
            rank0_print(rank, f"  fam->future examples/epoch: {n_train}")
            rank0_print(rank, f"  synchronized optimizer updates/epoch: {local_batches}")

        if val_sampler is not None:
            n_val = len(val_sampler)
            if cfg.max_val_examples is not None and int(cfg.max_val_examples) > 0:
                n_val = min(n_val, int(cfg.max_val_examples))
            usable_val = (n_val // world_size) * world_size
            rank0_print(rank, f"Validation tuples per epoch: {usable_val} / {len(val_sampler)}")

        opt = build_optimizer(model, cfg)
        updates_per_epoch = math.ceil(
            local_batches / cfg.grad_accum_steps
        )
        
        start_epoch = 0
        global_update = 0
        best = float("inf")
        
        if cfg.resume:
            # Restore model, EMA target encoder, and AdamW state,
            # but NOT the old LR scheduler.
            ck = load_checkpoint(
                cfg.resume,
                model,
                target,
                opt=opt,
                sched=None,
            )
        
            start_epoch = int(ck.get("epoch", -1)) + 1
            global_update = int(ck.get("global_update", 0))
            best = float(ck.get("best_val", best))
        
            # Reset optimizer LRs to the configured base values.
            # Adam moments remain loaded from the checkpoint.
            for group in opt.param_groups:
                if group.get("name") == "predictor":
                    group["lr"] = cfg.predictor_lr
                    group["initial_lr"] = cfg.predictor_lr
        
                elif group.get("name") == "encoder":
                    group["lr"] = cfg.encoder_lr
                    group["initial_lr"] = cfg.encoder_lr
        
        # Build the LR schedule only over the epochs that remain.
        remaining_epochs = cfg.epochs - start_epoch
        
        if remaining_epochs <= 0:
            raise ValueError(
                f"No epochs left to train: checkpoint is already at "
                f"epoch {start_epoch}, but --epochs={cfg.epochs}"
            )
        
        remaining_updates = (
            remaining_epochs * updates_per_epoch
        )
        
        sched = build_scheduler(
            opt,
            remaining_updates,
            cfg.warmup_fraction,
        )
        
        rank0_print(
            rank,
            f"Starting at epoch {start_epoch + 1}/{cfg.epochs}; "
            f"{remaining_epochs} epoch(s) remaining; "
            f"{remaining_updates} scheduled optimizer updates remaining."
        )
        dist.barrier()

        lf = None
        writer = None
        if rank == 0:
            log_path = out / "training_log.csv"
            exists = log_path.exists()
            lf = open(log_path, "a", newline="", buffering=1)
            writer = csv.DictWriter(
                lf,
                fieldnames=[
                    "epoch", "batch", "global_batch_size", "rank0_example_indices",
                    "global_update", "masked_loss", "fam_future_loss", "combined_loss",
                    "seconds_per_batch", "masked_seconds", "fam_future_seconds",
                    "sync_optimizer_seconds", "global_examples_per_second", "max_peak_gpu_gib",
                    "lr_predictor", "lr_encoder",
                    "val_loss", "val_masked", "val_fam_future", "val_n",
                ],
            )
            if not exists:
                writer.writeheader(); lf.flush()

        model.train()
        opt.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)

        for epoch in range(start_epoch, cfg.epochs):
            # Same global shuffle on both ranks, then deterministic disjoint split.
            global_indices = list(range(len(train_sampler)))
            random.Random(cfg.seed + epoch).shuffle(global_indices)
            global_indices = global_indices[:n_train]
            rank_indices = global_indices[rank::world_size]

            train_sampler.reseed(cfg.seed + 200_000 + epoch + rank * 10_000)
            gen = torch.Generator(device=device.type)
            gen.manual_seed(cfg.seed + 300_000 + epoch + rank * 10_000)

            batch_list = list(iter_chunks(rank_indices, cfg.batch_size))
            if len(batch_list) != local_batches:
                raise RuntimeError("Ranks disagree on number of batches")

            local_combined_sum = local_masked_sum = local_future_sum = 0.0
            local_count = local_masked_n = local_future_n = 0
            epoch_start = time.perf_counter()

            rank0_print(
                rank,
                f"\nEpoch {epoch+1}/{cfg.epochs} — {n_train} global tuples; "
                f"{local_batches} synchronized batches; global batch <= {cfg.batch_size * world_size}",
            )

            for batch_i, batch_indices in enumerate(batch_list):
                bs = len(batch_indices)
                batch_start = time.perf_counter()
                masked_value = None
                future_value = None
                masked_sec = 0.0
                fam_sec = 0.0
                sync_opt_sec = 0.0

                # ---------------------------------------------------------
                # Objective A: masked JEPA, batched independently per GPU.
                # ---------------------------------------------------------
                if cfg.mode in ("masked", "mixed"):
                    cuda_sync(device)
                    t0 = time.perf_counter()
                    with autocast_context(model):
                        loss_masked, _ = masked_batch_loss(
                            model, target, processor, train_sampler,
                            batch_indices, cfg, gen,
                        )
                    (w_masked * loss_masked / cfg.grad_accum_steps).backward()
                    cuda_sync(device)
                    masked_sec = time.perf_counter() - t0

                    masked_value = float(loss_masked.detach().cpu())
                    local_masked_sum += masked_value * bs
                    local_masked_n += bs
                    del loss_masked

                # ---------------------------------------------------------
                # Objective B: same tuples, memory-safe fam microbatches.
                # ---------------------------------------------------------
                if cfg.mode in ("fam_future", "mixed"):
                    cuda_sync(device)
                    t0 = time.perf_counter()
                    fam_weighted_sum = 0.0
                    for micro_indices in iter_chunks(batch_indices, cfg.fam_microbatch_size):
                        with autocast_context(model):
                            loss_future, _ = fam_microbatch_loss(
                                model, target, processor, train_sampler,
                                micro_indices, cfg,
                            )
                        micro_fraction = len(micro_indices) / float(bs)
                        scale = w_future * micro_fraction / cfg.grad_accum_steps
                        (scale * loss_future).backward()
                        fv = float(loss_future.detach().cpu())
                        fam_weighted_sum += fv * len(micro_indices)
                        local_future_sum += fv * len(micro_indices)
                        local_future_n += len(micro_indices)
                        del loss_future
                    cuda_sync(device)
                    fam_sec = time.perf_counter() - t0
                    future_value = fam_weighted_sum / bs

                if cfg.mode == "mixed":
                    combined_value = w_masked * masked_value + w_future * future_value
                elif cfg.mode == "masked":
                    combined_value = masked_value
                else:
                    combined_value = future_value

                local_combined_sum += combined_value * bs
                local_count += bs

                do_update = (
                    ((batch_i + 1) % cfg.grad_accum_steps == 0)
                    or (batch_i + 1 == len(batch_list))
                )
                if do_update:
                    cuda_sync(device)
                    t0 = time.perf_counter()
                    sync_gradients(model, world_size)
                    params = [
                        p for p in model.parameters()
                        if p.requires_grad and p.grad is not None
                    ]
                    if cfg.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    sched.step()
                    ema_update(target, model.encoder, cfg.ema)
                    global_update += 1
                    cuda_sync(device)
                    sync_opt_sec = time.perf_counter() - t0

                cuda_sync(device)
                batch_sec = time.perf_counter() - batch_start

                # Global loss values for logging. Very small collective.
                metric = reduce_sum_tensor(
                    [
                        combined_value * bs,
                        (masked_value if masked_value is not None else 0.0) * bs,
                        (future_value if future_value is not None else 0.0) * bs,
                        bs,
                    ],
                    device,
                )
                global_bs = int(round(metric[3].item()))
                global_combined = metric[0].item() / max(1, global_bs)
                global_masked = metric[1].item() / max(1, global_bs) if masked_value is not None else None
                global_future = metric[2].item() / max(1, global_bs) if future_value is not None else None

                timing = reduce_max_tensor(
                    [
                        batch_sec, masked_sec, fam_sec, sync_opt_sec,
                        torch.cuda.max_memory_allocated(device) / 1024**3,
                    ],
                    device,
                ).tolist()
                max_batch_sec, max_masked_sec, max_fam_sec, max_sync_sec, max_peak = timing
                global_eps = global_bs / max_batch_sec if max_batch_sec > 0 else float("nan")

                lrs = {g.get("name", "group"): g["lr"] for g in opt.param_groups}
                should_print = (
                    batch_i == 0
                    or (batch_i + 1) % max(1, cfg.print_every) == 0
                    or batch_i + 1 == len(batch_list)
                )

                if rank == 0 and writer is not None:
                    writer.writerow({
                        "epoch": epoch,
                        "batch": batch_i,
                        "global_batch_size": global_bs,
                        "rank0_example_indices": ";".join(map(str, batch_indices)),
                        "global_update": global_update,
                        "masked_loss": global_masked if global_masked is not None else "",
                        "fam_future_loss": global_future if global_future is not None else "",
                        "combined_loss": global_combined,
                        "seconds_per_batch": max_batch_sec,
                        "masked_seconds": max_masked_sec,
                        "fam_future_seconds": max_fam_sec,
                        "sync_optimizer_seconds": max_sync_sec,
                        "global_examples_per_second": global_eps,
                        "max_peak_gpu_gib": max_peak,
                        "lr_predictor": lrs.get("predictor", ""),
                        "lr_encoder": lrs.get("encoder", ""),
                        "val_loss": "", "val_masked": "", "val_fam_future": "", "val_n": "",
                    })

                if should_print and rank == 0:
                    msg = (
                        f"  batch {batch_i+1:5d}/{len(batch_list):5d} | "
                        f"global tuples {global_bs:2d} | combined {global_combined:.6f}"
                    )
                    if global_masked is not None:
                        msg += f" | masked {global_masked:.6f}"
                    if global_future is not None:
                        msg += f" | fam_future {global_future:.6f}"
                    msg += (
                        f" | total {max_batch_sec:.2f}s"
                        f" | masked {max_masked_sec:.2f}s"
                        f" | fam {max_fam_sec:.2f}s"
                        f" | sync+opt {max_sync_sec:.2f}s"
                        f" | {global_eps:.2f} global ex/s"
                        f" | peak {max_peak:.2f} GiB/GPU"
                    )
                    print(msg, flush=True)
                    lf.flush()

            # Global epoch means.
            epoch_totals = reduce_sum_tensor(
                [
                    local_combined_sum, local_masked_sum, local_future_sum,
                    local_count, local_masked_n, local_future_n,
                ],
                device,
            ).tolist()
            combined_sum, masked_sum, future_sum, count, masked_n, future_n = epoch_totals
            count = int(round(count)); masked_n = int(round(masked_n)); future_n = int(round(future_n))
            train_combined = combined_sum / max(1, count)
            train_masked = masked_sum / masked_n if masked_n else None
            train_future = future_sum / future_n if future_n else None

            dist.barrier()
            epoch_sec = time.perf_counter() - epoch_start
            rank0_print(rank, f"Epoch combined train loss: {train_combined:.6f} ({epoch_sec/60:.2f} min)")
            if train_masked is not None:
                rank0_print(rank, f"  masked JEPA: {train_masked:.6f} over {masked_n} tuples")
            if train_future is not None:
                rank0_print(rank, f"  fam->future: {train_future:.6f} over {future_n} tuples")

            v, vdetails = validate_distributed(
                model, target, processor, val_sampler, cfg,
                rank, world_size, device,
            )

            if v is not None and rank == 0:
                print(f"Validation combined loss: {v:.6f} over {vdetails['n']} tuples", flush=True)
                if vdetails["masked"] is not None:
                    print(
                        f"  validation masked JEPA: {vdetails['masked']:.6f} "
                        f"over {vdetails['masked_n']} tuples",
                        flush=True,
                    )
                if vdetails["fam_future"] is not None:
                    print(
                        f"  validation fam->future: {vdetails['fam_future']:.6f} "
                        f"over {vdetails['fam_future_n']} tuples",
                        flush=True,
                    )

                writer.writerow({
                    "epoch": epoch, "batch": "VAL", "global_batch_size": "",
                    "rank0_example_indices": "", "global_update": global_update,
                    "masked_loss": "", "fam_future_loss": "", "combined_loss": "",
                    "seconds_per_batch": "", "masked_seconds": "",
                    "fam_future_seconds": "", "sync_optimizer_seconds": "",
                    "global_examples_per_second": "", "max_peak_gpu_gib": "",
                    "lr_predictor": "", "lr_encoder": "",
                    "val_loss": v, "val_masked": vdetails["masked"],
                    "val_fam_future": vdetails["fam_future"], "val_n": vdetails["n"],
                })
                lf.flush()

                if v < best:
                    best = v
                    print(
                        "Saved improved:",
                        save_checkpoint(
                            out, model, target, opt, sched, cfg,
                            epoch, global_update, best, "best.pt",
                        ),
                        flush=True,
                    )

                if cfg.save_every_epoch > 0 and (epoch + 1) % cfg.save_every_epoch == 0:
                    print(
                        "Saved:",
                        save_checkpoint(
                            out, model, target, opt, sched, cfg,
                            epoch, global_update, best, f"epoch_{epoch+1:03d}.pt",
                        ),
                        flush=True,
                    )

            # Keep ranks together while rank 0 writes checkpoints.
            dist.barrier()

        if rank == 0:
            final = save_checkpoint(
                out, model, target, opt, sched, cfg,
                cfg.epochs - 1, global_update, best, "final.pt",
            )
            hf = out / "final_hf_model"
            model.save_pretrained(hf)
            processor.save_pretrained(hf)
            print("\nTraining complete", flush=True)
            print("Checkpoint:", final, flush=True)
            print("HF model:", hf, flush=True)
            if lf is not None:
                lf.flush(); lf.close()

        dist.barrier()

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Fine-tune BabyView V-JEPA2 on BIB2. One epoch is one shuffled "
            "visit to every training tuple exactly once, grouped into mini-batches. "
            "In mixed mode every tuple contributes to BOTH objectives."
        )
    )

    p.add_argument("--train-json-dir", required=True)
    p.add_argument("--val-json-dir", default=None)
    p.add_argument("--project-root", default=None)
    p.add_argument("--train-pattern", default="index_dict_train_*.json")
    p.add_argument("--val-pattern", default="index_dict_val_*.json")

    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--output-dir", default="vjepa2_bib2_finetune")
    p.add_argument("--mode", choices=["masked", "fam_future", "mixed"], default="mixed")
    p.add_argument("--device", default=None)

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument(
        "--max-train-examples",
        type=int,
        default=None,
        help="Debug cap. Default/0 = use every training tuple each epoch.",
    )
    p.add_argument(
        "--max-val-examples",
        type=int,
        default=None,
        help="Optional validation cap. Default/0 = use every validation tuple.",
    )
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument(
        "--batch-size", type=int, default=8,
        help="Training tuples in each masked-JEPA batch PER GPU. Default: 8.",
    )
    p.add_argument(
        "--fam-microbatch-size", type=int, default=2,
        help=(
            "Number of same-history fam->future examples processed at once. "
            "Per-GPU fam->future microbatch. Default: 2 for A100-40GB."
        ),
    )
    p.add_argument(
        "--print-every", type=int, default=10,
        help="Print/flush progress every N training batches.",
    )
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--predictor-lr", type=float, default=DEFAULT_PREDICTOR_LR)
    p.add_argument("--encoder-lr", type=float, default=DEFAULT_ENCODER_LR)
    p.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--warmup-fraction", type=float, default=0.05)
    p.add_argument("--ema", type=float, default=DEFAULT_EMA)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--unfreeze-last-n", type=int, default=0)
    p.add_argument("--gradient-checkpointing", action="store_true")

    # Method A
    p.add_argument("--clip-frames", type=int, default=DEFAULT_CLIP_FRAMES)
    p.add_argument("--clip-fps", type=float, default=DEFAULT_CLIP_FPS)
    p.add_argument("--mask-ratio", type=float, default=DEFAULT_MASK_RATIO)

    # Method B
    p.add_argument("--n-fam-trials", type=int, default=DEFAULT_N_FAM_TRIALS)
    p.add_argument("--frames-per-fam", type=int, default=DEFAULT_FRAMES_PER_FAM)
    p.add_argument("--test-fps", type=float, default=DEFAULT_TEST_FPS)
    p.add_argument("--min-history-tubelets", type=int, default=1)
    p.add_argument("--max-history-tubelets", type=int, default=None)
    p.add_argument("--fam-end-trim-seconds", type=float, default=DEFAULT_FAM_END_TRIM_SECONDS)
    p.add_argument("--test-end-trim-seconds", type=float, default=DEFAULT_TEST_END_TRIM_SECONDS)

    p.add_argument(
        "--mixed-masked-weight",
        "--mixed-masked-fraction",
        dest="mixed_masked_weight",
        type=float,
        default=0.5,
        help=(
            "In mixed mode, weight on masked-JEPA loss. The fam->future "
            "weight is 1 minus this value. Default: 0.5. "
            "--mixed-masked-fraction is retained as a backward-compatible alias."
        ),
    )
    p.add_argument("--save-every-epoch", type=int, default=1)
    p.add_argument("--resume", default=None)

    return p.parse_args()

def main():
    cfg = TrainConfig(**vars(parse_args()))
    # torchrun supplies RANK/LOCAL_RANK/WORLD_SIZE. Only rank 0 prints config.
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(asdict(cfg), indent=2), flush=True)
    train(cfg)

if __name__=="__main__": main()
