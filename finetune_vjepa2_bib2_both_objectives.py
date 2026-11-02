#!/usr/bin/env python3
"""Fine-tune BabyView V-JEPA2 on BIB2 with two self-supervised objectives.

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
  optimizer sees 32,000 tuple-level combined-gradient steps
  (before any gradient accumulation).

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

import argparse, copy, csv, json, math, os, random
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import numpy as np
import torch
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
    print_every: int = 100
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

        print(f"Loaded {len(self.examples)} data tuples from {len(files)} JSON files")
        for name, count in self.file_counts.items():
            print(f"  {name}: {count}")

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

        h = self.rng.randint(min_hist, max_hist)
        hf = h * tubelet_size

        return {
            "fam_context": fam_context,
            "current_test": test[:hf],
            "target_test": test[hf:hf + tubelet_size],
            "history_tubelets": h,
            "video_path": str(path),
            "example_index": int(example_index),
        }

def load_model_and_processor(model_id, device=None):
    if device is None: device="cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32
    processor=AutoVideoProcessor.from_pretrained(model_id)
    model=AutoModel.from_pretrained(model_id, dtype=dtype, attn_implementation="sdpa").to(device)
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
    print(f"Trainable predictor parameters: {npred:,}")
    print(f"Trainable encoder parameters:   {nenc:,}")


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
    pixels=processor_pixels(processor,[frames],model); B=pixels.size(0)
    vis,tgt=make_block_mask_ids(model.config,B,pixels.size(1),mask_ratio,pixels.device,torch_generator)
    ctx=encode_visible_tokens(model.encoder,pixels,vis)
    pred=predict_from_visible_tokens(model.predictor,ctx,vis,tgt)
    with torch.no_grad():
        full=target_encoder(pixel_values_videos=pixels).last_hidden_state
        target=gather_tokens(full,tgt); target=F.layer_norm(target.float(),(target.shape[-1],))
    return (pred.float()-target).abs().mean()


def fam_future_loss(model,target_encoder,processor,fam_context,current_test,target_test):
    tubelet=int(model.config.tubelet_size)
    if len(fam_context)%tubelet or len(current_test)%tubelet: raise ValueError("Context lengths must contain complete tubelets")
    if len(target_test)!=tubelet: raise ValueError("target_test must be exactly one tubelet")
    context=torch.cat([fam_context,current_test],0); full=torch.cat([context,target_test],0)
    cp=processor_pixels(processor,[context],model); fp=processor_pixels(processor,[full],model)
    ch=model.encoder(pixel_values_videos=cp).last_hidden_state; B,N,_=ch.shape
    grid=int(model.config.crop_size//model.config.patch_size); tpt=grid*grid
    cids=torch.arange(N,device=ch.device).unsqueeze(0).repeat(B,1)
    tids=torch.arange(N,N+tpt,device=ch.device).unsqueeze(0).repeat(B,1)
    pred=model.predictor(encoder_hidden_states=ch,context_mask=[cids],target_mask=[tids]).last_hidden_state
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


@torch.no_grad()
def validate(model, target_encoder, processor, sampler, cfg):
    """
    Validate on the same deterministic tuple/transform set every epoch.

    In mixed mode EVERY validation tuple is evaluated with BOTH objectives,
    and masked/fam_future losses are reported separately in addition to their
    weighted combined loss.
    """
    if sampler is None:
        return None, None

    model.eval()
    target_encoder.eval()

    n_val = len(sampler)
    if cfg.max_val_examples is not None and int(cfg.max_val_examples) > 0:
        n_val = min(n_val, int(cfg.max_val_examples))

    indices = list(range(n_val))

    # Same stochastic transformations at every validation epoch.
    sampler.reseed(cfg.seed + 600_000)
    dev = next(model.parameters()).device
    gen = torch.Generator(device=dev.type)
    gen.manual_seed(cfg.seed + 700_000)

    masked_vals = []
    future_vals = []
    combined_vals = []

    if cfg.mode == "mixed":
        w_masked, w_future = mixed_weights(cfg)
    elif cfg.mode == "masked":
        w_masked, w_future = 1.0, 0.0
    elif cfg.mode == "fam_future":
        w_masked, w_future = 0.0, 1.0
    else:
        raise ValueError("mode must be masked, fam_future, or mixed")

    for example_index in indices:
        if cfg.mode in ("masked", "mixed"):
            with autocast_context(model):
                loss_masked, _ = compute_one_loss(
                    "masked",
                    sampler,
                    example_index,
                    model,
                    target_encoder,
                    processor,
                    cfg,
                    gen,
                )
            masked_value = float(loss_masked.detach().cpu())
            masked_vals.append(masked_value)
        else:
            masked_value = None

        if cfg.mode in ("fam_future", "mixed"):
            with autocast_context(model):
                loss_future, _ = compute_one_loss(
                    "fam_future",
                    sampler,
                    example_index,
                    model,
                    target_encoder,
                    processor,
                    cfg,
                    gen,
                )
            future_value = float(loss_future.detach().cpu())
            future_vals.append(future_value)
        else:
            future_value = None

        if cfg.mode == "mixed":
            combined_vals.append(
                w_masked * masked_value + w_future * future_value
            )
        elif cfg.mode == "masked":
            combined_vals.append(masked_value)
        else:
            combined_vals.append(future_value)

    model.train()

    details = {
        "overall": float(np.mean(combined_vals)),
        "masked": float(np.mean(masked_vals)) if masked_vals else None,
        "fam_future": float(np.mean(future_vals)) if future_vals else None,
        "n": len(indices),
        "masked_n": len(masked_vals),
        "fam_future_n": len(future_vals),
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
    seed_everything(cfg.seed)

    if cfg.mode not in ("masked", "fam_future", "mixed"):
        raise ValueError("mode must be masked, fam_future, or mixed")
    if cfg.clip_frames % 2 or cfg.frames_per_fam % 2:
        raise ValueError("clip_frames and frames_per_fam must be even")

    if cfg.mode == "mixed":
        w_masked, w_future = mixed_weights(cfg)
    elif cfg.mode == "masked":
        w_masked, w_future = 1.0, 0.0
    else:
        w_masked, w_future = 0.0, 1.0

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "train_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    print("Loading:", cfg.model_id)
    processor, model = load_model_and_processor(cfg.model_id, cfg.device)
    print(
        "Device:",
        next(model.parameters()).device,
        "dtype:",
        next(model.parameters()).dtype,
    )

    configure_trainable_parameters(model, cfg.unfreeze_last_n)

    target = copy.deepcopy(model.encoder).to(
        next(model.parameters()).device
    ).eval()
    for p in target.parameters():
        p.requires_grad = False

    if cfg.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")
        except Exception as e:
            print("WARNING: gradient checkpointing failed:", e)

    train_sampler = BIB2Sampler(
        cfg.train_json_dir,
        pattern=cfg.train_pattern,
        project_root=cfg.project_root,
        seed=cfg.seed,
    )
    val_sampler = (
        BIB2Sampler(
            cfg.val_json_dir,
            pattern=cfg.val_pattern,
            project_root=cfg.project_root,
            seed=cfg.seed + 12345,
        )
        if cfg.val_json_dir
        else None
    )

    n_train = len(train_sampler)
    if cfg.max_train_examples is not None and int(cfg.max_train_examples) > 0:
        n_train = min(n_train, int(cfg.max_train_examples))
    if n_train <= 0:
        raise ValueError("Training split is empty")

    print(f"Training tuples per epoch: {n_train} / {len(train_sampler)}")
    if cfg.mode == "mixed":
        print(
            "Mixed objective: EVERY selected training tuple is used for BOTH "
            f"objectives (masked weight={w_masked:.3f}, "
            f"fam_future weight={w_future:.3f})."
        )
        print(
            f"  masked JEPA examples/epoch: {n_train}\n"
            f"  fam->future examples/epoch: {n_train}\n"
            f"  tuple-level gradient steps/epoch before accumulation: {n_train}"
        )
    else:
        print(f"Objective examples/epoch ({cfg.mode}): {n_train}")

    if val_sampler is not None:
        n_val = len(val_sampler)
        if cfg.max_val_examples is not None and int(cfg.max_val_examples) > 0:
            n_val = min(n_val, int(cfg.max_val_examples))
        print(f"Validation tuples per epoch: {n_val} / {len(val_sampler)}")
        if cfg.mode == "mixed":
            print(
                f"  validation masked examples: {n_val}\n"
                f"  validation fam->future examples: {n_val}"
            )

    opt = build_optimizer(model, cfg)

    # One optimizer opportunity per tuple, not per objective. In mixed mode the
    # two objective gradients are accumulated into that tuple-level update.
    total_micro_steps = cfg.epochs * n_train
    total_updates = math.ceil(total_micro_steps / cfg.grad_accum_steps)
    sched = build_scheduler(opt, total_updates, cfg.warmup_fraction)

    start_epoch = 0
    global_update = 0
    best = float("inf")

    if cfg.resume:
        ck = load_checkpoint(cfg.resume, model, target, opt, sched)
        start_epoch = int(ck.get("epoch", -1)) + 1
        global_update = int(ck.get("global_update", 0))
        best = float(ck.get("best_val", best))

    log_path = out / "training_log.csv"
    exists = log_path.exists()
    lf = open(log_path, "a", newline="")
    writer = csv.DictWriter(
        lf,
        fieldnames=[
            "epoch",
            "step",
            "example_index",
            "global_update",
            "masked_loss",
            "fam_future_loss",
            "combined_loss",
            "lr_predictor",
            "lr_encoder",
            "val_loss",
            "val_masked",
            "val_fam_future",
            "val_n",
        ],
    )
    if not exists:
        writer.writeheader()
        lf.flush()

    model.train()
    opt.zero_grad(set_to_none=True)
    dev = next(model.parameters()).device

    try:
        for epoch in range(start_epoch, cfg.epochs):
            # Every tuple appears exactly once per epoch.
            indices = list(range(len(train_sampler)))
            random.Random(cfg.seed + epoch).shuffle(indices)
            indices = indices[:n_train]

            # Reproducible but different stochastic training transforms per epoch.
            train_sampler.reseed(cfg.seed + 200_000 + epoch)
            gen = torch.Generator(device=dev.type)
            gen.manual_seed(cfg.seed + 300_000 + epoch)

            combined_losses = []
            masked_losses = []
            future_losses = []

            print(
                f"\nEpoch {epoch + 1}/{cfg.epochs} — full shuffled pass over "
                f"{len(indices)} tuples"
            )

            for step, example_index in enumerate(indices):
                masked_value = None
                future_value = None

                # -------------------------------------------------------------
                # Objective A: masked JEPA
                # -------------------------------------------------------------
                if cfg.mode in ("masked", "mixed"):
                    with autocast_context(model):
                        loss_masked, _ = compute_one_loss(
                            "masked",
                            train_sampler,
                            example_index,
                            model,
                            target,
                            processor,
                            cfg,
                            gen,
                        )

                    # Backward immediately so we do not hold Method A's graph
                    # while building the much longer fam->future graph.
                    scale = w_masked / cfg.grad_accum_steps
                    (scale * loss_masked).backward()

                    masked_value = float(loss_masked.detach().cpu())
                    masked_losses.append(masked_value)
                    del loss_masked

                # -------------------------------------------------------------
                # Objective B: familiarization -> future
                # -------------------------------------------------------------
                if cfg.mode in ("fam_future", "mixed"):
                    with autocast_context(model):
                        loss_future, _ = compute_one_loss(
                            "fam_future",
                            train_sampler,
                            example_index,
                            model,
                            target,
                            processor,
                            cfg,
                            gen,
                        )

                    scale = w_future / cfg.grad_accum_steps
                    (scale * loss_future).backward()

                    future_value = float(loss_future.detach().cpu())
                    future_losses.append(future_value)
                    del loss_future

                # Weighted scalar for logging/checkpoint interpretation.
                if cfg.mode == "mixed":
                    combined_value = (
                        w_masked * masked_value
                        + w_future * future_value
                    )
                elif cfg.mode == "masked":
                    combined_value = masked_value
                else:
                    combined_value = future_value

                combined_losses.append(combined_value)

                do_update = (
                    ((step + 1) % cfg.grad_accum_steps == 0)
                    or (step + 1 == len(indices))
                )

                if do_update:
                    params = [
                        p
                        for p in model.parameters()
                        if p.requires_grad and p.grad is not None
                    ]
                    if cfg.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            params,
                            cfg.max_grad_norm,
                        )
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    sched.step()
                    ema_update(target, model.encoder, cfg.ema)
                    global_update += 1

                if (
                    step == 0
                    or (step + 1) % max(1, cfg.print_every) == 0
                    or step + 1 == len(indices)
                ):
                    msg = (
                        f"  tuple {step + 1:6d}/{len(indices):6d} | "
                        f"combined {combined_value:.6f}"
                    )
                    if masked_value is not None:
                        msg += f" | masked {masked_value:.6f}"
                    if future_value is not None:
                        msg += f" | fam_future {future_value:.6f}"
                    msg += (
                        f" | mean{min(len(combined_losses), cfg.print_every)} "
                        f"{np.mean(combined_losses[-cfg.print_every:]):.6f}"
                    )
                    print(msg)

                lrs = {
                    g.get("name", "group"): g["lr"]
                    for g in opt.param_groups
                }
                writer.writerow({
                    "epoch": epoch,
                    "step": step,
                    "example_index": example_index,
                    "global_update": global_update,
                    "masked_loss": (
                        masked_value if masked_value is not None else ""
                    ),
                    "fam_future_loss": (
                        future_value if future_value is not None else ""
                    ),
                    "combined_loss": combined_value,
                    "lr_predictor": lrs.get("predictor", ""),
                    "lr_encoder": lrs.get("encoder", ""),
                    "val_loss": "",
                    "val_masked": "",
                    "val_fam_future": "",
                    "val_n": "",
                })

                if (step + 1) % max(1, cfg.print_every) == 0:
                    lf.flush()

            train_combined = float(np.mean(combined_losses))
            train_masked = (
                float(np.mean(masked_losses))
                if masked_losses
                else None
            )
            train_future = (
                float(np.mean(future_losses))
                if future_losses
                else None
            )

            print(f"Epoch combined train loss: {train_combined:.6f}")
            if train_masked is not None:
                print(
                    f"  masked JEPA: {train_masked:.6f} "
                    f"over {len(masked_losses)} tuples"
                )
            if train_future is not None:
                print(
                    f"  fam->future: {train_future:.6f} "
                    f"over {len(future_losses)} tuples"
                )

            v, vdetails = validate(
                model,
                target,
                processor,
                val_sampler,
                cfg,
            )

            if v is not None:
                print(
                    f"Validation combined loss: {v:.6f} "
                    f"over {vdetails['n']} tuples"
                )
                if vdetails["masked"] is not None:
                    print(
                        f"  validation masked JEPA: "
                        f"{vdetails['masked']:.6f} "
                        f"over {vdetails['masked_n']} tuples"
                    )
                if vdetails["fam_future"] is not None:
                    print(
                        f"  validation fam->future: "
                        f"{vdetails['fam_future']:.6f} "
                        f"over {vdetails['fam_future_n']} tuples"
                    )

                writer.writerow({
                    "epoch": epoch,
                    "step": "VAL",
                    "example_index": "",
                    "global_update": global_update,
                    "masked_loss": "",
                    "fam_future_loss": "",
                    "combined_loss": "",
                    "lr_predictor": "",
                    "lr_encoder": "",
                    "val_loss": v,
                    "val_masked": vdetails["masked"],
                    "val_fam_future": vdetails["fam_future"],
                    "val_n": vdetails["n"],
                })
                lf.flush()

                if v < best:
                    best = v
                    print(
                        "Saved improved:",
                        save_checkpoint(
                            out,
                            model,
                            target,
                            opt,
                            sched,
                            cfg,
                            epoch,
                            global_update,
                            best,
                            "best.pt",
                        ),
                    )

            if (
                cfg.save_every_epoch > 0
                and (epoch + 1) % cfg.save_every_epoch == 0
            ):
                print(
                    "Saved:",
                    save_checkpoint(
                        out,
                        model,
                        target,
                        opt,
                        sched,
                        cfg,
                        epoch,
                        global_update,
                        best,
                        f"epoch_{epoch + 1:03d}.pt",
                    ),
                )

        final = save_checkpoint(
            out,
            model,
            target,
            opt,
            sched,
            cfg,
            cfg.epochs - 1,
            global_update,
            best,
            "final.pt",
        )
        hf = out / "final_hf_model"
        model.save_pretrained(hf)
        processor.save_pretrained(hf)

        print("\nTraining complete")
        print("Checkpoint:", final)
        print("HF model:", hf)

        return {
            "model": model,
            "target_encoder": target,
            "processor": processor,
            "final_checkpoint": str(final),
            "hf_model_dir": str(hf),
            "best_val": best,
        }
    finally:
        lf.close()


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Fine-tune BabyView V-JEPA2 on BIB2. One epoch is one shuffled "
            "visit to every training tuple exactly once. In mixed mode every "
            "tuple contributes to BOTH objectives."
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
    p.add_argument("--print-every", type=int, default=100)
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
    cfg=TrainConfig(**vars(parse_args())); print(json.dumps(asdict(cfg),indent=2)); train(cfg)

if __name__=="__main__": main()
