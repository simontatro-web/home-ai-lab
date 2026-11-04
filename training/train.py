"""Training loop for the from-scratch transformer.

    # one GPU (or CPU for a smoke test)
    python training/train.py --config training/configs/tiny_quickstart.yaml

    # both GPUs (data parallel)
    torchrun --nproc_per_node=2 training/train.py --config training/configs/model5_500M.yaml

Every hyperparameter lives in the YAML config, so diffing two configs shows
exactly what changed between runs. Each run writes to its out_dir:
    log.csv     step, train loss, val loss, learning rate, tokens seen, elapsed time
    ckpt.pt     latest checkpoint (model + optimizer + step), used by --resume
    config.yaml copy of the config the run used

Mixed precision: bf16 where the GPU supports it, otherwise fp16 with loss
scaling (V100s have no bf16 hardware, so they use fp16), fp32 on CPU.
"""

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.transformer import Transformer  # noqa: E402

DEFAULTS = {
    "schedule": "cosine",     # "cosine" (linear warmup, cosine decay) or "constant"
    "warmup_steps": 0,
    "min_lr_frac": 0.1,       # cosine decays to lr * min_lr_frac
    "weight_decay": 0.1,
    "grad_clip": 1.0,         # 0 disables clipping
    "grad_accum": 1,
    "eval_every": 500,
    "eval_batches": 50,
    "log_every": 50,
    "seed": 1337,
}


# ------------------------------------------------------------------ data
class TokenData:
    """Random fixed-length windows from uint16 token files (memory-mapped)."""

    def __init__(self, data_dir):
        with open(os.path.join(data_dir, "meta.json")) as f:
            self.meta = json.load(f)
        self.splits = {
            s: np.memmap(os.path.join(data_dir, f"{s}.bin"), dtype=np.uint16, mode="r")
            for s in ("train", "val")
        }

    def batch(self, split, batch_size, ctx, device):
        data = self.splits[split]
        if len(data) <= ctx + 1:
            raise ValueError(f"{split} split has {len(data)} tokens, need more than ctx+1={ctx + 1}")
        ix = torch.randint(len(data) - ctx - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i + ctx].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + ctx].astype(np.int64)) for i in ix])
        if device.type == "cuda":
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)


# ------------------------------------------------------------- schedule
def lr_at(step, cfg):
    lr, warmup = cfg["lr"], cfg["warmup_steps"]
    if warmup and step < warmup:
        return lr * (step + 1) / warmup
    if cfg["schedule"] == "constant":
        return lr
    min_lr = lr * cfg["min_lr_frac"]
    progress = min(1.0, (step - warmup) / max(1, cfg["max_steps"] - warmup))
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, data, cfg, device, autocast):
    model.eval()
    losses = []
    for _ in range(cfg["eval_batches"]):
        x, y = data.batch("val", cfg["batch_size"], cfg["model"]["ctx"], device)
        with autocast:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# ----------------------------------------------------------------- main
def train(cfg, resume=False):
    cfg = {**DEFAULTS, **cfg}

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
    else:
        rank, world = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    master = rank == 0
    torch.manual_seed(cfg["seed"] + rank)

    if device.type == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False):
        dtype = torch.bfloat16
    elif device.type == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32
    autocast = torch.autocast(device.type, dtype=dtype) if dtype != torch.float32 else nullcontext()
    scaler = torch.amp.GradScaler(device.type, enabled=(dtype == torch.float16))

    data = TokenData(cfg["data_dir"])
    m = cfg["model"]
    model = Transformer(
        vocab_size=data.meta["vocab_size"],
        n_layer=m["n_layer"], d_model=m["d_model"], n_heads=m["n_heads"],
        n_kv_heads=m.get("n_kv_heads"), ctx=m["ctx"],
    ).to(device)

    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg["lr"], betas=(0.9, 0.95), fused=(device.type == "cuda"),
    )

    out_dir = cfg["out_dir"]
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    start_step = 0
    if resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1

    raw_model = model
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index])

    tokens_per_step = cfg["batch_size"] * m["ctx"] * cfg["grad_accum"] * world
    if master:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"{cfg['run_name']}: {raw_model.num_params() / 1e6:.1f}M params "
              f"({raw_model.num_params(non_embedding=True) / 1e6:.1f}M non-embedding), "
              f"{world} x {device.type}, {dtype}, {tokens_per_step:,} tokens/step, "
              f"{cfg['max_steps'] * tokens_per_step / 1e9:.2f}B tokens total")
        log_path = os.path.join(out_dir, "log.csv")
        new_log = not (resume and os.path.exists(log_path))
        log_file = open(log_path, "w" if new_log else "a", newline="")
        log = csv.writer(log_file)
        if new_log:
            log.writerow(["step", "train_loss", "val_loss", "lr", "tokens", "elapsed_s"])

    def save(step):
        torch.save({"model": raw_model.state_dict(), "optimizer": opt.state_dict(),
                    "step": step, "config": cfg}, ckpt_path + ".tmp")
        shutil.move(ckpt_path + ".tmp", ckpt_path)

    t0 = time.time()
    model.train()
    for step in range(start_step, cfg["max_steps"]):
        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr

        for micro in range(cfg["grad_accum"]):
            x, y = data.batch("train", cfg["batch_size"], m["ctx"], device)
            sync = (not ddp) or micro == cfg["grad_accum"] - 1
            with (nullcontext() if sync else model.no_sync()):
                with autocast:
                    _, loss = model(x, y)
                scaler.scale(loss / cfg["grad_accum"]).backward()

        if cfg["grad_clip"]:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        last = step == cfg["max_steps"] - 1
        do_eval = step % cfg["eval_every"] == 0 or last
        if master and (step % cfg["log_every"] == 0 or do_eval):
            train_loss = loss.item()
            val_loss = evaluate(raw_model, data, cfg, device, autocast) if do_eval else ""
            elapsed = time.time() - t0
            log.writerow([step, f"{train_loss:.4f}", f"{val_loss:.4f}" if val_loss != "" else "",
                          f"{lr:.3e}", (step + 1) * tokens_per_step, f"{elapsed:.0f}"])
            log_file.flush()
            msg = f"step {step:7d} | loss {train_loss:.4f} | lr {lr:.2e} | {elapsed / 60:.1f} min"
            print(msg + (f" | val {val_loss:.4f}" if val_loss != "" else ""))
            if not math.isfinite(train_loss):
                print("loss is not finite, stopping (lower the LR or check grad_clip)")
                save(step)
                break
            if do_eval:
                save(step)

    if master:
        log_file.close()
        print(f"done in {(time.time() - t0) / 3600:.2f} h -> {out_dir}")
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true", help="continue from out_dir/ckpt.pt")
    ap.add_argument("--max-steps", type=int, help="override max_steps (for short test runs)")
    ap.add_argument("--out-dir", help="override out_dir")
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("run_name", os.path.basename(a.config).replace(".yaml", ""))
    if a.max_steps:
        cfg["max_steps"] = a.max_steps
    if a.out_dir:
        cfg["out_dir"] = a.out_dir
    train(cfg, resume=a.resume)
