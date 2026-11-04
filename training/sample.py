"""Generate text from a trained checkpoint.

    python training/sample.py --ckpt runs/tiny_quickstart/ckpt.pt --prompt "The server"
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.tokenizer import BPETokenizer  # noqa: E402
from model.transformer import Transformer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--tokenizer", default="model/tokenizer.json")
ap.add_argument("--prompt", default="The")
ap.add_argument("--tokens", type=int, default=200)
ap.add_argument("--temperature", type=float, default=0.8)
ap.add_argument("--top-k", type=int, default=50)
a = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load(a.ckpt, map_location=device)
state = ckpt["model"]
m = ckpt["config"]["model"]
model = Transformer(vocab_size=state["wte.weight"].shape[0], n_layer=m["n_layer"],
                    d_model=m["d_model"], n_heads=m["n_heads"],
                    n_kv_heads=m.get("n_kv_heads"), ctx=m["ctx"]).to(device)
model.load_state_dict(state)
model.eval()

tok = BPETokenizer.load(a.tokenizer)
idx = torch.tensor([tok.encode(a.prompt)], device=device)
out = model.generate(idx, a.tokens, temperature=a.temperature, top_k=a.top_k)[0].tolist()
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # partial characters are possible
print(tok.decode([i for i in out if i < tok.vocab_size]))
