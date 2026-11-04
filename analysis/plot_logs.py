"""Plot validation loss from real training logs (each run's log.csv).

    python analysis/plot_logs.py runs/model3_123M runs/model4_236M_raw --out docs/images/clean_vs_raw.png
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+", help="run folders containing log.csv")
ap.add_argument("--out", required=True)
a = ap.parse_args()

fig, ax = plt.subplots(figsize=(7, 4.5))
for run in a.runs:
    with open(os.path.join(run, "log.csv")) as f:
        rows = [r for r in csv.DictReader(f) if r["val_loss"]]
    if not rows:
        raise SystemExit(f"{run} has no validation results yet")
    tokens = [int(r["tokens"]) / 1e9 for r in rows]
    losses = [float(r["val_loss"]) for r in rows]
    name = os.path.basename(run.rstrip("/\\"))
    ax.plot(tokens, losses, label=f"{name} (final {losses[-1]:.2f})")
    print(f"{name}: final val loss {losses[-1]:.4f} after {tokens[-1]:.2f}B tokens")

ax.set_xlabel("tokens seen (billions)")
ax.set_ylabel("validation loss")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(a.out, dpi=150)
print(f"wrote {a.out}")
