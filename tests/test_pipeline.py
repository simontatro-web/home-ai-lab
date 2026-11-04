import csv
import json
import random
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True, capture_output=True, text=True)


def test_prepare_train_resume_sample(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    rng = random.Random(0)
    words = "the server trains a small model on clean data and the loss goes down".split()
    for i in range(60):
        (corpus / f"d{i}.txt").write_text(" ".join(rng.choice(words) for _ in range(150)))
    original = (corpus / "d0.txt").read_text()
    (corpus / "copy.txt").write_text(original.replace(" ", "  "))  # same text, messier spacing

    tok, data = tmp_path / "tok.json", tmp_path / "data"
    run("training/prepare_data.py", "--src", corpus, "--out", data, "--tokenizer", tok,
        "--train-tokenizer", "--vocab-size", 300, "--val-frac", 0.1)
    meta = json.loads((data / "meta.json").read_text())
    assert meta["stats"]["duplicates"] == 1

    raw = tmp_path / "raw"
    run("training/prepare_data.py", "--src", corpus, "--out", raw, "--tokenizer", tok,
        "--no-clean", "--val-from", data)
    assert (raw / "val.bin").read_bytes() == (data / "val.bin").read_bytes()
    # the 6 validation documents, plus any messy copy of one of them
    assert json.loads((raw / "meta.json").read_text())["stats"]["held_out_for_val"] >= 6

    out = tmp_path / "run"
    cfg = {"run_name": "t", "model": {"n_layer": 1, "d_model": 32, "n_heads": 2, "ctx": 16},
           "data_dir": str(data), "lr": 1e-3, "warmup_steps": 2, "max_steps": 6, "batch_size": 2,
           "log_every": 1, "eval_every": 3, "eval_batches": 1, "out_dir": str(out)}
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg))
    run("training/train.py", "--config", path)

    cfg["max_steps"] = 8
    path.write_text(yaml.safe_dump(cfg))
    run("training/train.py", "--config", path, "--resume")
    with open(out / "log.csv") as f:
        rows = list(csv.DictReader(f))
    assert int(rows[0]["step"]) == 0 and int(rows[-1]["step"]) == 7

    run("training/sample.py", "--ckpt", out / "ckpt.pt", "--tokenizer", tok, "--tokens", 3)


def test_plot_reads_real_log(tmp_path):
    run_dir = tmp_path / "m"
    run_dir.mkdir()
    (run_dir / "log.csv").write_text(
        "step,train_loss,val_loss,lr,tokens,elapsed_s\n0,6,5.9,1e-3,10,1\n1,5,,1e-3,20,2\n2,4,3.9,1e-3,30,3\n"
    )
    png = tmp_path / "p.png"
    run("analysis/plot_logs.py", run_dir, "--out", png)
    assert png.stat().st_size > 0
