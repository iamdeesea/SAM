"""
Split juliflora dataset into train / test txt files for SAM‑Adapter.
Run:  python tools/split_dataset.py
"""
import random, os, pathlib, json, argparse
ROOT = pathlib.Path("load/juliflora")
TRAIN_TXT = ROOT / "train.txt"
TEST_TXT  = ROOT / "test.txt"
RATIO = 0.8                         # 80 % train, 20 % test

samples = sorted([p.name for p in ROOT.iterdir() if p.name.startswith("sample_")])
random.shuffle(samples)
split = int(len(samples) * RATIO)
TRAIN_TXT.write_text("\n".join(samples[:split]))
TEST_TXT.write_text("\n".join(samples[split:]))

print(f"✔ Saved {TRAIN_TXT} ({split}) and {TEST_TXT} ({len(samples)-split})")
