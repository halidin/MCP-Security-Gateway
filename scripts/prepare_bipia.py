from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

from bipia.data import AutoPIABuilder
from interceptor.io import load_jsonl, save_jsonl

DOMAINS = ["email", "table", "code"]


def to_str(v: object) -> str:
    return " ".join(str(x) for x in v) if isinstance(v, list) else str(v or "")


def build_malicious(data_dir: Path, seed: int, split: str) -> list[dict]:
    frames = []
    for domain in DOMAINS:
        attack_file = data_dir / (f"code_attack_{split}.json" if domain == "code" else f"text_attack_{split}.json")
        builder = AutoPIABuilder.from_name(domain)(seed=seed)
        df = builder(str(data_dir / domain / f"{split}.jsonl"), str(attack_file), enable_stealth=False)
        df["domain"] = domain
        frames.append(df)
        print(f"  {domain}: {len(df)} malicious samples")
    combined = pd.concat(frames, ignore_index=True)
    return [
        {"user_goal": str(r["question"]), "context": str(r["context"]), "label": 1, "domain": str(r["domain"]), "source": "bipia"}
        for _, r in combined.iterrows()
    ]


def build_bipia_benign(data_dir: Path) -> list[dict]:
    rows, seen = [], set()
    for split in ["train", "test"]:
        for domain in DOMAINS:
            path = data_dir / domain / f"{split}.jsonl"
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                question = to_str(row.get("question") or row.get("error") or "").strip()
                context = to_str(row.get("context", "")).strip()
                key = (question, context)
                if question and context and key not in seen:
                    seen.add(key)
                    rows.append({"user_goal": question, "context": context, "label": 0, "domain": domain, "source": "bipia"})
    print(f"  {len(rows)} BIPIA benign contexts")
    return rows


def load_synthetic_benign(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping synthetic benign.")
        return []
    rows = load_jsonl(path)
    for r in rows:
        r["label"] = 0
        r["source"] = "synthetic"
    print(f"  {len(rows)} synthetic benign samples")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bipia_dir", type=Path, default=Path("data/bipia/benchmark"))
    parser.add_argument("--synthetic_benign", type=Path, default=Path("data/processed/benign_synthetic.jsonl"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--train_size", type=int, default=5000, help="Total train samples (half malicious, half benign).")
    parser.add_argument("--test_size", type=int, default=5000, help="Total test samples (half malicious, half benign).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.bipia_dir.exists():
        print(f"ERROR: {args.bipia_dir} not found.")
        print("Clone with: git clone https://github.com/microsoft/BIPIA.git data/bipia")
        sys.exit(1)

    rng = random.Random(args.seed)

    print("Building malicious samples...")
    print("  [train split + train attacks]")
    train_mal = build_malicious(args.bipia_dir, args.seed, split="train")
    print("  [test split + test attacks]")
    test_mal = build_malicious(args.bipia_dir, args.seed, split="test")

    print("\nBuilding benign pool...")
    bipia_ben = build_bipia_benign(args.bipia_dir)
    synth_ben = load_synthetic_benign(args.synthetic_benign)

    all_benign = bipia_ben + synth_ben
    rng.shuffle(all_benign)
    print(f"  Total benign pool: {len(all_benign)}")

    n_train = args.train_size // 2
    n_test = args.test_size // 2

    if len(all_benign) < n_train + n_test:
        print(f"WARNING: only {len(all_benign)} benign available, needed {n_train + n_test}")
        n_train = len(all_benign) // 2
        n_test = len(all_benign) - n_train

    train_ben = all_benign[:n_train]
    test_ben = all_benign[n_train:n_train + n_test]

    rng.shuffle(train_mal)
    rng.shuffle(test_mal)

    train_samples = train_mal[:n_train] + train_ben
    test_samples = test_mal[:n_test] + test_ben

    rng.shuffle(train_samples)
    rng.shuffle(test_samples)

    for i, s in enumerate(train_samples):
        s["id"] = f"train-{i}"
    for i, s in enumerate(test_samples):
        s["id"] = f"test-{i}"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(train_samples, args.out_dir / "train.jsonl")
    save_jsonl(test_samples, args.out_dir / "test.jsonl")

    mal_train = sum(1 for s in train_samples if s["label"] == 1)
    ben_train = sum(1 for s in train_samples if s["label"] == 0)
    mal_test = sum(1 for s in test_samples if s["label"] == 1)
    ben_test = sum(1 for s in test_samples if s["label"] == 0)

    print(f"\nTrain: {len(train_samples)} total | {mal_train} malicious | {ben_train} benign")
    print(f"Test:  {len(test_samples)} total  | {mal_test} malicious  | {ben_test} benign")
    print(f"\nSaved -> {args.out_dir / 'train.jsonl'}")
    print(f"Saved -> {args.out_dir / 'test.jsonl'}")


if __name__ == "__main__":
    main()
