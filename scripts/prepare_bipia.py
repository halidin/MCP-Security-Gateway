from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

from bipia.data import AutoPIABuilder
from interceptor.io import save_jsonl

DOMAINS = ["email", "table", "code"]


def to_str(v: object) -> str:
    return " ".join(str(x) for x in v) if isinstance(v, list) else str(v or "")


def build_malicious(data_dir: Path, seed: int, split: str) -> pd.DataFrame:
    frames = []
    for domain in DOMAINS:
        attack_file = data_dir / (f"code_attack_{split}.json" if domain == "code" else f"text_attack_{split}.json")
        builder = AutoPIABuilder.from_name(domain)(seed=seed)
        df = builder(str(data_dir / domain / f"{split}.jsonl"), str(attack_file), enable_stealth=False)
        df["domain"] = domain
        frames.append(df)
        print(f"  {domain}: {len(df)} malicious samples")
    return pd.concat(frames, ignore_index=True)


def build_benign(data_dir: Path, splits: list[str]) -> list[dict]:
    rows = []
    seen = set()
    for split in splits:
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
                    rows.append({"question": question, "context": context, "domain": domain})
    print(f"  {len(rows)} benign contexts ({'+'.join(splits)})")
    return rows


def make_samples(mal_df: pd.DataFrame, benign_rows: list[dict], target: int, rng: random.Random) -> list[dict]:
    malicious = [
        {"user_goal": str(r["question"]), "context": str(r["context"]), "label": 1, "domain": str(r["domain"]), "source": "bipia"}
        for _, r in mal_df.iterrows()
    ]
    benign = [
        {"user_goal": r["question"], "context": r["context"], "label": 0, "domain": r["domain"], "source": "bipia"}
        for r in benign_rows
    ]
    n_each = min(target // 2, len(malicious), len(benign))
    rng.shuffle(malicious)
    rng.shuffle(benign)
    samples = malicious[:n_each] + benign[:n_each]
    rng.shuffle(samples)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bipia_dir", type=Path, default=Path("data/bipia_repo/benchmark"),
                        help="Path to the BIPIA benchmark directory. Clone with: git clone https://github.com/microsoft/BIPIA.git data/bipia_repo")
    parser.add_argument("--out_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--train_size", type=int, default=6000)
    parser.add_argument("--test_size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.bipia_dir.exists():
        print(f"ERROR: {args.bipia_dir} not found.")
        print("Clone the BIPIA repo first:")
        print("  git clone https://github.com/microsoft/BIPIA.git data/bipia_repo")
        sys.exit(1)

    rng = random.Random(args.seed)

    print("Building train malicious samples (train attacks)...")
    train_mal_df = build_malicious(args.bipia_dir, args.seed, split="train")

    print("\nBuilding test malicious samples (test attacks)...")
    test_mal_df = build_malicious(args.bipia_dir, args.seed, split="test")

    print("\nBuilding benign samples...")
    all_benign = build_benign(args.bipia_dir, splits=["train", "test"])

    train_samples = make_samples(train_mal_df, all_benign, args.train_size, rng)
    test_samples = make_samples(test_mal_df, all_benign, args.test_size, rng)

    for i, s in enumerate(train_samples):
        s["id"] = f"train-{i}"
    for i, s in enumerate(test_samples):
        s["id"] = f"test-{i}"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(train_samples, args.out_dir / "train.jsonl")
    save_jsonl(test_samples, args.out_dir / "test.jsonl")

    domain_counts: dict[str, int] = {}
    for s in train_samples:
        domain_counts[s["domain"]] = domain_counts.get(s["domain"], 0) + 1

    print(f"\nSaved {len(train_samples)} train -> {args.out_dir / 'train.jsonl'}")
    print(f"Saved {len(test_samples)} test  -> {args.out_dir / 'test.jsonl'}")
    print("Domain distribution (train):", domain_counts)


if __name__ == "__main__":
    main()
