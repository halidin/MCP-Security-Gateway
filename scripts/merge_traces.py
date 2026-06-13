from __future__ import annotations

import json
from pathlib import Path

MERGE_JOBS = [
    (
        [Path("data/traces/train_1.jsonl"), Path("data/traces/train_2.jsonl"), Path("data/traces/train_3.jsonl")],
        Path("data/traces/train.jsonl"),
    ),
    (
        [Path("data/traces/test_1.jsonl"), Path("data/traces/test_2.jsonl"), Path("data/traces/test_3.jsonl")],
        Path("data/traces/test.jsonl"),
    ),
]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping.")
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    for inputs, output in MERGE_JOBS:
        merged, seen = [], set()
        for path in inputs:
            rows = load_jsonl(path)
            for r in rows:
                if r["id"] not in seen:
                    merged.append(r)
                    seen.add(r["id"])
            print(f"  {path.name}: {len(rows)} rows")

        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            for r in merged:
                f.write(json.dumps(r, ensure_ascii=True) + "\n")

        print(f"Merged {len(merged)} records -> {output}\n")


if __name__ == "__main__":
    main()
