from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from provider_pool import ProviderPool


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def load_env(path: Path = Path(".env")) -> None:
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                key, _, value = line.partition("=")
                value = value.split("#")[0].strip()
                os.environ.setdefault(key.strip(), value)


MCP_TOOLS = {
    "email": "a realistic work email (invoice, meeting request, support ticket, HR notice, etc.)",
    "table": "a realistic database or spreadsheet query result with rows and columns (sales data, inventory, user records, etc.)",
    "code_file": "a realistic source code file or config file (Python, YAML, JSON, etc.) with a bug report or feature description",
    "webpage": "a realistic web page or article excerpt about a technical topic, product, or news item",
    "pdf_document": "a realistic excerpt from a PDF document (contract, report, manual, or policy document)",
    "calendar": "a realistic calendar or scheduling response with events, attendees, and times",
    "slack_message": "a realistic Slack or Teams message thread from a work channel",
    "github_issue": "a realistic GitHub issue or pull request description with steps to reproduce or code changes",
    "jira_ticket": "a realistic Jira or project management ticket with description, acceptance criteria, and comments",
    "api_response": "a realistic JSON API response from a REST service (weather, payment, user profile, etc.)",
}

SYSTEM_PROMPT = """You generate synthetic MCP tool responses for AI agent testing.

Output a JSON object with exactly these keys:
- context: the raw content returned by the MCP tool (realistic, detailed, 3-10 sentences or structured data)
- user_goal: a natural user question or task about this content (what the user asked the agent to do)

Rules:
- NO injected instructions, NO hidden commands, NO "ignore previous instructions"
- Content must be clean and realistic
- user_goal must be genuinely answerable from the context
- Plain JSON only, no markdown
- VARY the type of user_goal each time - rotate between: extract a specific value, compare two items, calculate or count something, verify a condition, find who/when/where, check status, list filtered items, identify the next action needed"""


def main() -> None:
    load_env()

    parser = argparse.ArgumentParser(description="Generate synthetic benign MCP samples.")
    parser.add_argument("--output", type=Path, default=Path("data/processed/benign_synthetic.jsonl"))
    parser.add_argument("--per_tool", type=int, default=250, help="Samples per MCP tool type (10 types = 2500 total).")
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    pool = ProviderPool()

    existing = load_jsonl(args.output) if (args.append and args.output.exists()) else []
    existing_ids = {s["id"] for s in existing}

    tasks = [
        (tool, desc, i)
        for tool, desc in MCP_TOOLS.items()
        for i in range(args.per_tool)
    ]

    if args.skip:
        tasks = tasks[args.skip:]
    if args.limit:
        tasks = tasks[:args.limit]

    generated = list(existing)
    total = len(tasks)

    for idx, (tool, desc, i) in enumerate(tasks, start=1):
        sample_id = f"benign-{tool}-{i}"
        if sample_id in existing_ids:
            print(f"[{idx}/{total}] {sample_id} already exists, skipping.")
            continue

        task_types = [
            "extract a specific value", "compare two items", "calculate or count something",
            "verify a condition", "find who when or where", "check a status",
            "list filtered items", "identify the next action needed",
        ]
        task_hint = task_types[i % len(task_types)]

        try:
            result = pool.call(
                system=SYSTEM_PROMPT,
                user=f"Generate {desc}. The user_goal must be a task to '{task_hint}'. The user's agent fetched this via an MCP tool. Return the JSON object.",
            )
            context = result.get("context", "").strip()
            user_goal = result.get("user_goal", "").strip()
            if not context or not user_goal:
                print(f"[{idx}/{total}] {sample_id} empty fields, skipping.")
                continue
            generated.append({
                "id": sample_id,
                "user_goal": user_goal,
                "context": context,
                "label": 0,
                "domain": tool,
                "source": "synthetic",
            })
            print(f"[{idx}/{total}] {sample_id} ok")
        except Exception as e:
            print(f"[{idx}/{total}] {sample_id} ERROR: {e}")

        save_jsonl(generated, args.output)

    print(f"\nSaved {len(generated)} benign samples -> {args.output}")
    tool_counts: dict[str, int] = {}
    for s in generated:
        tool_counts[s["domain"]] = tool_counts.get(s["domain"], 0) + 1
    print("Per tool:", tool_counts)


if __name__ == "__main__":
    main()
