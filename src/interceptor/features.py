from __future__ import annotations


def combine_goal_and_trace(user_goal: str, agent_trace: str) -> str:
    """Build a single text field used by vector-based classifiers."""
    user_goal = (user_goal or "").strip()
    agent_trace = (agent_trace or "").strip()
    return f"GOAL: {user_goal}\nTRACE: {agent_trace}"


def parse_trace_steps(agent_trace: str) -> list[str]:
    """Fallback step parser when explicit trace_steps are unavailable."""
    text = (agent_trace or "").strip()
    if not text:
        return []

    parts = [p.strip() for p in text.replace("\n", ". ").split(".")]
    return [p for p in parts if p]
