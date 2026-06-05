from __future__ import annotations

import argparse

from interceptor.guard import MCPGuard
from interceptor.model import DriftClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single guard decision demo.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--resource", default="")
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()

    clf = DriftClassifier.load(args.model_path)
    guard = MCPGuard(classifier=clf, threshold=args.threshold)

    decision = guard.inspect(
        user_goal=args.goal,
        agent_trace=args.trace,
        target_resource=args.resource,
    )

    print("allow:", decision.allow)
    print("reason:", decision.reason)
    print("malicious_probability:", round(decision.malicious_probability, 4))


if __name__ == "__main__":
    main()
