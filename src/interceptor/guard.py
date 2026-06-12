from __future__ import annotations
from dataclasses import dataclass
from .model import DriftClassifier


@dataclass
class GuardDecision:
    allow: bool
    reason: str
    malicious_probability: float


class MCPGuard:
    """Online guard that blocks risky tool calls before execution."""

    def __init__(self, classifier: DriftClassifier, threshold: float = 0.5, honey_resources: list[str] | None = None):
        self.classifier = classifier
        self.threshold = threshold
        #Can be customized with specific sensitive resources relevant to the application domain
        default_honey_resources = [
            "secrets.txt",
            "credentials.pdf",
            "credentials.json",
            ".env",
            "id_rsa",
            "config.yaml",
            "passwords.xlsx",
            "api_keys.txt",
            "wallet.dat",
            "backup.sql",
        ]
        self.honey_resources = set(x.lower() for x in (honey_resources or default_honey_resources))

    def inspect(self, user_goal: str, agent_trace: str, target_resource: str | None = None) -> GuardDecision:
        target = (target_resource or "").lower().strip()
        if target and target in self.honey_resources:
            return GuardDecision(
                allow=False,
                reason="Kill-switch: attempted access to honey-resource.",
                malicious_probability=1.0,
            )

        score = self.classifier.predict_proba([user_goal], [agent_trace])[0]
        if score >= self.threshold:
            return GuardDecision(
                allow=False,
                reason="Blocked: semantic drift indicates potential prompt injection.",
                malicious_probability=score,
            )

        return GuardDecision(
            allow=True,
            reason="Allowed: no malicious drift detected.",
            malicious_probability=score,
        )
