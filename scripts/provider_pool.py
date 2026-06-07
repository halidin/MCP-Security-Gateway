from __future__ import annotations

import os
import time
import json
import requests


PROVIDERS = [
    {
        "name": "Cerebras",
        "url": "https://api.cerebras.ai/v1/chat/completions",
        "model": "gpt-oss-120b",
        "key_env": "CEREBRAS_API_KEY",
    },
    {
        "name": "Groq",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "model": "openai/gpt-oss-120b",
        "key_env": "GROQ_API_KEY",
    },
    {
        "name": "NVIDIA",
        "url": "https://integrate.api.nvidia.com/v1/chat/completions",
        "model": "openai/gpt-oss-120b",
        "key_env": "NVIDIA_API_KEY",
    },
]


def _expand_providers() -> list[dict]:
    "Expand providers - supports PROVIDER_KEY and PROVIDER_KEY_1, _2, _3..."
    "You can add multiple keys for the same provider in order to parallelize calls and increase throughput."
    expanded = []
    for p in PROVIDERS:
        base = p["key_env"]
        # check numbered variants first: KEY_1, KEY_2, ...
        numbered = [f"{base}_{i}" for i in range(1, 6) if os.getenv(f"{base}_{i}")]
        if numbered:
            for env in numbered:
                expanded.append({**p, "name": f"{p['name']}-{env[-1]}", "key_env": env})
        elif os.getenv(base):
            expanded.append(p)
    return expanded


class ProviderPool:
    def __init__(self) -> None:
        self.providers = _expand_providers()
        if not self.providers:
            raise SystemExit("No API keys found. Set CEREBRAS_API_KEY, GROQ_API_KEY, or NVIDIA_API_KEY.")
        self._index = 0
        print(f"Loaded providers: {[p['name'] for p in self.providers]}")

    def _next(self) -> dict:
        p = self.providers[self._index % len(self.providers)]
        self._index += 1
        return p

    def call(self, system: str, user: str, temperature: float = 0.9, retries: int = 3) -> dict:
        tried = 0
        while tried < len(self.providers) * retries:
            p = self._next()
            api_key = os.getenv(p["key_env"])
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": p["model"],
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
            try:
                response = requests.post(p["url"], headers=headers, json=payload, timeout=60)
                if response.status_code == 429:
                    print(f"  [{p['name']}] 429 rate limit, switching provider...")
                    tried += 1
                    time.sleep(2)
                    continue
                response.raise_for_status()
                result = json.loads(response.json()["choices"][0]["message"]["content"])
                print(f"  [{p['name']}]", end=" ")
                return result
            except Exception as e:
                print(f"  [{p['name']}] ERROR: {e}, switching...")
                tried += 1
                time.sleep(2)
                continue

        raise RuntimeError("All providers failed after retries.")
