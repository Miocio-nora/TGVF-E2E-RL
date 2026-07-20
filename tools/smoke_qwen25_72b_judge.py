"""Validate the pinned Qwen2.5-72B OpenAI-compatible judge service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from urllib import request


EXPECTED_MODEL = "Qwen2.5-72B-Instruct"


def _json_request(url: str, payload: dict[str, object] | None = None) -> dict[str, object]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, method="GET" if body is None else "POST")
    req.add_header("Content-Type", "application/json")
    with request.urlopen(req, timeout=120) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8012/v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    start = time.monotonic()
    models = _json_request(f"{base_url}/models")
    served = [item.get("id") for item in models.get("data", [])]
    if EXPECTED_MODEL not in served:
        raise RuntimeError(f"expected {EXPECTED_MODEL}, service returned {served}")

    completion = _json_request(
        f"{base_url}/chat/completions",
        {
            "model": EXPECTED_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with only the text TGVF_JUDGE_READY.",
                }
            ],
            "temperature": 0,
            "max_tokens": 32,
            "seed": 42,
        },
    )
    choices = completion.get("choices", [])
    if not choices or not choices[0].get("message", {}).get("content"):
        raise RuntimeError("judge service returned no assistant content")

    result = {
        "status": "pass",
        "model": EXPECTED_MODEL,
        "served_models": served,
        "response": choices[0]["message"]["content"],
        "finish_reason": choices[0].get("finish_reason"),
        "usage": completion.get("usage"),
        "elapsed_seconds": time.monotonic() - start,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
