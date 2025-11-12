from __future__ import annotations

import json
import os
from typing import Any, Dict, Protocol


class LLMProvider(Protocol):
    def generate_json(
        self, system: str, user: str, *, temperature: float = 0.2, max_tokens: int = 8000
    ) -> Dict[str, Any]: ...


class MockProvider:
    """
    Deterministic provider for tests and offline dev.
    Returns a small set of plausible suggestions with the required schema.
    """

    def generate_json(
        self, system: str, user: str, *, temperature: float = 0.2, max_tokens: int = 8000
    ) -> Dict[str, Any]:
        payload = {
            "suggestions": [
                {
                    "type": "rewrite_bullet",
                    "target_section": "experience",
                    "original": "",
                    "proposed": "Containerized Django REST APIs with Docker; deployed to AWS EC2; improved p95 latency by 35%.",
                    "rationale": "Aligns to JD: Docker, AWS, RESTful APIs, measurable impact.",
                    "needs_evidence": True,
                },
                {
                    "type": "summary_tweak",
                    "target_section": "summary",
                    "original": "",
                    "proposed": "Backend engineer skilled in Python, Django/DRF, PostgreSQL; delivered RESTful APIs and CI on GitHub Actions.",
                    "rationale": "JD emphasizes DRF, SQL (PostgreSQL), and delivery.",
                    "needs_evidence": False,
                },
            ],
            "notes": [
                "Verify AWS deployment details before adding.",
                "Add Postgres migrations bullet if applicable.",
            ],
            "guardrails": ["Do not invent employers, dates, or certifications."],
        }
        return payload


def _load_openai():
    try:
        from openai import OpenAI

        return OpenAI
    except Exception as e:
        raise RuntimeError("OpenAI client not installed. Run: pip install openai>=1.0.0") from e


class OpenAIProvider:
    """
    Uses the new OpenAI client (>=1.0). Requires OPENAI_API_KEY in env.
    Model: set via OPENAI_MODEL (default: gpt-4o-mini).
    """

    def __init__(self, model: str | None = None):
        OpenAI = _load_openai()
        self.client = OpenAI()
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def generate_json(
        self, system: str, user: str, *, temperature: float = 0.2, max_tokens: int = 8000
    ) -> Dict[str, Any]:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content
        try:
            return json.loads(content)
        except Exception as e:
            raise ValueError(
                f"Provider returned non-JSON or invalid JSON: {content[:200]}..."
            ) from e


def provider_from_env() -> LLMProvider:
    """
    GENAI_PROVIDER: 'openai' | 'mock' (default: mock)
    """
    which = os.getenv("GENAI_PROVIDER", "mock").lower().strip()
    if which == "openai":
        # fail early if no key
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set in environment.")
        return OpenAIProvider()
    return MockProvider()
