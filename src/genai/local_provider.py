import os
import json
import requests
import regex
from typing import Any, Dict

raw_sample = {
    "suggestions": [
        {
            "type": "summary_tweak",
            "target_section": "summary",
            "original": "",
            "proposed": "Backend engineer skilled in Python, Django/DRF and RESTful APIs, with experience optimizing SQL queries and improving production API latency.",
            "rationale": "Aligns directly with JD focus on Python, Django/DRF, REST APIs, and performant SQL while staying honest to existing backend work.",
            "needs_evidence": False,
        },
        {
            "type": "rewrite_bullet",
            "target_section": "experience",
            "original": "Refactored business logic into DRF serializers/viewsets, improving maintainability and reducing median API latency by 25% through optimized request handling techniques.",
            "proposed": "Refactored Django REST Framework serializers/viewsets and optimized SQL queries, reducing median API latency by 25% for production endpoints.",
            "rationale": "Injects explicit mention of Django REST Framework and SQL performance, echoing JD emphasis on DRF and performant database access.",
            "needs_evidence": False,
        },
        {
            "type": "add_bullet",
            "target_section": "experience",
            "original": "",
            "proposed": "Containerized a Django REST service with Docker and deployed on Linux, streamlining local development and test environments.",
            "rationale": "Covers JD requirements for Docker and Linux; phrased so candidate can adopt it if they have genuinely done this work.",
            "needs_evidence": True,
        },
        {
            "type": "keyword_injection",
            "target_section": "skills",
            "original": "",
            "proposed": "Skills: Python, Django, Django REST Framework, PostgreSQL/MySQL, RESTful APIs, Docker, Linux, Git, AWS (basic).",
            "rationale": "Aligns skills section terminology with JD phrasing (RESTful APIs, PostgreSQL/MySQL, Docker, Linux, Git, AWS basics).",
            "needs_evidence": False,
        },
    ],
    "notes": [
        "Confirm you have actually used Docker, Linux, and AWS in a meaningful way before adopting those bullets verbatim.",
        "Where possible, add concrete metrics (latency, throughput, error rates) to 1–2 bullets per project or role.",
    ],
    "guardrails": [
        "Do not invent new employers, positions, or dates.",
        "Do not claim PostgreSQL, Docker, or AWS experience unless the candidate has actually used them.",
        "Keep all bullets concise, action-oriented, and technically accurate.",
    ],
}


class LocalProvider:
    """
    Simple provider that calls a local Ollama server at http://localhost:11434/api/generate.
    Works fully offline once a model is pulled with `ollama pull <model>`.
    """

    def __init__(self):
        self.url = os.getenv("GENAI_LOCAL_URL", "http://localhost:11434/api/generate")
        self.model = os.getenv("LOCAL_MODEL", "phi3")

    def generate_json(
        self, system: str, user: str, *, temperature: float = 0.2, max_tokens: int = 800
    ) -> Dict[str, Any]:

        prompt = (
            f"{system}\n\n"
            f"{user}\n\n"
            "Return STRICT JSON ONLY. Do not add commentary. "
            "Do not add code fences. Do not add explanations."
        )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "num_predict": max_tokens,
        }

        r = requests.post(self.url, json=payload, timeout=300)
        r.raise_for_status()
        raw = r.json().get("response") or ""

        # --- 1) Try direct JSON ---

        try:
            return raw.copy()  # json.loads(raw)
        except Exception:
            pass

        # --- 2) Extract first JSON object using regex ---
        json_match = regex.search(r"\{(?:[^{}]|(?R))*\}", raw, flags=regex.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except Exception:
                pass

        # --- 3) Fallback: return safe structure ---
        return {
            "suggestions": [],
            "notes": [f"Ollama returned non-JSON text. Raw output (truncated): {raw[:180]}"],
            "guardrails": ["Ensure model prompt enforces JSON-only output."],
        }
