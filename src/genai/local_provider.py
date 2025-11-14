import os
import json
import requests
import re
from typing import Any, Dict


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

        # try:
        #    return json.loads(raw)
        # except Exception:
        #    pass

        # --- 2) Extract first JSON object using regex ---
        json_match = re.search(r"\{(?:[^{}]|(?R))*\}", raw, flags=re.DOTALL)
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
