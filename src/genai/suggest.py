from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from src.genai.llm import provider_from_env, LLMProvider
from src.genai.analyzer import build_gap_report, split_resume_sections
from src.genai.postcheck import estimate_lifts


SYSTEM_PROMPT = """You are a resume improvement assistant for the Indian job market.
Only propose edits grounded in the user's resume—do not invent employers, dates, roles, projects, or certifications.
If a suggestion requires evidence or user confirmation, set "needs_evidence": true.
Each bullet must be ≤ 25 words, use a strong verb, name the tech, and include a metric if available.
Return strictly valid JSON according to the requested schema, with no extra text."""


def _format_user_prompt(jd: Dict, resume_text: str, gap_report: Dict) -> str:
    # tiny splitter so the LLM sees some structure
    sections = split_resume_sections(resume_text)
    summary = sections.get("summary", "")
    skills = sections.get("skills", "")
    experience = sections.get("experience", sections.get("work experience", ""))
    projects = sections.get("projects", "")

    return f"""JD_TITLE: {jd.get('title','')}
JD_TEXT:
{jd.get('text','')}

JD_SKILLS: {', '.join(jd.get('skills', []))}

RESUME_SECTIONS:
Summary:
{summary}

Skills:
{skills}

Experience:
{experience}

Projects:
{projects}

GAP_REPORT (JSON):
{json.dumps(gap_report, ensure_ascii=False, indent=2)}

CONSTRAINTS:
- Use JD phrasing where truthful (e.g., RESTful, DRF, PostgreSQL).
- Do not change dates, employers, or degrees.
- If information is missing, propose a phrasing with "needs_evidence": true.

Return JSON with this shape:
{{
  "suggestions": [
    {{
      "type": "rewrite_bullet" | "add_bullet" | "summary_tweak" | "keyword_injection" | "section_order",
      "target_section": "summary" | "skills" | "experience" | "projects",
      "original": "<optional>",
      "proposed": "<concise ATS-friendly sentence>",
      "rationale": "<why this helps for the JD>",
      "needs_evidence": true | false
    }}
  ],
  "notes": ["..."],
  "guardrails": ["..."]
}}"""


REQUIRED_SUG_KEYS = {"type", "target_section", "proposed", "rationale", "needs_evidence"}
ALLOWED_TYPES = {
    "rewrite_bullet",
    "add_bullet",
    "summary_tweak",
    "keyword_injection",
    "section_order",
}
ALLOWED_SECTIONS = {"summary", "skills", "experience", "projects"}


def _validate_response(
    payload: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    if not isinstance(payload, dict):
        raise ValueError("LLM response is not a JSON object.")

    suggestions = payload.get("suggestions", [])
    notes = payload.get("notes", [])
    guardrails = payload.get("guardrails", [])

    if not isinstance(suggestions, list):
        raise ValueError("`suggestions` must be a list.")

    cleaned: List[Dict[str, Any]] = []
    for s in suggestions:
        if not isinstance(s, dict):
            continue
        missing = REQUIRED_SUG_KEYS - set(s.keys())
        if missing:
            continue
        if s["type"] not in ALLOWED_TYPES:
            continue
        if s["target_section"] not in ALLOWED_SECTIONS:
            continue
        if not isinstance(s["needs_evidence"], bool):
            continue
        # keep only minimal fields + optional original
        item = {
            "type": s["type"],
            "target_section": s["target_section"],
            "proposed": s["proposed"].strip(),
            "rationale": s["rationale"].strip(),
            "needs_evidence": bool(s["needs_evidence"]),
        }
        if "original" in s and isinstance(s["original"], str):
            item["original"] = s["original"].strip()
        cleaned.append(item)

    return (
        cleaned,
        [n for n in notes if isinstance(n, str)],
        [g for g in guardrails if isinstance(g, str)],
    )


def generate_improvements(resume_text: str, jd: Dict) -> Dict[str, Any]:
    """
    Main entrypoint:
    - builds gap report
    - creates prompt
    - calls the provider (mock/openai)
    - validates shape
    - estimates lifts per suggestion
    """
    gap = build_gap_report(resume_text, jd)
    system = SYSTEM_PROMPT
    user = _format_user_prompt(jd, resume_text, gap)

    provider: LLMProvider = provider_from_env()
    raw = provider.generate_json(system, user, temperature=0.2, max_tokens=1200)
    suggestions, notes, guardrails = _validate_response(raw)

    # enrich with estimated lifts (semantic+skills deltas)
    enriched = estimate_lifts(resume_text, jd, suggestions)

    return {
        "gap_report": gap,
        "suggestions": enriched,
        "notes": notes,
        "guardrails": guardrails,
    }
