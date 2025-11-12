from __future__ import annotations

import numpy as np
from typing import Dict, Iterable, List

from src.skills import find_skills, load_skill_aliases
from src.score_embed import embed_text  # uses MiniLM embed
from src.genai.analyzer import split_resume_sections


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _semantic_similarity(text_a: str, text_b: str) -> float:
    va = embed_text(text_a)
    vb = embed_text(text_b)
    return _cos(va, vb)


def estimate_snippet_lift(
    resume_text: str, jd: Dict, snippet: str, target_section: str | None = None
) -> Dict:
    """
    Estimate delta in semantic similarity and skills coverage if 'snippet' were added.
    We don't rewrite the whole resume; we approximate by appending snippet to the chosen section (or to body).
    """
    jd_text = jd.get("text", "") or ""
    jd_skills = jd.get("skills", []) or []

    # baseline
    base_sem = _semantic_similarity(resume_text, jd_text)
    aliases = load_skill_aliases()
    base_matched = find_skills(resume_text, jd_skills, alias_map=aliases)
    base_cov = (len(base_matched) / len(jd_skills)) if jd_skills else 0.0

    # apply snippet
    sections = split_resume_sections(resume_text)
    ts = (target_section or "body").lower()
    combined = dict(sections)
    combined[ts] = (combined.get(ts, "") + "\n" + (snippet or "")).strip()
    # reconstruct naive resume text order
    order = [
        "summary",
        "skills",
        "experience",
        "projects",
        "education",
        "certifications",
        "work experience",
        "body",
    ]
    new_resume = "\n\n".join([combined[s] for s in order if s in combined])

    new_sem = _semantic_similarity(new_resume, jd_text)
    new_matched = find_skills(new_resume, jd_skills, alias_map=aliases)
    new_cov = (len(new_matched) / len(jd_skills)) if jd_skills else 0.0

    return {
        "baseline": {"semantic": round(base_sem, 4), "skills": round(base_cov, 4)},
        "new": {"semantic": round(new_sem, 4), "skills": round(new_cov, 4)},
        "delta": {"semantic": round(new_sem - base_sem, 4), "skills": round(new_cov - base_cov, 4)},
    }


def estimate_lifts(resume_text: str, jd: Dict, suggestions: Iterable[Dict]) -> List[Dict]:
    """
    suggestions: iterable of {"proposed": str, "target_section": str}
    returns: each item + {"est_lift": {...}}
    """
    out = []
    for s in suggestions:
        snippet = s.get("proposed", "")
        section = s.get("target_section")
        lift = estimate_snippet_lift(resume_text, jd, snippet, section)
        item = dict(s)
        item["est_lift"] = lift["delta"]
        out.append(item)
    return out
