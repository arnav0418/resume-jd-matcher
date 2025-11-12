from __future__ import annotations

import re
from typing import Dict, List

from src.skills import find_skills, load_skill_aliases


# --- Resume sectioning (very lightweight) ---

SECTION_PAT = re.compile(
    r"(?im)^\s*(summary|objective|skills|experience|work experience|projects|education|certifications)\s*[:\-]?\s*$"
)


def split_resume_sections(resume_text: str) -> Dict[str, str]:
    """
    Heuristic splitter by common headings (case-insensitive).
    Returns dict with keys like 'summary','skills','experience','projects',... (lowercase).
    Unmatched text goes into 'body'.
    """
    text = resume_text or ""
    parts = {}
    current = "body"
    buf: List[str] = []
    for line in text.splitlines():
        m = SECTION_PAT.match(line)
        if m:
            # flush previous
            if buf:
                parts[current] = (parts.get(current, "") + "\n".join(buf)).strip()
                buf = []
            current = m.group(1).lower()
        else:
            buf.append(line)
    if buf:
        parts[current] = (parts.get(current, "") + "\n".join(buf)).strip()
    return parts


# --- JD term extraction & underuse detection ---

WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\+\#\.\-]*")


def _terms(text: str) -> List[str]:
    return [t.lower() for t in WORD_RE.findall(text or "")]


def _count_occurrences(hay: str, needle: str) -> int:
    # word-ish search that respects boundaries; +/#/. allowed inside token
    pat = re.compile(rf"(?<![A-Za-z0-9]){re.escape(needle)}(?![A-Za-z0-9])", re.IGNORECASE)
    return len(pat.findall(hay or ""))


def find_underused_keywords(
    resume_text: str, jd_text: str, jd_skills: List[str], min_count: int = 1
) -> List[str]:
    """
    Keywords that appear in JD but < min_count in resume.
    Includes canonical JD skills + top unigrams/bigrams from JD (simple heuristic).
    """
    resume_lo = (resume_text or "").lower()
    jd_lo = (jd_text or "").lower()

    # seed with jd skills (canonical only; aliases are matched during skills detection)
    seed = {s.lower() for s in (jd_skills or [])}

    # add common key terms from JD (verbs & nouns-ish; keep short)
    toks = _terms(jd_lo)
    # naive filter to remove stop-ish words
    stop = {
        "and",
        "or",
        "the",
        "a",
        "an",
        "to",
        "in",
        "of",
        "for",
        "on",
        "with",
        "using",
        "we",
        "our",
        "is",
        "are",
        "as",
        "by",
    }
    nouns = [t for t in toks if len(t) >= 3 and t not in stop]
    # frequency threshold—pick top 15 terms
    from collections import Counter

    top_terms = [w for w, _ in Counter(nouns).most_common(15)]
    seed.update(top_terms)

    # evaluate underuse
    underused = []
    for term in sorted(seed):
        c = _count_occurrences(resume_lo, term)
        if c < min_count:
            underused.append(term)
    return underused


# --- Unquantified bullet detection ---

BULLET_LINE = re.compile(r"^\s*([\-•\*\u2022])\s+(.*)$")
HAS_METRIC = re.compile(r"(\b\d+(\.\d+)?\s*(%|ms|s|sec|mins?|hrs?|x|X|k|K|MB|GB)\b)|(\b\d{2,}\b)")

STRONG_VERB = {
    "led",
    "owned",
    "designed",
    "delivered",
    "shipped",
    "built",
    "implemented",
    "optimized",
    "reduced",
    "increased",
    "automated",
    "migrated",
    "deployed",
    "containerized",
    "instrumented",
    "refactored",
    "debugged",
    "launched",
    "architected",
    "integrated",
}


def find_unquantified_bullets(resume_text: str, limit: int = 5) -> List[str]:
    """
    Return up to N bullet lines that lack obvious quantification tokens.
    """
    bad: List[str] = []
    for line in (resume_text or "").splitlines():
        m = BULLET_LINE.match(line)
        if not m:
            continue
        content = m.group(2).strip()
        if not HAS_METRIC.search(content):
            bad.append(content)
        if len(bad) >= limit:
            break
    return bad


# --- Main gap report ---


def build_gap_report(resume_text: str, jd: Dict) -> Dict:
    jd_text = jd.get("text", "") or ""
    jd_skills = jd.get("skills", []) or []

    aliases = load_skill_aliases()
    matched = find_skills(resume_text, jd_skills, alias_map=aliases)
    missing = [s for s in jd_skills if s not in matched]

    underused = find_underused_keywords(resume_text, jd_text, jd_skills, min_count=1)
    unquant = find_unquantified_bullets(resume_text, limit=5)

    sections = split_resume_sections(resume_text)
    has_summary = "summary" in sections
    has_projects = "projects" in sections

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "underused_keywords": underused[:20],
        "unquantified_bullets": unquant,
        "sections": {
            "has_summary": has_summary,
            "has_projects": has_projects,
            "present": sorted(list(sections.keys())),
        },
        "notes": [
            "Quantify impact in top 1–2 bullets per role.",
            "Prefer JD phrasing where truthful (e.g., RESTful, DRF, PostgreSQL).",
        ],
    }
