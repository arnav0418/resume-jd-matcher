from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List


def _repo_root() -> Path:
    # .../src/skills.py -> repo root
    return Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def load_skill_aliases(path: str | None = None) -> Dict[str, List[str]]:
    """
    Load alias map once. Keys = canonical skills (lowercase).
    Values = list of aliases/variants (any case). Missing file -> {}.
    """
    if path is None:
        path = str(_repo_root() / "data" / "skill_aliases.json")
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    # normalize keys to lowercase, keep alias strings as-is
    out: Dict[str, List[str]] = {}
    for k, vals in data.items():
        out[k.lower()] = list(dict.fromkeys([v for v in vals if isinstance(v, str)]))
    return out


def _alias_to_regex(alias: str) -> str:
    """
    Convert an alias string into a regex that:
      - is case-insensitive (handled by flags in search)
      - respects word-ish boundaries so 'git' ≠ 'digital'
      - allows flexible whitespace inside multi-word aliases
    """
    alias = alias.strip()
    if not alias:
        return ""

    # Escape, then make spaces flexible
    esc = re.escape(alias)
    # allow whitespace/hyphen between words (e.g., "react js" / "react.js")
    esc = esc.replace(r"\ ", r"\s+")

    # Some hand-tuned expansions for common tech tokens
    # REST -> REST or RESTful
    if re.fullmatch(r"(?i)rest", alias, flags=re.IGNORECASE):
        esc = r"REST(?:ful)?"

    # Node.js: accept Node.js / Nodejs / Node JS
    if re.fullmatch(r"(?i)node\.?js", alias, flags=re.IGNORECASE):
        esc = r"Node(?:\.?\s*JS)?"

    # React.js: accept React / React.js / ReactJS
    if re.fullmatch(r"(?i)react(?:\.?js)?", alias, flags=re.IGNORECASE):
        esc = r"React(?:\.?JS)?"

    # PostgreSQL: accept PostgreSQL / Postgres / PSQL
    if re.fullmatch(r"(?i)postgres(?:ql)?", alias, flags=re.IGNORECASE):
        esc = r"(?:PostgreSQL|Postgres|PSQL)"

    # AWS: accept AWS / Amazon Web Services
    if re.fullmatch(r"(?i)aws", alias, flags=re.IGNORECASE):
        esc = r"(?:AWS|Amazon\s+Web\s+Services)"

    # Build custom "word" boundaries: do not allow letters/digits to touch on either side
    # Keep +, #, . as part of the token so C++ / C# / Node.js match correctly.
    return rf"(?<![A-Za-z0-9]){esc}(?![A-Za-z0-9])"


def _compile_patterns(canonical: str, aliases: Iterable[str]) -> List[re.Pattern]:
    variants = [canonical] + list(aliases)
    uniq: List[str] = []
    seen = set()
    for v in variants:
        key = v.lower().strip()
        if key and key not in seen:
            seen.add(key)
            uniq.append(v)

    pats: List[re.Pattern] = []
    for a in uniq:
        pat = _alias_to_regex(a)
        if pat:
            pats.append(re.compile(pat, flags=re.IGNORECASE))
    return pats


def find_skills(
    resume_text: str, jd_skills: List[str], alias_map: Dict[str, List[str]] | None = None
) -> List[str]:
    """
    Return canonical skills (as passed in jd_skills order) found in resume_text.
    - alias_map: mapping from canonical (lowercase) -> list of aliases
    """
    txt = resume_text or ""
    alias_map = alias_map or load_skill_aliases()
    found: List[str] = []

    for skill in jd_skills:
        key = (skill or "").lower()
        aliases = alias_map.get(key, [])
        patterns = _compile_patterns(skill, aliases)
        if any(p.search(txt) for p in patterns):
            found.append(skill)

    # keep JD order, de-dup just in case
    seen = set()
    ordered = []
    for s in found:
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    return ordered
