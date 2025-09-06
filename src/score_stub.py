import re
from typing import Dict, List


STOPWORDS = {
    "a",
    "an",
    "and",
    "the",
    "to",
    "in",
    "of",
    "with",
    "for",
    "on",
    "at",
    "by",
    "from",
    "is",
    "are",
    "was",
    "were",
    "be",
    "as",
    "it",
    "this",
    "that",
    "we",
    "you",
    "our",
    "using",
    "experience",
    "basic",
    "basics",
}


def _tokenize(text: str) -> List[str]:
    # words with letters/numbers, length >= 2
    return [
        w for w in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(w) >= 2 and w not in STOPWORDS
    ]


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union


def _skills_found(resume_text: str, skills: List[str]) -> List[str]:
    found = []
    lower = resume_text.lower()
    for s in skills:
        # Approx: case-insensitive substring; escape special chars
        pat = re.escape(s.lower())
        if re.search(pat, lower):
            found.append(s)
    return sorted(list(dict.fromkeys(found)))  # dedupe, keep order


def compute_stub_scores(resume_text: str, jd: Dict, top_n: int = 3) -> Dict:
    jd_text = jd.get("text", "")
    jd_skills = jd.get("skills", [])

    # Semantic (stub): Jaccard overlap of tokens
    sem = _jaccard(_tokenize(resume_text), _tokenize(jd_text))

    # Skills coverage
    matched = _skills_found(resume_text, jd_skills)
    coverage = (len(matched) / len(jd_skills)) if jd_skills else 0.0
    missing = [s for s in jd_skills if s not in matched]

    # Final score
    final = 100.0 * (0.7 * sem + 0.3 * coverage)

    # Sentence-level “explanations” (stub): pick JD sentences most overlapping with resume
    sentences = re.split(r"(?<=[.!?])\s+", jd_text.strip())
    scored = []
    for s in sentences:
        sim = _jaccard(_tokenize(resume_text), _tokenize(s))
        scored.append({"sentence": s, "similarity": round(sim, 4)})
    scored.sort(key=lambda x: x["similarity"], reverse=True)
    top_sent = scored[:top_n] if sentences else []

    return {
        "jd_id": jd.get("id"),
        "overall_score": round(final, 2),
        "semantic_similarity": round(sem, 4),
        "skills_coverage": round(coverage, 4),
        "matched_skills": matched,
        "missing_skills": missing,
        "resume_text_preview": resume_text,
        "top_matching_jd_sentences": top_sent,
    }
