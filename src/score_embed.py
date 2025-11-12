from __future__ import annotations

import re
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

# add this import near the top
from skills import find_skills, load_skill_aliases


# ---- model loading (lazy singletons) ----
_MODEL: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        # Fast, small, great baseline. Downloads once to cache.
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")  # ~80MB
    return _MODEL


# ---- small helpers ----
def _tokenize_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


# public helper for other modules (post-check uses this)
def embed_text(text: str):
    return _embed_text(text)


def _chunk_words(words: List[str], size: int = 250, overlap: int = 50) -> List[str]:
    """
    Break a long text into ~word-sized chunks so we don't lose information
    to tokenizer truncation. ~250 words loosely maps to model's max tokens.
    """
    if not words:
        return []
    chunks = []
    i = 0
    n = len(words)
    while i < n:
        j = min(i + size, n)
        chunks.append(" ".join(words[i:j]))
        if j == n:
            break
        i = j - overlap  # slide with overlap
        if i < 0:
            i = 0
    return chunks


def _embed_text(text: str) -> np.ndarray:
    """
    Embed (possibly long) text by chunking -> mean-pooling chunk embeddings.
    Returns a single L2-normalized vector.
    """
    model = _get_model()
    words = _tokenize_words(text)
    if not words:
        # fall back to embedding of empty string (will be zero-ish vector)
        vec = model.encode([""], normalize_embeddings=True)[0]
        return vec.astype(np.float32)

    chunks = _chunk_words(words, size=250, overlap=50)
    embs = model.encode(chunks, normalize_embeddings=True)
    # mean-pool then renormalize to unit vector
    mean_vec = np.mean(embs, axis=0)
    norm = np.linalg.norm(mean_vec) + 1e-12
    return (mean_vec / norm).astype(np.float32)


"""
def _skills_found(resume_text: str, skills: List[str]) -> List[str]:
    found = []
    lower = resume_text.lower()
    for s in skills:
        pat = re.escape(s.lower())
        if re.search(pat, lower):
            found.append(s)
    # dedupe, keep order of first occurrence
    return sorted(list(dict.fromkeys(found)), key=lambda x: found.index(x))
"""


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def compute_embed_scores(resume_text: str, jd: Dict, top_n: int = 3) -> Dict:
    jd_text = jd.get("text", "") or ""
    jd_skills = jd.get("skills", []) or []

    # Semantic similarity via embeddings
    resume_vec = _embed_text(resume_text)
    jd_vec = _embed_text(jd_text)
    semantic = _cosine(resume_vec, jd_vec)  # already normalized → cosine in [~0,1]

    # Skills coverage
    """
    matched = _skills_found(resume_text, jd_skills)
    coverage = (len(matched) / len(jd_skills)) if jd_skills else 0.0
    missing = [s for s in jd_skills if s not in matched]
    """
    # Skills coverage (alias-aware, word-boundary safe)
    aliases = load_skill_aliases()
    matched = find_skills(resume_text, jd_skills, alias_map=aliases)
    coverage = (len(matched) / len(jd_skills)) if jd_skills else 0.0
    missing = [s for s in jd_skills if s not in matched]

    # Final score (same formula)
    final = 100.0 * (0.7 * semantic + 0.3 * coverage)

    # Explainability: sentence-level sims (embed each JD sentence vs resume vector)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", jd_text.strip()) if s]
    top_sent = []
    if sentences:
        model = _get_model()
        sent_embs = model.encode(sentences, normalize_embeddings=True)
        sims = np.dot(sent_embs, resume_vec)  # cosine per sentence
        order = np.argsort(-sims)[:top_n]
        top_sent = [{"sentence": sentences[i], "similarity": float(sims[i])} for i in order]

    return {
        "jd_id": jd.get("id"),
        "overall_score": round(final, 2),
        "semantic_similarity": round(semantic, 4),
        "skills_coverage": round(coverage, 4),
        "matched_skills": matched,
        "missing_skills": missing,
        "resume_text_preview": resume_text,
        "top_matching_jd_sentences": top_sent,
    }
