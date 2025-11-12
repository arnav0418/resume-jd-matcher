from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Dict
import uuid


RESULT_SCHEMA_VERSION = "1.0.0"


@dataclass
class ScoreWeights:
    semantic: float = 0.7
    skills: float = 0.3

    def as_dict(self) -> Dict[str, float]:
        return {"semantic": self.semantic, "skills": self.skills}


def wrap_result(
    core: Dict,  # expects keys: overall_score, semantic_similarity, skills_coverage, matched_skills, missing_skills, resume_text_preview, top_matching_jd_sentences, jd_id
    *,
    jd_title: str,
    backend: str,
    weights: ScoreWeights,
    latency_ms: int,
    resume_char_count: int,
    preview_limit: int = 2000,
) -> Dict:
    """
    Returns a normalized, versioned JSON payload with metadata.
    """
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    preview = (core.get("resume_text_preview") or "")[:preview_limit]
    truncated = resume_char_count > preview_limit

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": str(uuid.uuid4())[:8],
        "timestamp_utc": ts,
        "backend": backend,  # "embeddings:minilm-l6-v2" | "stub:token-overlap"
        "weights": weights.as_dict(),  # {"semantic": 0.7, "skills": 0.3}
        "latency_ms": int(latency_ms),
        "jd": {
            "id": core.get("jd_id"),
            "title": jd_title,
        },
        "metrics": {
            "overall_score": float(core.get("overall_score", 0.0)),
            "semantic_similarity": float(core.get("semantic_similarity", 0.0)),
            "skills_coverage": float(core.get("skills_coverage", 0.0)),
        },
        "skills": {
            "matched": list(core.get("matched_skills", [])),
            "missing": list(core.get("missing_skills", [])),
        },
        "explanations": {
            "top_matching_jd_sentences": list(core.get("top_matching_jd_sentences", [])),
        },
        "resume": {
            "char_count": int(resume_char_count),
            "preview_first_n": preview_limit,
            "preview": preview,
            "truncated_preview": bool(truncated),
        },
    }
