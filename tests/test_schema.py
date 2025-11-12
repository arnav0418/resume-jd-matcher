from src.schema import wrap_result, ScoreWeights


def test_wraps_core_with_metadata():
    core = {
        "jd_id": "backend",
        "overall_score": 82.5,
        "semantic_similarity": 0.74,
        "skills_coverage": 0.63,
        "matched_skills": ["python", "django", "rest", "sql"],
        "missing_skills": ["docker", "aws"],
        "resume_text_preview": "hello" * 600,  # > 2000 chars after repeat
        "top_matching_jd_sentences": [{"sentence": "Use Django/DRF", "similarity": 0.8}],
    }
    payload = wrap_result(
        core,
        jd_title="Backend SDE (Django/REST)",
        backend="embeddings:minilm-l6-v2",
        weights=ScoreWeights(),
        latency_ms=123,
        resume_char_count=5000,
        preview_limit=2000,
    )
    assert payload["schema_version"] == "1.0.0"
    assert payload["jd"]["id"] == "backend"
    assert payload["metrics"]["overall_score"] == 82.5
    assert payload["resume"]["truncated_preview"] is True
    assert "run_id" in payload and len(payload["run_id"]) == 8
