import json
from src.score_stub import compute_stub_scores, _tokenize, _jaccard, _skills_found


def test_compute_stub_scores_backend():
    jd = json.load(open("data/jds.json"))[0]
    resume_text = (
        "Python developer skilled in Django and REST APIs. "
        "Experience with SQL, Docker, Linux, and Git."
    )
    result = compute_stub_scores(resume_text, jd)

    expected_sem = round(_jaccard(_tokenize(resume_text), _tokenize(jd["text"])), 4)
    expected_matched = _skills_found(resume_text, jd["skills"])
    expected_cov = round(len(expected_matched) / len(jd["skills"]), 4)
    expected_overall = round(100 * (0.7 * expected_sem + 0.3 * expected_cov), 2)

    assert result["semantic_similarity"] == expected_sem
    assert result["skills_coverage"] == expected_cov
    assert result["overall_score"] == expected_overall
