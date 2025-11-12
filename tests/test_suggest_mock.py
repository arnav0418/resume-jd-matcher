from src.genai.suggest import generate_improvements


def test_generate_improvements_mock_provider(monkeypatch):
    monkeypatch.setenv("GENAI_PROVIDER", "mock")  # force mock

    jd = {
        "id": "backend",
        "title": "Backend SDE (Django/REST)",
        "skills": ["python", "django", "rest", "sql", "postgres", "docker", "linux", "git", "aws"],
        "text": "We build RESTful APIs using Python and Django/DRF. Experience with SQL and PostgreSQL. Docker, Linux, Git, and AWS preferred.",
    }
    resume = "Backend engineer with Python and Django. Built APIs; familiar with Git."

    out = generate_improvements(resume, jd)
    assert "suggestions" in out and isinstance(out["suggestions"], list)
    assert len(out["suggestions"]) >= 1

    item = out["suggestions"][0]
    # Each suggestion should carry an est_lift delta added by orchestrator
    assert "est_lift" in item
    assert set(["semantic", "skills"]).issubset(item["est_lift"].keys())
