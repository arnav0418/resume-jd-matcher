from src.genai.analyzer import (
    split_resume_sections,
    find_underused_keywords,
    find_unquantified_bullets,
    build_gap_report,
)
from src.genai.postcheck import estimate_snippet_lift

JD = {
    "id": "backend",
    "title": "Backend SDE (Django/REST)",
    "skills": ["python", "django", "rest", "sql", "postgres", "docker", "linux", "git", "aws"],
    "text": "We build RESTful APIs using Python and Django/DRF. Experience with SQL and PostgreSQL. Docker, Linux, Git, and AWS preferred.",
}

RESUME = """
SUMMARY
Backend engineer with experience building APIs in Python and Django.

SKILLS
Python, Django, SQL, Git

EXPERIENCE
- Built APIs with Django.
- Improved performance.

PROJECTS
- Personal service with GitHub Actions.
"""


def test_split_sections_basic():
    parts = split_resume_sections(RESUME)
    assert "summary" in parts and "skills" in parts and "experience" in parts


def test_underused_keywords_simple():
    under = find_underused_keywords(RESUME, JD["text"], JD["skills"], min_count=1)
    # we expect many underused (resume is thin); at least a few known terms should be underused
    assert "aws" in under and "docker" in under and "linux" in under


def test_unquantified_bullets():
    bad = find_unquantified_bullets(RESUME, limit=5)
    assert any("Improved performance" in b for b in bad)


def test_gap_report_includes_missing_skills():
    gap = build_gap_report(RESUME, JD)
    assert "missing_skills" in gap and "aws" in [s.lower() for s in gap["missing_skills"]]


def test_postcheck_lift_increases_with_relevant_snippet():
    snippet = "Containerized Django REST APIs with Docker; deployed on AWS EC2; improved p95 latency by 35%."
    lift = estimate_snippet_lift(RESUME, JD, snippet, target_section="experience")
    assert lift["delta"]["semantic"] >= 0.0
    assert lift["delta"]["skills"] > 0.0  # should cover docker/aws
