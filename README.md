# Resume ↔ JD Matching Demo

Lightweight, demo-ready system that accepts a resume, lets the user pick a JD, and returns:
- Overall match score (0–100)
- Semantic similarity
- Skills coverage (matched vs missing)
- Optional explanations (top matching JD sentences)

## v1 Scope (In/Out)
- In: PDF/DOCX/TXT upload (non-OCR), 2–3 preloaded JDs, embedding-based similarity + skills overlap, single-page UI, JSON download, basic validation
- Out: OCR, multi-JD ranking at scale, advanced NER/reasoning, auth/databases/analytics

## Non-Functional
- Simplicity + speed (≤3–5s typical), reproducible pinned deps, portable local/free-tier deploy, no file storage by default

## Stack
- Python 3.10+
- Streamlit (UI), sentence-transformers, torch, pdfplumber, python-docx, numpy, regex
- Optional API later (FastAPI/Flask)

## Scoring (Plan B)
final = 100 * (0.7 * semantic_similarity + 0.3 * skills_coverage)

## Runbook (will update as we go)
1. Step 0: Repo + hygiene ✅
2. Step 1: Walking skeleton (thin end-to-end UI) ⏳
3. Step 2+: Vertical slices (extraction → embeddings → scoring → explainability)

## Running the demo

Install the project in editable mode from the repository root:

```bash
pip install -e .
```

Then launch the Streamlit app:

```bash
streamlit run app/app.py
```
