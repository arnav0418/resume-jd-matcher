# Resume ↔ JD Matching Demo

A lightweight but production-minded prototype that scores how well a resume matches a job description (JD) and suggests grounded edits to close the gap. The Streamlit UI wires together text extraction, semantic scoring, skills coverage, explainability, and optional GenAI suggestions in a single-page experience.

## Key capabilities

- **Multi-format resume ingestion** – parse PDF, DOCX, or TXT files (non-OCR) and normalize whitespace for consistent downstream processing.
- **Embeddings-based similarity** – `sentence-transformers` (MiniLM) encoder with chunking and mean pooling keeps long resumes under the model token limit for robust semantic scores.
- **Alias-aware skills detection** – configurable canonical skills plus regex-safe alias mapping prevent false positives while covering variants such as “React.js” or “Amazon Web Services.”
- **Explainable outputs** – normalized JSON payload captures scores, matched/missing skills, resume previews, and top JD sentences for transparency and easy export.
- **GenAI improvement workflow** – optional providers (mock, local Ollama, OpenAI) generate resume edits, enriched with estimated lifts in similarity and skills coverage before display.

## Repository structure

```text
.
  README.md
  pyproject.toml
  requirements.txt
  data/
    jds.json                # Demo job descriptions with id/title/skills/text
    skill_aliases.json      # Canonical skill -> alias list used by skills.py
  src/
    __init__.py
    extract.py              # File parsers + text normalization helpers
    jds.py                  # Load/select JD definitions
    schema.py               # ScoreWeights dataclass + response wrapper
    score_embed.py          # Embedding backend with semantic + skills scoring
    score_stub.py           # Fast token-overlap scoring baseline
    skills.py               # Alias-aware skill extraction utilities
    genai/
      analyzer.py           # Resume sectioning + gap/keyword analysis
      llm.py                # Provider selection (mock/local/OpenAI)
      local_provider.py     # Ollama-backed provider (offline-friendly)
      postcheck.py          # Similarity/skills lift estimation per suggestion
      suggest.py            # GenAI orchestration & validation pipeline
  examples/
    sample_resume_backend.txt
    sample_resume_data_ml.txt
  tests/                   # Pytest suite exercising extract, skills, scoring, schema, and GenAI flow
    ...
  app/
    app.py                 # Streamlit UI wiring the full workflow
```

## Quick start

1. **Install dependencies** (Python 3.10+ recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Run the Streamlit app**:
   ```bash
   streamlit run app/app.py
   ```
3. **Interact**:
   - Choose a JD from the built-in dataset (`data/jds.json`).
   - Upload a resume (PDF/DOCX/TXT) or use one of the provided samples.
   - Select the scoring backend (`Embeddings` for production-quality scores, `Stub` for fast demos).
   - Click **Match** to view scores, matched/missing skills, top JD sentences, and exportable JSON.
   - (Optional) Click **✨ Improve with GenAI** after matching to see provider-generated suggestions with estimated impact.

## Testing

The project uses `pytest`. Run the full suite with:
```bash
pytest
```
Tests cover normalization, skills alias handling, scoring math, schema wrapping, analyzer/post-check heuristics, and the GenAI mock orchestration.

## Implementation guide

### 1. Resume ingestion
- `extract.extract_text_from_file` branches on file extension and delegates to PDF (`pdfplumber`), DOCX (`python-docx`), or raw text readers. The helper `_normalize` collapses whitespace and enforces a configurable length cap so downstream components receive clean input.
- The Streamlit uploader forwards `max_chars` from the UI slider to this function, keeping long resumes performant in demos.

### 2. JD management
- `data/jds.json` stores demo job descriptions with `id`, `title`, `skills`, and free-text `text`. Load them via `jds.load_jds` and select a specific JD using `jds.get_jd_by_id` (O(n) scan is sufficient for small curated lists).
- Extend the dataset by appending new objects; keep skill names consistent with aliases for best coverage.

### 3. Scoring backends
- **Embedding backend** (`score_embed.compute_embed_scores`):
  - Performs word tokenization, chunking (~250 tokens with 50-word overlap), and mean pooling to avoid truncation. Encodings are normalized so cosine similarity equals dot product.
  - Skills coverage uses `skills.find_skills` for alias-aware matching; adjust `data/skill_aliases.json` to add variants (cached via `load_skill_aliases`).
  - Produces overall score using `final = 100 * (0.7 * semantic + 0.3 * coverage)`; tweak weights through `ScoreWeights` if desired.
- **Stub backend** (`score_stub.compute_stub_scores`):
  - Tokenizes with simple regex + stopword filtering and computes Jaccard overlap as a fast semantic proxy.
  - Shares the same skills pipeline and scoring formula for consistency.
- Switching backends in the UI flips between these implementations; both return the same core schema consumed by `wrap_result`.

### 4. Result packaging & download
- `schema.wrap_result` enriches the raw scoring dictionary with metadata (schema version, run id, timestamp, backend, weights, latency, resume preview) and structuring for UI display + JSON download.

### 5. GenAI suggestion pipeline (optional)
- `genai/analyzer.py` splits resume sections heuristically, identifies missing JD skills, underused JD keywords, and unquantified bullets to produce a gap report shown to the LLM.
- `genai/suggest.py` builds the system/user prompts, selects a provider (`mock`, `local`, or `openai`), validates the JSON schema, and attaches estimated lifts per suggestion via `genai/postcheck.py`.
- Providers:
  - `MockProvider` – deterministic responses for tests/offline demos (default).
  - `LocalProvider` – targets a local Ollama server (e.g., `ollama run phi3`); simple JSON extraction fallback keeps output robust.
  - `OpenAIProvider` – wraps the official client and enforces JSON-only responses; requires `OPENAI_API_KEY` and optional `OPENAI_MODEL` env vars.
- Set the provider at runtime via the UI selector, which mutates `GENAI_PROVIDER` (and related env vars).

### 6. Estimating suggestion impact
- `genai/postcheck.estimate_snippet_lift` reuses MiniLM embeddings and skills matching to estimate the delta in semantic similarity and skills coverage if a suggestion were applied to a target section. The UI surfaces these deltas alongside each proposed edit.

## Configuration & customization

| Setting | Location | Notes |
| --- | --- | --- |
| Resume length cap | `extract.extract_text_from_file(max_chars)` | Controlled via Streamlit slider; adjust default `MAX_DEFAULT` in `extract.py` if needed. |
| Score weighting | `schema.ScoreWeights` | Update defaults or expose sliders to favor semantic vs. skills coverage. |
| Skill aliases | `data/skill_aliases.json` | Add variants per canonical skill; cache is auto-invalidated when process restarts. |
| GenAI provider | `GENAI_PROVIDER`, `LOCAL_MODEL`, `OPENAI_MODEL` | UI sets env vars; can also export in shell before launching Streamlit. |
| Output schema | `schema.RESULT_SCHEMA_VERSION` | Bump version and extend wrapper when introducing breaking changes. |

## Extending the project

- **Add new scoring engines**: implement `compute_<name>_scores(resume_text, jd, top_n)` returning the same keys as existing backends, then plug it into `app/app.py` selection logic and optionally expand tests with deterministic fixtures.
- **Integrate ATS exports**: the JSON payload already captures metadata and resume previews; adapt `wrap_result` or add new serializers for CSV/Excel as needed.
- **Enhance GenAI prompting**: customize the prompt template in `genai/suggest.py` or enrich the gap report to include more structured evidence (e.g., bullet-level context) before sending to providers.
- **Production hardening**: introduce persistence for uploaded resumes, authentication, or rate limiting by wrapping the scoring logic in a FastAPI service (UI can call the API instead of running in-process).

## Troubleshooting

- **Model downloads**: the first MiniLM run downloads ~80 MB to the local cache; ensure the host has internet access when using the embedding backend.
- **PDF parsing**: only text-based PDFs are supported; scanned documents require an OCR pass before upload.
- **Ollama integration**: verify `ollama serve` is running and the specified `LOCAL_MODEL` is pulled; otherwise `LocalProvider` returns a fallback message with debugging notes.
- **OpenAI provider errors**: missing keys or malformed JSON raise explicit exceptions so the UI can surface actionable messages. Confirm billing and quota when using managed models.

---

This README targets contributors who need to understand the full resume↔JD matching workflow, how modules fit together, and where to extend the system for more realistic deployments.
