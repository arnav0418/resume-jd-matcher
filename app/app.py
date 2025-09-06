# make repo root importable (so `import src...` works when run via streamlit)
import sys
from pathlib import Path

import json

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.extract import extract_text_from_file  # noqa: E402
from src.jds import load_jds, get_jd_by_id  # noqa: E402
from src.score_embed import compute_embed_scores  # noqa: E402


st.set_page_config(page_title="Resume ↔ JD Matching Demo", layout="centered")

DATA_DIR = Path("data")
JDS_PATH = DATA_DIR / "jds.json"

st.title("Resume ↔ JD Matching Demo (v1 • walking skeleton)")

# Load JDs
try:
    jds = load_jds(JDS_PATH)
    jd_options = [f"{jd['id']} — {jd['title']}" for jd in jds]
except Exception as e:
    st.error(f"Failed to load JDs: {e}")
    st.stop()

col1, col2 = st.columns([1, 1])
with col1:
    jd_choice = st.selectbox("Choose a Job Description (JD)", jd_options, index=0)

with col2:
    max_chars = st.number_input(
        "Max resume characters (for speed)",
        min_value=1000,
        max_value=20000,
        value=10000,
        step=500,
        help="Trim very long resumes for a fast demo. Will be documented in README.",
    )

backend = st.radio(
    "Scoring backend",
    options=["Embeddings (MiniLM)", "Stub (token overlap)"],
    index=0,
    help="Use embeddings for real semantic similarity. Stub is fast but simplistic.",
)

uploaded = st.file_uploader(
    "Upload your resume (PDF / DOCX / TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=False,
    help="v1 supports text-based PDFs only (no OCR).",
)

run = st.button("Match")

if run:
    if not uploaded:
        st.warning("Please upload a resume file.")
        st.stop()

    jd_id = jd_choice.split(" — ", 1)[0]
    jd = get_jd_by_id(jds, jd_id)
    if jd is None:
        st.error("Invalid JD selection.")
        st.stop()

    with st.spinner("Parsing and scoring..."):
        try:
            resume_text = extract_text_from_file(uploaded, max_chars=max_chars)
            if not resume_text or resume_text.strip() == "":
                st.error("Could not extract text. Ensure the PDF is text-based (not scanned).")
                st.stop()

            if backend.startswith("Embedding"):
                results = compute_embed_scores(resume_text, jd, top_n=3)
            else:
                from src.score_stub import compute_stub_scores  # lazy import keeps app fast

                results = compute_stub_scores(resume_text, jd, top_n=3)

            st.subheader("Results")
            c1, c2, c3 = st.columns(3)
            c1.metric("Overall Score (0–100)", f"{results['overall_score']:.1f}")
            c2.metric("Semantic similarity (0–1)", f"{results['semantic_similarity']:.2f}")
            c3.metric("Skills coverage (0–1)", f"{results['skills_coverage']:.2f}")

            st.write("**Matched skills:**", ", ".join(results["matched_skills"]) or "—")
            st.write("**Missing skills:**", ", ".join(results["missing_skills"]) or "—")

            if results.get("top_matching_jd_sentences"):
                st.write("**Top matching JD sentences (stub):**")
                for item in results["top_matching_jd_sentences"]:
                    st.write(f"- “{item['sentence']}” — sim {item['similarity']:.2f}")

            # Preview for transparency
            with st.expander("Resume text preview"):
                st.text_area(
                    "Preview",
                    results.get("resume_text_preview", ""),
                    height=300,  # adjust as needed
                    label_visibility="collapsed",  # hides the "Preview" label
                )

            # Download JSON
            payload = json.dumps(results, ensure_ascii=False, indent=2)
            st.download_button(
                "Download JSON",
                data=payload.encode("utf-8"),
                file_name=f"match_{jd_id}.json",
                mime="application/json",
            )

        except Exception as e:
            st.exception(e)
            st.stop()

st.caption(
    "Note: This is a skeleton with stub scoring (token overlap + skills match). "
    "Embeddings (MiniLM) arrive in Step 2."
)
