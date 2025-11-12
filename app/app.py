# make repo root importable (so `import src...` works when run via streamlit)
import sys
from pathlib import Path
from datetime import datetime

import json
import time

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.extract import extract_text_from_file  # noqa: E402
from src.jds import load_jds, get_jd_by_id  # noqa: E402
from src.score_embed import compute_embed_scores  # noqa: E402
from src.score_stub import compute_stub_scores  # noqa: E402
from src.schema import wrap_result, ScoreWeights  # noqa: E402


st.set_page_config(page_title="Resume ↔ JD Matching Demo", layout="centered")

DATA_DIR = Path("data")
JDS_PATH = DATA_DIR / "jds.json"

st.title("Resume ↔ JD Matching Demo")

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

weights = ScoreWeights(semantic=0.7, skills=0.3)

tabs = st.tabs(["Upload file", "Try a sample"])
with tabs[0]:
    uploaded = st.file_uploader(
        "Upload your resume (PDF / DOCX / TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=False,
        help="v1 supports text-based PDFs only (no OCR).",
    )

with tabs[1]:
    st.write("Use a sample resume to quickly test the pipeline.")
    sample_choice = st.radio("Sample", ["Backend sample", "Data/ML sample"], horizontal=True)
    sample_path = Path("examples") / (
        "sample_resume_backend.txt" if "Backend" in sample_choice else "sample_resume_data_ml.txt"
    )
    sample_text = ""
    try:
        sample_text = sample_path.read_text(encoding="utf-8")
    except Exception:
        sample_text = "Sample file missing. Add examples/ files."
    sample_area = st.text_area("Sample resume text", value=sample_text, height=200)


run = st.button("Match")

if run:
    jd_id = jd_choice.split(" — ", 1)[0]
    jd = get_jd_by_id(jds, jd_id)
    if jd is None:
        st.error("Invalid JD selection.")
        st.stop()

    # Determine source: uploader vs sample tab
    resume_text = ""
    if uploaded is not None and uploaded.size > 0:
        try:
            resume_text = extract_text_from_file(uploaded, max_chars=int(max_chars))
        except Exception as e:
            st.error(f"Failed to parse uploaded file: {e}")
            st.stop()
    else:
        # take from sample tab textarea
        resume_text = (sample_area or "").strip()

    if not resume_text:
        st.warning("Please upload a resume OR use the sample text.")
        st.stop()

    with st.spinner("Parsing and scoring..."):
        t0 = time.perf_counter()

        if backend.startswith("Embedding"):
            core = compute_embed_scores(resume_text, jd, top_n=3)
            backend_id = "embeddings:minilm-l6-v2"
        else:
            core = compute_stub_scores(resume_text, jd, top_n=3)
            backend_id = "stub:token-overlap"

        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        payload = wrap_result(
            core,
            jd_title=jd.get("title", ""),
            backend=backend_id,
            weights=weights,
            latency_ms=elapsed_ms,
            resume_char_count=len(resume_text),
            preview_limit=2000,
        )

        st.success(f"Done in {elapsed_ms} ms")

        st.subheader("Results")
        c1, c2, c3 = st.columns(3)
        c1.metric("Overall Score (0–100)", f"{payload['metrics']['overall_score']:.1f}")
        c2.metric("Semantic similarity (0–1)", f"{payload['metrics']['semantic_similarity']:.2f}")
        c3.metric("Skills coverage (0–1)", f"{payload['metrics']['skills_coverage']:.2f}")

        st.write("**Matched skills:**", ", ".join(payload["skills"]["matched"]) or "—")
        st.write("**Missing skills:**", ", ".join(payload["skills"]["missing"]) or "—")

        if payload["explanations"]["top_matching_jd_sentences"]:
            st.write("**Top matching JD sentences:**")
            for item in payload["explanations"]["top_matching_jd_sentences"]:
                st.write(f"- “{item['sentence']}” — sim {item['similarity']:.2f}")

        with st.expander("Resume preview"):
            st.text(payload["resume"]["preview"])

        with st.expander("Raw JSON"):
            st.json(payload)

        # Better download filename
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"match_{payload['jd']['id']}_{payload['backend'].split(':')[0]}_{ts}.json"
        st.download_button(
            "Download JSON",
            data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=fname,
            mime="application/json",
        )


st.caption(
    "Note: This is a skeleton with stub scoring (token overlap + skills match). "
    "Embeddings (MiniLM) arrive in Step 2."
)
