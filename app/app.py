# make repo root importable (so `import src...` works when run via streamlit)
import sys
import os
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
from src.genai.suggest import generate_improvements  # noqa: E402


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

        st.divider()
        st.subheader("GenAI: Improve resume to better match the JD")

        # Explain provider selection (optional)
        st.caption(
            "Provider: set GENAI_PROVIDER=mock (default) or openai; "
            "for OpenAI also set OPENAI_API_KEY and optionally OPENAI_MODEL."
        )

        provider_choice = st.selectbox(
            "Provider",
            ["mock (default)", "openai"],
            index=0,
            help="mock requires no API key; openai needs OPENAI_API_KEY",
        )
        if provider_choice.startswith("openai"):
            os.environ["GENAI_PROVIDER"] = "openai"
        else:
            os.environ["GENAI_PROVIDER"] = "mock"

        colA, colB = st.columns([1, 3])
        with colA:
            run_genai = st.button("✨ Improve with GenAI", type="primary")
        with colB:
            st.write(
                "Get targeted, truthful suggestions with estimated impact on similarity and skills coverage."
            )

        if run_genai:
            with st.spinner("Generating suggestions..."):
                try:
                    improve_payload = generate_improvements(resume_text, jd)
                except Exception as e:
                    st.exception(e)
                    st.stop()

            suggestions = improve_payload.get("suggestions", [])
            gap_report = improve_payload.get("gap_report", {})
            notes = improve_payload.get("notes", [])
            guardrails = improve_payload.get("guardrails", [])

            if not suggestions:
                st.info("No suggestions returned. Try switching provider or using a longer resume.")
            else:
                st.write(f"**Suggestions** ({len(suggestions)})")

                # Render suggestion cards
                for i, s in enumerate(suggestions, start=1):
                    with st.container(border=True):
                        top = st.columns([5, 2, 2])
                        with top[0]:
                            st.markdown(
                                f"**{i}. {s['type'].replace('_',' ').title()} → {s['target_section'].title()}**"
                            )
                        with top[1]:
                            # needs evidence badge
                            if s.get("needs_evidence"):
                                st.markdown(":orange[**needs evidence**]")
                            else:
                                st.markdown(":green[**grounded**]")
                        with top[2]:
                            # estimated lift chips
                            lift = s.get("est_lift", {})
                            sem = lift.get("semantic", 0.0)
                            skl = lift.get("skills", 0.0)
                            st.markdown(f"Δ semantic: **{sem:+.2f}**, Δ skills: **{skl:+.2f}**")

                        # Proposed line
                        st.markdown("**Proposed:**")
                        st.code(s["proposed"], language="text")

                        # Optional original
                        if s.get("original"):
                            with st.expander("Original"):
                                st.code(s["original"], language="text")

                        # Rationale
                        st.markdown(f"**Why:** {s['rationale']}")

                        # Copy button
                        st.button(
                            "Copy proposed",
                            key=f"copy_{i}",
                            use_container_width=False,
                            on_click=lambda txt=s["proposed"]: st.session_state.update(
                                {"_copy_last": txt}
                            ),
                            help="Copies the proposed line to session state (you can manually copy from the code block).",
                        )

                # Gap report summary (collapsible)
                with st.expander("Gap report"):
                    st.write(gap_report)

                # Notes & guardrails
                if notes:
                    st.write("**Notes:**")
                    for n in notes:
                        st.write(f"- {n}")
                if guardrails:
                    st.write("**Guardrails:**")
                    for g in guardrails:
                        st.write(f"- {g}")

                # Download suggestions JSON
                from datetime import datetime

                ts2 = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname2 = f"suggestions_{jd['id']}_{ts2}.json"
                st.download_button(
                    "Download suggestions JSON",
                    data=json.dumps(improve_payload, ensure_ascii=False, indent=2).encode("utf-8"),
                    file_name=fname2,
                    mime="application/json",
                )


st.caption(
    "Note: This is a skeleton with stub scoring (token overlap + skills match). "
    "Embeddings (MiniLM) arrive in Step 2."
)
