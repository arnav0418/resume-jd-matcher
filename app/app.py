from pathlib import Path

import streamlit as st


st.set_page_config(page_title="Resume _ JD Matching demo", layout="centered")

DATA_DIR = Path("data")
JDS_PATH = DATA_DIR / "jds.json"

st.title("Resume _ JD Matching demo (v1 . walking skeleton)")
