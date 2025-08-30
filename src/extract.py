import io
import re

import pdfplumber
from docx import Document


MAX_DEFAULT = 10_000


def _normalize(text: str, max_chars: int = MAX_DEFAULT) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def _read_pdf_bytes(b: bytes) -> str:
    out = []
    with pdfplumber.open(io.BytesIO(b)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            out.append(t)
    return "\n".join(out)


def _read_docx_bytes(b: bytes) -> str:
    doc = Document(io.BytesIO(b))
    paras = [p.text for p in doc.paragraphs]
    return "\n".join(paras)


def _read_txt_bytes(b: bytes) -> str:
    try:
        return b.decode("utf-8", errors="ignore")
    except Exception:
        return b.decode("latin-1", errors="ignore")


def extract_text_from_file(uploaded_file, max_chars: int = MAX_DEFAULT) -> str:
    """
    uploaded_file: streamlit UploadedFile (has .name, .type, .read()) or a similar object.
    """
    name = (uploaded_file.name or "").lower()
    b = uploaded_file.read()

    if name.endswith(".pdf"):
        text = _read_pdf_bytes(b)
    elif name.endswith(".docx"):
        text = _read_docx_bytes(b)
    elif name.endswith(".txt"):
        text = _read_txt_bytes(b)
    else:
        raise ValueError("Unsupported file type. Use PDF, DOCX, or TXT.")

    return _normalize(text, max_chars=max_chars)
