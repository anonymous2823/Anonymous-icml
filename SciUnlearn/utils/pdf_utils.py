import re
from pathlib import Path
from typing import List, Optional

try:
    import PyPDF2 as pypdf2
except ImportError:
    import pypdf2  # fallback name


def extract_corpus_id_from_filename(filename: str) -> Optional[int]:
    """
    Extract a plausible Corpus ID from the PDF filename by taking the longest
    (or first sufficiently long) integer substring (>= 6 digits).
    Example: '119529102.pdf' -> 119529102
    """
    nums = re.findall(r"\d+", filename)
    if not nums:
        return None

    nums.sort(key=lambda s: (-len(s), s))
    for s in nums:
        if len(s) >= 6:
            try:
                return int(s)
            except ValueError:
                continue

    try:
        return int(nums[0])
    except ValueError:
        return None


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text using PyPDF2 first.
    If that fails or returns too little, fall back to PyMuPDF.
    """
    # 1) Try PyPDF2
    try:
        text_parts: List[str] = []
        with open(pdf_path, "rb") as f:
            reader = pypdf2.PdfReader(f, strict=False)
            for i, page in enumerate(reader.pages):
                try:
                    txt = page.extract_text() or ""
                except Exception:
                    txt = ""
                text_parts.append(f"\n\n=== Page {i + 1} ===\n{txt}")

        text = "\n".join(text_parts).strip()
        if text:
            return text
    except Exception as e:
        print(f"[WARN] PyPDF2 failed on {pdf_path.name}: {e}")

    # 2) Fallback: PyMuPDF
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(pdf_path))
        text_parts = []
        for i, page in enumerate(doc):
            text_parts.append(f"\n\n=== Page {i + 1} ===\n{page.get_text('text')}")
        text = "".join(text_parts).strip()
        if text:
            return text
    except Exception as e:
        print(f"[WARN] PyMuPDF also failed on {pdf_path.name}: {e}")

    return ""