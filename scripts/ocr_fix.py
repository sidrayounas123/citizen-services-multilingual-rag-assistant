"""
Stage 1b: OCR fix for scanned PDF(s) that pdfplumber couldn't extract properly.
Runs OCR (Urdu + English) on the problematic file and merges results into corpus.json.
"""

import json
import re
from pathlib import Path
from pdf2image import convert_from_path
import pytesseract
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

# ---- CONFIG: your local paths ----
POPPLER_PATH = r"C:\Users\Sidra Younas\Downloads\Release-26.02.0-0\poppler-26.02.0\Library\bin"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# -----------------------------------

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
CORPUS_PATH = OUT_DIR / "corpus.json"

# The file(s) that need OCR (scanned images, pdfplumber failed on these)
FILES_NEEDING_OCR = [
    "2025-04-15-online_safety_guide_urdu_08-03-2024.pdf",
]


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def detect_language(text: str) -> str:
    if len(text) < 20:
        return "unknown"
    try:
        lang = detect(text)
        if lang in ("ur", "en"):
            return lang
        return "other"
    except Exception:
        return "unknown"


def ocr_pdf(pdf_path: Path) -> list[dict]:
    """Convert each page to an image, then OCR it with Urdu+English language pack."""
    records = []
    print(f"Converting {pdf_path.name} to images (this can take a minute)...")
    pages = convert_from_path(str(pdf_path), poppler_path=POPPLER_PATH, dpi=300)

    for page_num, image in enumerate(pages, start=1):
        # 'urd+eng' tells tesseract to recognize both Urdu and English on the page
        raw_text = pytesseract.image_to_string(image, lang="urd+eng")
        cleaned = clean_text(raw_text)
        if len(cleaned) < 30:
            continue
        language = detect_language(cleaned)
        records.append({
            "source": pdf_path.name,
            "page": page_num,
            "language": language,
            "text": cleaned
        })
        print(f"  Page {page_num}: {len(cleaned)} chars, detected language: {language}")

    return records


def main():
    # Load existing corpus
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    # Remove old (bad) records for the files we're about to re-OCR
    before_count = len(corpus)
    corpus = [r for r in corpus if r["source"] not in FILES_NEEDING_OCR]
    removed = before_count - len(corpus)
    print(f"Removed {removed} old low-quality records for files being re-processed.\n")

    # OCR each problematic file and add fresh records
    new_records = []
    for filename in FILES_NEEDING_OCR:
        pdf_path = RAW_DIR / filename
        if not pdf_path.exists():
            print(f"WARNING: {pdf_path} not found, skipping.")
            continue
        records = ocr_pdf(pdf_path)
        new_records.extend(records)
        ur_count = sum(1 for r in records if r["language"] == "ur")
        en_count = sum(1 for r in records if r["language"] == "en")
        print(f"\n{filename}: {len(records)} pages OCR'd (ur: {ur_count}, en: {en_count})\n")

    corpus.extend(new_records)

    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print(f"Updated corpus saved. Total records now: {len(corpus)}")


if __name__ == "__main__":
    main()
