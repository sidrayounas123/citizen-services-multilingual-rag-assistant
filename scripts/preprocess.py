"""
Stage 1: Data Collection & Preprocessing for Multilingual (Urdu/English) RAG pipeline
Extracts text from PDFs, detects language per page, cleans it, and saves as structured JSON.
"""

import pdfplumber
import json
import re
from pathlib import Path
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0  # makes langdetect deterministic

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    """Remove extra whitespace, fix common encoding artifacts."""
    text = re.sub(r"\s+", " ", text)          # collapse whitespace
    text = re.sub(r"-\n", "", text)            # fix hyphenated line breaks
    text = text.strip()
    return text


def detect_language(text: str) -> str:
    """Detect language of a text snippet. Returns 'ur', 'en', or 'unknown'."""
    if len(text) < 20:
        return "unknown"
    try:
        lang = detect(text)
        if lang in ("ur", "en"):
            return lang
        return "other"
    except Exception:
        return "unknown"


def extract_pdf(pdf_path: Path) -> list[dict]:
    """Extract text page-by-page with metadata, ready for chunking later."""
    records = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                raw_text = page.extract_text() or ""
                cleaned = clean_text(raw_text)
                if len(cleaned) < 30:  # skip near-empty pages
                    continue
                language = detect_language(cleaned)
                records.append({
                    "source": pdf_path.name,
                    "page": page_num,
                    "language": language,
                    "text": cleaned
                })
    except Exception as e:
        print(f"  ERROR processing {pdf_path.name}: {e}")
    return records


def main():
    all_records = []
    pdf_files = list(RAW_DIR.rglob("*.pdf"))  # rglob = also checks subfolders

    if not pdf_files:
        print(f"No PDFs found in {RAW_DIR}. Add your source PDFs there first.")
        return

    print(f"Found {len(pdf_files)} PDF(s). Processing...\n")

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        records = extract_pdf(pdf_path)
        all_records.extend(records)

        ur_count = sum(1 for r in records if r["language"] == "ur")
        en_count = sum(1 for r in records if r["language"] == "en")
        other_count = len(records) - ur_count - en_count
        print(f"  -> {len(records)} pages extracted (ur: {ur_count}, en: {en_count}, other/unknown: {other_count})")

    out_path = OUT_DIR / "corpus.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    total_ur = sum(1 for r in all_records if r["language"] == "ur")
    total_en = sum(1 for r in all_records if r["language"] == "en")
    print(f"\nDone. {len(all_records)} total page-records saved to {out_path}")
    print(f"Language breakdown -> Urdu: {total_ur}, English: {total_en}, Other/Unknown: {len(all_records) - total_ur - total_en}")

    if total_ur == 0:
        print("\nWARNING: No Urdu pages detected. If your Urdu PDFs are scanned images")
        print("(not selectable text), pdfplumber can't extract them - you'll need OCR")
        print("(pytesseract) for those files. Let me know if this happens.")


if __name__ == "__main__":
    main()
