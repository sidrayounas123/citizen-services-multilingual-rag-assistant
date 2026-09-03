"""
Diagnostic: compare pdfplumber vs PyMuPDF (fitz) text extraction on the Urdu PDFs.
Just prints sample text from each - no file changes yet.
"""

import fitz  # PyMuPDF
from pathlib import Path

RAW_DIR = Path("data/raw")

TEST_FILES = [
    "Birth_Death_Public_Guide_KP.pdf",
    "digital_ID_Citizen_Guide_Urdu.pdf",
]

for filename in TEST_FILES:
    path = RAW_DIR / filename
    if not path.exists():
        print(f"Missing: {filename}")
        continue

    print(f"\n{'='*60}")
    print(f"FILE: {filename}")
    print('='*60)

    doc = fitz.open(path)
    # grab first page that has substantial text
    for page_num in range(min(3, len(doc))):
        text = doc[page_num].get_text()
        if len(text.strip()) > 30:
            print(f"\n--- Page {page_num + 1} (PyMuPDF) ---")
            print(text[:400])
            break
    doc.close()
