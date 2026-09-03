# ---- Stage 1: Build the React frontend ----
FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: Python backend + built frontend ----
FROM python:3.11-slim
WORKDIR /app

# System dependency needed by pdf2image if you ever re-run OCR inside the container
# (not required at runtime for the API itself, safe to omit if you only ship
# the already-built data/chroma_db - included here for completeness)
# RUN apt-get update && apt-get install -y poppler-utils tesseract-ocr tesseract-ocr-urd

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY data/chroma_db ./data/chroma_db
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
