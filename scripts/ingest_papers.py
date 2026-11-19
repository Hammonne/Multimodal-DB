import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rag.ingestion.pdf_reader import extract_native_pdf
from rag.ingestion.cleaner import clean_document_pages
from rag.ingestion.chunker import chunk_pages

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

METADATA_PATH = Path("data/rag/metadata.csv")
CHUNKS_OUT_PATH = Path("data/rag/chunks.jsonl")
REJECTED_LOG_PATH = Path("data/rag/rejected_ingestion.jsonl")


def run_ingestion(limit: int | None = None) -> None:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"No existe {METADATA_PATH}. Ejecuta primero scripts/audit_availability.py"
        )

    with METADATA_PATH.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if limit is not None:
        rows = rows[:limit]

    CHUNKS_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    papers_ok = 0
    papers_rejected = 0
    total_chunks = 0

    with (
        CHUNKS_OUT_PATH.open("w", encoding="utf-8") as chunks_file,
        REJECTED_LOG_PATH.open("w", encoding="utf-8") as rejected_file,
    ):
        for row in rows:
            paper_id = row["paper_id"]
            pdf_path = Path(row["local_path"])

            # 1. Extracción con PyMuPDF (sort=True)
            extraction = extract_native_pdf(pdf_path)
            if not extraction.is_valid_pdf or not extraction.has_sufficient_text:
                papers_rejected += 1
                rejected_file.write(
                    json.dumps(
                        {
                            "paper_id": paper_id,
                            "local_path": str(pdf_path),
                            "reason": extraction.error or "Texto insuficiente",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            # 2. Limpieza académica
            cleaned_pages = clean_document_pages(extraction.pages)
            if not cleaned_pages:
                papers_rejected += 1
                rejected_file.write(
                    json.dumps(
                        {
                            "paper_id": paper_id,
                            "local_path": str(pdf_path),
                            "reason": "Sin contenido útil tras limpieza de headers/references",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            # 3. Chunking (200-500 palabras) conservando page_start y page_end
            chunks = chunk_pages(paper_id=paper_id, pages=cleaned_pages)
            if not chunks:
                papers_rejected += 1
                continue

            for ch in chunks:
                record: Dict[str, Any] = {
                    "chunk_id": ch.chunk_id,
                    "paper_id": ch.paper_id,
                    "chunk_index": ch.chunk_index,
                    "title": row.get("title", ""),
                    "authors": row.get("authors", "Unknown"),
                    "year": row.get("year", ""),
                    "conference": row.get("conference", ""),
                    "source_url": row.get("source_url", ""),
                    "page_start": ch.page_start,
                    "page_end": ch.page_end,
                    "word_count": ch.word_count,
                    "text": ch.text,
                    "citation_text": ch.citation_text,
                }
                chunks_file.write(json.dumps(record, ensure_ascii=False) + "\n")

            papers_ok += 1
            total_chunks += len(chunks)

    logger.info(
        f"Ingesta completada: {papers_ok} papers indexables ({total_chunks} chunks generados) | "
        f"{papers_rejected} papers rechazados."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extrae, limpia y divide PDFs en chunks para el RAG.")
    parser.add_argument("--limit", type=int, default=None, help="Limitar número de papers a ingestar (ej. 50).")
    args = parser.parse_args()
    run_ingestion(limit=args.limit)