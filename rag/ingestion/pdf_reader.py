from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import pymupdf


@dataclass
class PageText:
    page_number: int  # 1-indexed
    text: str
    char_count: int


@dataclass
class PDFExtractionResult:
    is_valid_pdf: bool
    has_sufficient_text: bool
    total_pages: int = 0
    total_chars: int = 0
    pages: List[PageText] = field(default_factory=list)
    title_meta: str = ""
    author_meta: str = ""
    error: Optional[str] = None


def extract_native_pdf(
    pdf_path: Path,
    min_total_chars: int = 500,
    min_chars_per_page: int = 40,
) -> PDFExtractionResult:
    """
    Abre un PDF con PyMuPDF, verifica legibilidad estructural y extrae texto nativo.
    """
    if not pdf_path.exists() or not pdf_path.is_file():
        return PDFExtractionResult(
            is_valid_pdf=False,
            has_sufficient_text=False,
            error="Archivo no encontrado en disco",
        )

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        return PDFExtractionResult(
            is_valid_pdf=False,
            has_sufficient_text=False,
            error=f"Error al abrir PDF con PyMuPDF: {e}",
        )

    try:
        if doc.is_encrypted:
            return PDFExtractionResult(
                is_valid_pdf=False,
                has_sufficient_text=False,
                error="PDF encriptado / protegido con contraseña",
            )

        total_pages = len(doc)
        if total_pages == 0:
            return PDFExtractionResult(
                is_valid_pdf=False,
                has_sufficient_text=False,
                error="PDF sin páginas",
            )

        meta = doc.metadata or {}
        title_meta = (meta.get("title") or "").strip()
        author_meta = (meta.get("author") or "").strip()

        extracted_pages: List[PageText] = []
        total_chars = 0

        for idx in range(total_pages):
            page = doc[idx]
            raw_text = page.get_text("text") or ""
            clean_text = raw_text.strip()
            c_count = len(clean_text)

            if c_count >= min_chars_per_page:
                extracted_pages.append(
                    PageText(
                        page_number=idx + 1,
                        text=clean_text,
                        char_count=c_count,
                    )
                )
            total_chars += c_count

        has_sufficient = total_chars >= min_total_chars and len(extracted_pages) > 0

        return PDFExtractionResult(
            is_valid_pdf=True,
            has_sufficient_text=has_sufficient,
            total_pages=total_pages,
            total_chars=total_chars,
            pages=extracted_pages,
            title_meta=title_meta,
            author_meta=author_meta,
        )
    except Exception as e:
        return PDFExtractionResult(
            is_valid_pdf=False,
            has_sufficient_text=False,
            error=f"Fallo leyendo páginas: {e}",
        )
    finally:
        doc.close()