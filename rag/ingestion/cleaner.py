import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Set
from rag.ingestion.pdf_reader import PageText


@dataclass
class CleanedPage:
    page_number: int
    original_text: str  # Conservado para citas verificables
    cleaned_text: str   # Normalizado para chunking e indexación


# Patrones de encabezado de sección bibliográfica al final del paper
REFERENCES_HEADER_RE = re.compile(
    r"^\s*(?:\d+\.?\s+)?(?:references|bibliography|works\s+cited)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _normalize_line_signature(line: str) -> str:
    """Normaliza números de página para detectar encabezados/pies repetidos."""
    s = re.sub(r"\d+", "#", line.strip().lower())
    return re.sub(r"\s+", " ", s)


def detect_repeated_headers_footers(
    pages: List[PageText],
    edge_lines: int = 3,
    min_ratio: float = 0.4,
) -> Set[str]:
    """
    Detecta líneas que se repiten en las primeras o últimas `edge_lines` líneas
    de al menos `min_ratio` (40%) de las páginas (ej. títulos de conferencia, autores).
    """
    if len(pages) < 3:
        return set()

    counter: Counter[str] = Counter()
    for p in pages:
        lines = [ln.strip() for ln in p.raw_text.splitlines() if ln.strip()]
        if not lines:
            continue
        candidates = set()
        for ln in lines[:edge_lines] + lines[-edge_lines:]:
            sig = _normalize_line_signature(ln)
            if 3 <= len(sig) <= 120:
                candidates.add(sig)
        for sig in candidates:
            counter[sig] += 1

    threshold = max(2, int(len(pages) * min_ratio))
    return {sig for sig, count in counter.items() if count >= threshold}


def dehyphenate(text: str) -> str:
    """Une palabras cortadas por guion al final de línea (ej. 'convolu-\\ntional' -> 'convolutional')."""
    return re.sub(r"([A-Za-z]{2,})-\s*\n\s*([a-z]{2,})", r"\1\2", text)


def normalize_spaces(text: str) -> str:
    """Colapsa saltos de línea simples en espacios preservando párrafos y limpia espacios múltiples."""
    # Preservar saltos dobles de párrafo
    text = re.sub(r"\n{2,}", "\n\n", text)
    # Convertir saltos simples dentro de un párrafo en espacio
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Limpiar espacios/tabs repetidos
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def clean_document_pages(pages: List[PageText]) -> List[CleanedPage]:
    """
    Ejecuta la limpieza completa sobre las páginas de un paper:
    1. Elimina encabezados y pies repetidos.
    2. Corta la sección References y todo lo posterior.
    3. Une palabras cortadas por guion.
    4. Normaliza espacios conservando `original_text` para citas.
    """
    repeated_sigs = detect_repeated_headers_footers(pages)
    cleaned_pages: List[CleanedPage] = []
    references_reached = False

    for p in pages:
        if references_reached:
            break

        # Filtrar encabezados y pies repetidos
        kept_lines: List[str] = []
        for raw_ln in p.raw_text.splitlines():
            stripped_ln = raw_ln.strip()
            if stripped_ln and _normalize_line_signature(stripped_ln) in repeated_sigs:
                continue
            kept_lines.append(raw_ln)

        page_body = "\n".join(kept_lines)

        # Detectar inicio de sección References (solo después del primer tercio del paper)
        if p.page_number > max(1, len(pages) // 3):
            match = REFERENCES_HEADER_RE.search(page_body)
            if match:
                page_body = page_body[: match.start()]
                references_reached = True

        original_preserved = page_body.strip()
        if not original_preserved:
            continue

        # Unir guiones y normalizar espacios
        cleaned = dehyphenate(original_preserved)
        cleaned = normalize_spaces(cleaned)

        if len(cleaned) >= 40:
            cleaned_pages.append(
                CleanedPage(
                    page_number=p.page_number,
                    original_text=original_preserved,
                    cleaned_text=cleaned,
                )
            )

    return cleaned_pages