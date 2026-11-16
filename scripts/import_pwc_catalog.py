import csv
import io
import logging
import re
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse, urlunparse

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Constantes
PWC_CSV_URL = "https://raw.githubusercontent.com/zziz/pwc/master/src/pwc.csv"
OUTPUT_PATH = Path("data/rag/pwc_catalog.csv")

TARGET_COLUMNS = [
    "paper_title",
    "paper_link",
    "conference",
    "year",
    "repo_link",
    "repo_star",
    "last_update",
]


def normalize_whitespace(text: Optional[str]) -> str:
    """Elimina espacios extra al inicio/final y colapsa múltiples espacios internos."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_title_key(title: str) -> str:
    """Genera una clave normalizada para deduplicar por título (minúsculas y alfanumérico)."""
    clean = normalize_whitespace(title).lower()
    return re.sub(r"[^a-z0-9\s]", "", clean).strip()


def normalize_url(url: Optional[str]) -> str:
    """Limpia espacios, unifica esquema a https si aplica y remueve trailing slashes."""
    clean = normalize_whitespace(url)
    if not clean:
        return ""
    
    # Si no tiene esquema, asumimos https://
    if not clean.startswith(("http://", "https://")):
        clean = "https://" + clean

    parsed = urlparse(clean)
    # Normalizamos dominio a minúsculas y quitamos '/' al final del path
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    
    return urlunparse((parsed.scheme, netloc, path, parsed.params, parsed.query, ""))


def normalize_year(year_raw: Optional[str]) -> str:
    """Extrae un año válido de 4 dígitos (ej. 1990-2030) o retorna vacío."""
    clean = normalize_whitespace(year_raw)
    if not clean:
        return ""
    match = re.search(r"\b(19\d{2}|20\d{2})\b", clean)
    return match.group(1) if match else ""


def download_csv(url: str, timeout: int = 30) -> str:
    """Descarga el contenido del CSV en UTF-8."""
    logger.info(f"Descargando catálogo desde {url}...")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "chutaDB-RAG-CatalogImporter/1.0"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8-sig", errors="replace")


def process_catalog(raw_csv: str) -> List[Dict[str, str]]:
    """Parsea, limpia y deduplica los registros del catálogo."""
    reader = csv.DictReader(io.StringIO(raw_csv))
    
    if not reader.fieldnames:
        raise ValueError("El CSV descargado está vacío o no tiene encabezados.")

    # Verificamos que existan las columnas esperadas
    missing_cols = [col for col in TARGET_COLUMNS if col not in reader.fieldnames]
    if missing_cols:
        logger.warning(f"Columnas faltantes en el origen (se rellenarán vacías): {missing_cols}")

    seen_urls: Set[str] = set()
    seen_titles: Set[str] = set()
    cleaned_rows: List[Dict[str, str]] = []
    
    total_read = 0
    duplicates_skipped = 0
    empty_skipped = 0

    for row in reader:
        total_read += 1
        
        title = normalize_whitespace(row.get("paper_title"))
        paper_link = normalize_url(row.get("paper_link"))
        
        # Descartamos filas completamente rotas sin título ni enlace
        if not title and not paper_link:
            empty_skipped += 1
            continue

        title_key = normalize_title_key(title)

        # Deduplicación:
        # 1 Por URL si existe
        # 2 Respaldo por título normalizado
        if paper_link:
            if paper_link in seen_urls:
                duplicates_skipped += 1
                continue
        elif title_key:
            if title_key in seen_titles:
                duplicates_skipped += 1
                continue

        # Registramos claves vistas
        if paper_link:
            seen_urls.add(paper_link)
        if title_key:
            seen_titles.add(title_key)

        cleaned_row = {
            "paper_title": title,
            "paper_link": paper_link,
            "conference": normalize_whitespace(row.get("conference")),
            "year": normalize_year(row.get("year")),
            "repo_link": normalize_url(row.get("repo_link")),
            "repo_star": normalize_whitespace(row.get("repo_star")),
            "last_update": normalize_whitespace(row.get("last_update")),
        }
        cleaned_rows.append(cleaned_row)

    logger.info(
        f"Procesamiento completado: {total_read} leídos | "
        f"{len(cleaned_rows)} conservados | "
        f"{duplicates_skipped} duplicados omitidos | "
        f"{empty_skipped} vacíos omitidos."
    )
    return cleaned_rows


def save_catalog(rows: List[Dict[str, str]], output_path: Path) -> None:
    """Guarda el catálogo limpio en disco creando los directorios necesarios."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TARGET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
        
    logger.info(f"Catálogo guardado exitosamente en: {output_path.resolve()}")


def main() -> None:
    try:
        raw_content = download_csv(PWC_CSV_URL)
        catalog_rows = process_catalog(raw_content)
        save_catalog(catalog_rows, OUTPUT_PATH)
    except Exception as e:
        logger.error(f"Error al importar el catálogo de Papers With Code: {e}")
        raise


if __name__ == "__main__":
    main()