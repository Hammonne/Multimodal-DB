import argparse
import csv
import hashlib
import json
import logging
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Rutas por defecto
CATALOG_PATH = Path("data/rag/pwc_catalog.csv")
PAPERS_DIR = Path("data/papers")
MANIFEST_PATH = Path("data/rag/download_manifest.jsonl")
FAILED_PATH = Path("data/rag/failed_downloads.jsonl")

# Parámetros de red
DEFAULT_TIMEOUT = 25
MAX_RETRIES = 3
BASE_DELAY_SEC = 1.5  # Espera entre solicitudes

RETRYABLE_HTTP_CODES = {408, 429, 500, 502, 503, 504}


def compute_url_hash(url: str) -> str:
    """Genera el hash SHA-256 de la URL."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def is_valid_http_url(url: str) -> bool:
    """Verifica que la URL tenga esquema http o https y un host válido."""
    if not url:
        return False
    parsed = urlparse(url.strip())
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def normalize_arxiv_url(url: str) -> str:
    """Convierte enlaces de abstract de arXiv (/abs/...) a PDF directo (/pdf/....pdf) si aplica."""
    parsed = urlparse(url)
    if "arxiv.org" in parsed.netloc.lower() and parsed.path.startswith("/abs/"):
        paper_id = parsed.path[len("/abs/"):].strip("/")
        return f"https://arxiv.org/pdf/{paper_id}.pdf"
    return url


def is_pdf_resource(url: str, content_type: str, first_bytes: bytes) -> bool:
    """Identifica si el recurso es PDF mediante firma %PDF, Content-Type o extensión."""
    # 1. Firma mágica binaria (la prueba definitiva)
    if first_bytes.startswith(b"%PDF"):
        return True

    # 2. Header Content-Type
    ct_lower = content_type.lower()
    if "application/pdf" in ct_lower or "application/x-pdf" in ct_lower:
        return True

    # 3. Extensión en la ruta URL (solo respaldo si los bytes no dicen lo contrario)
    parsed_path = urlparse(url).path.lower()
    if parsed_path.endswith(".pdf") and not first_bytes.lstrip().startswith((b"<!DOCTYPE", b"<html", b"<HTML")):
        return True

    return False


def load_manifest(manifest_path: Path) -> Dict[str, Dict[str, Any]]:
    """Carga el manifiesto existente indexado por catalog_id."""
    records: Dict[str, Dict[str, Any]] = {}
    if not manifest_path.exists():
        return records

    with manifest_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                cid = entry.get("catalog_id")
                if cid:
                    records[cid] = entry
            except json.JSONDecodeError:
                logger.warning(f"Línea corrupta en manifiesto (línea {line_num}), ignorada.")
    return records


def save_manifest(records: Dict[str, Dict[str, Any]], manifest_path: Path) -> None:
    """Persiste el manifiesto actualizado en formato JSONL."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for entry in records.values():
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def append_failed_log(entry: Dict[str, Any], failed_path: Path) -> None:
    """Agrega un registro de fallo al log dedicado sin detener el lote."""
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    with failed_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def download_single_paper(
    url: str,
    dest_path: Path,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = MAX_RETRIES,
) -> Tuple[str, Optional[int], str, Optional[str], int, int, Optional[str]]:
    """
    Intenta descargar un PDF con reintentos limitados.
    Retorna: (status, http_status, content_type, sha256, bytes_written, attempts, error_msg)
    """
    fetch_url = normalize_arxiv_url(url)
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; chutaDB-AcademicRAG/1.0)",
        "Accept": "application/pdf,*/*;q=0.8",
    }

    last_http_status: Optional[int] = None
    last_content_type: str = ""
    last_error: Optional[str] = None

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(fetch_url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                http_status = getattr(resp, "status", 200)
                content_type = resp.headers.get("Content-Type", "").split(";")[0].strip()
                last_http_status = http_status
                last_content_type = content_type

                # Leemos el primer bloque para validar firma %PDF antes de bajar todo
                first_chunk = resp.read(8192)
                if not first_chunk:
                    return ("invalid_pdf", http_status, content_type, None, 0, attempt, "Respuesta vacía")

                if not is_pdf_resource(fetch_url, content_type, first_chunk):
                    return ("not_pdf", http_status, content_type, None, 0, attempt, None)

                if not first_chunk.startswith(b"%PDF"):
                    # Decía ser PDF por header/URL, pero no empieza con %PDF
                    return ("invalid_pdf", http_status, content_type, None, 0, attempt, "Falta cabecera %PDF")

                # Escribimos a archivo temporal y calculamos SHA-256 en streaming
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = dest_path.with_suffix(".tmp")
                hasher = hashlib.sha256()
                total_bytes = 0

                with tmp_path.open("wb") as out_file:
                    out_file.write(first_chunk)
                    hasher.update(first_chunk)
                    total_bytes += len(first_chunk)

                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        hasher.update(chunk)
                        total_bytes += len(chunk)

                # Validación mínima de tamaño (un PDF real difícilmente pesa < 1 KB)
                if total_bytes < 1024:
                    tmp_path.unlink(missing_ok=True)
                    return ("invalid_pdf", http_status, content_type, None, total_bytes, attempt, "Archivo demasiado pequeño (<1KB)")

                tmp_path.replace(dest_path)
                return ("downloaded", http_status, content_type, hasher.hexdigest(), total_bytes, attempt, None)

        except urllib.error.HTTPError as e:
            last_http_status = e.code
            last_content_type = e.headers.get("Content-Type", "").split(";")[0].strip() if e.headers else ""
            last_error = f"HTTP {e.code}: {e.reason}"
            if e.code not in RETRYABLE_HTTP_CODES or attempt == max_retries:
                return ("http_error", last_http_status, last_content_type, None, 0, attempt, last_error)

        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            if attempt == max_retries:
                return ("failed", last_http_status, last_content_type, None, 0, attempt, last_error)

        # Backoff exponencial con jitter antes del siguiente reintento
        sleep_time = (BASE_DELAY_SEC * (2 ** (attempt - 1))) + random.uniform(0.2, 0.8)
        logger.debug(f"Reintentando ({attempt}/{max_retries}) en {sleep_time:.1f}s -> {url}")
        time.sleep(sleep_time)

    return ("failed", last_http_status, last_content_type, None, 0, max_retries, last_error)


def run_batch(
    catalog_path: Path,
    papers_dir: Path,
    manifest_path: Path,
    failed_path: Path,
    limit: Optional[int],
    resume: bool,
    delay: float,
) -> None:
    if not catalog_path.exists():
        raise FileNotFoundError(f"No se encontró el catálogo en {catalog_path}. Ejecuta import_pwc_catalog.py primero.")

    manifest = load_manifest(manifest_path)

    # Conjunto de checksums ya descargados en disco para evitar duplicados de contenido
    existing_checksums: Set[str] = {
        entry["sha256"]
        for entry in manifest.values()
        if entry.get("sha256") and entry.get("status") in ("downloaded", "extracted")
    }

    with catalog_path.open("r", encoding="utf-8") as f:
        catalog_rows = list(csv.DictReader(f))

    # Inicializamos entradas como 'cataloged' si no estaban en el manifiesto
    for row in catalog_rows:
        source_url = (row.get("paper_link") or "").strip()
        if not is_valid_http_url(source_url):
            continue
        url_hash = compute_url_hash(source_url)
        catalog_id = f"sha256:{url_hash}"
        if catalog_id not in manifest:
            manifest[catalog_id] = {
                "catalog_id": catalog_id,
                "paper_title": (row.get("paper_title") or "").strip(),
                "source_url": source_url,
                "local_path": None,
                "status": "cataloged",
                "http_status": None,
                "content_type": None,
                "sha256": None,
                "bytes": 0,
                "attempts": 0,
            }

    # Estados terminales que se saltan al usar --resume
    terminal_statuses = {"downloaded", "extracted", "not_pdf", "http_error", "invalid_pdf"}

    downloaded_count = 0
    processed_in_run = 0

    for row in catalog_rows:
        if limit is not None and downloaded_count >= limit:
            logger.info(f"Límite alcanzado ({limit} descargas exitosas en esta ejecución).")
            break

        # Regla: usar solo paper_link, nunca repo_link
        source_url = (row.get("paper_link") or "").strip()
        if not is_valid_http_url(source_url):
            continue

        url_hash = compute_url_hash(source_url)
        catalog_id = f"sha256:{url_hash}"
        current_entry = manifest[catalog_id]
        current_status = current_entry.get("status", "cataloged")

        # Si ya tiene checksum registrado y el archivo existe, no volver a descargar
        if current_entry.get("sha256") in existing_checksums:
            local_p = current_entry.get("local_path")
            if local_p and Path(local_p).exists():
                continue

        # Con --resume saltamos registros ya resueltos (o fallidos definitivamente)
        if resume and current_status in terminal_statuses:
            continue

        dest_path = papers_dir / f"{url_hash}.pdf"
        logger.info(f"[{downloaded_count + 1}/{limit or 'inf'}] Descargando: {current_entry['paper_title'][:65]}...")

        status, http_status, content_type, sha256, num_bytes, attempts, error_msg = download_single_paper(
            url=source_url,
            dest_path=dest_path,
        )

        # Verificación extra de deduplicación por contenido (mismo PDF bajo distinta URL)
        if status == "downloaded" and sha256 in existing_checksums:
            logger.info(f"  -> Contenido duplicado detectado (sha256={sha256[:12]}...), eliminando copia redundante.")
            dest_path.unlink(missing_ok=True)
            # Apuntamos al archivo existente que ya tenía ese sha256
            existing_path = next(
                (e["local_path"] for e in manifest.values() if e.get("sha256") == sha256 and e.get("local_path")),
                str(dest_path),
            )
            local_path_str: Optional[str] = existing_path
        elif status == "downloaded":
            existing_checksums.add(sha256)  # type: ignore[arg-type]
            local_path_str = dest_path.as_posix()
            downloaded_count += 1
            logger.info(f"  -> OK ({num_bytes:,} bytes) guardado en {local_path_str}")
        else:
            local_path_str = None
            logger.warning(f"  -> Estado: {status} | {error_msg or content_type}")

        # Actualizamos registro del manifiesto
        updated_entry: Dict[str, Any] = {
            "catalog_id": catalog_id,
            "paper_title": current_entry["paper_title"],
            "source_url": source_url,
            "local_path": local_path_str,
            "status": status,
            "http_status": http_status,
            "content_type": content_type or None,
            "sha256": sha256,
            "bytes": num_bytes,
            "attempts": attempts,
        }
        manifest[catalog_id] = updated_entry

        if status in ("http_error", "invalid_pdf", "failed"):
            failed_record = {**updated_entry, "error": error_msg}
            append_failed_log(failed_record, failed_path)

        # Persistimos incrementalmente para que un Ctrl+C no pierda progreso al usar --resume
        save_manifest(manifest, manifest_path)
        processed_in_run += 1

        # Espera cortés entre solicitudes
        time.sleep(delay + random.uniform(0.1, 0.5))

    logger.info(
        f"Lote finalizado. Solicitudes ejecutadas: {processed_in_run} | "
        f"PDFs nuevos descargados: {downloaded_count}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Descarga controlada de PDFs académicos desde data/rag/pwc_catalog.csv"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Número máximo de PDFs a descargar exitosamente en esta ejecución (ej. --limit 50).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Omite registros ya procesados en data/rag/download_manifest.jsonl.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=BASE_DELAY_SEC,
        help=f"Segundos base de espera entre solicitudes (default: {BASE_DELAY_SEC}s).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_batch(
        catalog_path=CATALOG_PATH,
        papers_dir=PAPERS_DIR,
        manifest_path=MANIFEST_PATH,
        failed_path=FAILED_PATH,
        limit=args.limit,
        resume=args.resume,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()