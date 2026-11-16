import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Asegura que el paquete `rag` en la raíz sea importable al ejecutar desde scripts/ o raíz
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rag.ingestion.pdf_reader import extract_native_pdf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

CATALOG_PATH = Path("data/rag/pwc_catalog.csv")
MANIFEST_PATH = Path("data/rag/download_manifest.jsonl")
METADATA_OUT_PATH = Path("data/rag/metadata.csv")
CHUNKS_PATH = Path("data/rag/chunks.jsonl")
EVAL_SET_PATH = Path("data/rag/eval_ground_truth.jsonl")  # Opcional si ya existe


def load_catalog(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Falta {path}. Ejecuta primero import_pwc_catalog.py")
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_manifest(path: Path) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                records[entry["catalog_id"]] = entry
    return records


def save_manifest(records: Dict[str, Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for entry in records.values():
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def count_indexed_papers(chunks_path: Path) -> int:
    """Cuenta cuántos papers distintos tienen chunks válidos generados."""
    if not chunks_path.exists():
        return 0
    paper_ids = set()
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                pid = obj.get("paper_id") or obj.get("catalog_id")
                if pid:
                    paper_ids.add(pid)
    return len(paper_ids)


def count_eval_papers(eval_path: Path) -> int:
    """Cuenta cuántos papers distintos están en el set de evaluación ground truth."""
    if not eval_path.exists():
        return 0
    paper_ids = set()
    with eval_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                pid = obj.get("paper_id") or obj.get("catalog_id")
                if pid:
                    paper_ids.add(pid)
    return len(paper_ids)


def run_audit() -> None:
    catalog_rows = load_catalog(CATALOG_PATH)
    manifest = load_manifest(MANIFEST_PATH)

    # Indexamos catálogo por URL para cruzar metadata (año, conferencia, título)
    catalog_by_url = {
        (row.get("paper_link") or "").strip(): row
        for row in catalog_rows
        if row.get("paper_link")
    }

    # Contadores explícitos de las 6 capas (Sección 5.4)
    layer_catalog = len(catalog_rows)
    layer_downloadable = 0  # HTTP 200 válido
    layer_valid_pdf = 0     # PyMuPDF lo abre y es un PDF estructuralmente íntegro
    layer_extracted = 0     # PyMuPDF extrajo texto nativo suficiente (>= 500 chars)
    layer_indexable = 0     # Tiene chunks + metadata utilizable

    metadata_rows: List[Dict[str, Any]] = []
    html_saved_as_pdf = 0
    seen_sha256 = set()
    duplicate_files = 0

    for cid, entry in manifest.items():
        http_status = entry.get("http_status")
        status = entry.get("status")
        local_path_str = entry.get("local_path")
        sha256 = entry.get("sha256")

        if sha256:
            if sha256 in seen_sha256:
                duplicate_files += 1
            seen_sha256.add(sha256)

        # Capa 2: Descargable (respuesta HTTP válida 2xx)
        if http_status is not None and 200 <= int(http_status) < 300:
            layer_downloadable += 1

        # Si tiene archivo local descargado, auditamos con PyMuPDF
        if local_path_str and status in ("downloaded", "extracted", "invalid_pdf"):
            pdf_path = Path(local_path_str)

            # Chequeo de criterio 5.5: ningún HTML guardado como PDF
            if pdf_path.exists():
                with pdf_path.open("rb") as fb:
                    head = fb.read(64).lstrip()
                if head.startswith((b"<!DOCTYPE", b"<html", b"<HTML")):
                    html_saved_as_pdf += 1
                    entry["status"] = "not_pdf"
                    continue

            res = extract_native_pdf(pdf_path)

            if res.is_valid_pdf:
                layer_valid_pdf += 1

                if res.has_sufficient_text:
                    layer_extracted += 1
                    entry["status"] = "extracted"

                    cat_info = catalog_by_url.get(entry.get("source_url", ""), {})
                    title = (cat_info.get("paper_title") or res.title_meta or entry.get("paper_title") or "").strip()
                    year = (cat_info.get("year") or "").strip()
                    authors = res.author_meta or "Unknown"

                    # Es indexable si tiene texto suficiente y al menos título utilizable
                    has_usable_meta = bool(title)
                    if has_usable_meta:
                        layer_indexable += 1

                    metadata_rows.append(
                        {
                            "paper_id": cid,
                            "title": title,
                            "authors": authors,
                            "year": year,
                            "conference": (cat_info.get("conference") or "").strip(),
                            "source_url": entry.get("source_url", ""),
                            "local_path": local_path_str,
                            "sha256": sha256 or "",
                            "total_pages": res.total_pages,
                            "total_chars": res.total_chars,
                            "indexable": str(has_usable_meta).lower(),
                        }
                    )
                else:
                    # PDF válido estructuralmente pero escaneado / sin texto nativo suficiente
                    entry["status"] = "invalid_pdf"
            else:
                entry["status"] = "invalid_pdf"

    # Persistimos estados actualizados en download_manifest.jsonl
    save_manifest(manifest, MANIFEST_PATH)

    # Guardamos data/rag/metadata.csv con los documentos validados
    if metadata_rows:
        METADATA_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(metadata_rows[0].keys())
        with METADATA_OUT_PATH.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metadata_rows)

    # Si ya existe chunks.jsonl, tomamos el conteo real de indexados; si no, los candidatos validados
    indexed_real = count_indexed_papers(CHUNKS_PATH)
    layer_indexable_final = indexed_real if indexed_real > 0 else layer_indexable
    layer_evaluation = count_eval_papers(EVAL_SET_PATH)

    # Reporte Sección 5.4
    print("\n" + "=" * 68)
    print("  REPORTE DE CAPAS DE DISPONIBILIDAD (SECCIÓN 5.4)")
    print("=" * 68)
    print(f"{'Capa':<20} | {'Conteo':>8} | {'Significado'}")
    print("-" * 68)
    print(f"{'Catálogo':<20} | {layer_catalog:>8} | Registros en pwc_catalog.csv")
    print(f"{'Descargable':<20} | {layer_downloadable:>8} | Respuesta HTTP válida (2xx)")
    print(f"{'PDF válido':<20} | {layer_valid_pdf:>8} | Archivo contiene PDF legible")
    print(f"{'Texto extraído':<20} | {layer_extracted:>8} | PyMuPDF extrajo texto suficiente")
    print(f"{'Indexable':<20} | {layer_indexable_final:>8} | Chunks válidos y metadata utilizable")
    print(f"{'Evaluación':<20} | {layer_evaluation:>8} | Seleccionados para ground truth")
    print("=" * 68)

    # Verificación Sección 5.5
    print("\nVERIFICACIÓN DE CRITERIOS DE ACEPTACIÓN (SECCIÓN 5.5):")
    checks = [
        ("Catálogo importado (> 0 registros)", layer_catalog > 0),
        ("Intentos registrados en manifiesto", len(manifest) > 0),
        ("Sin enlaces HTML guardados como PDF", html_saved_as_pdf == 0),
        ("Sin duplicados por SHA-256 en manifiesto", duplicate_files == 0),
        ("Corpus validado listo para ingesta (20-50 recomendado)", layer_extracted >= 20),
    ]
    for desc, passed in checks:
        mark = "[OK]" if passed else "[REVISAR]"
        print(f"  {mark:<9} {desc}")
    print()


if __name__ == "__main__":
    run_audit()