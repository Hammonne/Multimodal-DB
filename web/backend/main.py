from __future__ import annotations

import json
import os
import base64
import csv
import io
import re
import sys
import re
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "parser"))
sys.path.insert(1, str(ROOT))
os.chdir(ROOT)

from parser import Parser  # noqa: E402
from scanner import Scanner  # noqa: E402
from executor import ExecuteVisitor  # noqa: E402
from storage.storage_manager import StorageManager  # noqa: E402
from web.backend.app_catalogo import catalogo  # noqa: E402
from web.backend.text_search import text_search  # noqa: E402
from web.backend.multimedia_search import multimedia_search  # noqa: E402


class QueryRequest(BaseModel):
    sql: str = Field(min_length=1, max_length=100_000)


class CsvUploadRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    # Base64 ocupa aproximadamente 4/3 del tamaño original. Esto permite
    # cargar el corpus JSONL de 100 000 documentos sin truncarlo.
    content_base64: str = Field(min_length=1, max_length=70_000_000)


class CsvColumn(BaseModel):
    name: str
    type: str


class CsvUploadResponse(BaseModel):
    filename: str
    path: str
    columns: list[CsvColumn] = []
    rows: int
    kind: str = "csv"
    sql: str | None = None


MEDIA_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp",
    ".mp3", ".wav", ".ogg", ".flac", ".m4a",
}

class QueryResponse(BaseModel):
    message: str = ""
    columns: list[str] = []
    rows: list[list[Any]] = []
    plan: list[dict[str, Any]] = []
    row_count: int = 0
    affected_rows: int = 0
    duration_ms: float = 0


class TextIndexRequest(BaseModel):
    dataset: str = Field(default="1k")


class TextSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10_000)
    algorithm: str = Field(default="bm25")
    dataset: str = Field(default="1k")
    top_k: int = Field(default=10, ge=1, le=100)


class MultimediaFileSearchRequest(BaseModel):
    index: str = Field(default="hnsw")
    metric: str = Field(default="coseno")
    filename: str = Field(min_length=1, max_length=255)
    content_base64: str = Field(min_length=1, max_length=30_000_000)
    top_k: int = Field(default=10, ge=1, le=100)


class MultimediaSearchRequest(BaseModel):
    kind: str = Field(default="sift")
    index: str = Field(default="hnsw")
    metric: str = Field(default="coseno")
    query_id: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=10, ge=1, le=100)


BENCHMARK_DIR = ROOT / "benchmark" / "multimedia" / "graficas"
TEXT_BENCHMARK_PATH = ROOT / "benchmark" / "text" / "results_gin.json"


app = FastAPI(
    title="Chuta DB API",
    version="1.0.0",
    description="API HTTP para consultar el motor de almacenamiento de Chuta DB.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1):\d+$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_storage = StorageManager(wal_path=os.getenv("WAL_PATH", "chuta_wal.log"))
_sessions: dict[str, ExecuteVisitor] = {}
UPLOAD_DIR = ROOT / "data" / "uploads"
SQL_INT_MIN = -(2**31)
SQL_INT_MAX = 2**31 - 1


def json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    return str(value)


def table_metadata(name: str, metadata: dict[str, Any]) -> dict[str, Any]:
    indexes = _storage.catalog.get_table_indexes(name)
    return {
        "name": name,
        "file_type": metadata["file_type"],
        "key_index": metadata["key_index"],
        "columns": [
            {
                "name": column_name,
                "type": column_type,
                "position": position,
                "primary_key": position == metadata["key_index"],
            }
            for position, (column_name, column_type) in enumerate(
                zip(metadata["column_names"], metadata["schema"])
            )
        ],
        "indexes": indexes,
    }


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/import/csv", response_model=CsvUploadResponse)
def upload_csv(request: CsvUploadRequest) -> CsvUploadResponse:
    filename = Path(request.filename).name
    suffix = Path(filename).suffix.lower()
    if suffix not in {".csv", ".json", ".jsonl"} | MEDIA_EXTENSIONS or not re.fullmatch(
        r"[A-Za-z0-9._-]+", filename
    ):
        raise HTTPException(
            status_code=400,
            detail="solo se permiten archivos CSV, JSON o JSONL con nombre simple",
        )
    try:
        content = base64.b64decode(request.content_base64, validate=True)
    except (ValueError, base64.binascii.Error) as error:
        raise HTTPException(status_code=400, detail="el contenido no es Base64 válido") from error
    max_upload_bytes = 50 * 1024 * 1024
    if len(content) > max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail="el archivo supera el límite de 50 MB",
        )
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise HTTPException(status_code=400, detail="el CSV debe estar codificado en UTF-8") from error
    if suffix in MEDIA_EXTENSIONS:
        media_dir = ROOT / "data" / "uploads" / "media"
        media_dir.mkdir(parents=True, exist_ok=True)
        target = media_dir / filename
        target.write_bytes(content)
        media_id = Path(filename).stem
        kind = "sift" if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"} else "mfcc"
        table = "demo_images" if kind == "sift" else "demo_audio"
        sql = (
            f"CREATE TABLE {table} (\n"
            "  id VARCHAR(255) PRIMARY KEY,\n"
            "  path VARCHAR(500)\n"
            ") USING HEAP;\n\n"
            f"INSERT INTO {table} VALUES ('{media_id}', "
            f"'uploads/media/{filename}');"
        )
        return CsvUploadResponse(
            filename=filename,
            path=f"data/uploads/media/{filename}",
            rows=1,
            kind="media_asset",
            sql=sql,
        )
    if suffix in {".json", ".jsonl"}:
        try:
            if suffix == ".json":
                payload = json.loads(text)
                records = payload if isinstance(payload, list) else [payload]
            else:
                records = [json.loads(line) for line in text.splitlines() if line.strip()]
        except json.JSONDecodeError as error:
            raise HTTPException(status_code=400, detail=f"JSON inválido: {error}") from error
        if not records or not all(isinstance(record, dict) for record in records):
            raise HTTPException(status_code=400, detail="el corpus JSON debe contener objetos")
        normalized = []
        for position, record in enumerate(records, start=1):
            if "id" not in record:
                raise HTTPException(status_code=400, detail=f"documento {position} no tiene id")
            text_value = record.get("lyrics", record.get("text", ""))
            if not isinstance(text_value, str) or not text_value.strip():
                raise HTTPException(
                    status_code=400,
                    detail=f"documento {position} no tiene lyrics o text",
                )
            normalized.append({
                "id": record["id"],
                "title": str(record.get("title", record.get("name", f"Documento {record['id']}"))),
                "lyrics": text_value,
            })
        corpus_path = ROOT / "data" / "lyrics.jsonl"
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        with corpus_path.open("w", encoding="utf-8") as handle:
            for record in normalized:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        # El motor SQL importa CSV. Generamos una representación tabular del
        # mismo corpus para que el upload también pueda probar CREATE INDEX.
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        sql_filename = "lyrics.jsonl.csv"
        sql_target = UPLOAD_DIR / sql_filename
        with sql_target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "title", "lyrics"])
            writer.writeheader()
            writer.writerows(normalized)
        sql = (
            "CREATE TABLE demo_lyrics (\n"
            "  id VARCHAR(64) PRIMARY KEY,\n"
            "  title VARCHAR(256),\n"
            "  lyrics VARCHAR(512)\n"
            ") USING HEAP;\n\n"
            "IMPORT INTO demo_lyrics FROM 'data/uploads/lyrics.jsonl.csv';"
        )
        return CsvUploadResponse(
            filename=filename,
            path="data/lyrics.jsonl",
            rows=len(normalized),
            kind="text_corpus",
            sql=sql,
        )

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or any(not column for column in reader.fieldnames):
        raise HTTPException(status_code=400, detail="el CSV debe tener un encabezado válido")
    sample = list(reader)
    if not sample:
        raise HTTPException(status_code=400, detail="el CSV no contiene filas")

    def infer_type(column: str) -> str:
        values = [row.get(column, "").strip() for row in sample if row.get(column, "").strip()]
        normalized_column = column.strip().lower()
        if normalized_column == "id" or normalized_column.endswith("_id"):
            # External identifiers (Deezer, albums, tracks, etc.) are not
            # bounded by the storage engine's signed 32-bit INT.
            max_length = max([len(value) for value in values] + [1])
            return f"VARCHAR({min(max(max_length + 20, 32), 1000)})"
        if values and all(value.lower() in {"true", "false", "1", "0", "t", "f", "yes", "no"} for value in values):
            return "BOOL"
        if values:
            try:
                integers = [int(value) for value in values]
                if all(SQL_INT_MIN <= value <= SQL_INT_MAX for value in integers):
                    return "INT"
                # The storage engine's INT is a signed 32-bit value. Keep
                # larger identifiers as text so their exact value is preserved.
                max_length = max(len(value) for value in values)
                return f"VARCHAR({min(max(max_length + 20, 32), 1000)})"
            except ValueError:
                try:
                    [float(value) for value in values]
                    return "FLOAT"
                except ValueError:
                    pass
        max_length = max([len(value) for value in values] + [1])
        return f"VARCHAR({min(max(max_length + 20, 32), 1000)})"

    columns = [
        {"name": column, "type": infer_type(column)}
        for column in reader.fieldnames
    ]
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target = UPLOAD_DIR / filename
    target.write_bytes(content)
    return CsvUploadResponse(
        filename=filename,
        path=f"data/uploads/{filename}",
        columns=[CsvColumn(**column) for column in columns],
        rows=len(sample),
        kind="csv",
    )


@app.post("/api/text/index")
def build_text_index(request: TextIndexRequest) -> dict[str, Any]:
    try:
        return text_search.build(request.dataset)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/text/search")
def search_text(request: TextSearchRequest) -> dict[str, Any]:
    try:
        return text_search.search(
            request.query,
            request.algorithm,
            request.dataset,
            request.top_k,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/text/benchmarks")
def text_benchmarks() -> dict[str, Any]:
    if not TEXT_BENCHMARK_PATH.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"no existe el benchmark GIN: {TEXT_BENCHMARK_PATH}. "
            "Ejecute python -m benchmark.text_gin.",
        )
    try:
        return json.loads(TEXT_BENCHMARK_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=500, detail="results_gin.json no es JSON válido") from error


@app.post("/api/multimedia/search")
def search_multimedia(request: MultimediaSearchRequest) -> dict[str, Any]:
    try:
        return multimedia_search.search(
            request.kind,
            request.index,
            request.metric,
            request.query_id,
            request.top_k,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.on_event("startup")
def precalentar_multimedia() -> None:
    multimedia_search.precalentar()


# app: la vista tipo reproductor de música (web/app/musica/page.tsx)

class AppReconocerRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    content_base64: str = Field(min_length=1, max_length=30_000_000)
    k: int = Field(default=10, ge=1, le=50)


def _app(funcion, *args):
    try:
        return funcion(*args)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/app/inicio")
def app_inicio() -> dict[str, Any]:
    return _app(catalogo.inicio)


@app.get("/api/app/artista/{artista_id}")
def app_artista(artista_id: str) -> dict[str, Any]:
    return _app(catalogo.artista, artista_id)


@app.get("/api/app/artista/{artista_id}/se-parece")
def app_artista_se_parece(artista_id: str, k: int = 10) -> dict[str, Any]:
    return _app(catalogo.se_parece, artista_id, k)


@app.get("/api/app/album/{album_id}")
def app_album(album_id: str) -> dict[str, Any]:
    return _app(catalogo.album, album_id)


@app.get("/api/app/album/{album_id}/parecidos")
def app_album_parecidos(album_id: str, k: int = 10) -> dict[str, Any]:
    return _app(catalogo.portadas_parecidas, album_id, k)


@app.get("/api/app/cancion/{track_id}")
def app_cancion(track_id: str) -> dict[str, Any]:
    return _app(catalogo.cancion, track_id)


@app.get("/api/app/cancion/{track_id}/parecidas")
def app_cancion_parecidas(track_id: str, k: int = 10) -> dict[str, Any]:
    return _app(catalogo.suena_parecido, track_id, k)


@app.post("/api/app/reconocer")
def app_reconocer(request: AppReconocerRequest) -> dict[str, Any]:
    filename = Path(request.filename).name
    suffix = Path(filename).suffix.lower()
    if suffix not in MEDIA_EXTENSIONS:
        raise HTTPException(status_code=400, detail="sube una foto (jpg, png...) o un audio (mp3, wav...)")
    try:
        content = base64.b64decode(request.content_base64, validate=True)
    except (ValueError, base64.binascii.Error) as error:
        raise HTTPException(status_code=400, detail="el contenido no es Base64 válido") from error
    destino = ROOT / "data" / "uploads" / "consultas"
    destino.mkdir(parents=True, exist_ok=True)
    seguro = re.sub(r"[^A-Za-z0-9._-]", "_", Path(filename).stem)[:80] or "consulta"
    path = destino / f"{int(time.time() * 1000)}_{seguro}{suffix}"
    path.write_bytes(content)
    return _app(catalogo.reconocer, path, True, False, request.k)


class AppCamaraRequest(BaseModel):
    # un cuadro de la cámara (o una foto) en base64, sin guardarlo en data/
    content_base64: str = Field(min_length=1, max_length=20_000_000)
    en_vivo: bool = False
    k: int = Field(default=10, ge=1, le=50)


def _bytes(contenido: str) -> bytes:
    try:
        return base64.b64decode(contenido, validate=True)
    except (ValueError, base64.binascii.Error) as error:
        raise HTTPException(status_code=400, detail="el contenido no es Base64 válido") from error


@app.post("/api/app/reconocer-cara")
def app_reconocer_cara(request: AppCamaraRequest) -> dict[str, Any]:
    return _app(catalogo.reconocer_caras, _bytes(request.content_base64), request.en_vivo, request.k)


@app.post("/api/app/reconocer-portada")
def app_reconocer_portada(request: AppCamaraRequest) -> dict[str, Any]:
    return _app(catalogo.reconocer_portada_bytes, _bytes(request.content_base64), request.en_vivo, request.k)


@app.get("/api/app/cara-recorte")
def app_cara_recorte(path: str):
    from fastapi.responses import Response
    contenido = _app(catalogo.recorte_cara, path)
    return Response(content=contenido, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


# cómo funciona: cada paso del pipeline sobre una portada o canción real (web/app/como-funciona)

@app.get("/api/explica/resumen")
def explica_resumen() -> dict[str, Any]:
    from web.backend import explica
    return _app(explica.resumen)


@app.get("/api/explica/ejemplos")
def explica_ejemplos(tipo: str = "sift", n: int = 12) -> dict[str, Any]:
    from web.backend import explica
    return {"ids": _app(explica.ejemplos, tipo, max(1, min(n, 40)))}


@app.get("/api/explica/sift")
def explica_sift(path: str) -> dict[str, Any]:
    from web.backend import explica
    return _app(explica.sift, path)


@app.get("/api/explica/mfcc")
def explica_mfcc(track: str) -> dict[str, Any]:
    from web.backend import explica
    return _app(explica.mfcc, track)


@app.get("/api/explica/dog")
def explica_dog(path: str) -> dict[str, Any]:
    from web.backend import explica
    return _app(explica.dog, path)


@app.get("/api/explica/palabras2d")
def explica_palabras2d(path: str) -> dict[str, Any]:
    from web.backend import explica
    return _app(explica.palabras2d, path)


@app.get("/api/explica/ivf")
def explica_ivf(consulta: int = 0, nprobe: int = 3) -> dict[str, Any]:
    from web.backend import explica
    return _app(explica.ivf, consulta, nprobe)


@app.get("/api/explica/hnsw")
def explica_hnsw(consulta: int = 0) -> dict[str, Any]:
    from web.backend import explica
    return _app(explica.hnsw, consulta)


@app.get("/api/explica/caras-ejemplos")
def explica_caras_ejemplos(n: int = 12) -> dict[str, Any]:
    from web.backend import explica
    return {"ids": _app(explica.caras_ejemplos, max(1, min(n, 30)))}


@app.get("/api/explica/caras")
def explica_caras(path: str) -> dict[str, Any]:
    from web.backend import explica
    return _app(explica.caras, path)


@app.get("/api/app/actividad")
def app_actividad(limite: int = 100) -> dict[str, Any]:
    return {"actividad": catalogo.actividad(limite)}


@app.post("/api/multimedia/search-file")
def search_multimedia_file(request: MultimediaFileSearchRequest) -> dict[str, Any]:
    # la consulta es una imagen o un audio subido desde la web, no un objeto de la base
    filename = Path(request.filename).name
    suffix = Path(filename).suffix.lower()
    if suffix not in MEDIA_EXTENSIONS:
        raise HTTPException(status_code=400, detail="sube una imagen (jpg, png...) o un audio (mp3, wav...)")
    try:
        content = base64.b64decode(request.content_base64, validate=True)
    except (ValueError, base64.binascii.Error) as error:
        raise HTTPException(status_code=400, detail="el contenido no es Base64 válido") from error
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="el archivo supera el límite de 20 MB")
    destino = ROOT / "data" / "uploads" / "consultas"
    destino.mkdir(parents=True, exist_ok=True)
    seguro = re.sub(r"[^A-Za-z0-9._-]", "_", Path(filename).stem)[:80] or "consulta"
    path = destino / f"{int(time.time() * 1000)}_{seguro}{suffix}"
    path.write_bytes(content)
    try:
        return multimedia_search.search_file(path, request.index, request.metric, request.top_k)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (ValueError, RuntimeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/multimedia/consultas-ejemplo")
def multimedia_consultas_ejemplo() -> dict[str, Any]:
    # fotos reales de discogs y portadas modificadas (data augmentation) para probar "subir archivo"
    data = ROOT / "data"
    salida = []

    def filas(nombre: str) -> list[dict[str, str]]:
        ruta = data / nombre
        if not ruta.is_file():
            return []
        with ruta.open(newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    reales = [f for f in filas("fotos_reales.csv") if f.get("tipo") == "principal"]
    for f in reales[:6]:
        salida.append({"path": f["path"], "etiqueta": f"foto real ({f['formato']})", "original": f["cover_path"]})
    aumentadas = filas("consultas_aumentadas.csv")
    if aumentadas:
        primera = aumentadas[0]["album_id"]
        for f in aumentadas:
            if f["album_id"] == primera:
                salida.append({"path": f["path"], "etiqueta": f["cambio"], "original": f["original"]})
    return {"consultas": salida}


@app.get("/api/multimedia/examples")
def multimedia_examples(kind: str = "sift", grupo: str = "", offset: int = 0, limit: int = 16) -> dict[str, Any]:
    try:
        return {"kind": kind, "grupo": grupo,
                **multimedia_search.examples(kind, max(1, min(limit, 64)), grupo, max(0, offset))}
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/api/multimedia/assets/{asset_path:path}")
def multimedia_asset(asset_path: str) -> FileResponse:
    """Sirve únicamente archivos multimedia ubicados dentro de data/."""
    data_root = (ROOT / "data").resolve()
    asset = (data_root / asset_path).resolve()
    if data_root not in asset.parents or asset.suffix.lower() not in MEDIA_EXTENSIONS:
        raise HTTPException(status_code=404, detail="recurso multimedia inválido")
    if not asset.is_file():
        raise HTTPException(status_code=404, detail="recurso multimedia no encontrado")
    return FileResponse(asset)


@app.get("/api/multimedia/benchmarks")
def multimedia_benchmarks() -> dict[str, Any]:
    result_path = BENCHMARK_DIR / "resultados.json"
    if not result_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"no existe el benchmark multimedia: {result_path}",
        )
    try:
        return json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=500, detail="resultados.json no es JSON válido") from error


@app.get("/api/multimedia/benchmarks/gallery/{gallery_key}")
def multimedia_benchmark_gallery(gallery_key: str) -> dict[str, Any]:
    result_path = BENCHMARK_DIR / "resultados.json"
    if not result_path.is_file():
        raise HTTPException(status_code=404, detail="no existe resultados.json")
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=500, detail="resultados.json no es JSON válido") from error
    gallery = payload.get("galeria", {}).get(gallery_key)
    if gallery is None:
        raise HTTPException(status_code=404, detail=f"galería inexistente: {gallery_key}")
    asset_name = f"galeria_{gallery_key}.png"
    return {
        "key": gallery_key,
        "cases": gallery,
        "asset": asset_name if (BENCHMARK_DIR / asset_name).is_file() else None,
    }


@app.get("/api/multimedia/benchmarks/assets/{filename}")
def multimedia_benchmark_asset(filename: str) -> FileResponse:
    asset = (BENCHMARK_DIR / filename).resolve()
    if asset.parent != BENCHMARK_DIR.resolve() or not asset.is_file():
        raise HTTPException(status_code=404, detail="recurso de benchmark no encontrado")
    return FileResponse(asset)


@app.get("/api/tables")
def list_tables() -> dict[str, list[dict[str, Any]]]:
    tables = []
    for name in ("sys_tables", "sys_columns", "sys_indexes"):
        metadata = _storage.catalog.get_table_info(name)
        if metadata:
            tables.append(table_metadata(name, metadata))
    for record in _storage.catalog.sys_tables.scan():
        name = _storage.catalog._clean_str(record[1][0])
        if name not in {table["name"] for table in tables}:
            metadata = _storage.catalog.get_table_info(name)
            if metadata:
                tables.append(table_metadata(name, metadata))
    return {"tables": tables}


@app.get("/api/spatial/points")
def spatial_points(table: str | None = None) -> dict[str, Any]:
    """Returns rows that expose longitude/latitude or POINT columns for map rendering."""
    points = []
    metadata_items = []
    for record in _storage.catalog.sys_tables.scan():
        name = _storage.catalog._clean_str(record[1][0])
        if name.startswith("sys_") or (table and name != table):
            continue
        metadata = _storage.catalog.get_table_info(name)
        if metadata:
            metadata_items.append((name, metadata))

    for name, metadata in metadata_items:
        names = [column.lower() for column in metadata["column_names"]]
        types = [t.lower() for t in metadata["schema"]]
        
        # Buscar columna de tipo POINT nativo
        point_index = next((i for i, t in enumerate(types) if t == "point"), None)
        # O fallback a columnas lon/lat separadas
        lon_index = next((names.index(column) for column in ("longitude", "lon", "lng", "x") if column in names), None)
        lat_index = next((names.index(column) for column in ("latitude", "lat", "y") if column in names), None)
        
        if point_index is None and (lon_index is None or lat_index is None):
            continue
            
        table_obj = _storage.open_table(name)
        for rid, values in table_obj.scan():
            try:
                if point_index is not None:
                    pt = values[point_index]
                    if not isinstance(pt, (tuple, list)) or len(pt) != 2:
                        continue
                    longitude, latitude = float(pt[0]), float(pt[1])
                else:
                    longitude = float(values[lon_index])
                    latitude = float(values[lat_index])
            except (TypeError, ValueError, IndexError):
                continue
                
            points.append({
                "table": name,
                "rid": [rid[0], rid[1]],
                "longitude": longitude,
                "latitude": latitude,
                "label": str(values[0]) if values else name,
                "values": [json_value(value) for value in values],
            })
    return {"points": points}


def affected_rows(message: str, fallback: int) -> int:
    match = re.search(r"(\d+)\s+fila", message or "")
    return int(match.group(1)) if match else fallback


@app.post("/api/query", response_model=QueryResponse)
def execute_query(
    request: QueryRequest,
    session_id: str = Header(default="default", alias="X-Session-ID"),
) -> QueryResponse:
    started = time.perf_counter()
    try:
        program = Parser(Scanner(request.sql)).parse_program()
        visitor = _sessions.setdefault(session_id, ExecuteVisitor(_storage))
        results = visitor.ejecutar(program)
    except Exception as error:
        detail = str(error)
        raise HTTPException(status_code=400, detail=detail) from error

    if not results:
        return QueryResponse(
            message="Consulta ejecutada",
            duration_ms=(time.perf_counter() - started) * 1000,
        )
    result = results[-1]
    duration_ms = (time.perf_counter() - started) * 1000
    rows = len(result.filas)
    return QueryResponse(
        message=result.mensaje,
        columns=result.columnas,
        rows=[[json_value(value) for value in row] for row in result.filas],
        plan=result.plan,
        row_count=rows,
        affected_rows=affected_rows(result.mensaje, rows),
        duration_ms=duration_ms,
    )


@app.get("/api/transactions")
def list_transactions() -> dict[str, list[dict[str, Any]]]:
    transactions = []
    for transaction in _storage.transaction_manager._transactions.values():
        transactions.append({
            "transaction_id": transaction.transaction_id,
            "status": transaction.status.name,
            "last_lsn": transaction.last_lsn,
        })
    return {"transactions": transactions}


@app.get("/api/transactions/{transaction_id}")
def transaction_detail(transaction_id: int) -> dict[str, Any]:
    try:
        transaction = _storage.transaction_manager.get(transaction_id)
    except Exception as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    records = [
        record for record in _storage.log_manager.iter_records()
        if record.transaction_id == transaction_id
    ]
    locks = [
        lock for lock in _storage.lock_manager.snapshot()
        if lock["transaction_id"] == transaction_id
    ]
    return {
        "transaction_id": transaction.transaction_id,
        "status": transaction.status.name,
        "last_lsn": transaction.last_lsn,
        "locks": locks,
        "events": [
            {
                "type": record.record_type.name,
                "lsn": record.lsn,
                "prev_lsn": record.prev_lsn,
                "operation": record.operation,
                "resource": record.file_name,
                "resource_type": record.resource_type,
                "page_id": record.page_id,
            }
            for record in records
        ],
    }


@app.get("/api/locks")
def list_locks() -> dict[str, list[dict[str, Any]]]:
    return {"locks": _storage.lock_manager.snapshot()}


@app.get("/api/wal")
def list_wal(transaction_id: int | None = None) -> dict[str, list[dict[str, Any]]]:
    records = []
    for record in _storage.log_manager.iter_records():
        if transaction_id is not None and record.transaction_id != transaction_id:
            continue
        records.append({
            "lsn": record.lsn,
            "prev_lsn": record.prev_lsn,
            "transaction_id": record.transaction_id,
            "type": record.record_type.name,
            "operation": record.operation,
            "resource": record.file_name,
            "file_name": record.file_name,
            "resource_type": record.resource_type,
            "page_id": record.page_id,
            "before_bytes": len(record.before),
            "after_bytes": len(record.after),
        })
    return {"records": records}


@app.get("/api/recovery/status")
def recovery_status() -> dict[str, Any]:
    records = list(_storage.log_manager.iter_records())
    return {
        "recovered_transactions": _storage.recovered_transactions,
        "last_checkpoint": next((record.lsn for record in reversed(records) if record.record_type.name == "CHECKPOINT"), None),
        "redo_records": sum(record.record_type.name == "UPDATE" and record.transaction_id not in _storage.recovered_transactions for record in records),
        "clr_records": sum(record.record_type.name == "CLR" for record in records),
    }


@app.on_event("shutdown")
def close_storage() -> None:
    _storage.close()
