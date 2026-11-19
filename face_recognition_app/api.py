"""
Router REST de la app de reconocimiento facial (/api/faces).

web/backend/main.py lo monta (sin cambios) con:
    create_face_recognition_tables(_storage)
    _face_db = FaceRecognitionDB(_storage)
    app.include_router(create_face_router(_face_db))

Solo traduce HTTP <-> FaceService; la lógica vive en service.py.
También sirve el frontend estático en /api/faces/ui/.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field

from face_recognition_app import config
from face_recognition_app.db import FaceRecognitionDB
from face_recognition_app.embeddings import NoFaceError
from face_recognition_app.service import FaceService

FRONTEND_DIR = config.APP_DIR / "frontend"


class RebuildIndexRequest(BaseModel):
    model: str = config.DEFAULT_MODEL
    index: str = Field(default="hnsw", pattern="^(hnsw|ivf)$")
    metric: str = config.DEFAULT_METRIC
    M: int | None = Field(default=None, ge=2, le=128)
    ef_construction: int | None = Field(default=None, ge=4, le=1000)
    nlist: int | None = Field(default=None, ge=1)


class RunEvaluationRequest(BaseModel):
    limit: int | None = Field(default=200, ge=10, le=5000)
    k: int = Field(default=10, ge=1, le=config.MAX_K)
    split: str = Field(default="test", pattern="^(test|val|all)$")


def _call(fn, *args, **kwargs):
    """Ejecuta una operación del servicio traduciendo errores a HTTP."""
    try:
        return fn(*args, **kwargs)
    except NoFaceError as exc:
        raise HTTPException(422, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(404, str(exc.args[0]) if exc.args else "No encontrado.") from exc
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc


async def _read_uploads(files: list[UploadFile]) -> list[tuple[str, bytes]]:
    out = []
    for f in files:
        data = await f.read()
        if data:
            out.append((f.filename or "foto.jpg", data))
    return out


def create_face_router(db: FaceRecognitionDB, service: FaceService | None = None,
                       warmup: bool | None = None) -> APIRouter:
    """
    db: FaceRecognitionDB sobre el StorageManager del backend.
    service: inyectable (los tests usan uno con un motor de IA falso).
    warmup: cargar modelos en segundo plano al crear el router
            (por defecto sí, salvo FACE_APP_WARMUP=0 o si se inyecta service).
    """
    svc = service or FaceService(db)
    if warmup is None:
        warmup = service is None and os.getenv("FACE_APP_WARMUP", "1") != "0"
    if warmup:
        threading.Thread(target=svc.warmup, name="face-warmup", daemon=True).start()

    router = APIRouter(prefix="/api/faces", tags=["Face Recognition"])

    # -- estado / configuración --------------------------------------------

    @router.get("/health")
    def health():
        return svc.health()

    @router.get("/config")
    def get_config():
        return svc.get_config()

    # -- personas --------------------------------------------------------

    @router.get("/stats")
    def stats():
        return svc.stats()

    @router.get("/cache")
    def cache_stats():
        """Estado del cache de personas frecuentes: backend, aciertos y fallos."""
        return svc.cache.stats()

    @router.post("/cache/clear")
    def cache_clear():
        """Vacia el cache. Util al cambiar la galeria o para medir en frio."""
        svc.cache.invalidate()
        return svc.cache.stats()

    @router.get("/persons")
    def list_persons(q: str = "", limit: int = 24, offset: int = 0, sort: str = "name_asc",
                     source: str | None = None):
        return _call(svc.list_persons, q, max(1, min(limit, 1000)), source or None, offset, sort)

    @router.get("/persons/{person_id}")
    def get_person(person_id: int):
        return _call(svc.person_detail, person_id)

    @router.post("/persons")
    async def create_person(name: str = Form(...), photos: list[UploadFile] = File(...)):
        images = await _read_uploads(photos)
        return _call(svc.register_person, name, images)

    @router.post("/persons/{person_id}/photos")
    async def add_photos(person_id: int, photos: list[UploadFile] = File(...)):
        images = await _read_uploads(photos)
        return _call(svc.add_photos, person_id, images)

    @router.delete("/persons/{person_id}")
    def delete_person(person_id: int):
        return _call(svc.delete_person, person_id)

    @router.get("/images/{photo_id}")
    def get_image(photo_id: int):
        return FileResponse(_call(svc.photo_file, photo_id))

    # -- reconocimiento / verificación ------------------------------------

    @router.post("/recognize")
    async def recognize(
        image: UploadFile = File(...),
        model: str = Form(config.DEFAULT_MODEL),
        index: str = Form(config.DEFAULT_INDEX),
        metric: str = Form(config.DEFAULT_METRIC),
        k: int = Form(config.DEFAULT_K),
        threshold: float | None = Form(None),
        ef_search: int = Form(config.DEFAULT_EF_SEARCH),
        nprobe: int = Form(config.DEFAULT_NPROBE),
    ):
        data = await image.read()
        return _call(svc.recognize, data, model=model, index=index, metric=metric, k=k,
                     threshold=threshold, ef_search=ef_search, nprobe=nprobe)

    @router.post("/compare")
    async def compare(
        image: UploadFile = File(...),
        models: str = Form(",".join(config.MODELS)),
        indexes: str = Form(",".join(config.INDEX_TYPES)),
        metrics: str = Form(config.DEFAULT_METRIC),
        k: int = Form(config.DEFAULT_K),
        threshold: float | None = Form(None),
        threshold_model: str | None = Form(None),
        ef_search: int = Form(config.DEFAULT_EF_SEARCH),
        nprobe: int = Form(config.DEFAULT_NPROBE),
    ):
        split = lambda s: [x.strip() for x in s.split(",") if x.strip()]
        data = await image.read()
        return _call(svc.compare, data, models=split(models), indexes=split(indexes),
                     metrics=split(metrics), k=k, threshold=threshold,
                     threshold_model=threshold_model, ef_search=ef_search, nprobe=nprobe)

    @router.post("/verify")
    async def verify(images: list[UploadFile] = File(...), threshold: float | None = Form(None)):
        return _call(svc.verify, await _read_uploads(images), threshold)

    # -- índices / evaluación ---------------------------------------------

    @router.get("/indexes")
    def index_status():
        return {"indexes": svc.index_status()}

    @router.post("/indexes/rebuild")
    def rebuild_index(request: RebuildIndexRequest):
        return _call(svc.rebuild_index, request.model, request.index, request.metric,
                     request.M, request.ef_construction, request.nlist)

    @router.get("/evaluation/last")
    def last_evaluation():
        result = svc.last_evaluation()
        if result is None:
            raise HTTPException(404, "Todavía no se corrió la evaluación.")
        return result

    @router.get("/evaluation/status")
    def evaluation_status():
        return svc.evaluation_status()

    @router.post("/evaluation/run")
    def run_evaluation(request: RunEvaluationRequest):
        return _call(svc.start_evaluation, request.limit, request.k, request.split)

    # -- frontend estático --------------------------------------------------

    @router.get("/ui", include_in_schema=False)
    def ui_root():
        return RedirectResponse("/api/faces/ui/")

    @router.get("/ui/{path:path}", include_in_schema=False)
    def ui_file(path: str = ""):
        target = (FRONTEND_DIR / (path or "index.html")).resolve()
        if FRONTEND_DIR.resolve() not in target.parents or not target.is_file():
            raise HTTPException(404, "No encontrado.")
        return FileResponse(target, headers={"Cache-Control": "no-cache"})

    return router
