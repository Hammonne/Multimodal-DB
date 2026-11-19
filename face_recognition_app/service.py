"""
Lógica de la aplicación (independiente de HTTP).

    registrar persona/fotos  -> embeddings (todos los modelos) -> FaceImage -> índices
    reconocer imagen          -> embeddings -> Exacto/HNSW/IVF -> FaceImage -> Person
                                 -> % similitud, agrupación por persona, umbral
    comparar configuraciones  -> misma consulta con modelos x índices x métricas elegidos
    verificar fotos (1:1)     -> matriz de similitud entre N fotos
    evaluación                -> conjunto de consultas LFW (en segundo plano)
"""
from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from face_recognition_app import config
from face_recognition_app.db import FaceRecognitionDB, Person
from face_recognition_app.embeddings import ImageAnalysis, NoFaceError
from face_recognition_app.cache import FaceCache
from face_recognition_app.vector_store import (
    SearchResult,
    TaskTracker,
    VectorStore,
    canonical_metric,
    validate_index_type,
)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_PHOTOS_PER_REQUEST = 20
MAX_VERIFY_IMAGES = 6
PERSON_SORTS = ("name_asc", "name_desc", "photos_desc", "recent")


def similarity_pct(similarity: float) -> float:
    """Similitud coseno -> porcentaje 0..100 (los negativos cuentan como 0 %)."""
    return round(float(np.clip(similarity, 0.0, 1.0)) * 100, 1)


class FaceService:
    def __init__(
        self,
        db: FaceRecognitionDB,
        store: VectorStore | None = None,
        engine_factory: Callable | None = None,
        data_dir: Path | str | None = None,
        cache: FaceCache | None = None,
    ):
        self.db = db
        self.data_dir = Path(data_dir or config.DATA_DIR)
        self.photos_dir = self.data_dir / config.PHOTOS_SUBDIR
        self.photos_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = store.tracker if store is not None else TaskTracker()
        self.store = store or VectorStore(db, self.data_dir / config.INDEXES_SUBDIR, tracker=self.tracker)
        if engine_factory is None:
            from face_recognition_app.embeddings import get_engine
            engine_factory = get_engine
        self._engine_factory = engine_factory
        self._engine = None
        self._engine_state = "idle"          # idle | loading | ready | error
        self._engine_error: str | None = None
        self._engine_lock = threading.Lock()
        self._write_lock = threading.RLock()
        # Cache de personas frecuentes: evita la busqueda en el indice cuando
        # la cara ya se reconocio antes con holgura. Ver cache.py.
        self.cache = cache if cache is not None else FaceCache()
        self._evaluation = {"state": "idle"}  # idle | running | done | error
        self.models = list(config.MODELS)

    # ------------------------------------------------------------------
    # Motor de IA (carga perezosa)
    # ------------------------------------------------------------------

    @property
    def engine(self):
        with self._engine_lock:
            if self._engine is None:
                self._engine_state = "loading"
                try:
                    with self.tracker.track("Cargando modelos de IA (ArcFace, FaceNet)"):
                        engine = self._engine_factory()
                        if hasattr(engine, "warmup"):
                            engine.warmup(self.models)
                    self._engine = engine
                    self._engine_state = "ready"
                    self._engine_error = None
                except Exception as exc:  # pragma: no cover - depende del entorno
                    self._engine_state = "error"
                    self._engine_error = f"{type(exc).__name__}: {exc}"
                    raise
            return self._engine

    @property
    def engine_ready(self) -> bool:
        return self._engine is not None

    def warmup(self) -> None:
        """Carga galerías, modelos y los índices por defecto (en segundo plano al arrancar)."""
        for m in self.models:
            self.store.gallery(m)
        try:
            self.engine
        except Exception:
            return
        for m in self.models:
            if self.store.gallery(m).size:
                for index_type in ("hnsw", "ivf"):
                    self.store.ensure_index(m, index_type, config.DEFAULT_METRIC)

    # ------------------------------------------------------------------
    # Configuración / umbrales / estado
    # ------------------------------------------------------------------

    def thresholds(self) -> dict[str, float]:
        out = dict(config.DEFAULT_THRESHOLDS)
        path = self.data_dir / config.CALIBRATION_FILE
        if path.exists():
            try:
                for model, info in json.loads(path.read_text(encoding="utf-8")).items():
                    out[model] = float(info["threshold"])
            except (OSError, ValueError, KeyError, TypeError):
                pass
        return out

    def _threshold(self, model: str, threshold: float | None) -> float:
        if threshold is None:
            return self.thresholds().get(model, 0.5)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("El umbral debe estar entre 0 y 1.")
        return float(threshold)

    def _validate_model(self, model: str) -> str:
        model = str(model).strip().lower()
        if model not in self.models:
            raise ValueError(f"Modelo no soportado: {model!r} (usa {', '.join(self.models)})")
        return model

    def get_config(self) -> dict:
        return {
            "models": [{"id": m, "label": config.MODEL_LABELS.get(m, m)} for m in self.models],
            "indexes": [{"id": i, "label": config.INDEX_LABELS.get(i, i)} for i in config.INDEX_TYPES],
            "metrics": [{"id": m, "label": config.METRIC_LABELS.get(m, m)} for m in config.METRICS],
            "defaults": {
                "model": config.DEFAULT_MODEL,
                "index": config.DEFAULT_INDEX,
                "metric": config.DEFAULT_METRIC,
                "k": config.DEFAULT_K,
                "max_k": config.MAX_K,
                "ef_search": config.DEFAULT_EF_SEARCH,
                "nprobe": config.DEFAULT_NPROBE,
                "M": config.DEFAULT_M,
                "ef_construction": config.DEFAULT_EF_CONSTRUCTION,
            },
            "thresholds": self.thresholds(),
            "max_photos_per_request": MAX_PHOTOS_PER_REQUEST,
            "max_photos_per_person": config.MAX_PHOTOS_PER_PERSON,
            "max_verify_images": MAX_VERIFY_IMAGES,
        }

    def health(self) -> dict:
        tasks = self.tracker.list()
        if self._evaluation.get("state") == "running":
            ev = self._evaluation
            tasks.append({"label": f"Evaluando {ev.get('n_queries', 0)} consultas "
                                   f"({ev.get('done', 0)}/{ev.get('total', 0)} configuraciones)",
                          "elapsed_s": round(time.time() - ev.get("started", time.time()), 1)})
        return {
            "status": "ok",
            "service": "face_recognition",
            "engine": self._engine_state,
            "engine_ready": self.engine_ready,
            "engine_error": self._engine_error,
            "tasks": tasks,
            "persons": self.db.count_persons(),
            "faces": {m: self.db.count_faces(m) for m in self.models},
        }

    def stats(self) -> dict:
        """Qué hay cargado en chutaDB (pestaña Carga de datos)."""
        persons = self.db.list_persons()
        by_source = {"own": 0, "lfw": 0}
        for p in persons:
            by_source[p.source] = by_source.get(p.source, 0) + 1
        photos = {f.photo_id for m in self.models for f in self._faces_of_model(m)}
        models = []
        for m in self.models:
            g = self.store.gallery(m)
            models.append({
                "id": m, "label": config.MODEL_LABELS.get(m, m),
                "embeddings": g.size, "dim": int(g.X.shape[1]) if g.size else None,
                "threshold": self.thresholds().get(m),
            })
        lfw = None
        manifest = self.data_dir / config.LFW_SUBDIR / "manifest.json"
        if manifest.exists():
            mf = json.loads(manifest.read_text(encoding="utf-8"))
            queries = mf.get("queries", [])
            lfw = {
                "gallery_people": len({g["person"] for g in mf.get("gallery", [])}),
                "gallery_photos": len(mf.get("gallery", [])),
                "known_queries": sum(q["known"] for q in queries),
                "unknown_queries": sum(not q["known"] for q in queries),
                "unknown_people": len({q["person"] for q in queries if not q["known"]}),
                "params": mf.get("params", {}),
            }
        calibration = None
        cal_path = self.data_dir / config.CALIBRATION_FILE
        if cal_path.exists():
            calibration = json.loads(cal_path.read_text(encoding="utf-8"))
        return {
            "persons": {"total": len(persons), **by_source},
            "photos": len(photos),
            "face_rows": self.db.count_faces(),
            "models": models,
            "lfw": lfw,
            "calibration": calibration,
            "indexes": self.store.status(),
        }

    def _faces_of_model(self, model: str):
        return self.store.gallery(model).faces

    # ------------------------------------------------------------------
    # Personas
    # ------------------------------------------------------------------

    def _person_summary(self, person: Person) -> dict:
        photos = sorted({f.photo_id for f in self.db.faces_of_person(person.person_id)})
        return {**person.to_dict(), "photo_count": len(photos), "photo_ids": photos}

    def list_persons(self, query: str = "", limit: int = 100, source: str | None = None,
                     offset: int = 0, sort: str = "name_asc") -> dict:
        if sort not in PERSON_SORTS:
            raise ValueError(f"Orden no soportado: {sort!r} (usa {', '.join(PERSON_SORTS)})")
        persons = [self._person_summary(p) for p in self.db.list_persons(query)
                   if not source or p.source == source]
        if sort == "name_desc":
            persons.sort(key=lambda p: p["name"].lower(), reverse=True)
        elif sort == "photos_desc":
            persons.sort(key=lambda p: (-p["photo_count"], p["name"].lower()))
        elif sort == "recent":
            persons.sort(key=lambda p: -p["person_id"])
        offset = max(0, int(offset))
        return {"total": len(persons), "offset": offset, "limit": limit,
                "persons": persons[offset:offset + limit]}

    def person_detail(self, person_id: int) -> dict:
        person = self.db.get_person(person_id)
        if person is None:
            raise KeyError(f"No existe la persona con person_id={person_id}.")
        faces = self.db.faces_of_person(person_id)
        photos: dict[int, dict] = {}
        for f in faces:
            p = photos.setdefault(f.photo_id, {"photo_id": f.photo_id, "image_path": f.image_path,
                                               "models": []})
            p["models"].append(f.model)
        return {**person.to_dict(), "photos": sorted(photos.values(), key=lambda p: p["photo_id"])}

    @staticmethod
    def _extension(filename: str | None) -> str:
        ext = Path(filename or "").suffix.lower()
        return ext if ext in ALLOWED_EXTENSIONS else ".jpg"

    def _analyze_uploads(self, images: list[tuple[str, bytes]]) -> list[dict]:
        """Embeddings de la cara principal de cada foto (sin tocar la BD)."""
        if not images:
            raise ValueError("Debes enviar al menos una foto.")
        if len(images) > MAX_PHOTOS_PER_REQUEST:
            raise ValueError(f"Máximo {MAX_PHOTOS_PER_REQUEST} fotos por solicitud.")
        out = []
        for filename, data in images:
            item = {"filename": filename, "data": data, "status": "ok", "faces_detected": 0}
            try:
                analysis = self.engine.analyze(data, models=self.models, max_faces=0)
                item["faces_detected"] = len(analysis.faces)
                if not analysis.faces:
                    raise NoFaceError("No se detectó ningún rostro.")
                item["face"] = analysis.faces[0]       # la más grande
                item["bbox"] = analysis.faces[0].face.to_dict(analysis.scale)["bbox"]
                if len(analysis.faces) > 1:
                    item["warning"] = (f"Se detectaron {len(analysis.faces)} rostros; "
                                       "se registró el más grande.")
            except (ValueError, FileNotFoundError) as exc:
                item["status"] = "error"
                item["error"] = str(exc)
            out.append(item)
        return out

    def _store_photos(self, person: Person, analyzed: list[dict]) -> list[dict]:
        """Guarda fotos válidas: archivo + filas FaceImage (una por modelo) + índices."""
        new_faces, new_vectors, report = [], [], []
        for item in analyzed:
            entry = {k: item[k] for k in ("filename", "status", "faces_detected") if k in item}
            for k in ("error", "warning", "bbox"):
                if k in item:
                    entry[k] = item[k]
            if item["status"] == "ok":
                photo_id = self.db.next_photo_id()
                dest = self.photos_dir / str(person.person_id) / f"{photo_id}{self._extension(item['filename'])}"
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(item["data"])
                rel = dest.relative_to(self.data_dir).as_posix()
                for model in self.models:
                    vec = item["face"].embeddings[model]
                    new_faces.append(self.db.add_face(person.person_id, photo_id, rel, model, vec))
                    new_vectors.append(vec)
                entry["photo_id"] = photo_id
            report.append(entry)
        self.db.flush()                      # durable aunque el backend no cierre ordenadamente
        self.store.add(new_faces, new_vectors)
        return report

    def _check_photo_limit(self, current: int, new: int) -> None:
        limit = config.MAX_PHOTOS_PER_PERSON
        if current + new > limit:
            raise ValueError(f"Máximo {limit} fotos por persona (ya tiene {current}, se enviaron {new}).")

    def register_person(self, name: str, images: list[tuple[str, bytes]], source: str = "own") -> dict:
        self._check_photo_limit(0, len(images))
        analyzed = self._analyze_uploads(images)
        if not any(i["status"] == "ok" for i in analyzed):
            errors = "; ".join(f"{i['filename']}: {i.get('error')}" for i in analyzed)
            raise ValueError(f"Ninguna foto tiene un rostro válido ({errors}).")
        with self._write_lock:
            person = self.db.create_person(name, source=source)
            report = self._store_photos(person, analyzed)    # incluye db.flush()
        return {"person": self._person_summary(person), "photos": report}

    def add_photos(self, person_id: int, images: list[tuple[str, bytes]]) -> dict:
        person = self.db.get_person(person_id)
        if person is None:
            raise KeyError(f"No existe la persona con person_id={person_id}.")
        self._check_photo_limit(self._person_summary(person)["photo_count"], len(images))
        analyzed = self._analyze_uploads(images)
        with self._write_lock:
            report = self._store_photos(person, analyzed)
        return {"person": self._person_summary(person), "photos": report}

    def delete_person(self, person_id: int) -> dict:
        with self._write_lock:
            removed = self.db.delete_person(person_id)
            self.db.flush()
            self.store.remove(removed)
        shutil.rmtree(self.photos_dir / str(int(person_id)), ignore_errors=True)
        return {"deleted": int(person_id), "faces_removed": len(removed)}

    def photo_file(self, photo_id: int) -> Path:
        faces = self.db.faces_of_photo(photo_id)
        if not faces:
            raise KeyError(f"No existe la foto {photo_id}.")
        path = (self.data_dir / faces[0].image_path).resolve()
        if self.data_dir.resolve() not in path.parents or not path.is_file():
            raise KeyError(f"No se encontró el archivo de la foto {photo_id}.")
        return path

    # ------------------------------------------------------------------
    # Reconocimiento
    # ------------------------------------------------------------------

    def _format(self, res: SearchResult, threshold: float, exact_ids: list[int] | None = None,
                exact_person_ids: list[int] | None = None) -> dict:
        started = time.perf_counter()
        matches, persons = [], {}
        for rank, hit in enumerate(res.hits, 1):
            person = self.db.get_person(hit.face.person_id)
            name = person.name if person else "?"
            matches.append({
                "rank": rank,
                "face_id": hit.face.face_id,
                "photo_id": hit.face.photo_id,
                "person_id": hit.face.person_id,
                "name": name,
                "distance": round(hit.distance, 6),
                "similarity": round(hit.similarity, 6),
                "similarity_pct": similarity_pct(hit.similarity),
            })
            agg = persons.get(hit.face.person_id)
            if agg is None:
                persons[hit.face.person_id] = {
                    "person_id": hit.face.person_id,
                    "name": name,
                    "similarity": round(hit.similarity, 6),
                    "similarity_pct": similarity_pct(hit.similarity),
                    "distance": round(hit.distance, 6),
                    "best_photo_id": hit.face.photo_id,
                    "votes": 1,
                }
            else:
                agg["votes"] += 1
        ranked = sorted(persons.values(), key=lambda p: (-p["similarity"], -p["votes"]))
        for i, p in enumerate(ranked, 1):
            p["rank"] = i

        best = ranked[0] if ranked else None
        known = best is not None and best["similarity"] >= threshold
        decision = {
            "status": "known" if known else "unknown",
            "person_id": best["person_id"] if known else None,
            "name": best["name"] if known else None,
            "closest_name": best["name"] if best else None,
            "similarity_pct": best["similarity_pct"] if best else 0.0,
            "threshold": threshold,
            "threshold_pct": round(threshold * 100, 1),
        }
        out = {
            "model": res.model,
            "index": res.index_type,
            "metric": res.metric,
            "k": res.k,
            "params": res.params,
            "search_ms": round(res.search_ms, 3),
            "candidates": res.candidates,
            "gallery_size": res.gallery_size,
            "decision": decision,
            "persons": ranked,
            "matches": matches,
        }
        if exact_ids is not None:
            got = {m["face_id"] for m in matches}
            out["recall_vs_exact"] = round(len(got & set(exact_ids)) / len(exact_ids), 4) if exact_ids else None
            out["in_exact"] = [m["face_id"] in set(exact_ids) for m in matches]
        if exact_person_ids is not None:
            out["in_exact_person"] = [p["person_id"] in set(exact_person_ids) for p in ranked]
        out["db_ms"] = round((time.perf_counter() - started) * 1000, 3)
        return out

    def _format_cached(self, cached, model: str, index: str, metric: str,
                       k: int, threshold: float) -> dict | None:
        """
        Arma la misma respuesta que `_format` a partir de un acierto del cache.

        Devuelve None si la cara cacheada ya no existe en la base (por ejemplo,
        si se borro a la persona): en ese caso el llamador sigue con la
        busqueda normal, de modo que un cache desactualizado nunca produce un
        resultado equivocado.
        """
        started = time.perf_counter()
        face = self.db.get_face(cached.face_id)
        if face is None:
            return None
        person = self.db.get_person(face.person_id)
        if person is None:
            return None

        match = {
            "rank": 1,
            "face_id": face.face_id,
            "photo_id": face.photo_id,
            "person_id": face.person_id,
            "name": person.name,
            "distance": round(1.0 - cached.similarity, 6),
            "similarity": round(cached.similarity, 6),
            "similarity_pct": similarity_pct(cached.similarity),
        }
        persona = {
            "person_id": face.person_id,
            "name": person.name,
            "similarity": match["similarity"],
            "similarity_pct": match["similarity_pct"],
            "distance": match["distance"],
            "best_photo_id": face.photo_id,
            "votes": 1,
            "rank": 1,
        }
        return {
            "model": model,
            "index": index,
            "metric": metric,
            "k": int(k),
            "params": {},
            # el cache resuelve sin tocar el indice: no hay busqueda que medir
            "search_ms": 0.0,
            "candidates": 0,
            "gallery_size": self.store.gallery(model).size,
            "decision": {
                "status": "known",
                "person_id": face.person_id,
                "name": person.name,
                "closest_name": person.name,
                "similarity_pct": match["similarity_pct"],
                "threshold": threshold,
                "threshold_pct": round(threshold * 100, 1),
            },
            "persons": [persona],
            "matches": [match],
            "source": "cache",
            "cache_ms": round(cached.lookup_ms, 3),
            "db_ms": round((time.perf_counter() - started) * 1000, 3),
        }

    def _check_search_params(self, k, ef_search, nprobe) -> None:
        if not 1 <= int(k) <= config.MAX_K:
            raise ValueError(f"k debe estar entre 1 y {config.MAX_K}.")
        if ef_search < 1 or nprobe < 1:
            raise ValueError("ef_search y nprobe deben ser >= 1.")

    @staticmethod
    def _timings(analysis: ImageAnalysis, started: float) -> dict:
        return {
            "detect_ms": round(analysis.detect_ms, 2),
            "embed_ms": {m: round(v, 2) for m, v in analysis.embed_ms.items()},
            "total_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    def recognize(
        self,
        image: bytes,
        model: str = config.DEFAULT_MODEL,
        index: str = config.DEFAULT_INDEX,
        metric: str = config.DEFAULT_METRIC,
        k: int = config.DEFAULT_K,
        threshold: float | None = None,
        ef_search: int = config.DEFAULT_EF_SEARCH,
        nprobe: int = config.DEFAULT_NPROBE,
    ) -> dict:
        model = self._validate_model(model)
        index = validate_index_type(index)
        metric = canonical_metric(metric)
        self._check_search_params(k, ef_search, nprobe)
        th = self._threshold(model, threshold)

        started = time.perf_counter()
        analysis: ImageAnalysis = self.engine.analyze(image, models=[model])
        faces_out = []
        version = self.db.version(model)
        for analyzed in analysis.faces:
            entry = analyzed.face.to_dict(analysis.scale)
            embedding = analyzed.embeddings[model]

            # 1) conjunto caliente: si la cara ya se reconocio antes con
            #    holgura, no hace falta tocar el indice
            cached = self.cache.lookup(model, version, embedding, th)
            if cached is not None:
                formatted = self._format_cached(cached, model, index, metric, k, th)
                if formatted is not None:
                    entry["result"] = formatted
                    faces_out.append(entry)
                    continue

            # 2) camino normal: busqueda en el indice vectorial
            with self._write_lock:
                res = self.store.search(model, embedding, index, metric, k,
                                        ef_search, nprobe)
                entry["result"] = self._format(res, th)

            # 3) si el reconocimiento fue bueno, la cara pasa a estar caliente
            if entry["result"]["decision"]["status"] == "known" and res.hits:
                self.cache.record(model, version, res.hits[0].face.face_id, embedding)

            faces_out.append(entry)

        return {
            "image": {"width": analysis.image_size[0], "height": analysis.image_size[1]},
            "params": {"model": model, "index": index, "metric": metric, "k": int(k),
                       "threshold": th, "ef_search": int(ef_search), "nprobe": int(nprobe)},
            "faces": faces_out,
            "timings": self._timings(analysis, started),
        }

    def compare(
        self,
        image: bytes,
        models: list[str] | None = None,
        indexes: list[str] | None = None,
        metrics: list[str] | None = None,
        k: int = config.DEFAULT_K,
        threshold: float | None = None,
        threshold_model: str | None = None,
        ef_search: int = config.DEFAULT_EF_SEARCH,
        nprobe: int = config.DEFAULT_NPROBE,
    ) -> dict:
        """
        Misma consulta con cada combinación (modelo, índice, métrica) elegida.
        `threshold` se aplica a `threshold_model`; los demás modelos usan su umbral calibrado.
        Recall@k de cada fila = coincidencia con el top-k exacto del mismo modelo y métrica.
        """
        models = [self._validate_model(m) for m in (models or self.models)]
        indexes = [validate_index_type(i) for i in (indexes or list(config.INDEX_TYPES))]
        metrics = [canonical_metric(m) for m in (metrics or [config.DEFAULT_METRIC])]
        if not models or not indexes or not metrics:
            raise ValueError("Elige al menos un modelo, un índice y una métrica.")
        self._check_search_params(k, ef_search, nprobe)
        thresholds = {m: self._threshold(m, threshold if m == threshold_model else None)
                      for m in models}

        started = time.perf_counter()
        analysis: ImageAnalysis = self.engine.analyze(image, models=models)
        faces_out = []
        for analyzed in analysis.faces:
            entry = analyzed.face.to_dict(analysis.scale)
            rows = []
            with self._write_lock:
                for m in models:
                    for met in metrics:
                        exact = self.store.search(m, analyzed.embeddings[m], "exact", met, k)
                        exact_ids = [h.face.face_id for h in exact.hits]
                        exact_persons = [h.face.person_id for h in exact.hits]
                        for idx in indexes:
                            res = exact if idx == "exact" else self.store.search(
                                m, analyzed.embeddings[m], idx, met, k, ef_search, nprobe)
                            rows.append(self._format(res, thresholds[m], exact_ids, exact_persons))
            entry["configs"] = rows
            faces_out.append(entry)

        return {
            "image": {"width": analysis.image_size[0], "height": analysis.image_size[1]},
            "params": {"models": models, "indexes": indexes, "metrics": metrics, "k": int(k),
                       "thresholds": thresholds, "ef_search": int(ef_search), "nprobe": int(nprobe)},
            "faces": faces_out,
            "timings": self._timings(analysis, started),
        }

    # ------------------------------------------------------------------
    # Verificación (comparar fotos entre sí, 1:1)
    # ------------------------------------------------------------------

    def verify(self, images: list[tuple[str, bytes]], threshold: float | None = None) -> dict:
        if not 2 <= len(images) <= MAX_VERIFY_IMAGES:
            raise ValueError(f"Envía entre 2 y {MAX_VERIFY_IMAGES} fotos.")
        items, vectors = [], {m: [] for m in self.models}
        for filename, data in images:
            item = {"filename": filename, "status": "ok"}
            try:
                analysis = self.engine.analyze(data, models=self.models, max_faces=0)
                if not analysis.faces:
                    raise NoFaceError("No se detectó ningún rostro.")
                face = analysis.faces[0]
                item["faces_detected"] = len(analysis.faces)
                item["bbox"] = face.face.to_dict(analysis.scale)["bbox"]
                for m in self.models:
                    vectors[m].append(face.embeddings[m])
            except (ValueError, FileNotFoundError) as exc:
                item["status"] = "error"
                item["error"] = str(exc)
                for m in self.models:
                    vectors[m].append(None)
            items.append(item)

        results = {}
        for m in self.models:
            th = self._threshold(m, threshold)
            n = len(images)
            matrix = [[None] * n for _ in range(n)]
            same = [[None] * n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    a, b = vectors[m][i], vectors[m][j]
                    if a is None or b is None:
                        continue
                    s = float(a @ b)
                    matrix[i][j] = similarity_pct(s)
                    same[i][j] = s >= th
            # foto "atípica": su similitud media con las demás válidas no alcanza el umbral
            outliers = []
            for i in range(n):
                others = [matrix[i][j] for j in range(n) if j != i and matrix[i][j] is not None]
                if others and np.mean(others) / 100 < th:
                    outliers.append(i)
            results[m] = {"threshold": th, "threshold_pct": round(th * 100, 1),
                          "similarity_pct": matrix, "same_person": same, "outliers": outliers}
        return {"images": items, "models": results}

    # ------------------------------------------------------------------
    # Índices
    # ------------------------------------------------------------------

    def rebuild_index(self, model: str, index: str, metric: str, M: int | None = None,
                      ef_construction: int | None = None, nlist: int | None = None) -> dict:
        model = self._validate_model(model)
        with self._write_lock:
            return self.store.build_index(
                model, index, metric,
                M=M or config.DEFAULT_M,
                ef_construction=ef_construction or config.DEFAULT_EF_CONSTRUCTION,
                nlist=nlist,
            )

    def index_status(self) -> list[dict]:
        return self.store.status()

    # ------------------------------------------------------------------
    # Evaluación con un conjunto de consultas (LFW)
    # ------------------------------------------------------------------

    def _evaluation_path(self) -> Path:
        return self.data_dir / config.EVALUATION_SUBDIR / "last.json"

    def last_evaluation(self) -> dict | None:
        path = self._evaluation_path()
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def evaluation_status(self) -> dict:
        return {k: v for k, v in self._evaluation.items() if k != "thread"}

    def _evaluation_inputs(self, split: str, limit: int | None):
        lfw = self.data_dir / config.LFW_SUBDIR
        manifest = lfw / "manifest.json"
        if not manifest.exists():
            raise ValueError("No hay consultas LFW preparadas (falta data/lfw/manifest.json; ver README).")
        mf = json.loads(manifest.read_text(encoding="utf-8"))
        caches = {}
        for m in self.models:
            path = lfw / f"cache_{m}.npz"
            if not path.exists():
                raise ValueError(f"Faltan los embeddings de consulta de {m} (prepare_lfw --embed).")
            z = np.load(path, allow_pickle=False)
            caches[m] = {str(f): v for f, v in zip(z["files"], z["vectors"])}
        queries = [{"person": q["person"].replace("_", " "), "known": q["known"], "file": q["file"]}
                   for q in mf["queries"] if split == "all" or q["split"] == split]
        if limit:
            # muestra estable que mantiene la proporción conocidas / desconocidas
            rng = np.random.default_rng(0)
            idx = np.sort(rng.permutation(len(queries))[:limit])
            queries = [queries[i] for i in idx]
        return queries, caches

    def start_evaluation(self, limit: int | None = 200, k: int = 10, split: str = "test") -> dict:
        from face_recognition_app.evaluation import evaluate, to_markdown

        if self._evaluation.get("state") == "running":
            raise ValueError("Ya hay una evaluación en curso.")
        if split not in ("test", "val", "all"):
            raise ValueError("split debe ser test, val o all.")
        if not 1 <= int(k) <= config.MAX_K:
            raise ValueError(f"k debe estar entre 1 y {config.MAX_K}.")
        queries, caches = self._evaluation_inputs(split, limit)
        if not queries:
            raise ValueError("No hay consultas para evaluar.")

        state = {"state": "running", "started": time.time(), "done": 0,
                 "total": len(self.models) * len(config.INDEX_TYPES) * len(config.METRICS),
                 "n_queries": len(queries)}
        self._evaluation = state

        def progress(done, total):
            state["done"], state["total"] = done, total

        def run():
            try:
                result = evaluate(self, queries, caches, k=int(k), progress=progress, verbose=False)
                result["split"] = split
                out = self._evaluation_path()
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(result, indent=2), encoding="utf-8")
                out.with_suffix(".md").write_text(to_markdown(result), encoding="utf-8")
                state["state"] = "done"
            except Exception as exc:  # pragma: no cover - se informa al usuario
                state["state"] = "error"
                state["error"] = f"{type(exc).__name__}: {exc}"
            state["elapsed_s"] = round(time.time() - state["started"], 1)

        threading.Thread(target=run, name="face-evaluation", daemon=True).start()
        return self.evaluation_status()
