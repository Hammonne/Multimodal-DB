"""
Búsqueda vectorial de la app sobre las estructuras de chutaDB.

- "exact": búsqueda secuencial (knn_exacto del core) sobre los embeddings leídos
  de FaceImage. Es el ground truth.
- "hnsw" / "ivf": VectorSecondaryIndex del core (construir_desde_vectores,
  buscar_embedding), uno por (modelo, tipo, métrica).

Los índices se guardan en data/indexes como archivos derivados: si faltan o no
coinciden con la tabla, se reconstruyen desde FaceImage.embedding. No se
registran en sys_indexes (ver PLAN.md, hallazgo H1), pero sí se exponen en
storage.index_manager.open_indexes para buscar con search_embedding_index.
"""
from __future__ import annotations

import json
import math
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from indexes.metricas import get_metrica, knn_exacto
from storage.rid import RID
from storage.vector_index import VectorSecondaryIndex

from face_recognition_app import codec, config
from face_recognition_app.db import FaceRecognitionDB, FaceRecord

APPROX_INDEX_TYPES = ("hnsw", "ivf")


def canonical_metric(metric: str) -> str:
    name = get_metrica(metric).nombre
    if name not in config.METRICS:
        raise ValueError(f"Métrica no soportada: {metric!r}")
    return name


def validate_index_type(index_type: str) -> str:
    index_type = str(index_type).strip().lower()
    if index_type not in config.INDEX_TYPES:
        raise ValueError(f"Índice no soportado: {index_type!r} (usa {', '.join(config.INDEX_TYPES)})")
    return index_type


def index_name(model: str, index_type: str, metric: str) -> str:
    return f"face_{model}_{index_type}_{metric}"


def default_nlist(n: int) -> int:
    return max(1, min(n, round(math.sqrt(n))))


# ----------------------------------------------------------------------
# Resultados
# ----------------------------------------------------------------------


@dataclass
class Hit:
    face: FaceRecord
    distance: float      # distancia de la métrica usada (menor = más parecido)
    similarity: float    # similitud coseno (embeddings normalizados), -1..1


@dataclass
class SearchResult:
    model: str
    index_type: str
    metric: str
    k: int
    hits: list[Hit]
    search_ms: float
    candidates: int
    gallery_size: int
    params: dict = field(default_factory=dict)


@dataclass
class Gallery:
    faces: list[FaceRecord]
    X: np.ndarray
    version: int = 0      # FaceRecognitionDB.version(model) con la que está sincronizada

    @property
    def size(self) -> int:
        return len(self.faces)


# ----------------------------------------------------------------------
# VectorStore
# ----------------------------------------------------------------------


class TaskTracker:
    """Tareas largas en curso (p. ej. construir un índice), para informarlas en /health."""

    def __init__(self):
        self._lock = threading.Lock()
        self._tasks: dict[int, tuple[str, float]] = {}
        self._next = 0

    @contextmanager
    def track(self, label: str):
        with self._lock:
            self._next += 1
            task_id = self._next
            self._tasks[task_id] = (label, time.time())
        try:
            yield
        finally:
            with self._lock:
                self._tasks.pop(task_id, None)

    def list(self) -> list[dict]:
        now = time.time()
        with self._lock:
            return [{"label": label, "elapsed_s": round(now - started, 1)}
                    for label, started in self._tasks.values()]


class VectorStore:
    def __init__(self, db: FaceRecognitionDB, index_dir: Path | str | None = None,
                 tracker: TaskTracker | None = None):
        self.db = db
        self.index_dir = Path(index_dir or config.DATA_DIR / config.INDEXES_SUBDIR)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = tracker or TaskTracker()
        self._lock = threading.RLock()
        self._galleries: dict[str, Gallery] = {}
        self._indexes: dict[str, VectorSecondaryIndex] = {}

    # ------------------------------------------------------------------
    # Galería (embeddings leídos de la tabla)
    # ------------------------------------------------------------------

    def gallery(self, model: str) -> Gallery:
        """
        Embeddings del modelo en memoria. Si la tabla cambió sin pasar por
        add()/remove(), se relee y se descartan los índices abiertos que ya no
        coinciden (se recargan o reconstruyen al usarse).
        """
        with self._lock:
            current = self.db.version(model)
            cached = self._galleries.get(model)
            if cached is not None and cached.version == current:
                return cached
            faces, X = self.db.load_embeddings(model)
            self._galleries[model] = Gallery(faces, X, current)
            if cached is not None:
                live = {f.rid for f in faces}
                for name in [n for n in self._indexes if n.startswith(f"face_{model}_")]:
                    vsi = self._indexes[name]
                    if {r for i, r in enumerate(vsi.rids) if i not in vsi.borrados} != live:
                        self._drop_open(name)
            return self._galleries[model]

    def _sync_version(self, model: str) -> None:
        if model in self._galleries:
            self._galleries[model].version = self.db.version(model)

    def invalidate(self, model: str | None = None) -> None:
        with self._lock:
            if model is None:
                self._galleries.clear()
            else:
                self._galleries.pop(model, None)

    # ------------------------------------------------------------------
    # Rutas de los índices
    # ------------------------------------------------------------------

    def _base_path(self, name: str) -> Path:
        return self.index_dir / name

    def _meta_path(self, name: str) -> Path:
        return self.index_dir / f"{name}.json"

    def _files(self, name: str) -> list[Path]:
        base = self._base_path(name)
        return [Path(f"{base}.npz"), Path(f"{base}_indice.npz"), self._meta_path(name)]

    def _persisted_names(self, model: str) -> list[str]:
        prefix = f"face_{model}_"
        return sorted(p.stem for p in self.index_dir.glob(f"{prefix}*.json"))

    def _register_open(self, name: str, vsi: VectorSecondaryIndex) -> None:
        self._indexes[name] = vsi
        self.db.index_manager.open_indexes[name] = vsi

    def _drop_open(self, name: str) -> None:
        self._indexes.pop(name, None)
        self.db.index_manager.open_indexes.pop(name, None)

    # ------------------------------------------------------------------
    # Construcción / carga de índices
    # ------------------------------------------------------------------

    def build_index(
        self,
        model: str,
        index_type: str,
        metric: str = config.DEFAULT_METRIC,
        M: int = config.DEFAULT_M,
        ef_construction: int = config.DEFAULT_EF_CONSTRUCTION,
        nlist: int | None = config.DEFAULT_NLIST,
    ) -> dict:
        """(Re)construye un índice HNSW/IVF desde los embeddings de la tabla."""
        index_type = validate_index_type(index_type)
        if index_type not in APPROX_INDEX_TYPES:
            raise ValueError("Solo se construyen índices hnsw o ivf.")
        metric = canonical_metric(metric)
        name = index_name(model, index_type, metric)

        with self._lock:
            gallery = self.gallery(model)
            if gallery.size == 0:
                raise ValueError(f"No hay embeddings del modelo '{model}' para indexar.")

            nlist = default_nlist(gallery.size) if not nlist else min(int(nlist), gallery.size)
            self._drop_open(name)
            for path in self._files(name):
                path.unlink(missing_ok=True)

            label = (f"Construyendo índice {config.INDEX_LABELS.get(index_type, index_type)} de "
                     f"{config.MODEL_LABELS.get(model, model)} ({config.METRIC_LABELS.get(metric, metric)}, "
                     f"{gallery.size} rostros)")
            with self.tracker.track(label):
                started = time.perf_counter()
                vsi = VectorSecondaryIndex(f"{self._base_path(name)}.npz", index_type)
                vsi.construir_desde_vectores(
                    gallery.X,
                    [f.rid for f in gallery.faces],
                    nlist=nlist,
                    M=M,
                    ef_construction=ef_construction,
                    metrica=metric,
                )
                build_ms = (time.perf_counter() - started) * 1000

            meta = {
                "name": name,
                "model": model,
                "index_type": index_type,
                "metric": metric,
                "M": M,
                "ef_construction": ef_construction,
                "nlist": nlist if index_type == "ivf" else None,
                "built_size": gallery.size,
                "build_ms": round(build_ms, 2),
                "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            self._meta_path(name).write_text(json.dumps(meta, indent=2), encoding="utf-8")
            self._register_open(name, vsi)
            return self.index_info(model, index_type, metric)

    def _load_index(self, name: str, index_type: str, gallery: Gallery) -> VectorSecondaryIndex | None:
        """Carga un índice persistido si coincide exactamente con la galería dada."""
        if not self._meta_path(name).exists():
            return None
        vsi = VectorSecondaryIndex(f"{self._base_path(name)}.npz", index_type)
        try:
            if not vsi.cargar(gallery.size):
                return None
        except Exception:
            return None
        live = {r for i, r in enumerate(vsi.rids) if i not in vsi.borrados}
        if live != {f.rid for f in gallery.faces}:
            return None
        return vsi

    def ensure_index(self, model: str, index_type: str, metric: str) -> VectorSecondaryIndex:
        """Devuelve el índice abierto; si no existe o está desfasado, lo (re)construye."""
        index_type = validate_index_type(index_type)
        metric = canonical_metric(metric)
        name = index_name(model, index_type, metric)
        with self._lock:
            gallery = self.gallery(model)   # descarta índices abiertos desfasados
            if name in self._indexes:
                return self._indexes[name]
            vsi = self._load_index(name, index_type, gallery)
            if vsi is not None:
                self._register_open(name, vsi)
                return vsi
            old = self._read_meta(name) or {}
            self.build_index(
                model, index_type, metric,
                M=old.get("M", config.DEFAULT_M),
                ef_construction=old.get("ef_construction", config.DEFAULT_EF_CONSTRUCTION),
                nlist=None,
            )
            return self._indexes[name]

    def _read_meta(self, name: str) -> dict | None:
        path = self._meta_path(name)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def index_info(self, model: str, index_type: str, metric: str) -> dict:
        metric = canonical_metric(metric)
        name = index_name(model, index_type, metric)
        meta = self._read_meta(name)
        if meta is None:
            return {"name": name, "model": model, "index_type": index_type,
                    "metric": metric, "built": False}
        size = sum(p.stat().st_size for p in self._files(name)[:2] if p.exists())
        vsi = self._indexes.get(name)
        return {
            **meta,
            "built": True,
            "loaded": vsi is not None,
            "size": len(vsi) if vsi is not None else None,
            "disk_bytes": size,
        }

    def status(self) -> list[dict]:
        out = []
        for model in config.MODELS:
            for index_type in APPROX_INDEX_TYPES:
                for metric in config.METRICS:
                    out.append(self.index_info(model, index_type, metric))
        return out

    # ------------------------------------------------------------------
    # Altas / bajas incrementales
    # ------------------------------------------------------------------

    def _open_all_persisted(self, model: str, gallery: Gallery) -> list[VectorSecondaryIndex]:
        """
        Abre los índices ya construidos del modelo que coinciden con `gallery`
        (sin construir nuevos). Los desfasados se borran: se reconstruirán desde
        la tabla en la próxima búsqueda que los use.
        """
        opened = []
        for name in self._persisted_names(model):
            if name not in self._indexes:
                meta = self._read_meta(name)
                vsi = None if meta is None else self._load_index(name, meta["index_type"], gallery)
                if vsi is None:
                    for path in self._files(name):
                        path.unlink(missing_ok=True)
                    continue
                self._register_open(name, vsi)
            opened.append(self._indexes[name])
        return opened

    def _incremental_gallery(self, model: str, n_changes: int) -> Gallery | None:
        """
        Galería en memoria si está exactamente n_changes versiones detrás de la
        tabla (o sea, solo le faltan los cambios que se están aplicando).
        Si no, la descarta: se releerá completa de la tabla.
        """
        cached = self._galleries.get(model)
        if cached is not None and cached.version + n_changes == self.db.version(model):
            return cached
        self._galleries.pop(model, None)
        for name in [n for n in self._indexes if n.startswith(f"face_{model}_")]:
            self._drop_open(name)
        return None

    def add(self, faces: list[FaceRecord], vectors) -> None:
        """
        Llamar después de insertar las filas en FaceImage (db.add_face).
        Agrega los embeddings a la galería y a todos los índices construidos de
        su modelo (inserción incremental del core: HNSW.agregar / IVF.agregar).
        """
        if not faces:
            return
        X = np.asarray(vectors, dtype=codec.DTYPE).reshape(len(faces), -1)
        by_model: dict[str, list[int]] = {}
        for i, face in enumerate(faces):
            by_model.setdefault(face.model, []).append(i)

        with self._lock:
            for model, positions in by_model.items():
                gallery = self._incremental_gallery(model, len(positions))
                if gallery is None:
                    continue
                # los índices deben coincidir con la galería previa a esta alta
                indexes = self._open_all_persisted(model, gallery)

                new_faces = [faces[i] for i in positions]
                new_X = X[positions]
                gallery.faces.extend(new_faces)
                gallery.X = new_X.copy() if gallery.X.size == 0 else np.vstack([gallery.X, new_X])
                self._sync_version(model)

                for vsi in indexes:
                    if vsi.indice is None:
                        continue
                    vsi.indice.agregar(new_X)
                    vsi.rids.extend(f.rid for f in new_faces)
                    vsi.guardar()

    def remove(self, faces: list[FaceRecord]) -> None:
        """
        Llamar después de borrar las filas de FaceImage (db.delete_person).
        Quita los embeddings de la galería y marca borrado lógico en los índices.
        """
        if not faces:
            return
        by_model: dict[str, set[RID]] = {}
        for face in faces:
            by_model.setdefault(face.model, set()).add(face.rid)

        with self._lock:
            for model, rids in by_model.items():
                gallery = self._incremental_gallery(model, len(rids))
                if gallery is None:
                    continue
                indexes = self._open_all_persisted(model, gallery)

                keep = [i for i, f in enumerate(gallery.faces) if f.rid not in rids]
                dim = gallery.X.shape[1] if gallery.X.ndim == 2 else 0
                gallery.faces = [gallery.faces[i] for i in keep]
                gallery.X = gallery.X[keep] if keep else np.zeros((0, dim), dtype=codec.DTYPE)
                self._sync_version(model)

                for vsi in indexes:
                    for rid in rids:
                        vsi.delete_ref(None, rid)
                    vsi.guardar()

    # ------------------------------------------------------------------
    # Búsqueda
    # ------------------------------------------------------------------

    def search(
        self,
        model: str,
        query,
        index_type: str = config.DEFAULT_INDEX,
        metric: str = config.DEFAULT_METRIC,
        k: int = config.DEFAULT_K,
        ef_search: int = config.DEFAULT_EF_SEARCH,
        nprobe: int = config.DEFAULT_NPROBE,
    ) -> SearchResult:
        index_type = validate_index_type(index_type)
        metric = canonical_metric(metric)
        k = int(k)
        if k < 1:
            raise ValueError("k debe ser >= 1.")
        q = codec.normalize(query)

        with self._lock:
            gallery = self.gallery(model)
            params: dict = {}
            if gallery.size == 0:
                return SearchResult(model, index_type, metric, k, [], 0.0, 0, 0, params)
            if q.size != gallery.X.shape[1]:
                raise ValueError(
                    f"La consulta tiene {q.size} dimensiones y la galería de "
                    f"'{model}' tiene {gallery.X.shape[1]}."
                )

            if index_type == "exact":
                started = time.perf_counter()
                ids, _ = knn_exacto(q, gallery.X, k=min(k, gallery.size), metrica=metric)
                search_ms = (time.perf_counter() - started) * 1000
                found = [(gallery.faces[int(i)], gallery.X[int(i)]) for i in ids]
                candidates = gallery.size
            else:
                vsi = self.ensure_index(model, index_type, metric)
                name = index_name(model, index_type, metric)
                if index_type == "hnsw":
                    params["ef_search"] = int(ef_search)
                else:
                    params["nprobe"] = int(nprobe)
                started = time.perf_counter()
                raw = self.db.index_manager.search_embedding_index(
                    name, q, k=k, nprobe=int(nprobe), ef_search=int(ef_search)
                )
                search_ms = (time.perf_counter() - started) * 1000
                candidates = int(vsi.candidatos)
                found = []
                for rid, vector in raw:
                    face = self.db.face_by_rid(rid)
                    if face is not None and face.model == model:
                        found.append((face, np.asarray(vector, dtype=codec.DTYPE)))

            hits: list[Hit] = []
            if found:
                V = np.vstack([v for _, v in found])
                distances = np.atleast_1d(get_metrica(metric).distancia(q, V))
                similarities = V @ q
                hits = [
                    Hit(face, float(d), float(s))
                    for (face, _), d, s in zip(found, distances, similarities)
                ]

            return SearchResult(
                model=model,
                index_type=index_type,
                metric=metric,
                k=k,
                hits=hits,
                search_ms=search_ms,
                candidates=candidates,
                gallery_size=gallery.size,
                params=params,
            )
