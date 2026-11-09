from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from indexes.hnsw import HNSW
from indexes.ivf import IVF
from multimedia import features


class MultimediaSearchManager:
    def __init__(self, root: Path) -> None:
        self.data = Path(os.getenv("MULTIMEDIA_DATA", root / "data"))
        self._indexes: dict[tuple[str, str, str], tuple[Any, np.ndarray, np.ndarray]] = {}
        self._locks: dict[tuple[str, str, str], threading.Lock] = {}
        self._locks_lock = threading.Lock()

    def _load_base(self, kind: str) -> dict[str, np.ndarray]:
        requested = int(os.getenv("MULTIMEDIA_K", features.K_DEFECTO))
        path = Path(features.ruta_base(kind, requested, str(self.data)))
        if not path.is_file():
            candidates = list(
                (self.data / "features").glob(f"tfidf_{kind}_k*.npz")
            )
            if candidates:
                candidates.sort(
                    key=lambda candidate: int(candidate.stem.rsplit("_k", 1)[1]),
                    reverse=True,
                )
                path = candidates[0]
                requested = int(path.stem.rsplit("_k", 1)[1])
        if not path.is_file():
            raise FileNotFoundError(
                f"no existe la base multimedia {path}. "
                f"Ejecute primero el pipeline: python -m multimedia.sift --source covers; "
                f"python -m multimedia.codebook --tipo {kind} --ks 1024; "
                f"python -m multimedia.features --tipo {kind}."
            )
        return features.cargar_base(kind, requested, str(self.data))

    def _ruta_cache(self, kind: str, index_name: str, metric: str, k: int, n: int) -> Path:
        # el hnsw de 10k audios tarda ~1.5 min y el de 20k portadas ~3 min en python:
        # se guarda la primera vez y después se carga en ~1 s
        return self.data / "features" / "indices_web" / f"{kind}_{index_name}_{metric}_k{k}_n{n}.npz"

    def _build(self, kind: str, index_name: str, metric: str) -> tuple[Any, np.ndarray, np.ndarray]:
        key = (kind, index_name, metric)
        with self._locks_lock:
            lock = self._locks.setdefault(key, threading.Lock())
        # si llegan dos búsquedas a la vez, la segunda espera al mismo índice en vez de armar otro
        with lock:
            if key in self._indexes:
                return self._indexes[key]
            base = self._load_base(kind)
            X = np.asarray(base["vectores"], dtype=np.float32)
            if len(X) == 0:
                raise ValueError("la base multimedia está vacía")
            if index_name not in {"ivf", "hnsw"}:
                raise ValueError("índice inválido: use ivf o hnsw")
            clase = IVF if index_name == "ivf" else HNSW
            cache = self._ruta_cache(kind, index_name, metric, int(base["k"]), len(X))
            started = time.perf_counter()
            if cache.is_file():
                index = clase.cargar(str(cache))
                meta = cache.with_suffix(".json")
                index.build_ms = json.loads(meta.read_text())["build_ms"] if meta.is_file() else 0.0
            else:
                if index_name == "ivf":
                    nlist = max(1, min(int(round(np.sqrt(len(X)))), len(X)))
                    index = IVF(nlist=nlist, metrica=metric).entrenar(X).agregar(X)
                else:
                    index = HNSW(M=16, ef_construction=100, metrica=metric).agregar(X)
                index.build_ms = (time.perf_counter() - started) * 1000
                cache.parent.mkdir(parents=True, exist_ok=True)
                index.guardar(str(cache))
                cache.with_suffix(".json").write_text(json.dumps({"build_ms": index.build_ms}))
            self._indexes[key] = (index, base["ids"], X)
            return self._indexes[key]

    def precalentar(self) -> None:
        # arma (o carga) de fondo los índices más usados, así la primera búsqueda no espera
        def trabajo() -> None:
            for kind in ("sift", "mfcc"):
                for index_name in ("ivf", "hnsw"):
                    try:
                        self._build(kind, index_name, "coseno")
                    except (FileNotFoundError, ValueError):
                        pass
        threading.Thread(target=trabajo, daemon=True).start()

    @staticmethod
    def _validar(kind: str, index_name: str, metric: str, top_k: int) -> tuple[str, str, str]:
        kind, index_name, metric = kind.lower(), index_name.lower(), metric.lower()
        if kind not in {"sift", "mfcc"}:
            raise ValueError("tipo inválido: use sift o mfcc")
        if index_name not in {"ivf", "hnsw"}:
            raise ValueError("índice inválido: use ivf o hnsw")
        if metric not in {"coseno", "euclidiana", "producto_punto"}:
            raise ValueError("métrica inválida")
        if not 1 <= top_k <= 100:
            raise ValueError("top_k debe estar entre 1 y 100")
        return kind, index_name, metric

    def _vecinos(self, kind: str, index_name: str, metric: str, query: np.ndarray, top_k: int,
                 query_position: int | None, query_id: str) -> dict[str, Any]:
        key = (kind, index_name, metric)
        index, ids, vectors = self._indexes.get(key) or self._build(kind, index_name, metric)
        started = time.perf_counter()
        if index_name == "ivf":
            result_ids, distances = index.buscar(query, top_k, nprobe=min(10, index.nlist))
        else:
            result_ids, distances = index.buscar(query, top_k, ef_search=max(50, top_k))
        # el índice devuelve posiciones en la base, el id real (número o ruta) sale de ids
        results = [{
            "rank": rank,
            "id": str(ids[int(position)]),
            "distance": float(distance),
            "is_query": int(position) == query_position,
        } for rank, (position, distance) in enumerate(zip(result_ids, distances), start=1)]
        return {
            "kind": kind,
            "index": index_name,
            "metric": metric,
            "query_id": query_id,
            "results": results,
            "documents": len(ids),
            "dimension": int(vectors.shape[1]),
            "build_ms": float(getattr(index, "build_ms", 0)),
            "duration_ms": (time.perf_counter() - started) * 1000,
            "candidates": int(getattr(index, "candidatos", 0)),
        }

    def search(self, kind: str, index_name: str, metric: str, query_id: str, top_k: int) -> dict[str, Any]:
        # consulta = un objeto que ya está en la base
        kind, index_name, metric = self._validar(kind, index_name, metric, top_k)
        _, ids, vectors = self._indexes.get((kind, index_name, metric)) or self._build(kind, index_name, metric)
        matches = np.flatnonzero(ids.astype(str) == str(query_id))
        if len(matches) == 0:
            raise ValueError(f"no existe el objeto multimedia {query_id!r}")
        position = int(matches[0])
        return self._vecinos(kind, index_name, metric, vectors[position], top_k, position, str(query_id))

    def search_file(self, path: Path, index_name: str, metric: str, top_k: int) -> dict[str, Any]:
        # consulta = un archivo subido que no está en la base: se le saca el vector con el mismo
        # codebook e idf que la colección
        kind = features.tipo_de(path)
        kind, index_name, metric = self._validar(kind, index_name, metric, top_k)
        k = int(self._load_base(kind)["k"])
        started = time.perf_counter()
        query = features.ExtractFeatures(str(path), k=k, data=str(self.data))
        extract_ms = (time.perf_counter() - started) * 1000
        result = self._vecinos(kind, index_name, metric, query, top_k, None, path.name)
        result["extract_ms"] = extract_ms
        return result

    def examples(self, kind: str, limit: int = 8) -> list[str]:
        base = self._load_base(kind)
        return [str(value) for value in base["ids"][:limit]]


multimedia_search = MultimediaSearchManager(Path(__file__).resolve().parents[2])
