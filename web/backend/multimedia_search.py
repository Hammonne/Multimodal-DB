from __future__ import annotations

import os
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

    def _build(self, kind: str, index_name: str, metric: str) -> tuple[Any, np.ndarray, np.ndarray]:
        base = self._load_base(kind)
        X = np.asarray(base["vectores"], dtype=np.float32)
        if len(X) == 0:
            raise ValueError("la base multimedia está vacía")
        started = time.perf_counter()
        if index_name == "ivf":
            nlist = max(1, min(int(round(np.sqrt(len(X)))), len(X)))
            index = IVF(nlist=nlist, metrica=metric).entrenar(X).agregar(X, base["ids"])
        elif index_name == "hnsw":
            index = HNSW(M=16, ef_construction=100, metrica=metric).agregar(X, base["ids"])
        else:
            raise ValueError("índice inválido: use ivf o hnsw")
        index.build_ms = (time.perf_counter() - started) * 1000
        self._indexes[(kind, index_name, metric)] = (index, base["ids"], X)
        return index, base["ids"], X

    def search(self, kind: str, index_name: str, metric: str, query_id: str, top_k: int) -> dict[str, Any]:
        kind = kind.lower()
        index_name = index_name.lower()
        metric = metric.lower()
        if kind not in {"sift", "mfcc"}:
            raise ValueError("tipo inválido: use sift o mfcc")
        if index_name not in {"ivf", "hnsw"}:
            raise ValueError("índice inválido: use ivf o hnsw")
        if metric not in {"coseno", "euclidiana", "producto_punto"}:
            raise ValueError("métrica inválida")
        if not 1 <= top_k <= 100:
            raise ValueError("top_k debe estar entre 1 y 100")
        key = (kind, index_name, metric)
        index, ids, vectors = self._indexes.get(key) or self._build(kind, index_name, metric)
        matches = np.flatnonzero(ids.astype(str) == str(query_id))
        if len(matches) == 0:
            raise ValueError(f"no existe el objeto multimedia {query_id!r}")
        query_position = int(matches[0])
        query = vectors[query_position]
        started = time.perf_counter()
        if index_name == "ivf":
            result_ids, distances = index.buscar(query, top_k, nprobe=min(10, index.nlist))
        else:
            result_ids, distances = index.buscar(query, top_k, ef_search=max(50, top_k))
        positions = {str(value): i for i, value in enumerate(ids)}
        results = [{
            "rank": rank,
            "id": str(result_id),
            "distance": float(distance),
            "is_query": positions.get(str(result_id)) == query_position,
        } for rank, (result_id, distance) in enumerate(zip(result_ids, distances), start=1)]
        return {
            "kind": kind,
            "index": index_name,
            "metric": metric,
            "query_id": str(query_id),
            "results": results,
            "documents": len(ids),
            "dimension": int(vectors.shape[1]),
            "build_ms": float(getattr(index, "build_ms", 0)),
            "duration_ms": (time.perf_counter() - started) * 1000,
            "candidates": int(getattr(index, "candidatos", 0)),
        }

    def examples(self, kind: str, limit: int = 8) -> list[str]:
        base = self._load_base(kind)
        return [str(value) for value in base["ids"][:limit]]


multimedia_search = MultimediaSearchManager(Path(__file__).resolve().parents[2])
