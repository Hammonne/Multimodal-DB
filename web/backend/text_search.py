from __future__ import annotations

import json
import os
import time
from pathlib import Path
from threading import Lock
from typing import Any

from text_based.bm25 import BM25
from text_based.spimi import SPIMIIndex, iter_documents
from text_based.tfidf import TFIDFCosine


DATASET_LIMITS = {"1k": 1_000, "10k": 10_000, "100k": 100_000}
ALGORITHMS = {"tfidf", "bm25"}


class TextSearchManager:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.source = Path(os.getenv("TEXT_SEARCH_SOURCE", root / "data" / "lyrics.jsonl"))
        self.index_root = Path(os.getenv("TEXT_SEARCH_INDEX_ROOT", root / "data" / "search_indexes"))
        self._lock = Lock()
        self._rankers: dict[tuple[str, str], TFIDFCosine | BM25] = {}

    def _key(self, dataset: str, algorithm: str) -> tuple[str, str]:
        dataset = dataset.lower()
        algorithm = algorithm.lower()
        if dataset not in DATASET_LIMITS:
            raise ValueError(f"dataset inválido: use uno de {', '.join(DATASET_LIMITS)}")
        if algorithm not in ALGORITHMS:
            raise ValueError("algoritmo inválido: use tfidf o bm25")
        return dataset, algorithm

    def _paths(self, dataset: str) -> tuple[Path, Path]:
        base = self.index_root / dataset
        return base / "spimi", base / "documents.json"

    @staticmethod
    def _index_size(index: SPIMIIndex) -> int:
        return sum(Path(path).stat().st_size for path in index.artifact_paths())

    def build(self, dataset: str) -> dict[str, Any]:
        dataset = dataset.lower()
        self._key(dataset, "tfidf")
        if not self.source.is_file():
            raise FileNotFoundError(
                f"no existe el corpus textual: {self.source}. "
                "Configure TEXT_SEARCH_SOURCE o genere data/lyrics.jsonl."
            )

        index_path, manifest_path = self._paths(dataset)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        documents: list[dict[str, Any]] = []
        for document in iter_documents(str(self.source)):
            if len(documents) >= DATASET_LIMITS[dataset]:
                break
            if "id" not in document:
                raise ValueError("el corpus contiene un documento sin id")
            text = str(document.get("lyrics", document.get("text", "")))
            documents.append({
                "id": document["id"],
                "text": text,
                "title": document.get("title", document.get("name", f"Documento {document['id']}")),
            })
        if not documents:
            raise ValueError("el corpus textual está vacío")

        index = SPIMIIndex(output_file=str(index_path))
        index.build({"id": doc["id"], "lyrics": doc["text"]} for doc in documents)
        manifest_path.write_text(json.dumps(documents, ensure_ascii=False), encoding="utf-8")
        self._rankers = {
            key: ranker for key, ranker in self._rankers.items()
            if key[0] != dataset
        }
        duration_ms = (time.perf_counter() - started) * 1000
        size_bytes = self._index_size(index)
        return {
            "dataset": dataset,
            "algorithm": "spimi",
            "documents": len(documents),
            "terms": len(index.term_offsets),
            "build_ms": duration_ms,
            "index_size_bytes": size_bytes,
        }

    def _ensure(self, dataset: str, algorithm: str) -> tuple[SPIMIIndex, TFIDFCosine | BM25, list[dict[str, Any]]]:
        dataset, algorithm = self._key(dataset, algorithm)
        index_path, manifest_path = self._paths(dataset)
        if not index_path.with_suffix(".json").is_file() or not manifest_path.is_file():
            self.build(dataset)
        key = (dataset, algorithm)
        with self._lock:
            ranker = self._rankers.get(key)
            index = SPIMIIndex(output_file=str(index_path)).load()
            if ranker is None:
                ranker = TFIDFCosine(index) if algorithm == "tfidf" else BM25(index)
                ranker.build_stats()
                self._rankers[key] = ranker
        documents = json.loads(manifest_path.read_text(encoding="utf-8"))
        return index, ranker, documents

    def search(self, query: str, algorithm: str, dataset: str, top_k: int) -> dict[str, Any]:
        if not query.strip():
            raise ValueError("la consulta no puede estar vacía")
        if not 1 <= top_k <= 100:
            raise ValueError("top_k debe estar entre 1 y 100")
        started = time.perf_counter()
        index, ranker, documents = self._ensure(dataset, algorithm)
        raw_results = ranker.search(query, top_k)
        by_id = {int(document["id"]): document for document in documents}
        results = []
        for rank, (doc_id, score) in enumerate(raw_results, start=1):
            document = by_id.get(int(doc_id), {"id": doc_id, "title": f"Documento {doc_id}", "text": ""})
            results.append({
                "rank": rank,
                "id": doc_id,
                "title": document["title"],
                "snippet": document["text"][:240],
                "score": score,
                "length": index.get_doc_length(doc_id),
            })
        return {
            "query": query,
            "algorithm": algorithm.lower(),
            "dataset": dataset.lower(),
            "results": results,
            "duration_ms": (time.perf_counter() - started) * 1000,
            "documents": index.doc_count,
            "terms": len(index.term_offsets),
            "index_size_bytes": self._index_size(index),
        }


text_search = TextSearchManager(Path(__file__).resolve().parents[2])
