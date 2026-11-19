from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from indexes.hnsw import HNSW
from text_based.bm25 import BM25


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: int
    score: float
    rank: int
    paper: dict[str, Any]
    content: str
    pages: str


class HybridRetriever:
    """Combina BM25 y HNSW con Reciprocal Rank Fusion."""

    def __init__(
        self,
        bm25: BM25 | None,
        hnsw: HNSW | None,
        chunks: Iterable[dict[str, Any]],
        query_encoder: Callable[[Iterable[str]], np.ndarray] | None,
        *,
        rrf_k: int = 60,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k debe ser positivo")
        self.bm25 = bm25
        self.hnsw = hnsw
        self.query_encoder = query_encoder
        self.rrf_k = rrf_k
        self.chunks = {int(chunk["chunk_id"]): dict(chunk) for chunk in chunks}

    def search(
        self,
        query: str,
        *,
        mode: str = "hybrid",
        top_k: int = 5,
        candidate_k: int = 20,
        year: int | None = None,
        author: str | None = None,
        language: str | None = None,
    ) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("la consulta no puede estar vacía")
        if mode not in {"bm25", "vector", "hybrid"}:
            raise ValueError("mode debe ser bm25, vector o hybrid")
        if top_k <= 0 or candidate_k <= 0:
            raise ValueError("top_k y candidate_k deben ser positivos")

        eligible = self._eligible(year=year, author=author, language=language)
        rankings: list[list[int]] = []
        if mode in {"bm25", "hybrid"}:
            if self.bm25 is None:
                raise RuntimeError("el índice textual no está disponible")
            rankings.append(
                [
                    int(chunk_id)
                    for chunk_id, _ in self.bm25.search(query, candidate_k)
                    if int(chunk_id) in eligible
                ]
            )
        if mode in {"vector", "hybrid"}:
            if self.hnsw is None or self.query_encoder is None:
                raise RuntimeError("el índice vectorial no está disponible")
            query_vector = np.asarray(self.query_encoder([query]), dtype=np.float32)
            ids, _ = self.hnsw.buscar(query_vector[0], candidate_k, ef_search=50)
            rankings.append([int(chunk_id) for chunk_id in ids if int(chunk_id) in eligible])

        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, chunk_id in enumerate(ranking, start=1):
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 1 / (
                    self.rrf_k + rank
                )

        ordered = sorted(fused, key=lambda chunk_id: fused[chunk_id], reverse=True)
        paper_counts: dict[Any, int] = {}
        results: list[RetrievalResult] = []
        for chunk_id in ordered:
            chunk = self.chunks[chunk_id]
            paper_id = chunk.get("paper_id")
            if paper_counts.get(paper_id, 0) >= 2:
                continue
            paper_counts[paper_id] = paper_counts.get(paper_id, 0) + 1
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    score=fused[chunk_id],
                    rank=len(results) + 1,
                    paper=dict(chunk.get("paper", {})),
                    content=str(chunk.get("content", "")),
                    pages=str(chunk.get("pages", "")),
                )
            )
            if len(results) >= top_k:
                break
        return results

    def _eligible(
        self,
        *,
        year: int | None,
        author: str | None,
        language: str | None,
    ) -> set[int]:
        eligible = set(self.chunks)
        for chunk_id, chunk in self.chunks.items():
            paper = chunk.get("paper", {})
            if year is not None and paper.get("year") != year:
                eligible.discard(chunk_id)
            if author is not None and author.casefold() not in str(
                paper.get("authors", "")
            ).casefold():
                eligible.discard(chunk_id)
            if language is not None and str(paper.get("language", "")).casefold() != language.casefold():
                eligible.discard(chunk_id)
        return eligible
