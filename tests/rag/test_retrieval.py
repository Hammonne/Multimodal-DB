from __future__ import annotations

import numpy as np

from rag.retrieval import HybridRetriever


class FakeBM25:
    def search(self, query, top_k):
        return [(1, 3.0), (2, 2.0), (3, 1.0)][:top_k]


class FakeHNSW:
    def buscar(self, query, k, ef_search):
        return np.array([2, 3, 4][:k]), np.zeros(min(k, 3), dtype=np.float32)


def test_hybrid_fuses_rankings_and_limits_chunks_per_paper():
    chunks = [
        {"chunk_id": 1, "paper_id": 10, "content": "uno", "pages": "1"},
        {"chunk_id": 2, "paper_id": 10, "content": "dos", "pages": "2"},
        {"chunk_id": 3, "paper_id": 10, "content": "tres", "pages": "3"},
        {"chunk_id": 4, "paper_id": 11, "content": "cuatro", "pages": "4"},
    ]
    retriever = HybridRetriever(
        FakeBM25(),
        FakeHNSW(),
        chunks,
        lambda queries: np.ones((len(list(queries)), 2), dtype=np.float32),
    )

    results = retriever.search("consulta", mode="hybrid", top_k=5, candidate_k=3)

    assert [result.chunk_id for result in results] == [2, 3, 4]
    assert [result.rank for result in results] == [1, 2, 3]
