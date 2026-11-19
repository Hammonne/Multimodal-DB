from __future__ import annotations

from storage.storage_manager import StorageManager

from rag.models import Chunk, Paper
from rag.storage.corpus_store import CorpusStore


def test_corpus_store_inserts_and_recovers_papers_and_chunks(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    with StorageManager(wal_path="rag_test_wal.log") as storage:
        store = CorpusStore(storage)
        paper_id = store.insert_paper(
            Paper(
                title="Approximate nearest neighbor search",
                authors="García et al.",
                year=2023,
                language="en",
                source_file="paper.pdf",
                n_pages=8,
            )
        )
        chunks_to_insert = [
            Chunk(
                paper_id=paper_id,
                chunk_idx=0,
                page_start=1,
                page_end=2,
                section="Introduction",
                content="HNSW construye un grafo navegable.",
                n_words=6,
            ),
            Chunk(
                paper_id=paper_id,
                chunk_idx=1,
                page_start=2,
                page_end=3,
                section="Methods",
                content="La búsqueda usa niveles jerárquicos.",
                n_words=6,
            ),
        ]
        store.insert_chunks(chunks_to_insert)

        assert [chunk.chunk_id for chunk in chunks_to_insert] == [1, 2]
        assert [chunk.chunk_id for chunk in store.get_chunks([2, 1])] == [2, 1]

    with StorageManager(wal_path="rag_test_wal.log") as storage:
        store = CorpusStore(storage)
        chunks = store.get_chunks([1, 2])

        assert chunks[0].paper_id == paper_id
        assert chunks[0].content.startswith("HNSW")
        assert chunks[1].section == "Methods"
