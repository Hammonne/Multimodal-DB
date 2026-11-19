from __future__ import annotations

import json
import hashlib
import os
import re
import time
from pathlib import Path
from typing import Any

from rag.embeddings.encoder import EmbeddingEncoder
from rag.llm import build_llm_client, build_rag_prompt
from rag.retrieval import HybridRetriever, RetrievalResult
from text_based.bm25 import BM25
from text_based.spimi import SPIMIIndex


class RAGService:
    """Orquesta corpus, índices, recuperación y generación de respuestas."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.data_dir = root / "data" / "rag"
        self.source_path = self.data_dir / "chunks.jsonl"
        self.normalized_path = self.data_dir / "chunks_index.jsonl"
        self.text_index_path = self.data_dir / "spimi"
        self.embeddings_path = self.data_dir / "embeddings.npz"
        self.hnsw_path = self.data_dir / "hnsw.npz"
        self.mapping_path = self.data_dir / "chunk_mapping.json"
        self.vector_manifest_path = self.data_dir / "vector_manifest.json"
        self.llm = build_llm_client()
        self._retriever: HybridRetriever | None = None

    def build(
        self, *, text: bool, vector: bool, max_papers: int | None = None
    ) -> dict[str, Any]:
        if max_papers is None and os.getenv("RAG_MAX_PAPERS"):
            max_papers = int(os.environ["RAG_MAX_PAPERS"])
        records = self._normalize_records(self._read_jsonl(self.source_path))
        if max_papers:
            keep = set(list(dict.fromkeys(r["paper_id"] for r in records))[:max_papers])
            records = [r for r in records if r["paper_id"] in keep]
        if not records:
            raise ValueError(f"no hay chunks en {self.source_path}")
        self._write_jsonl(self.normalized_path, records)

        result: dict[str, Any] = {
            "chunks": len(records),
            "text": False,
            "vector": False,
        }
        documents = (
            {"id": item["chunk_id"], "lyrics": item["content"], "title": item["title"]}
            for item in records
        )
        if text:
            index = SPIMIIndex(output_file=str(self.text_index_path))
            index.build(documents)
            result["text"] = True
            result["text_terms"] = len(index.term_offsets)
        if vector:
            model_name = os.getenv(
                "RAG_EMBEDDING_MODEL",
                "intfloat/multilingual-e5-small",
            )
            fingerprint = self._vector_fingerprint(records, model_name)
            if self._vector_index_is_current(fingerprint):
                result["vector"] = True
                result["vector_cached"] = True
                result["vector_dimension"] = EmbeddingEncoder.load(
                    self.embeddings_path
                ).shape[1]
            else:
                encoder = EmbeddingEncoder(model_name=model_name)
                encoder.build_hnsw(
                    [item["content"] for item in records],
                    [item["chunk_id"] for item in records],
                    embeddings_path=self.embeddings_path,
                    hnsw_path=self.hnsw_path,
                    mapping_path=self.mapping_path,
                    batch_size=int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "32")),
                )
                self.vector_manifest_path.write_text(
                    json.dumps({"fingerprint": fingerprint}, indent=2) + "\n",
                    encoding="utf-8",
                )
                result["vector"] = True
                result["vector_cached"] = False
                result["vector_dimension"] = EmbeddingEncoder.load(
                    self.embeddings_path
                ).shape[1]
        self._retriever = None
        return result

    def _vector_index_is_current(self, fingerprint: str) -> bool:
        if not all(
            path.is_file()
            for path in (
                self.embeddings_path,
                self.hnsw_path,
                self.mapping_path,
                self.vector_manifest_path,
            )
        ):
            return False
        try:
            manifest = json.loads(self.vector_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return manifest.get("fingerprint") == fingerprint

    @staticmethod
    def _vector_fingerprint(records: list[dict[str, Any]], model_name: str) -> str:
        digest = hashlib.sha256()
        digest.update(model_name.encode("utf-8"))
        max_length = int(os.getenv("RAG_EMBEDDING_MAX_LENGTH", "256"))
        if max_length != 512:  # 512 = comportamiento previo; conserva índices antiguos
            digest.update(f"|maxlen={max_length}".encode("utf-8"))
        for record in records:
            digest.update(str(record["chunk_id"]).encode("utf-8"))
            digest.update(b"\0")
            digest.update(record["content"].encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    def ask(
        self,
        question: str,
        *,
        mode: str,
        top_k: int,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        retriever = self._get_retriever(mode)
        retrieval_started = time.perf_counter()
        results = retriever.search(question, mode=mode, top_k=top_k, candidate_k=max(20, top_k))
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        context = self._context(results)
        prompt = build_rag_prompt(question, context)
        generation_started = time.perf_counter()
        answer = self.llm.generate(prompt)
        generation_ms = (time.perf_counter() - generation_started) * 1000
        return {
            "answer": answer,
            "sources": [self._source(result) for result in results],
            "latency_ms": {
                "embedding": 0.0 if mode == "bm25" else retrieval_ms,
                "text_retrieval": retrieval_ms if mode == "bm25" else 0.0,
                "vector_retrieval": retrieval_ms if mode == "vector" else 0.0,
                "generation": generation_ms,
                "total": (time.perf_counter() - started) * 1000,
            },
        }

    def papers(self) -> list[dict[str, Any]]:
        source = self.normalized_path if self.normalized_path.is_file() else self.source_path
        records = self._normalize_records(self._read_jsonl(source))
        papers: dict[int, dict[str, Any]] = {}
        for record in records:
            paper_id = int(record["paper_id"])
            paper = papers.setdefault(
                paper_id,
                {
                    "paper_id": paper_id,
                    "title": record["title"],
                    "authors": record["authors"],
                    "year": record["year"],
                    "language": record["language"],
                    "abstract": record["abstract"],
                    "source_file": record["source_file"],
                    "n_pages": record["page_end"] or record["page_start"] or 0,
                    "chunks": 0,
                },
            )
            paper["chunks"] += 1
        return list(papers.values())

    def _get_retriever(self, mode: str) -> HybridRetriever:
        if mode not in {"bm25", "vector", "hybrid"}:
            raise ValueError("mode debe ser bm25, vector o hybrid")
        if self._retriever is not None and (
            mode == "bm25" or self._retriever.hnsw is not None
        ):
            return self._retriever
        if not self.normalized_path.is_file():
            raise FileNotFoundError(
                f"no existe el corpus normalizado: {self.normalized_path}; "
                "construye primero los índices RAG"
            )
        records = self._normalize_records(self._read_jsonl(self.normalized_path))
        if not records:
            raise ValueError(f"el corpus normalizado está vacío: {self.normalized_path}")

        if mode in {"bm25", "hybrid"} and not self.text_index_path.with_suffix(".json").is_file():
            raise FileNotFoundError(
                f"no existe el índice textual: {self.text_index_path}"
            )
        index = (
            SPIMIIndex(output_file=str(self.text_index_path)).load()
            if self.text_index_path.with_suffix(".json").is_file()
            else None
        )
        hnsw = None
        encoder = None
        if mode in {"vector", "hybrid"}:
            if not self.hnsw_path.is_file():
                raise FileNotFoundError(f"no existe el índice HNSW: {self.hnsw_path}")
            from indexes.hnsw import HNSW

            hnsw = HNSW.cargar(self.hnsw_path)
            encoder = EmbeddingEncoder()
        self._retriever = HybridRetriever(
            BM25(index) if index is not None else None,
            hnsw,
            records,
            encoder.encode_queries if encoder else None,
        )
        return self._retriever

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            raise FileNotFoundError(f"no existe el corpus: {path}")
        records = []
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"JSON inválido en línea {line_number}") from error
                if not isinstance(value, dict):
                    raise ValueError(f"el registro {line_number} no es un objeto JSON")
                records.append(value)
        return records

    @staticmethod
    def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as target:
            for record in records:
                target.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _normalize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        records = RAGService._with_abstracts(records)
        normalized = []
        used_ids: set[int] = set()
        paper_ids: dict[str, int] = {}
        for position, record in enumerate(records, start=1):
            raw_paper_id = str(record.get("paper_id", ""))
            if raw_paper_id not in paper_ids:
                try:
                    paper_ids[raw_paper_id] = int(raw_paper_id)
                except ValueError:
                    paper_ids[raw_paper_id] = len(paper_ids) + 1
            raw_chunk_id = record.get("id", record.get("chunk_id", position))
            try:
                chunk_id = int(raw_chunk_id)
            except (TypeError, ValueError):
                chunk_id = position
            while chunk_id in used_ids:
                chunk_id += 1
            used_ids.add(chunk_id)
            normalized.append(
                {
                    "chunk_id": chunk_id,
                    "paper_id": paper_ids[raw_paper_id],
                    "title": str(record.get("title", "")),
                    "authors": str(record.get("authors", "Unknown")),
                    "year": int(record["year"]) if str(record.get("year", "")).isdigit() else None,
                    "language": str(record.get("language", "")),
                    "source_file": str(record.get("source_file", "")),
                    "content": str(record.get("text", record.get("content", record.get("lyrics", "")))),
                    "abstract": str(record.get("abstract", "")),
                    "page_start": record.get("page_start"),
                    "page_end": record.get("page_end"),
                    "pages": f"{record.get('page_start', '')}-{record.get('page_end', '')}",
                    "paper": {
                        "paper_id": paper_ids[raw_paper_id],
                        "abstract": str(record.get("abstract", "")),
                        "title": str(record.get("title", "")),
                        "authors": str(record.get("authors", "Unknown")),
                        "year": int(record["year"]) if str(record.get("year", "")).isdigit() else None,
                        "language": str(record.get("language", "")),
                    },
                }
            )
        return normalized

    @staticmethod
    def _with_abstracts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Propaga a todos los chunks de un paper el abstract extraído de su inicio."""
        abstracts: dict[str, str] = {}
        for record in records:
            key = str(record.get("paper_id", ""))
            if key in abstracts or record.get("abstract"):
                if record.get("abstract"):
                    abstracts.setdefault(key, str(record["abstract"]))
                continue
            text = str(record.get("text", record.get("content", "")))
            match = re.search(
                r"\bAbstract\b[:.\s\-—]*(.+?)(?:\b(?:1\.?\s+)?Introduction\b|\bKeywords?\b|$)",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if match and len(match.group(1).split()) >= 30:
                abstracts[key] = " ".join(match.group(1).split()[:300])
        return [
            {**record, "abstract": abstracts.get(str(record.get("paper_id", "")), "")}
            if not record.get("abstract")
            else record
            for record in records
        ]

    @staticmethod
    def _context(results: list[RetrievalResult]) -> str:
        return "\n\n".join(
            f"[Chunk {result.chunk_id}] {result.paper.get('title', '')}\n"
            f"Autores: {result.paper.get('authors') or 'Desconocido'} | "
            f"Año: {result.paper.get('year') or 's.f.'} | Páginas: {result.pages}\n"
            f"{result.content}"
            for result in results
        )

    @staticmethod
    def _source(result: RetrievalResult) -> dict[str, Any]:
        return {
            "chunk_id": result.chunk_id,
            "title": result.paper.get("title", ""),
            "authors": result.paper.get("authors", ""),
            "year": result.paper.get("year"),
            "pages": result.pages,
            "content": result.content,
            "score": result.score,
        }
