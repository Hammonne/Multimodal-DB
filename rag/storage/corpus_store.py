from __future__ import annotations

from collections.abc import Iterable

from storage.storage_manager import StorageManager

from rag.models import Chunk, Paper


class CorpusStore:
    """Adaptador de persistencia del corpus RAG sobre el motor propio."""

    PAPERS_TABLE = "rag_papers"
    CHUNKS_TABLE = "rag_chunks"

    _PAPERS_SCHEMA = [
        "integer",
        "varchar(500)",
        "varchar(1000)",
        "integer",
        "varchar(10)",
        "varchar(255)",
        "integer",
    ]
    _PAPERS_COLUMNS = [
        "paper_id",
        "title",
        "authors",
        "year",
        "language",
        "source_file",
        "n_pages",
    ]
    _CHUNKS_SCHEMA = [
        "integer",
        "integer",
        "integer",
        "integer",
        "integer",
        "varchar(255)",
        "varchar(3000)",
        "integer",
    ]
    _CHUNKS_COLUMNS = [
        "chunk_id",
        "paper_id",
        "chunk_idx",
        "page_start",
        "page_end",
        "section",
        "content",
        "n_words",
    ]

    def __init__(
        self,
        storage_manager: StorageManager,
        *,
        papers_table: str = PAPERS_TABLE,
        chunks_table: str = CHUNKS_TABLE,
    ) -> None:
        self.storage_manager = storage_manager
        self.papers_table_name = papers_table
        self.chunks_table_name = chunks_table
        self._ensure_table(
            papers_table,
            self._PAPERS_SCHEMA,
            self._PAPERS_COLUMNS,
        )
        self._ensure_table(
            chunks_table,
            self._CHUNKS_SCHEMA,
            self._CHUNKS_COLUMNS,
        )
        self._papers = self.storage_manager.open_table(papers_table)
        self._chunks = self.storage_manager.open_table(chunks_table)

    def insert_paper(self, paper: Paper) -> int:
        """Inserta un paper y devuelve su identificador persistido."""
        self._validate_paper(paper)
        existing = self._find_paper_by_source(paper.source_file)
        if existing is not None:
            raise ValueError(
                f"el paper '{paper.source_file}' ya existe con id {existing.paper_id}"
            )

        paper_id = paper.paper_id if paper.paper_id is not None else self._next_id(
            self._papers.scan()
        )
        if self._papers.search_by_key(paper_id):
            raise ValueError(f"el paper_id {paper_id} ya existe")

        self._papers.insert(
            [
                paper_id,
                paper.title,
                paper.authors,
                paper.year or 0,
                paper.language,
                paper.source_file,
                paper.n_pages,
            ]
        )
        return paper_id

    def insert_chunks(self, chunks: list[Chunk]) -> None:
        """Inserta chunks y asigna sus IDs a los objetos recibidos."""
        if not chunks:
            return

        paper_ids = {chunk.paper_id for chunk in chunks}
        existing_paper_ids = {
            values[0] for _, values in self._papers.scan()
        }
        missing_papers = paper_ids - existing_paper_ids
        if missing_papers:
            raise ValueError(
                f"no existen los paper_id requeridos: {sorted(missing_papers)}"
            )

        next_id = self._next_id(self._chunks.scan())
        seen_ids: set[int] = set()
        rows: list[list[object]] = []
        for chunk in chunks:
            self._validate_chunk(chunk)
            chunk_id = chunk.chunk_id if chunk.chunk_id is not None else next_id
            if chunk.chunk_id is None:
                next_id += 1
            if chunk_id in seen_ids or self._chunks.search_by_key(chunk_id):
                raise ValueError(f"el chunk_id {chunk_id} ya existe")
            seen_ids.add(chunk_id)
            chunk.chunk_id = chunk_id
            rows.append(
                [
                    chunk_id,
                    chunk.paper_id,
                    chunk.chunk_idx,
                    chunk.page_start or 0,
                    chunk.page_end or 0,
                    chunk.section or "",
                    chunk.content,
                    chunk.n_words,
                ]
            )

        for row in rows:
            self._chunks.insert(row)

    def get_chunks(self, chunk_ids: list[int]) -> list[Chunk]:
        """Recupera chunks en el orden solicitado."""
        if not chunk_ids:
            return []

        by_id = {
            values[0]: self._chunk_from_values(values)
            for _, values in self._chunks.scan()
        }
        missing = [chunk_id for chunk_id in chunk_ids if chunk_id not in by_id]
        if missing:
            raise KeyError(f"no existen los chunk_id: {missing}")
        return [by_id[chunk_id] for chunk_id in chunk_ids]

    def _ensure_table(
        self,
        name: str,
        schema: list[str],
        columns: list[str],
    ) -> None:
        table_info = self.storage_manager.catalog.get_table_info(name)
        if table_info is None:
            self.storage_manager.create_table(
                name=name,
                schema=schema,
                file_type="heap",
                key_index=0,
                column_names=columns,
            )
            return
        if (
            table_info["schema"] != schema
            or table_info["column_names"] != columns
            or table_info["key_index"] != 0
        ):
            raise ValueError(
                f"la tabla '{name}' existe con un esquema incompatible"
            )

    def _find_paper_by_source(self, source_file: str) -> Paper | None:
        for _, values in self._papers.scan():
            if values[5] == source_file:
                return self._paper_from_values(values)
        return None

    @staticmethod
    def _next_id(records: Iterable[tuple[object, list[object]]]) -> int:
        ids = [int(values[0]) for _, values in records]
        return max(ids, default=0) + 1

    @staticmethod
    def _validate_paper(paper: Paper) -> None:
        if paper.paper_id is not None and paper.paper_id <= 0:
            raise ValueError("paper_id debe ser positivo")
        if not paper.title.strip():
            raise ValueError("title no puede estar vacío")
        if not paper.source_file.strip():
            raise ValueError("source_file no puede estar vacío")
        if paper.year is not None and not 1000 <= paper.year <= 9999:
            raise ValueError("year debe ser un año de cuatro dígitos")
        if paper.n_pages < 0:
            raise ValueError("n_pages no puede ser negativo")
        CorpusStore._validate_text_length("title", paper.title, 500)
        CorpusStore._validate_text_length("authors", paper.authors, 1000)
        CorpusStore._validate_text_length("language", paper.language, 10)
        CorpusStore._validate_text_length("source_file", paper.source_file, 255)

    @staticmethod
    def _validate_chunk(chunk: Chunk) -> None:
        if chunk.paper_id <= 0:
            raise ValueError("paper_id debe ser positivo")
        if chunk.chunk_id is not None and chunk.chunk_id <= 0:
            raise ValueError("chunk_id debe ser positivo")
        if chunk.chunk_idx < 0:
            raise ValueError("chunk_idx no puede ser negativo")
        if not chunk.content.strip():
            raise ValueError("content no puede estar vacío")
        if chunk.n_words < 0:
            raise ValueError("n_words no puede ser negativo")
        CorpusStore._validate_text_length("section", chunk.section or "", 255)
        CorpusStore._validate_text_length("content", chunk.content, 3000)

    @staticmethod
    def _validate_text_length(name: str, value: str, max_bytes: int) -> None:
        if len(value.encode("utf-8")) > max_bytes:
            raise ValueError(
                f"{name} supera el máximo de {max_bytes} bytes del motor"
            )

    @staticmethod
    def _paper_from_values(values: list[object]) -> Paper:
        return Paper(
            paper_id=int(values[0]),
            title=str(values[1]),
            authors=str(values[2]),
            year=None if values[3] == 0 else int(values[3]),
            language=str(values[4]),
            source_file=str(values[5]),
            n_pages=int(values[6]),
        )

    @staticmethod
    def _chunk_from_values(values: list[object]) -> Chunk:
        return Chunk(
            chunk_id=int(values[0]),
            paper_id=int(values[1]),
            chunk_idx=int(values[2]),
            page_start=None if values[3] == 0 else int(values[3]),
            page_end=None if values[4] == 0 else int(values[4]),
            section=None if not values[5] else str(values[5]),
            content=str(values[6]),
            n_words=int(values[7]),
        )
