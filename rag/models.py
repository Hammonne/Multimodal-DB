from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Paper:
    title: str
    authors: str
    year: int | None
    language: str
    source_file: str
    n_pages: int
    paper_id: int | None = None


@dataclass(slots=True)
class Chunk:
    paper_id: int
    chunk_idx: int
    page_start: int | None
    page_end: int | None
    section: str | None
    content: str
    n_words: int
    chunk_id: int | None = None
