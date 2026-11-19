from dataclasses import dataclass
from typing import List
from rag.ingestion.cleaner import CleanedPage


@dataclass
class DocumentChunk:
    chunk_id: str
    paper_id: str
    chunk_index: int
    text: str           # Texto limpio para embeddings / BM25 / SPIMI
    citation_text: str  # Texto original conservado para mostrar citas exactas
    page_start: int
    page_end: int
    word_count: int


def chunk_pages(
    paper_id: str,
    pages: List[CleanedPage],
    min_words: int = 200,
    target_words: int = 350,
    max_words: int = 500,
    overlap_words: int = 50,
) -> List[DocumentChunk]:
    """
    Divide las páginas limpias en chunks de 200-500 palabras con overlap,
    rastreando `page_start` y `page_end` de cada chunk.
    """
    if not pages:
        return []

    # Tokenizamos por palabras asociando cada palabra a su página y texto original
    word_stream: List[tuple[str, str, int]] = []
    for p in pages:
        clean_words = p.cleaned_text.split()
        orig_words = p.original_text.split()
        # Alinear longitud por seguridad si difieren mínimamente tras dehyphenate
        for idx, cw in enumerate(clean_words):
            ow = orig_words[idx] if idx < len(orig_words) else cw
            word_stream.append((cw, ow, p.page_number))

    total_words = len(word_stream)
    if total_words == 0:
        return []

    chunks: List[DocumentChunk] = []
    start = 0
    chunk_idx = 0

    while start < total_words:
        end = min(start + target_words, total_words)

        # Si el remanente final es menor a min_words, lo fusionamos con el chunk actual (hasta max_words)
        remaining = total_words - end
        if 0 < remaining < min_words and (total_words - start) <= max_words:
            end = total_words

        window = word_stream[start:end]
        w_count = len(window)

        # Descartar fragmentos residuales diminutos (< 50 palabras) salvo que sea el único chunk del paper
        if w_count < 50 and chunks:
            break

        clean_text = " ".join(item[0] for item in window)
        citation_text = " ".join(item[1] for item in window)
        page_start = window[0][2]
        page_end = window[-1][2]

        chunks.append(
            DocumentChunk(
                chunk_id=f"{paper_id}::chunk_{chunk_idx:04d}",
                paper_id=paper_id,
                chunk_index=chunk_idx,
                text=clean_text,
                citation_text=citation_text,
                page_start=page_start,
                page_end=page_end,
                word_count=w_count,
            )
        )

        chunk_idx += 1
        if end >= total_words:
            break

        step = max(1, (end - start) - overlap_words)
        start += step

    return chunks