"""Generate the local JSONL corpus used by the text-search web demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TEMPLATES = (
    "base de datos vectorial para búsqueda eficiente de documentos de texto",
    "machine learning y procesamiento de lenguaje natural para análisis de información",
    "índice invertido con SPIMI, términos, frecuencias y ranking de documentos",
    "comparación de relevancia entre BM25 y TF-IDF con similitud del coseno",
    "sistema de almacenamiento con consultas, índices y recuperación de información",
)


def generate(output: Path, documents: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for document_id in range(1, documents + 1):
            template = TEMPLATES[(document_id - 1) % len(TEMPLATES)]
            record = {
                "id": document_id,
                "title": f"Documento textual {document_id}",
                "lyrics": (
                    f"{template}. Documento {document_id} contiene datos de prueba "
                    "para medir tiempo de consulta, construcción del índice y ranking."
                ),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/lyrics.jsonl"),
        help="archivo JSONL de salida (por defecto: data/lyrics.jsonl)",
    )
    parser.add_argument(
        "--documents",
        type=int,
        default=100_000,
        help="cantidad de documentos a generar (por defecto: 100000)",
    )
    args = parser.parse_args()
    if args.documents < 1:
        parser.error("--documents debe ser mayor que cero")
    generate(args.output, args.documents)
    print(f"Corpus generado: {args.output} ({args.documents} documentos)")


if __name__ == "__main__":
    main()
