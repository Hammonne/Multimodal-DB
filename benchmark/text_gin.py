"""Benchmark reproducible de PostgreSQL GIN contra el corpus textual.

Uso:
    python -m benchmark.text_gin --source data/lyrics.jsonl --sizes 1000 10000

Requiere PostgreSQL accesible mediante --dsn y psycopg. El archivo de salida
usa el mismo esquema base de los benchmarks web: filas con tiempos y tamaño.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from text_based.spimi import iter_documents


QUERIES = {
    "1 palabra": "vectorial",
    "3 palabras": "base de datos",
    "5+ palabras": "búsqueda eficiente en documentos de texto y datos vectoriales",
}


def load_documents(source: str, limit: int) -> list[tuple[int, str]]:
    documents = []
    for document in iter_documents(source):
        if len(documents) >= limit:
            break
        if "id" not in document:
            raise ValueError("el corpus contiene un documento sin id")
        text = str(document.get("lyrics", document.get("text", "")))
        documents.append((int(document["id"]), text))
    if not documents:
        raise ValueError(f"no hay documentos en {source}")
    return documents


def run_size(connection: Any, source: str, size: int) -> list[dict[str, Any]]:
    documents = load_documents(source, size)
    with connection.cursor() as cursor:
        cursor.execute("DROP TABLE IF EXISTS gin_documents")
        cursor.execute(
            "CREATE TABLE gin_documents (id BIGINT PRIMARY KEY, content TEXT NOT NULL, "
            "search_vector TSVECTOR GENERATED ALWAYS AS "
            "(to_tsvector('simple', content)) STORED)"
        )
        started = time.perf_counter()
        cursor.executemany(
            "INSERT INTO gin_documents (id, content) VALUES (%s, %s)",
            documents,
        )
        cursor.execute(
            "CREATE INDEX gin_documents_search_idx ON gin_documents USING GIN (search_vector)"
        )
        connection.commit()
        build_seconds = time.perf_counter() - started

        rows = []
        for query_type, query in QUERIES.items():
            started = time.perf_counter()
            cursor.execute(
                "SELECT id FROM gin_documents WHERE search_vector @@ "
                "plainto_tsquery('simple', %s) LIMIT 10",
                (query,),
            )
            result = cursor.fetchall()
            query_ms = (time.perf_counter() - started) * 1000
            cursor.execute(
                "SELECT pg_total_relation_size('gin_documents_search_idx')"
            )
            index_size = cursor.fetchone()[0]
            rows.append({
                "algoritmo": "gin_postgresql",
                "n": size,
                "consulta_tipo": query_type,
                "consulta": query,
                "construir_s": build_seconds,
                "consulta_ms": query_ms,
                "resultados": len(result),
                "espacio_bytes": index_size,
            })
        return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/lyrics.jsonl")
    parser.add_argument("--sizes", nargs="+", type=int, default=[1_000, 10_000, 100_000])
    parser.add_argument("--dsn", default=os.getenv("POSTGRES_DSN", "postgresql://postgres:postgres@localhost:5432/multimodal"))
    parser.add_argument("--output", default="benchmark/text/results_gin.json")
    args = parser.parse_args()

    try:
        import psycopg
    except ImportError as error:
        raise SystemExit("Instale psycopg[binary] para ejecutar el baseline GIN") from error

    with psycopg.connect(args.dsn) as connection:
        rows = []
        for size in args.sizes:
            rows.extend(run_size(connection, args.source, size))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "algoritmo": "gin_postgresql",
        "fuente": args.source,
        "tamanos": args.sizes,
        "filas": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Resultados GIN guardados en {output}")


if __name__ == "__main__":
    main()
