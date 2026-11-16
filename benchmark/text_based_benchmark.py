#!/usr/bin/env python3
"""Benchmark experimental de recuperacion de texto sobre letras de canciones.

Compara tres tecnicas con exactamente el mismo corpus y las mismas consultas:

  * TF-IDF + similitud del coseno (indice invertido SPIMI propio)
  * BM25 (indice invertido SPIMI propio)
  * GIN de PostgreSQL (baseline externo)

Para cada tamano de corpus mide:

  * tiempo de construccion;
  * tiempo de consulta (media y p95, con repeticiones);
  * Precision@10 y Recall@10 contra un ground truth definido antes de
    consultar;
  * espacio en disco y memoria pico del proceso Python.

Las consultas cubren tres longitudes: 1 palabra, 3 palabras y 5+ palabras.

Uso:
    python3 benchmark/text_based_benchmark.py
    python3 benchmark/text_based_benchmark.py --sizes 1000 2500 5000
    python3 benchmark/text_based_benchmark.py --data data/lyrics.jsonl

GIN requiere psycopg y un PostgreSQL accesible. El DSN se puede pasar con
--dsn o con la variable POSTGRES_DSN. Si PostgreSQL no esta disponible, el
benchmark sigue con TF-IDF y BM25 y avisa por consola.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import sys
import tempfile
import time
import tracemalloc
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from text_based.bm25 import BM25
from text_based.spimi import SPIMIIndex, iter_documents, normalize_doc_id, tokenize
from text_based.tfidf import TFIDFCosine


DEFAULT_SIZES = [1000, 2500, 5000, 7500, 10000]
DEFAULT_DATA = ROOT / "data" / "lyrics.jsonl"
DEFAULT_OUT = ROOT / "benchmark" / "text_based"
DEFAULT_DSN = os.getenv(
    "POSTGRES_DSN",
    "postgresql://postgres:postgres@localhost:5432/multimodal",
)

# Consultas de distinta longitud. Se eligieron terminos con distinta
# selectividad en las letras (algunos frecuentes, otros raros) para que las
# metricas no queden dominadas por una unica palabra.
QUERIES = {
    "1 palabra": [
        "love",
        "fire",
        "thunder",
        "midnight",
    ],
    "3 palabras": [
        "love heart baby",
        "fire water soul",
        "rain dream winter",
    ],
    "5+ palabras": [
        "love heart baby night fire",
        "love time never without your",
        "heart baby night rain dream",
    ],
}
QUERY_TYPES = list(QUERIES)

TECH_LABELS = {
    "tfidf": "TF-IDF + coseno",
    "bm25": "BM25",
    "gin": "GIN (PostgreSQL)",
}
TECH_COLORS = {
    "tfidf": "#2a78d6",
    "bm25": "#1baf7a",
    "gin": "#e07b39",
}
TECH_MARKERS = {"tfidf": "o", "bm25": "s", "gin": "^"}
REPEATS = 5


# ---------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------

def cargar_documentos(path: Path, cantidad: int) -> list[tuple[int, str]]:
    documentos: list[tuple[int, str]] = []
    for documento in iter_documents(str(path)):
        texto = documento.get("lyrics", documento.get("text", ""))
        if not texto or not str(texto).strip():
            continue
        if "id" not in documento:
            raise SystemExit(f"{path} contiene un documento sin el campo id")
        doc_id = normalize_doc_id(documento["id"])
        documentos.append((doc_id, str(texto).strip()))
        if len(documentos) >= cantidad:
            break

    if len(documentos) < cantidad:
        raise SystemExit(
            f"{path} solo tiene {len(documentos)} documentos con texto; "
            f"se necesitan {cantidad}."
        )
    return documentos


def percentil(valores: list[float], p: float) -> float:
    if not valores:
        return 0.0
    ordenados = sorted(valores)
    indice = min(len(ordenados) - 1, max(0, math.ceil(p * len(ordenados)) - 1))
    return ordenados[indice]


def medir(funcion):
    """Devuelve (resultado, milisegundos) de una llamada."""
    inicio = time.perf_counter()
    resultado = funcion()
    return resultado, (time.perf_counter() - inicio) * 1000.0


def tamano_indice(spimi: SPIMIIndex) -> int:
    return sum(Path(ruta).stat().st_size for ruta in spimi.artifact_paths())


def umbral_relevancia(cantidad_terminos: int) -> int:
    """Cantidad minima de terminos distintos para considerar relevante un doc.

    Con una sola palabra basta con contenerla. Para consultas de 3 palabras
    se exige coincidencia en al menos 2, y para 5+ en al menos 3. Es un
    criterio independiente del algoritmo: se calcula una sola vez y se usa
    igual para TF-IDF, BM25 y GIN.
    """
    if cantidad_terminos <= 1:
        return 1
    if cantidad_terminos <= 3:
        return 2
    return 3


def documentos_relevantes(spimi: SPIMIIndex, consulta: str) -> set[int]:
    terminos = list(dict.fromkeys(tokenize(consulta)))
    if not terminos:
        return set()

    coincidencias: Counter[int] = Counter()
    for termino in terminos:
        for doc_id, _frecuencia in spimi.get_postings(termino):
            coincidencias[doc_id] += 1

    umbral = umbral_relevancia(len(terminos))
    return {
        doc_id for doc_id, cantidad in coincidencias.items()
        if cantidad >= umbral
    }


def precision_recall_at_10(ids: list[int], relevantes: set[int]) -> tuple[float, float]:
    if not relevantes:
        return 0.0, 0.0
    aciertos = len(set(ids[:10]) & relevantes)
    precision = aciertos / 10.0
    recall = aciertos / len(relevantes)
    return precision, recall


# ---------------------------------------------------------------------
# Indices propios
# ---------------------------------------------------------------------

def construir_indices_python(
    documentos: list[tuple[int, str]],
    workdir: Path,
) -> dict:
    salida = workdir / "spimi"

    def construir():
        return SPIMIIndex(output_file=str(salida)).build(
            {"id": doc_id, "lyrics": texto}
            for doc_id, texto in documentos
        )

    # El tiempo se mide sin tracemalloc para no inflar la construccion.
    _, spimi_ms = medir(construir)

    # La memoria pico se mide en una segunda construccion. tracemalloc
    # agrega overhead, por eso no se usa para cronometrar.
    tracemalloc.start(1)
    construir()
    pico_spimi = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    indice = SPIMIIndex(output_file=str(salida)).load()

    tfidf = TFIDFCosine(indice)
    _, tfidf_stats_ms = medir(tfidf.build_stats)

    tracemalloc.start(1)
    TFIDFCosine(indice).build_stats()
    pico_tfidf = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    bm25 = BM25(indice)
    _, bm25_stats_ms = medir(bm25.build_stats)

    tracemalloc.start(1)
    BM25(indice).build_stats()
    pico_bm25 = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    return {
        "indice": indice,
        "tfidf": tfidf,
        "bm25": bm25,
        "metricas": {
            "spimi_ms": spimi_ms,
            "tfidf_stats_ms": tfidf_stats_ms,
            "bm25_stats_ms": bm25_stats_ms,
            "construccion_tfidf_ms": spimi_ms + tfidf_stats_ms,
            "construccion_bm25_ms": spimi_ms + bm25_stats_ms,
            "memoria_spimi_bytes": pico_spimi,
            "memoria_tfidf_bytes": pico_tfidf,
            "memoria_bm25_bytes": pico_bm25,
            "espacio_bytes": tamano_indice(indice),
        },
    }


def medir_consultas_python(
    indices: dict,
    consultas: dict[str, list[str]],
    repeticiones: int,
) -> list[dict]:
    spimi = indices["indice"]
    rankers = {"tfidf": indices["tfidf"], "bm25": indices["bm25"]}
    filas = []

    for tipo, lista in consultas.items():
        for consulta in lista:
            relevantes = documentos_relevantes(spimi, consulta)
            fila = {
                "tipo": tipo,
                "consulta": consulta,
                "relevantes": len(relevantes),
            }

            for tecnica, ranker in rankers.items():
                ranker.search(consulta, 10)  # calentamiento
                tiempos = []
                ids: list[int] = []
                for _ in range(repeticiones):
                    resultado, ms = medir(lambda: ranker.search(consulta, 10))
                    tiempos.append(ms)
                    ids = [doc_id for doc_id, _score in resultado]

                precision, recall = precision_recall_at_10(ids, relevantes)
                fila[tecnica] = {
                    "ms": statistics.fmean(tiempos),
                    "p95_ms": percentil(tiempos, 0.95),
                    "precision_10": precision,
                    "recall_10": recall,
                    "ids": ids,
                }
            filas.append(fila)
    return filas


# ---------------------------------------------------------------------
# GIN de PostgreSQL
# ---------------------------------------------------------------------

def importar_psycopg():
    try:
        import psycopg
        return psycopg
    except ImportError:
        try:
            import psycopg2 as psycopg
            return psycopg
        except ImportError:
            return None


def tsquery_or(consulta: str) -> str:
    terminos = list(dict.fromkeys(tokenize(consulta)))
    if not terminos:
        return ""
    return " | ".join(
        "'" + termino.replace("'", "''") + "'" for termino in terminos
    )


class GinBenchmark:
    def __init__(self, dsn: str, repeticiones: int):
        self.dsn = dsn
        self.repeticiones = repeticiones
        self.psycopg = importar_psycopg()
        self.conexion = None
        self.disponible = False

    def conectar(self) -> bool:
        if self.psycopg is None:
            print("aviso: psycopg no esta instalado; se omite GIN")
            return False
        try:
            self.conexion = self.psycopg.connect(self.dsn, autocommit=True)
            with self.conexion.cursor() as cursor:
                cursor.execute("SELECT 1")
            self.disponible = True
        except Exception as error:  # pragma: no cover - depende del entorno
            print(f"aviso: no se pudo conectar a PostgreSQL ({error}); se omite GIN")
            self.disponible = False
        return self.disponible

    def cerrar(self) -> None:
        if self.conexion is not None:
            self.conexion.close()

    def construir(self, documentos: list[tuple[int, str]]) -> dict:
        with self.conexion.cursor() as cursor:
            cursor.execute("DROP TABLE IF EXISTS text_bench_gin")
            cursor.execute(
                "CREATE TABLE text_bench_gin ("
                "id BIGINT PRIMARY KEY, content TEXT NOT NULL, "
                "search_vector TSVECTOR GENERATED ALWAYS AS "
                "(to_tsvector('simple', content)) STORED)"
            )

            inicio = time.perf_counter()
            cursor.executemany(
                "INSERT INTO text_bench_gin (id, content) VALUES (%s, %s)",
                documentos,
            )
            carga_ms = (time.perf_counter() - inicio) * 1000.0

            inicio = time.perf_counter()
            cursor.execute(
                "CREATE INDEX text_bench_gin_search_idx "
                "ON text_bench_gin USING GIN (search_vector)"
            )
            indice_ms = (time.perf_counter() - inicio) * 1000.0
            cursor.execute("ANALYZE text_bench_gin")

            cursor.execute(
                "SELECT pg_relation_size('text_bench_gin_search_idx')"
            )
            espacio_bytes = cursor.fetchone()[0]
            cursor.execute("SELECT pg_total_relation_size('text_bench_gin')")
            tabla_bytes = cursor.fetchone()[0]

        return {
            "carga_ms": carga_ms,
            "indice_ms": indice_ms,
            "construccion_ms": carga_ms + indice_ms,
            "espacio_bytes": espacio_bytes,
            "tabla_bytes": tabla_bytes,
        }

    def consultar(self, consulta: str, relevantes: set[int]) -> dict:
        tsquery = tsquery_or(consulta)
        sql = (
            "SELECT id FROM text_bench_gin "
            "WHERE search_vector @@ to_tsquery('simple', %s) "
            "ORDER BY ts_rank_cd(search_vector, "
            "to_tsquery('simple', %s)) DESC LIMIT 10"
        )
        with self.conexion.cursor() as cursor:
            cursor.execute(sql, (tsquery, tsquery))
            cursor.fetchall()  # calentamiento
            tiempos = []
            ids: list[int] = []
            for _ in range(self.repeticiones):
                inicio = time.perf_counter()
                cursor.execute(sql, (tsquery, tsquery))
                ids = [int(fila[0]) for fila in cursor.fetchall()]
                tiempos.append((time.perf_counter() - inicio) * 1000.0)

        precision, recall = precision_recall_at_10(ids, relevantes)
        return {
            "ms": statistics.fmean(tiempos),
            "p95_ms": percentil(tiempos, 0.95),
            "precision_10": precision,
            "recall_10": recall,
            "ids": ids,
        }

    def medir_consultas(
        self,
        consultas: dict[str, list[str]],
        relevantes_por_consulta: dict[str, set[int]],
    ) -> list[dict]:
        filas = []
        for tipo, lista in consultas.items():
            for consulta in lista:
                fila = {
                    "tipo": tipo,
                    "consulta": consulta,
                    "relevantes": len(relevantes_por_consulta[consulta]),
                    "gin": self.consultar(
                        consulta, relevantes_por_consulta[consulta]
                    ),
                }
                filas.append(fila)
        return filas


# ---------------------------------------------------------------------
# Medicion por tamano
# ---------------------------------------------------------------------

def medir_tamano(
    documentos: list[tuple[int, str]],
    gin: GinBenchmark | None,
    repeticiones: int,
    workdir: Path,
) -> dict:
    indices = construir_indices_python(documentos, workdir)
    spimi = indices["indice"]
    relevantes = {
        consulta: documentos_relevantes(spimi, consulta)
        for lista in QUERIES.values() for consulta in lista
    }
    filas_python = medir_consultas_python(indices, QUERIES, repeticiones)

    fila = {
        "n": len(documentos),
        "metricas": indices["metricas"],
        "consultas": filas_python,
    }

    if gin is not None and gin.disponible:
        metricas_gin = gin.construir(documentos)
        filas_gin = gin.medir_consultas(QUERIES, relevantes)
        por_consulta = {fila["consulta"]: fila["gin"] for fila in filas_gin}
        for fila_python in filas_python:
            fila_python["gin"] = por_consulta[fila_python["consulta"]]
        fila["metricas_gin"] = metricas_gin

    return fila


# ---------------------------------------------------------------------
# Graficas y tablas
# ---------------------------------------------------------------------

def guardar_grafica(
    ruta: Path,
    xs: list[int],
    series: list[tuple[str, list[float], str, str]],
    ylabel: str,
    titulo: str,
    log_y: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    for etiqueta, ys, color, marker in series:
        ax.plot(
            xs, ys, marker=marker, linewidth=2,
            label=etiqueta, color=color,
        )
    ax.set_xlabel("documentos")
    ax.set_ylabel(ylabel)
    ax.set_title(titulo)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def guardar_panel(
    ruta: Path,
    xs: list[int],
    datos: dict[str, dict[str, list[float]]],
    ylabel: str,
    titulo: str,
) -> None:
    fig, ejes = plt.subplots(1, 3, figsize=(12, 3.8), dpi=150, sharey=True)
    for eje, tipo in zip(ejes, QUERY_TYPES):
        for tecnica in TECH_LABELS:
            ys = datos[tipo].get(tecnica)
            if ys is None:
                continue
            eje.plot(
                xs, ys, marker=TECH_MARKERS[tecnica], linewidth=2,
                label=TECH_LABELS[tecnica], color=TECH_COLORS[tecnica],
            )
        eje.set_title(tipo)
        eje.set_xlabel("documentos")
        eje.grid(True, alpha=0.3)
    ejes[0].set_ylabel(ylabel)
    ejes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def promedios(resultados: list[dict], tecnica: str, campo: str) -> dict[str, list[float]]:
    datos = {}
    for tipo in QUERY_TYPES:
        valores = []
        for fila in resultados:
            del_tipo = [
                consulta for consulta in fila["consultas"]
                if consulta["tipo"] == tipo and tecnica in consulta
            ]
            if del_tipo:
                valores.append(
                    statistics.fmean(c[tecnica][campo] for c in del_tipo)
                )
            else:
                valores.append(float("nan"))
        datos[tipo] = valores
    return datos


def generar_graficas(resultados: list[dict], salida: Path) -> None:
    xs = [fila["n"] for fila in resultados]

    construccion = []
    for tecnica, campo in (
        ("tfidf", "construccion_tfidf_ms"),
        ("bm25", "construccion_bm25_ms"),
    ):
        construccion.append((
            TECH_LABELS[tecnica],
            [fila["metricas"][campo] for fila in resultados],
            TECH_COLORS[tecnica],
            TECH_MARKERS[tecnica],
        ))
    if all("metricas_gin" in fila for fila in resultados):
        construccion.append((
            TECH_LABELS["gin"],
            [fila["metricas_gin"]["construccion_ms"] for fila in resultados],
            TECH_COLORS["gin"],
            TECH_MARKERS["gin"],
        ))
    guardar_grafica(
        salida / "construccion_indice_ms.png",
        xs, construccion, "ms", "Construccion del indice",
        log_y=True,
    )

    for tecnica in ("tfidf", "bm25", "gin"):
        if not all(tecnica in c for fila in resultados for c in fila["consultas"]):
            continue
        datos = promedios(resultados, tecnica, "ms")
        guardar_panel(
            salida / f"consulta_{tecnica}_por_longitud_ms.png",
            xs, {tipo: {tecnica: datos[tipo]} for tipo in QUERY_TYPES},
            "ms por consulta",
            f"Tiempo de consulta: {TECH_LABELS[tecnica]}",
        )

    comparacion = {}
    for tipo in QUERY_TYPES:
        comparacion[tipo] = {}
        for tecnica in TECH_LABELS:
            if not all(tecnica in c for fila in resultados for c in fila["consultas"]):
                continue
            datos = promedios(resultados, tecnica, "ms")
            comparacion[tipo][tecnica] = datos[tipo]
    guardar_panel(
        salida / "comparacion_consultas_ms.png",
        xs, comparacion, "ms por consulta",
        "Tiempo de consulta por longitud",
    )

    for campo, nombre, titulo in (
        ("precision_10", "precision10_por_longitud.png", "Precision@10"),
        ("recall_10", "recall10_por_longitud.png", "Recall@10"),
    ):
        datos = {}
        for tipo in QUERY_TYPES:
            datos[tipo] = {}
            for tecnica in TECH_LABELS:
                if not all(tecnica in c for fila in resultados for c in fila["consultas"]):
                    continue
                promedios_tecnica = promedios(resultados, tecnica, campo)
                datos[tipo][tecnica] = promedios_tecnica[tipo]
        guardar_panel(
            salida / nombre, xs, datos, titulo, f"{titulo} por longitud",
        )

    espacio = []
    for tecnica, campo in (
        ("tfidf", "espacio_bytes"),
        ("bm25", "espacio_bytes"),
    ):
        espacio.append((
            TECH_LABELS[tecnica],
            [fila["metricas"][campo] / 1024 for fila in resultados],
            TECH_COLORS[tecnica],
            TECH_MARKERS[tecnica],
        ))
    if all("metricas_gin" in fila for fila in resultados):
        espacio.append((
            "GIN (indice + tabla)",
            [
                (
                    fila["metricas_gin"]["espacio_bytes"]
                    + fila["metricas_gin"]["tabla_bytes"]
                ) / 1024
                for fila in resultados
            ],
            TECH_COLORS["gin"],
            TECH_MARKERS["gin"],
        ))
    guardar_grafica(
        salida / "espacio_indice_kb.png",
        xs, espacio, "KB", "Espacio en disco (GIN incluye tabla y tsvector)",
        log_y=True,
    )


def generar_tabla(resultados: list[dict], salida: Path) -> None:
    lineas = []
    lineas.append("# Resultados del benchmark textual\n")
    lineas.append(
        "Ground truth: un documento es relevante si contiene al menos "
        "1 termino de una consulta de 1 palabra, 2 terminos de una de 3 "
        "palabras, o 3 terminos de una de 5+ palabras. El criterio se "
        "calcula una sola vez y se aplica igual a las tres tecnicas.\n"
    )

    lineas.append("## Construccion, espacio y memoria\n")
    lineas.append(
        "| N | TF-IDF + coseno | BM25 | GIN | Espacio TF-IDF/BM25 | "
        "Indice GIN | Tabla GIN | Total GIN | Memoria SPIMI | "
        "Memoria TF-IDF | Memoria BM25 |"
    )
    lineas.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for fila in resultados:
        m = fila["metricas"]
        gin = fila.get("metricas_gin", {})
        lineas.append(
            f"| {fila['n']} "
            f"| {m['construccion_tfidf_ms']:.1f} ms "
            f"| {m['construccion_bm25_ms']:.1f} ms "
            f"| {gin.get('construccion_ms', float('nan')):.1f} ms "
            f"| {m['espacio_bytes'] / 1024:.1f} KB "
            f"| {gin.get('espacio_bytes', float('nan')) / 1024:.1f} KB "
            f"| {gin.get('tabla_bytes', float('nan')) / 1024:.1f} KB "
            f"| {(gin.get('espacio_bytes', 0) + gin.get('tabla_bytes', 0)) / 1024:.1f} KB "
            f"| {m['memoria_spimi_bytes'] / 1024:.1f} KB "
            f"| {m['memoria_tfidf_bytes'] / 1024:.1f} KB "
            f"| {m['memoria_bm25_bytes'] / 1024:.1f} KB |"
        )

    lineas.append("\n## Tiempo de consulta, Precision@10 y Recall@10\n")
    lineas.append(
        "| N | Consulta | Tecnica | ms | p95 ms | P@10 | R@10 |"
    )
    lineas.append("|---:|---|---|---:|---:|---:|---:|")
    for fila in resultados:
        for consulta in fila["consultas"]:
            for tecnica in TECH_LABELS:
                resultado = consulta.get(tecnica)
                if resultado is None:
                    continue
                lineas.append(
                    f"| {fila['n']} | {consulta['consulta']} "
                    f"| {TECH_LABELS[tecnica]} "
                    f"| {resultado['ms']:.3f} "
                    f"| {resultado['p95_ms']:.3f} "
                    f"| {resultado['precision_10']:.3f} "
                    f"| {resultado['recall_10']:.3f} |"
                )

    lineas.append("\n## Ventajas y desventajas\n")
    lineas.append(
        "| Tecnica | Ventajas | Desventajas | Cuando conviene |\n"
        "|---|---|---|---|\n"
        "| TF-IDF + coseno | Simple e interpretable; el idf premia terminos "
        "raros; coseno compensa diferencias de longitud; no necesita "
        "parametros | No satura la frecuencia como BM25; un termino presente "
        "en todos los documentos tiene idf 0; sensible a la tokenizacion | "
        "Documentos cortos, baseline explicable y comparaciones didacticas |\n"
        "| BM25 | Satura la frecuencia y normaliza por longitud; suele rankear "
        "mejor texto heterogeneo; idf siempre positivo | Requiere calibrar "
        "`k1` y `b`; necesita la longitud media del corpus | Ranking textual "
        "principal con letras de distinta extension |\n"
        "| GIN (PostgreSQL) | Indice maduro y muy optimizado; sintaxis SQL; "
        "integrado con transacciones, concurrencia y mantenimiento | Depende "
        "de PostgreSQL; el ranking con `ts_rank_cd` no equivale a BM25; "
        "mayor costo de infraestructura | Produccion y baseline externo, no "
        "como sustituto del indice propio |\n"
    )
    (salida / "tabla.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=DEFAULT_SIZES,
        help="tamanos de corpus (por defecto 1000 2500 5000 7500 10000)",
    )
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--sin-gin", action="store_true", help="no ejecutar el baseline GIN",
    )
    parser.add_argument(
        "--solo-graficas", action="store_true",
        help="rehacer tabla y graficas desde resultados.json",
    )
    args = parser.parse_args()

    salida = args.output_dir
    if args.solo_graficas:
        documento = json.loads(
            (salida / "resultados.json").read_text(encoding="utf-8")
        )
        generar_graficas(documento["resultados"], salida)
        generar_tabla(documento["resultados"], salida)
        print(f"Tabla y graficas regeneradas en {salida}")
        return

    sizes = sorted(set(args.sizes))
    if not sizes or min(sizes) < 1:
        parser.error("--sizes debe contener enteros positivos")
    if args.repeats < 1:
        parser.error("--repeats debe ser mayor que cero")

    documentos = cargar_documentos(args.data, max(sizes))
    salida.mkdir(parents=True, exist_ok=True)

    gin = None
    if not args.sin_gin:
        gin = GinBenchmark(args.dsn, args.repeats)
        if not gin.conectar():
            gin = None

    resultados = []
    workdir = Path(tempfile.mkdtemp(prefix="text_bench_"))
    try:
        for n in sizes:
            print(f"midiendo N={n}...", end=" ", flush=True)
            subdir = workdir / f"n_{n}"
            subdir.mkdir()
            fila = medir_tamano(
                documentos[:n], gin, args.repeats, subdir
            )
            resultados.append(fila)
            m = fila["metricas"]
            texto = (
                f"tfidf={m['construccion_tfidf_ms']:.1f}ms "
                f"bm25={m['construccion_bm25_ms']:.1f}ms"
            )
            if "metricas_gin" in fila:
                texto += f" gin={fila['metricas_gin']['construccion_ms']:.1f}ms"
            print(texto)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        if gin is not None:
            gin.cerrar()

    documento = {
        "generado": datetime.now(timezone.utc).isoformat(),
        "corpus": str(args.data),
        "tamanos": sizes,
        "repeticiones": args.repeats,
        "consultas": QUERIES,
        "ground_truth": (
            "relevante si contiene al menos 1 termino (1 palabra), "
            "2 terminos (3 palabras) o 3 terminos (5+ palabras)"
        ),
        "resultados": resultados,
    }
    (salida / "resultados.json").write_text(
        json.dumps(documento, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    generar_graficas(resultados, salida)
    generar_tabla(resultados, salida)

    print(f"\nResultados en {salida / 'resultados.json'}")
    print(f"Tabla en {salida / 'tabla.md'}")
    print(f"Graficas en {salida}")


if __name__ == "__main__":
    main()
