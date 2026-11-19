"""
Pruebas de integracion entre el parser SQL y los indices de texto.

    PYTHONPATH=. python3 tests/text_based/test_sql_match.py

Cubren la gramatica de MATCH / SCORE / USING, la ejecucion con SPIMI,
TF-IDF + coseno y BM25, el orden por relevancia, LIMIT, la proyeccion
SELECT * con SCORE() y la creacion de indices de texto por SQL.
"""

import math
import os
import shutil
import tempfile

from parser.scanner import Scanner
from parser import Parser
from parser.executor import ExecuteVisitor, TEXT_INDEX_ROOT
from storage.storage_manager import StorageManager
from text_based.spimi import SPIMIIndex
from text_based.tfidf import TFIDFCosine
from text_based.bm25 import BM25


ARCHIVOS = [
    "sys_tables.dat", "sys_columns.dat", "sys_indexes.dat",
    "documentos.dat", "articulos.dat",
    "idx_documentos_contenido_spimi.bin",
    "idx_documentos_contenido_spimi.json",
    "idx_documentos_contenido_spimi.map.json",
    "idx_documentos_contenido_tfidf.bin",
    "idx_documentos_contenido_tfidf.json",
    "idx_documentos_contenido_tfidf.map.json",
    "idx_documentos_contenido_bm25.bin",
    "idx_documentos_contenido_bm25.json",
    "idx_documentos_contenido_bm25.map.json",
]


def limpiar():
    for nombre in ARCHIVOS:
        if os.path.exists(nombre):
            os.remove(nombre)
    for carpeta in (".", TEXT_INDEX_ROOT):
        if not os.path.isdir(carpeta):
            continue
        for nombre in os.listdir(carpeta):
            if nombre.startswith("idx_") and (
                ".bin" in nombre or ".json" in nombre
            ):
                os.remove(os.path.join(carpeta, nombre))
            if nombre.startswith("auto_") and (
                ".bin" in nombre or ".json" in nombre
            ):
                os.remove(os.path.join(carpeta, nombre))


def correr(sm, sql):
    return ExecuteVisitor(sm).ejecutar(Parser(Scanner(sql)).parse_p())


def _documentos():
    return [
        (1, "base de datos relacional"),
        (2, "base de datos vectorial"),
        (3, "machine learning y datos"),
        (4, "bases de datos vectoriales y busqueda"),
        (5, "cocina"),
    ]


def _sembrar_documentos(sm):
    correr(
        sm,
        "CREATE TABLE documentos (id INT PRIMARY KEY, "
        "contenido VARCHAR(200)) USING HEAP;",
    )
    for doc_id, texto in _documentos():
        correr(
            sm,
            f"INSERT INTO documentos VALUES ({doc_id}, '{texto}');",
        )


def _indice_temporal(documentos, algoritmo):
    workdir = tempfile.mkdtemp(prefix="sql_match_")
    base = os.path.join(workdir, "indice")
    index = SPIMIIndex(output_file=base)
    index.build([
        {"id": doc_id, "text": texto}
        for doc_id, texto in documentos
    ])
    buscador = TFIDFCosine(index) if algoritmo == "tfidf" else BM25(index)
    return workdir, buscador


# ---------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------

def test_parser_acepta_las_consultas_del_enunciado():
    sql_1 = (
        "SELECT * FROM documentos WHERE "
        "MATCH(contenido, 'base de datos vectorial') "
        "USING TF_IDF LIMIT 10;"
    )
    stmt = Parser(Scanner(sql_1)).parse_p().slist[0]
    assert stmt.select_all
    assert type(stmt.condicion).__name__ == "MatchExpr"
    assert stmt.limite == 10 and stmt.haylimite

    sql_2 = (
        "SELECT *, SCORE() as relevancia FROM articulos WHERE "
        "MATCH(texto, 'machine learning') USING BM25 "
        "ORDER BY relevancia DESC;"
    )
    stmt = Parser(Scanner(sql_2)).parse_p().slist[0]
    assert stmt.select_all
    assert len(stmt.proyeccion) == 1
    item = stmt.proyeccion[0]
    assert type(item.columna).__name__ == "SimilarityScoreExpr"
    assert item.alias == "relevancia"
    assert type(stmt.condicion).__name__ == "MatchExpr"
    assert stmt.order_by.columna == "relevancia"

    # SCORE() tambien se puede usar directamente en ORDER BY.
    stmt = Parser(
        Scanner(
            "SELECT id FROM documentos WHERE "
            "MATCH(contenido, 'datos') USING SPIMI ORDER BY SCORE() DESC;"
        )
    ).parse_p().slist[0]
    assert type(stmt.order_by).__name__ == "SimilarityScoreExpr"
    print("test_parser_acepta_las_consultas_del_enunciado: OK")


# ---------------------------------------------------------------------
# Ejecucion
# ---------------------------------------------------------------------

def test_match_tfidf_ordena_y_limita():
    limpiar()
    sm = StorageManager()
    _sembrar_documentos(sm)

    workdir, esperado = _indice_temporal(_documentos(), "tfidf")
    try:
        resultados = esperado.search("base de datos vectorial", 10)
    finally:
        shutil.rmtree(workdir)

    esperado_por_id = {
        _documentos()[doc_id - 1][0]: score
        for doc_id, score in resultados
    }

    res = correr(
        sm,
        "SELECT id, SCORE() AS relevancia FROM documentos WHERE "
        "MATCH(contenido, 'base de datos vectorial') USING TF_IDF "
        "ORDER BY relevancia DESC LIMIT 3;",
    )[0]
    assert res.columnas == ["id", "relevancia"]
    assert len(res.filas) == 3

    ids = [fila[0] for fila in res.filas]
    scores = [fila[1] for fila in res.filas]
    assert ids == list(esperado_por_id)[:3], (ids, esperado_por_id)
    for doc_id, score in zip(ids, scores):
        assert math.isclose(score, esperado_por_id[doc_id], rel_tol=1e-9)

    # Sin ORDER BY, MATCH ya devuelve los documentos por relevancia.
    res = correr(
        sm,
        "SELECT id FROM documentos WHERE "
        "MATCH(contenido, 'base de datos vectorial') USING TF_IDF LIMIT 10;",
    )[0]
    assert [fila[0] for fila in res.filas] == ids + [
        doc_id for doc_id in esperado_por_id if doc_id not in ids
    ]

    sm.close()
    limpiar()
    print("test_match_tfidf_ordena_y_limita: OK")


def test_match_bm25_select_estrella_y_score():
    limpiar()
    sm = StorageManager()
    correr(
        sm,
        "CREATE TABLE articulos (id INT PRIMARY KEY, "
        "texto VARCHAR(200)) USING HEAP;",
    )
    articulos = [
        (1, "machine learning"),
        (2, "machine learning machine learning"),
        (3, "datos y estadistica"),
        (4, "machine learning con datos"),
    ]
    for doc_id, texto in articulos:
        correr(sm, f"INSERT INTO articulos VALUES ({doc_id}, '{texto}');")

    res = correr(
        sm,
        "SELECT *, SCORE() as relevancia FROM articulos WHERE "
        "MATCH(texto, 'machine learning') USING BM25 "
        "ORDER BY relevancia DESC;",
    )[0]
    assert res.columnas == ["id", "texto", "relevancia"]
    assert {fila[0] for fila in res.filas} == {1, 2, 4}

    scores = [fila[-1] for fila in res.filas]
    assert scores == sorted(scores, reverse=True)

    workdir, esperado = _indice_temporal(articulos, "bm25")
    try:
        resultados = dict(esperado.search("machine learning", 10))
    finally:
        shutil.rmtree(workdir)
    for doc_id, _texto, score in res.filas:
        assert math.isclose(score, resultados[doc_id], rel_tol=1e-9)

    sm.close()
    limpiar()
    print("test_match_bm25_select_estrella_y_score: OK")


def test_match_spimi_devuelve_solo_documentos_con_terminos():
    limpiar()
    sm = StorageManager()
    _sembrar_documentos(sm)

    res = correr(
        sm,
        "SELECT id FROM documentos WHERE "
        "MATCH(contenido, 'vectorial') USING SPIMI;",
    )[0]
    assert sorted(fila[0] for fila in res.filas) == [2]

    res = correr(
        sm,
        "SELECT id FROM documentos WHERE "
        "MATCH(contenido, 'inexistente') USING SPIMI;",
    )[0]
    assert res.filas == []

    sm.close()
    limpiar()
    print("test_match_spimi_devuelve_solo_documentos_con_terminos: OK")


def test_match_se_combina_con_otras_condiciones():
    limpiar()
    sm = StorageManager()
    _sembrar_documentos(sm)

    res = correr(
        sm,
        "SELECT id FROM documentos WHERE "
        "MATCH(contenido, 'datos') USING TF_IDF AND id > 2;",
    )[0]
    assert sorted(fila[0] for fila in res.filas) == [3, 4]

    sm.close()
    limpiar()
    print("test_match_se_combina_con_otras_condiciones: OK")


def test_create_index_de_texto_por_sql():
    limpiar()
    sm = StorageManager()
    _sembrar_documentos(sm)

    for tipo, sufijo in (
        ("SPIMI", "spimi"),
        ("TF_IDF", "tfidf"),
        ("BM25", "bm25"),
    ):
        res = correr(
            sm, f"CREATE INDEX ON documentos (contenido) USING {tipo};"
        )[0]
        assert tipo in res.mensaje
        assert f"documentos(contenido)" in res.mensaje
        assert os.path.exists(
            os.path.join(
                TEXT_INDEX_ROOT,
                f"idx_documentos_contenido_{sufijo}.json",
            )
        )

    indices = sm.catalog.get_table_indexes("documentos")
    tipos = {meta["index_type"] for meta in indices}
    assert tipos == {"spimi", "tfidf", "bm25"}

    # La consulta funciona igual aunque haya varios indices de texto.
    res = correr(
        sm,
        "SELECT id FROM documentos WHERE "
        "MATCH(contenido, 'base de datos vectorial') USING BM25;",
    )[0]
    assert res.filas

    sm.close()
    limpiar()
    print("test_create_index_de_texto_por_sql: OK")


def test_explain_muestra_text_index_scan():
    limpiar()
    sm = StorageManager()
    _sembrar_documentos(sm)

    res = correr(
        sm,
        "EXPLAIN SELECT * FROM documentos WHERE "
        "MATCH(contenido, 'datos vectoriales') USING TF_IDF LIMIT 2;",
    )[0]
    assert "Text Index Scan using TF_IDF" in res.explain
    assert "MATCH(contenido, 'datos vectoriales')" in res.explain

    sm.close()
    limpiar()
    print("test_explain_muestra_text_index_scan: OK")


def test_el_indice_de_texto_se_actualiza_al_insertar():
    limpiar()
    sm = StorageManager()
    _sembrar_documentos(sm)

    res = correr(
        sm,
        "SELECT id FROM documentos WHERE "
        "MATCH(contenido, 'kubernetes') USING TF_IDF;",
    )[0]
    assert res.filas == []

    correr(
        sm,
        "INSERT INTO documentos VALUES (6, 'kubernetes y contenedores');",
    )

    res = correr(
        sm,
        "SELECT id FROM documentos WHERE "
        "MATCH(contenido, 'kubernetes') USING TF_IDF;",
    )[0]
    assert [fila[0] for fila in res.filas] == [6]

    sm.close()
    limpiar()
    print("test_el_indice_de_texto_se_actualiza_al_insertar: OK")


tests = [
    test_parser_acepta_las_consultas_del_enunciado,
    test_match_tfidf_ordena_y_limita,
    test_match_bm25_select_estrella_y_score,
    test_match_spimi_devuelve_solo_documentos_con_terminos,
    test_match_se_combina_con_otras_condiciones,
    test_create_index_de_texto_por_sql,
    test_explain_muestra_text_index_scan,
    test_el_indice_de_texto_se_actualiza_al_insertar,
]


if __name__ == "__main__":
    for test in tests:
        print(f"Corriendo {test.__name__}...", end=" ")
        test()
    print(f"\n{len(tests)} tests pasaron.")
