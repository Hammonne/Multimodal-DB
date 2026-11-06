
"""
Pruebas de BM25.

    PYTHONPATH=. python3 tests/text_based/test_bm25.py

Cubren construccion de estadisticas, frecuencias documentales,
longitud media, puntuaciones BM25, ranking, consultas vacias
y casos limite.
"""

import math
import os
import shutil
import tempfile

from text_based.spimi import SPIMIIndex
from text_based.bm25 import BM25


def create_bm25(workdir, documents, k1=1.2, b=0.75):
    output = os.path.join(workdir, "index")
    index = SPIMIIndex(output_file=output)
    index.build(documents)
    return BM25(index, k1=k1, b=b)


def test_construye_estadisticas():
    workdir = tempfile.mkdtemp(prefix="bm25_test_")
    try:
        bm25 = create_bm25(
            workdir,
            [
                {"id": 1, "lyrics": "gato perro gato"},
                {"id": 2, "lyrics": "gato raton"},
                {"id": 3, "lyrics": "perro raton"},
            ],
        )

        assert not bm25.ready

        bm25.build_stats()

        assert bm25.ready
        assert bm25.avg_doc_length > 0

        assert set(bm25.doc_frequencies.keys()) == {
            "gato", "perro", "raton"
        }

    finally:
        shutil.rmtree(workdir)

    print("OK: construye estadisticas del corpus")


def test_calcula_longitud_media():
    workdir = tempfile.mkdtemp(prefix="bm25_test_")
    try:
        bm25 = create_bm25(
            workdir,
            [
                {"id": 1, "lyrics": "gato perro gato"},
                {"id": 2, "lyrics": "gato raton"},
                {"id": 3, "lyrics": "perro raton"},
            ],
        )

        bm25.build_stats()

        expected = (
            bm25.index.doc_lengths[1]
            + bm25.index.doc_lengths[2]
            + bm25.index.doc_lengths[3]
        ) / 3

        assert math.isclose(
            bm25.avg_doc_length,
            expected,
            rel_tol=1e-9,
        )

    finally:
        shutil.rmtree(workdir)

    print("OK: calcula correctamente la longitud media")


def test_calcula_frecuencias_documentales():
    workdir = tempfile.mkdtemp(prefix="bm25_test_")
    try:
        bm25 = create_bm25(
            workdir,
            [
                {"id": 1, "lyrics": "gato perro gato"},
                {"id": 2, "lyrics": "gato raton"},
                {"id": 3, "lyrics": "perro raton"},
            ],
        )

        bm25.build_stats()

        # gato aparece en los documentos 1 y 2.
        assert bm25.doc_frequencies["gato"] == 2

        # perro aparece en los documentos 1 y 3.
        assert bm25.doc_frequencies["perro"] == 2

        # raton aparece en los documentos 2 y 3.
        assert bm25.doc_frequencies["raton"] == 2

    finally:
        shutil.rmtree(workdir)

    print("OK: calcula correctamente las frecuencias documentales")


def test_busca_termino_unico():
    workdir = tempfile.mkdtemp(prefix="bm25_test_")
    try:
        bm25 = create_bm25(
            workdir,
            [
                {"id": 1, "lyrics": "gato gato perro"},
                {"id": 2, "lyrics": "gato raton"},
                {"id": 3, "lyrics": "perro raton"},
            ],
        )

        results = bm25.search("gato", 10)

        assert len(results) == 2
        assert {doc_id for doc_id, score in results} == {1, 2}

        for doc_id, score in results:
            assert math.isfinite(score)

    finally:
        shutil.rmtree(workdir)

    print("OK: busca documentos que contienen un termino")


def test_calcula_puntuacion_bm25():
    workdir = tempfile.mkdtemp(prefix="bm25_test_")
    try:
        bm25 = create_bm25(
            workdir,
            [
                {"id": 1, "lyrics": "gato gato perro"},
                {"id": 2, "lyrics": "gato raton"},
                {"id": 3, "lyrics": "perro raton"},
                {"id": 4, "lyrics": "pajaro pez"},
            ],
        )

        results = bm25.search("gato", 10)
        scores = dict(results)

        # gato aparece en 2 de los 4 documentos.
        N = 4
        df = 2

        expected_idf = math.log(
            (N - df + 0.5) / (df + 0.5)
        )

        doc_id = 1
        tf = 2
        doc_length = bm25.index.doc_lengths[doc_id]

        thirdterm = (
            1 - bm25.b
            + bm25.b * doc_length / bm25.avg_doc_length
        )

        expected_tf = tf / (
            tf + bm25.k1 + thirdterm
        )

        expected_score = expected_tf * expected_idf

        assert math.isclose(
            scores[1],
            expected_score,
            rel_tol=1e-9,
        )

    finally:
        shutil.rmtree(workdir)

    print("OK: calcula correctamente las puntuaciones BM25")


def test_ranking():
    workdir = tempfile.mkdtemp(prefix="bm25_test_")
    try:
        bm25 = create_bm25(
            workdir,
            [
                {"id": 1, "lyrics": "gato gato perro"},
                {"id": 2, "lyrics": "gato raton"},
                {"id": 3, "lyrics": "perro raton"},
                {"id": 4, "lyrics": "pajaro pez"},
            ],
        )

        results = bm25.search("gato", 10)
        scores = [score for doc_id, score in results]

        assert scores == sorted(scores, reverse=True)

        # El documento 1 tiene dos apariciones de gato,
        # mientras que el documento 2 solamente tiene una.
        assert results[0][0] == 1

    finally:
        shutil.rmtree(workdir)

    print("OK: ordena los resultados por puntuacion descendente")


def test_top_k():
    workdir = tempfile.mkdtemp(prefix="bm25_test_")
    try:
        bm25 = create_bm25(
            workdir,
            [
                {"id": 1, "lyrics": "gato perro"},
                {"id": 2, "lyrics": "gato raton"},
                {"id": 3, "lyrics": "gato pajaro"},
            ],
        )

        assert len(bm25.search("gato", 2)) == 2
        assert len(bm25.search("gato", 1)) == 1

        assert bm25.search("gato", 0) == []
        assert bm25.search("gato", -1) == []

    finally:
        shutil.rmtree(workdir)

    print("OK: respeta top_k y rechaza valores no positivos")


def test_consulta_con_terminos_repetidos():
    workdir = tempfile.mkdtemp(prefix="bm25_test_")
    try:
        bm25 = create_bm25(
            workdir,
            [
                {"id": 1, "lyrics": "gato gato perro"},
                {"id": 2, "lyrics": "gato raton"},
                {"id": 3, "lyrics": "perro raton"},
                {"id": 4, "lyrics": "pajaro pez"},
            ],
        )

        single = bm25.search("gato", 10)
        repeated = bm25.search("gato gato", 10)

        assert len(single) == len(repeated)

        for (id1, score1), (id2, score2) in zip(
            single, repeated
        ):
            assert id1 == id2
            assert math.isclose(
                score1,
                score2,
                rel_tol=1e-9,
            )

    finally:
        shutil.rmtree(workdir)

    print("OK: maneja consultas con terminos repetidos")


def test_consulta_terminos_desconocidos():
    workdir = tempfile.mkdtemp(prefix="bm25_test_")
    try:
        bm25 = create_bm25(
            workdir,
            [
                {"id": 1, "lyrics": "gato perro"},
                {"id": 2, "lyrics": "raton"},
            ],
        )

        assert bm25.search("inexistente", 5) == []
        assert bm25.search("xyz abc", 5) == []

    finally:
        shutil.rmtree(workdir)

    print("OK: devuelve lista vacia para terminos desconocidos")


def test_consulta_mixta():
    workdir = tempfile.mkdtemp(prefix="bm25_test_")
    try:
        bm25 = create_bm25(
            workdir,
            [
                {"id": 1, "lyrics": "gato perro"},
                {"id": 2, "lyrics": "gato raton"},
                {"id": 3, "lyrics": "perro raton"},
                {"id": 4, "lyrics": "pajaro pez"},
            ],
        )

        results = bm25.search("gato inexistente", 5)
        expected = bm25.search("gato", 5)

        assert len(results) == len(expected)

        for (id1, score1), (id2, score2) in zip(
            results, expected
        ):
            assert id1 == id2
            assert math.isclose(
                score1,
                score2,
                rel_tol=1e-9,
            )

    finally:
        shutil.rmtree(workdir)

    print("OK: ignora terminos desconocidos en consultas mixtas")


def test_consulta_vacia():
    workdir = tempfile.mkdtemp(prefix="bm25_test_")
    try:
        bm25 = create_bm25(
            workdir,
            [{"id": 1, "lyrics": "gato perro"}],
        )

        assert bm25.search("", 5) == []
        assert bm25.search("   ", 5) == []

    finally:
        shutil.rmtree(workdir)

    print("OK: maneja consultas vacias")


def test_scores_validos():
    workdir = tempfile.mkdtemp(prefix="bm25_test_")
    try:
        bm25 = create_bm25(
            workdir,
            [
                {"id": 1, "lyrics": "gato perro gato"},
                {"id": 2, "lyrics": "gato raton"},
                {"id": 3, "lyrics": "perro raton"},
                {"id": 4, "lyrics": "pajaro pez"},
            ],
        )

        bm25.build_stats()

        assert math.isfinite(bm25.avg_doc_length)
        assert bm25.avg_doc_length > 0

        for df in bm25.doc_frequencies.values():
            assert df > 0

        results = bm25.search("gato perro raton", 10)

        for doc_id, score in results:
            assert math.isfinite(score)

    finally:
        shutil.rmtree(workdir)

    print("OK: genera estadisticas y puntuaciones finitas")


def test_corpus_vacio():
    workdir = tempfile.mkdtemp(prefix="bm25_test_")
    try:
        bm25 = create_bm25(workdir, [])

        bm25.build_stats()

        assert bm25.ready
        assert bm25.avg_doc_length == 0.0
        assert bm25.doc_frequencies == {}
        assert bm25.search("gato", 5) == []

    finally:
        shutil.rmtree(workdir)

    print("OK: maneja un corpus vacio")


def test_parametros_personalizados():
    workdir = tempfile.mkdtemp(prefix="bm25_test_")
    try:
        bm25 = create_bm25(
            workdir,
            [
                {"id": 1, "lyrics": "gato gato perro"},
                {"id": 2, "lyrics": "gato raton"},
                {"id": 3, "lyrics": "perro raton"},
                {"id": 4, "lyrics": "pajaro pez"},
            ],
            k1=2.0,
            b=0.5,
        )

        assert bm25.k1 == 2.0
        assert bm25.b == 0.5

        results = bm25.search("gato", 5)

        assert len(results) == 2

        for doc_id, score in results:
            assert math.isfinite(score)

    finally:
        shutil.rmtree(workdir)

    print("OK: permite utilizar parametros BM25 personalizados")


def test_carga_indice_desde_disco():
    workdir = tempfile.mkdtemp(prefix="bm25_test_")
    try:
        documents = [
            {"id": 1, "lyrics": "gato gato perro"},
            {"id": 2, "lyrics": "gato raton"},
            {"id": 3, "lyrics": "perro raton"},
            {"id": 4, "lyrics": "pajaro pez"},
        ]

        original = create_bm25(workdir, documents)
        original_results = original.search("gato perro", 5)

        loaded_index = SPIMIIndex(
            output_file=os.path.join(workdir, "index")
        ).load()

        loaded = BM25(loaded_index)
        loaded_results = loaded.search("gato perro", 5)

        assert len(original_results) == len(loaded_results)

        for (id1, score1), (id2, score2) in zip(
            original_results, loaded_results
        ):
            assert id1 == id2
            assert math.isclose(
                score1,
                score2,
                rel_tol=1e-9,
            )

    finally:
        shutil.rmtree(workdir)

    print("OK: obtiene los mismos resultados al cargar el indice")


tests = [
    test_construye_estadisticas,
    test_calcula_longitud_media,
    test_calcula_frecuencias_documentales,
    test_busca_termino_unico,
    test_calcula_puntuacion_bm25,
    test_ranking,
    test_top_k,
    test_consulta_con_terminos_repetidos,
    test_consulta_terminos_desconocidos,
    test_consulta_mixta,
    test_consulta_vacia,
    test_scores_validos,
    test_corpus_vacio,
    test_parametros_personalizados,
    test_carga_indice_desde_disco,
]

for test in tests:
    print(f"Corriendo {test.__name__}...", end=" ")
    test()
    print("OK")

print(f"\n{len(tests)} tests pasaron.")
