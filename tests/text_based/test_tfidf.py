"""
Pruebas de TF-IDF + similitud coseno.

    PYTHONPATH=. python3 tests/text_based/test_tfidf.py

Cubren construccion de estadisticas, pesos TF-IDF, normalizacion,
similitud coseno, ranking, consultas vacias y casos limite.
"""

import math
import os
import shutil
import tempfile

from text_based.spimi import SPIMIIndex
from text_based.tfidf import TFIDFCosine


def create_tfidf(workdir, documents):
    output = os.path.join(workdir, "index")
    index = SPIMIIndex(output_file=output)
    index.build(documents)
    return TFIDFCosine(index)


def test_construye_estadisticas():
    workdir = tempfile.mkdtemp(prefix="tfidf_test_")
    try:
        tfidf = create_tfidf(
            workdir,
            [
                {"id": 1, "lyrics": "gato perro gato"},
                {"id": 2, "lyrics": "gato raton"},
                {"id": 3, "lyrics": "perro raton"},
            ],
        )

        assert not tfidf.ready
        tfidf.build_stats()

        assert tfidf.ready
        assert set(tfidf.doc_norms) == {1, 2, 3}

        for norm in tfidf.doc_norms.values():
            assert norm > 0
    finally:
        shutil.rmtree(workdir)
    print("OK: construye estadisticas y normas de documentos")


def test_calcula_normas_tfidf():
    workdir = tempfile.mkdtemp(prefix="tfidf_test_")
    try:
        tfidf = create_tfidf(
            workdir,
            [
                {"id": 1, "lyrics": "gato gato perro"},
                {"id": 2, "lyrics": "gato raton"},
                {"id": 3, "lyrics": "perro raton"},
            ],
        )

        tfidf.build_stats()

        # df(gato) = 2
        # df(perro) = 2
        # tf(gato, doc 1) = 1 + log10(2)
        # tf(perro, doc 1) = 1

        expected = math.sqrt(
            ((1 + math.log10(2)) / 2) ** 2
            + (1 / 2) ** 2
        )

        assert math.isclose(
            tfidf.doc_norms[1],
            expected,
            rel_tol=1e-9,
        )
    finally:
        shutil.rmtree(workdir)
    print("OK: calcula correctamente las normas TF-IDF")


def test_consulta_termino_unico():
    workdir = tempfile.mkdtemp(prefix="tfidf_test_")
    try:
        tfidf = create_tfidf(
            workdir,
            [
                {"id": 1, "lyrics": "gato gato"},
                {"id": 2, "lyrics": "gato perro"},
                {"id": 3, "lyrics": "perro raton"},
            ],
        )

        results = tfidf.search("gato", 10)

        assert len(results) == 2
        assert {doc_id for doc_id, score in results} == {1, 2}

        for doc_id, score in results:
            assert 0 <= score <= 1
    finally:
        shutil.rmtree(workdir)
    print("OK: busca documentos que contienen un termino")


def test_similitud_coseno():
    workdir = tempfile.mkdtemp(prefix="tfidf_test_")
    try:
        tfidf = create_tfidf(
            workdir,
            [
                {"id": 1, "lyrics": "gato gato"},
                {"id": 2, "lyrics": "gato perro"},
                {"id": 3, "lyrics": "perro raton"},
            ],
        )

        results = tfidf.search("gato", 10)
        scores = dict(results)

        # Al consultar un unico termino, los documentos que lo
        # contienen tienen similitud coseno positiva.
        assert scores[1] > 0
        assert scores[2] > 0

        # El documento 1 solo contiene gato, por lo que su vector
        # tiene la misma direccion que la consulta.
        assert math.isclose(scores[1], 1.0, rel_tol=1e-9)
    finally:
        shutil.rmtree(workdir)
    print("OK: calcula correctamente la similitud coseno")


def test_ranking():
    workdir = tempfile.mkdtemp(prefix="tfidf_test_")
    try:
        tfidf = create_tfidf(
            workdir,
            [
                {"id": 1, "lyrics": "gato gato"},
                {"id": 2, "lyrics": "gato perro"},
                {"id": 3, "lyrics": "perro raton"},
            ],
        )

        results = tfidf.search("gato", 10)
        scores = [score for doc_id, score in results]

        assert scores == sorted(scores, reverse=True)
        assert results[0][0] == 1
    finally:
        shutil.rmtree(workdir)
    print("OK: ordena los resultados por similitud descendente")


def test_top_k():
    workdir = tempfile.mkdtemp(prefix="tfidf_test_")
    try:
        tfidf = create_tfidf(
            workdir,
            [
                {"id": 1, "lyrics": "gato"},
                {"id": 2, "lyrics": "gato"},
                {"id": 3, "lyrics": "gato"},
            ],
        )

        assert len(tfidf.search("gato", 2)) == 2
        assert len(tfidf.search("gato", 1)) == 1
        assert tfidf.search("gato", 0) == []
        assert tfidf.search("gato", -1) == []
    finally:
        shutil.rmtree(workdir)
    print("OK: respeta top_k y rechaza valores no positivos")


def test_consulta_con_terminos_repetidos():
    workdir = tempfile.mkdtemp(prefix="tfidf_test_")
    try:
        tfidf = create_tfidf(
            workdir,
            [
                {"id": 1, "lyrics": "gato gato perro"},
                {"id": 2, "lyrics": "gato raton"},
                {"id": 3, "lyrics": "perro raton"},
            ],
        )

        single = tfidf.search("gato", 10)
        repeated = tfidf.search("gato gato", 10)

        assert len(single) == len(repeated)

        for (id1, score1), (id2, score2) in zip(single, repeated):
            assert id1 == id2
            assert math.isclose(score1, score2, rel_tol=1e-9)
    finally:
        shutil.rmtree(workdir)
    print("OK: normaliza consultas con terminos repetidos")


def test_consulta_terminos_desconocidos():
    workdir = tempfile.mkdtemp(prefix="tfidf_test_")
    try:
        tfidf = create_tfidf(
            workdir,
            [
                {"id": 1, "lyrics": "gato perro"},
                {"id": 2, "lyrics": "raton"},
            ],
        )

        assert tfidf.search("inexistente", 5) == []
        assert tfidf.search("xyz abc", 5) == []
    finally:
        shutil.rmtree(workdir)
    print("OK: devuelve lista vacia para terminos desconocidos")


def test_consulta_mixta():
    workdir = tempfile.mkdtemp(prefix="tfidf_test_")
    try:
        tfidf = create_tfidf(
            workdir,
            [
                {"id": 1, "lyrics": "gato perro"},
                {"id": 2, "lyrics": "gato raton"},
                {"id": 3, "lyrics": "perro raton"},
            ],
        )

        results = tfidf.search("gato inexistente", 5)
        expected = tfidf.search("gato", 5)

        assert len(results) == len(expected)

        for (id1, score1), (id2, score2) in zip(results, expected):
            assert id1 == id2
            assert math.isclose(score1, score2, rel_tol=1e-9)
    finally:
        shutil.rmtree(workdir)
    print("OK: ignora terminos desconocidos en consultas mixtas")


def test_consulta_vacia():
    workdir = tempfile.mkdtemp(prefix="tfidf_test_")
    try:
        tfidf = create_tfidf(
            workdir,
            [{"id": 1, "lyrics": "gato perro"}],
        )

        assert tfidf.search("", 5) == []
        assert tfidf.search("   ", 5) == []
    finally:
        shutil.rmtree(workdir)
    print("OK: maneja consultas vacias")


def test_normas_y_scores_validos():
    workdir = tempfile.mkdtemp(prefix="tfidf_test_")
    try:
        tfidf = create_tfidf(
            workdir,
            [
                {"id": 1, "lyrics": "gato perro gato"},
                {"id": 2, "lyrics": "gato raton"},
                {"id": 3, "lyrics": "perro raton"},
            ],
        )

        tfidf.build_stats()

        for norm in tfidf.doc_norms.values():
            assert math.isfinite(norm)
            assert norm > 0

        results = tfidf.search("gato perro raton", 10)

        for doc_id, score in results:
            assert math.isfinite(score)
            assert 0 <= score <= 1 + 1e-10
    finally:
        shutil.rmtree(workdir)
    print("OK: genera normas y scores finitos y normalizados")


def test_corpus_vacio():
    workdir = tempfile.mkdtemp(prefix="tfidf_test_")
    try:
        tfidf = create_tfidf(workdir, [])
        tfidf.build_stats()

        assert tfidf.ready
        assert tfidf.doc_norms == {}
        assert tfidf.search("gato", 5) == []
    finally:
        shutil.rmtree(workdir)
    print("OK: maneja un corpus vacio")


def test_carga_indice_desde_disco():
    workdir = tempfile.mkdtemp(prefix="tfidf_test_")
    try:
        documents = [
            {"id": 1, "lyrics": "gato gato perro"},
            {"id": 2, "lyrics": "gato raton"},
            {"id": 3, "lyrics": "perro raton"},
        ]

        original = create_tfidf(workdir, documents)
        original_results = original.search("gato perro", 5)

        loaded_index = SPIMIIndex(
            output_file=os.path.join(workdir, "index")
        ).load()

        loaded = TFIDFCosine(loaded_index)
        loaded_results = loaded.search("gato perro", 5)

        assert len(original_results) == len(loaded_results)

        for (id1, score1), (id2, score2) in zip(
            original_results, loaded_results
        ):
            assert id1 == id2
            assert math.isclose(score1, score2, rel_tol=1e-9)
    finally:
        shutil.rmtree(workdir)
    print("OK: obtiene los mismos resultados al cargar el indice")


tests = [
    test_construye_estadisticas,
    test_calcula_normas_tfidf,
    test_consulta_termino_unico,
    test_similitud_coseno,
    test_ranking,
    test_top_k,
    test_consulta_con_terminos_repetidos,
    test_consulta_terminos_desconocidos,
    test_consulta_mixta,
    test_consulta_vacia,
    test_normas_y_scores_validos,
    test_corpus_vacio,
    test_carga_indice_desde_disco,
]

for test in tests:
    print(f"Corriendo {test.__name__}...", end=" ")
    test()
    print("OK")

print(f"\n{len(tests)} tests pasaron.")
