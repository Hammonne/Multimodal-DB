"""
Pruebas del indice invertido SPIMI.

    PYTHONPATH=. python3 tests/text_based/test_spimi.py

Cubren tokenizacion, lectura JSONL/arreglo, presupuesto en bytes, merge
progresivo, formato binario y consulta por offsets.
"""

import json
import os
import shutil
import tempfile

from text_based.spimi import (
    MAX_OPEN_RUNS,
    SPIMIIndex,
    iter_documents,
    normalize_doc_id,
    tokenize,
)


def write_jsonl(path, documents):
    with open(path, "w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def test_tokenize_unicode_y_signos():
    text = "Hola, MUNDO! don't  foo_bar  ñandú 2024"
    assert tokenize(text) == ["hola", "mundo", "don't", "foo", "bar", "ñandú", "2024"]
    assert tokenize("ROCK’n’ROLL") == ["rock’n’roll"]
    print("OK: tokeniza Unicode, minusculas, apostrofes y guiones bajos")


def test_normalize_doc_id():
    assert normalize_doc_id("123") == 123
    assert normalize_doc_id(" -4 ") == -4
    assert normalize_doc_id(7) == 7
    try:
        normalize_doc_id("abc")
        assert False, "un id de texto debia fallar"
    except ValueError as exc:
        assert "id" in str(exc)
    print("OK: normaliza ids numericos y rechaza ids de texto")


def test_lee_jsonl_y_arreglo():
    workdir = tempfile.mkdtemp(prefix="spimi_test_")
    try:
        jsonl = os.path.join(workdir, "docs.jsonl")
        write_jsonl(jsonl, [{"id": 1, "lyrics": "uno"}, {"id": 2, "lyrics": "dos"}])
        assert [doc["id"] for doc in iter_documents(jsonl)] == [1, 2]

        array = os.path.join(workdir, "docs.json")
        with open(array, "w", encoding="utf-8") as handle:
            json.dump(
                [{"id": 10, "lyrics": "uno"}, {"id": 2, "lyrics": "dos"}],
                handle,
            )
        docs = list(iter_documents(array))
        assert [doc["id"] for doc in docs] == [10, 2]
    finally:
        shutil.rmtree(workdir)
    print("OK: lee JSONL y arreglos JSON")


def test_construye_bloques_y_binario():
    workdir = tempfile.mkdtemp(prefix="spimi_test_")
    try:
        source = os.path.join(workdir, "docs.jsonl")
        output = os.path.join(workdir, "index")
        write_jsonl(
            source,
            [
                {"id": 1, "lyrics": "gato gato perro"},
                {"id": 2, "lyrics": "gato raton"},
                {"id": 3, "lyrics": "perro perro perro"},
            ],
        )

        index = SPIMIIndex(
            max_memory=1,
            output_file=output,
            max_open_runs=2,
        )
        target = index.build_from_json(source)
        assert target == output
        assert index.doc_count == 3
        assert index.block_count == 5
        assert index.loaded

        index_bin, metadata = index.artifact_paths()
        assert os.path.getsize(index_bin) > 0
        assert os.path.getsize(metadata) > 0

        data = read_json(metadata)
        assert data["doc_count"] == 3
        assert set(data["term_offsets"]) == {"gato", "perro", "raton"}
        assert data["doc_lengths"] == {"1": 3, "2": 2, "3": 3}

        loaded = SPIMIIndex(output_file=output).load()
        assert loaded.index == {}
        assert loaded.doc_count == 3
        assert loaded.doc_lengths == {1: 3, 2: 2, 3: 3}
        assert loaded.get_postings("gato") == [(1, 2), (2, 1)]
        assert loaded.get_postings("perro") == [(1, 1), (3, 3)]
        assert loaded.get_postings("raton") == [(2, 1)]
        assert loaded.get_postings("inexistente") == []
    finally:
        shutil.rmtree(workdir)
    print("OK: usa presupuesto en bytes, fusiona bloques y escribe binario")


def test_merge_termino_repetido_entre_bloques():
    workdir = tempfile.mkdtemp(prefix="spimi_test_")
    try:
        output = os.path.join(workdir, "index")
        index = SPIMIIndex(max_memory=1, output_file=output, max_open_runs=2)
        target = index.build(
            [
                {"id": 1, "lyrics": "comun"},
                {"id": 2, "lyrics": "comun"},
                {"id": 3, "lyrics": "comun"},
            ]
        )
        loaded = SPIMIIndex(output_file=target).load()
        assert loaded.get_postings("comun") == [(1, 1), (2, 1), (3, 1)]
        assert loaded.get_postings("comun") != loaded.get_postings("otro")
    finally:
        shutil.rmtree(workdir)
    print("OK: el merge suma un termino repetido en varios bloques")


def test_consulta_por_offsets():
    workdir = tempfile.mkdtemp(prefix="spimi_test_")
    try:
        output = os.path.join(workdir, "index")
        terms = ["alfa", "beta", "gato", "perro", "raton", "zorro", "ñandú"]
        SPIMIIndex(max_memory=1, output_file=output, max_open_runs=2).build(
            [{"id": 1, "lyrics": " ".join(terms)}]
        )
        loaded = SPIMIIndex(output_file=output).load()
        for term in terms:
            assert loaded.get_postings(term) == [(1, 1)]
        assert loaded.get_postings("abeja") == []
        assert loaded.get_postings("zzz") == []
        assert loaded.get_doc_length(1) == len(terms)
        assert loaded.get_doc_length("1") == len(terms)
        assert loaded.get_doc_length(99) is None
    finally:
        shutil.rmtree(workdir)
    print("OK: encuentra terminos por offset, incluidos Unicode")


def test_presupuesto_en_bytes_no_postings():
    workdir = tempfile.mkdtemp(prefix="spimi_test_")
    try:
        output = os.path.join(workdir, "index")
        documents = [{"id": doc_id, "lyrics": "comun"} for doc_id in range(20)]
        index = SPIMIIndex(max_memory=600, output_file=output)
        index.build(documents)
        assert index.block_count == 2
        loaded = SPIMIIndex(output_file=output).load()
        assert loaded.doc_count == 20
        assert loaded.get_postings("comun") == [
            (doc_id, 1) for doc_id in range(20)
        ]
    finally:
        shutil.rmtree(workdir)
    print("OK: el presupuesto se mide en bytes estimados, no en postings")


def test_corpus_vacio():
    workdir = tempfile.mkdtemp(prefix="spimi_test_")
    try:
        output = os.path.join(workdir, "index")
        target = SPIMIIndex(output_file=output).build([])
        index_bin, metadata = SPIMIIndex(output_file=target).artifact_paths()
        assert os.path.getsize(index_bin) == 0
        data = read_json(metadata)
        assert data["doc_count"] == 0
        assert data["term_offsets"] == {}
        assert data["doc_lengths"] == {}

        loaded = SPIMIIndex(output_file=target).load()
        assert loaded.doc_count == 0
        assert loaded.get_postings("cualquiera") == []
    finally:
        shutil.rmtree(workdir)
    print("OK: un corpus vacio produce un indice binario valido")


def test_documento_sin_id():
    workdir = tempfile.mkdtemp(prefix="spimi_test_")
    try:
        try:
            SPIMIIndex(output_file=os.path.join(workdir, "index")).build(
                [{"lyrics": "sin id"}]
            )
            assert False, "un documento sin id debia fallar"
        except ValueError as exc:
            assert "id" in str(exc)
        assert os.listdir(workdir) == []
    finally:
        shutil.rmtree(workdir)
    print("OK: rechaza documentos sin campo id y limpia temporales")


tests = [
    test_tokenize_unicode_y_signos,
    test_normalize_doc_id,
    test_lee_jsonl_y_arreglo,
    test_construye_bloques_y_binario,
    test_merge_termino_repetido_entre_bloques,
    test_consulta_por_offsets,
    test_presupuesto_en_bytes_no_postings,
    test_corpus_vacio,
    test_documento_sin_id,
]

for test in tests:
    print(f"Corriendo {test.__name__}...", end=" ")
    test()
    print("OK")

print(f"\n{len(tests)} tests pasaron.")
