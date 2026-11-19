"""Fase 2: VectorStore — Exacto / HNSW / IVF del core, incremental, persistencia."""
from __future__ import annotations

import numpy as np
import pytest

from face_recognition_app.vector_store import VectorStore, index_name

from conftest import identity_embeddings

N_PEOPLE = 20
PER_PERSON = 4
DIM = 64


def _seed(db, model="arcface", n_people=N_PEOPLE, per_person=PER_PERSON, seed=0):
    """Registra per_person-1 fotos por persona; la última queda como consulta."""
    E = identity_embeddings(n_people, per_person, DIM, seed=seed)
    queries = []
    for p in range(n_people):
        person = db.create_person(f"P{p}")
        for j in range(per_person - 1):
            db.add_face(person.person_id, db.next_photo_id(), f"{p}_{j}.jpg", model, E[p, j])
        queries.append((person.person_id, E[p, -1]))
    return queries


@pytest.fixture
def store(temp_db, tmp_path):
    return VectorStore(temp_db.db, tmp_path / "indexes")


@pytest.mark.parametrize("index_type", ["exact", "hnsw", "ivf"])
@pytest.mark.parametrize("metric", ["coseno", "euclidiana", "producto_punto"])
def test_search_finds_identity(temp_db, store, index_type, metric):
    queries = _seed(temp_db.db)
    correct = 0
    for person_id, q in queries:
        res = store.search("arcface", q, index_type=index_type, metric=metric, k=5, nprobe=4)
        assert len(res.hits) == 5
        assert res.gallery_size == N_PEOPLE * (PER_PERSON - 1)
        # ordenado por distancia de la métrica elegida
        d = [h.distance for h in res.hits]
        assert d == sorted(d)
        correct += res.hits[0].face.person_id == person_id
    assert correct >= 0.9 * len(queries)


def test_metrics_give_same_ranking_on_normalized_vectors(temp_db, store):
    queries = _seed(temp_db.db)
    q = queries[3][1]
    rankings = [
        [h.face.face_id for h in store.search("arcface", q, "exact", m, k=10).hits]
        for m in ("coseno", "euclidiana", "producto_punto")
    ]
    assert rankings[0] == rankings[1] == rankings[2]
    hits = store.search("arcface", q, "exact", "euclidiana", k=3).hits
    # similitud coseno y distancia L2 consistentes: d² = 2 - 2·cos
    for h in hits:
        assert h.distance ** 2 == pytest.approx(2 - 2 * h.similarity, abs=1e-4)


def test_index_registered_in_index_manager_and_persisted(temp_db, store, tmp_path):
    queries = _seed(temp_db.db)
    store.search("arcface", queries[0][1], "hnsw", "coseno")
    name = index_name("arcface", "hnsw", "coseno")
    assert name in temp_db.storage.index_manager.open_indexes
    info = store.index_info("arcface", "hnsw", "coseno")
    assert info["built"] and info["size"] == N_PEOPLE * (PER_PERSON - 1)
    assert (tmp_path / "indexes" / f"{name}_indice.npz").exists()

    expected = [h.face.face_id for h in store.search("arcface", queries[0][1], "hnsw", "coseno").hits]

    # reabrir chutaDB: el índice se carga de disco, no se reconstruye
    db = temp_db.reopen()
    store2 = VectorStore(db, tmp_path / "indexes")
    built_at = info["built_at"]
    got = [h.face.face_id for h in store2.search("arcface", queries[0][1], "hnsw", "coseno").hits]
    assert got == expected
    assert store2.index_info("arcface", "hnsw", "coseno")["built_at"] == built_at


@pytest.mark.parametrize("index_type", ["hnsw", "ivf"])
def test_incremental_add_and_remove(temp_db, store, index_type):
    db = temp_db.db
    _seed(db)
    store.gallery("arcface")
    store.ensure_index("arcface", index_type, "coseno")
    built_at = store.index_info("arcface", index_type, "coseno")["built_at"]

    # alta de una persona nueva con 2 fotos
    E = identity_embeddings(1, 3, DIM, seed=99)[0]
    newbie = db.create_person("Nueva")
    faces = [db.add_face(newbie.person_id, db.next_photo_id(), f"n{j}.jpg", "arcface", E[j])
             for j in range(2)]
    store.add(faces, E[:2])

    res = store.search("arcface", E[2], index_type, "coseno", k=3, nprobe=50)
    assert res.hits[0].face.person_id == newbie.person_id
    # no se reconstruyó: fue inserción incremental
    assert store.index_info("arcface", index_type, "coseno")["built_at"] == built_at
    assert store.index_info("arcface", index_type, "coseno")["size"] == res.gallery_size

    # baja
    removed = db.delete_person(newbie.person_id)
    store.remove(removed)
    res = store.search("arcface", E[2], index_type, "coseno", k=5, nprobe=50)
    assert all(h.face.person_id != newbie.person_id for h in res.hits)
    assert res.gallery_size == N_PEOPLE * (PER_PERSON - 1)


def test_stale_index_is_rebuilt_from_table(temp_db, store, tmp_path):
    db = temp_db.db
    queries = _seed(db)
    store.ensure_index("arcface", "ivf", "coseno")

    # filas agregadas a la tabla "a espaldas" del índice (p. ej. un corte)
    E = identity_embeddings(1, 2, DIM, seed=7)[0]
    newbie = db.create_person("Fantasma")
    db.add_face(newbie.person_id, db.next_photo_id(), "f.jpg", "arcface", E[0])

    db = temp_db.reopen()
    store2 = VectorStore(db, tmp_path / "indexes")
    res = store2.search("arcface", E[1], "ivf", "coseno", k=1, nprobe=50)
    assert res.hits[0].face.person_id == newbie.person_id
    assert store2.index_info("arcface", "ivf", "coseno")["size"] == res.gallery_size


def test_models_are_not_mixed(temp_db, store):
    db = temp_db.db
    _seed(db, model="arcface", n_people=5)
    p = db.create_person("Solo FaceNet")
    v = identity_embeddings(1, 1, DIM, seed=3)[0, 0]
    db.add_face(p.person_id, db.next_photo_id(), "x.jpg", "facenet", v)

    res = store.search("facenet", v, "hnsw", "coseno", k=10)
    assert [h.face.person_id for h in res.hits] == [p.person_id]
    res = store.search("arcface", v, "exact", "coseno", k=50)
    assert all(h.face.model == "arcface" for h in res.hits)


def test_rebuild_with_custom_params(temp_db, store):
    _seed(temp_db.db)
    info = store.build_index("arcface", "ivf", "coseno", nlist=4)
    assert info["nlist"] == 4
    info = store.build_index("arcface", "hnsw", "euclidiana", M=8, ef_construction=40)
    assert (info["M"], info["ef_construction"]) == (8, 40)


def test_empty_gallery_and_bad_input(temp_db, store):
    assert store.search("arcface", np.ones(DIM), "hnsw", "coseno").hits == []
    _seed(temp_db.db, n_people=3)
    with pytest.raises(ValueError):
        store.search("arcface", np.ones(DIM + 1), "exact", "coseno")
    with pytest.raises(ValueError):
        store.search("arcface", np.ones(DIM), "kdtree", "coseno")
    with pytest.raises(ValueError):
        store.search("arcface", np.ones(DIM), "exact", "manhattan")
