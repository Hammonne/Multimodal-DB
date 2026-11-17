"""Fase 2: esquema, codec y FaceRecognitionDB (sin modelos de IA)."""
from __future__ import annotations

import numpy as np
import pytest

from face_recognition_app import codec
from face_recognition_app.db_schema import (
    FACE_IMAGE_COLUMNS,
    FACE_IMAGE_TABLE,
    PERSON_COLUMNS,
    PERSON_TABLE,
    create_face_recognition_tables,
)

from conftest import identity_embeddings


# ----------------------------------------------------------------------
# codec
# ----------------------------------------------------------------------


def test_codec_roundtrip_is_exact():
    v = codec.normalize(np.random.default_rng(1).normal(size=512))
    text = codec.encode(v)
    assert len(text) <= codec.EMBEDDING_COLUMN_WIDTH
    assert np.array_equal(codec.decode(text, 512), v)
    # el relleno \x00 de los varchar no afecta
    assert np.array_equal(codec.decode(text + "\x00\x00", 512), v)


def test_codec_rejects_bad_vectors():
    with pytest.raises(ValueError):
        codec.normalize(np.zeros(8))
    with pytest.raises(ValueError):
        codec.encode(np.ones(513, dtype=np.float32))
    with pytest.raises(ValueError):
        codec.decode(codec.encode(np.ones(4)), 8)


# ----------------------------------------------------------------------
# esquema
# ----------------------------------------------------------------------


def test_tables_created_with_expected_columns(temp_db):
    catalog = temp_db.storage.catalog
    assert catalog.get_table_info(PERSON_TABLE)["column_names"] == PERSON_COLUMNS
    assert catalog.get_table_info(FACE_IMAGE_TABLE)["column_names"] == FACE_IMAGE_COLUMNS
    # idempotente
    create_face_recognition_tables(temp_db.storage)
    assert catalog.get_table_info(PERSON_TABLE)["column_names"] == PERSON_COLUMNS


def test_old_schema_is_recreated(tmp_path, monkeypatch):
    from storage.storage_manager import StorageManager

    monkeypatch.chdir(tmp_path)
    storage = StorageManager(wal_path=str(tmp_path / "wal.log"))
    # esquema de la versión anterior de la app
    storage.create_table("Person", ["integer", "varchar(100)"], "heap", 0, ["person_id", "name"])
    storage.open_table("Person").insert([1, "Vieja"])
    storage.close()

    storage = StorageManager(wal_path=str(tmp_path / "wal.log"))
    create_face_recognition_tables(storage)
    info = storage.catalog.get_table_info("Person")
    assert info["column_names"] == PERSON_COLUMNS
    assert list(storage.open_table("Person").scan()) == []
    storage.close()


def test_stale_catalog_embedding_index_is_dropped(tmp_path, monkeypatch):
    """Un embedding_* registrado (como dejaba la versión anterior) no rompe la apertura."""
    from storage.storage_manager import StorageManager

    monkeypatch.chdir(tmp_path)
    storage = StorageManager(wal_path=str(tmp_path / "wal.log"))
    create_face_recognition_tables(storage)
    storage.catalog.register_index("viejo_idx", FACE_IMAGE_TABLE, "embedding", 0, "embedding_hnsw")
    storage.close()

    storage = StorageManager(wal_path=str(tmp_path / "wal.log"))
    create_face_recognition_tables(storage)
    assert storage.catalog.get_table_indexes(FACE_IMAGE_TABLE) == []
    storage.open_table(FACE_IMAGE_TABLE)  # no lanza
    storage.close()


# ----------------------------------------------------------------------
# FaceRecognitionDB
# ----------------------------------------------------------------------


def _populate(db, n_people=3, per_person=2, dim=64):
    E = identity_embeddings(n_people, per_person, dim)
    persons = []
    for p in range(n_people):
        person = db.create_person(f"Persona {p}", source="lfw")
        persons.append(person)
        for j in range(per_person):
            photo_id = db.next_photo_id()
            for model in ("arcface", "facenet"):
                db.add_face(person.person_id, photo_id, f"photos/{person.person_id}/{photo_id}.jpg",
                            model, E[p, j])
    return persons, E


def test_person_and_faces_crud(temp_db):
    db = temp_db.db
    persons, E = _populate(db)

    assert [p.person_id for p in persons] == [1, 2, 3]
    assert db.count_persons() == 3
    assert db.count_faces() == 12
    assert db.count_faces("arcface") == 6
    assert db.get_person(2).name == "Persona 1"
    assert [p.name for p in db.list_persons("persona 2")] == ["Persona 2"]

    faces = db.faces_of_person(1, "arcface")
    assert len(faces) == 2
    assert db.face_by_rid(faces[0].rid) == faces[0]
    # la misma foto tiene una fila por modelo
    assert {f.model for f in db.faces_of_photo(faces[0].photo_id)} == {"arcface", "facenet"}
    # el embedding se guarda en la tabla y vuelve normalizado
    assert np.allclose(db.get_embedding(faces[0].face_id), E[0, 0], atol=1e-6)

    with pytest.raises(ValueError):
        db.add_face(999, 1, "x.jpg", "arcface", E[0, 0])
    with pytest.raises(ValueError):
        db.create_person("   ")


def test_load_embeddings_reads_from_table(temp_db):
    db = temp_db.db
    _, E = _populate(db)
    faces, X = db.load_embeddings("facenet")
    assert X.shape == (6, 64)
    assert all(f.model == "facenet" for f in faces)
    assert np.allclose(np.linalg.norm(X, axis=1), 1, atol=1e-5)


def test_delete_person(temp_db):
    db = temp_db.db
    _populate(db)
    removed = db.delete_person(2)
    assert len(removed) == 4
    assert db.get_person(2) is None
    assert db.count_faces() == 8
    assert all(f.person_id != 2 for _, f in [(0, x) for x in db.load_embeddings("arcface")[0]])
    with pytest.raises(KeyError):
        db.delete_person(2)


def test_reset_then_insert_survives_reopen(temp_db):
    """Regresión: tras reset, los datos nuevos no deben perderse al reabrir."""
    from face_recognition_app.db import FaceRecognitionDB
    from face_recognition_app.db_schema import reset_face_recognition_tables

    _populate(temp_db.db)
    storage = temp_db.storage
    reset_face_recognition_tables(storage)
    db = FaceRecognitionDB(storage)
    assert db.count_faces() == 0
    _populate(db, n_people=2)

    db = temp_db.reopen()
    db = temp_db.reopen()              # el BufferManager compartido cierra todo lo conocido
    assert db.count_persons() == 2
    assert db.count_faces() == 8
    assert len(db.load_embeddings("arcface")[0]) == 4


def test_persistence_after_reopen(temp_db):
    db = temp_db.db
    _populate(db)
    db.delete_person(1)
    before = {f.face_id: f for f in db.load_embeddings("arcface")[0]}

    db = temp_db.reopen()
    assert db.count_persons() == 2
    assert db.count_faces() == 8
    after = {f.face_id: f for f in db.load_embeddings("arcface")[0]}
    assert after == before
    # los IDs siguen creciendo sin repetirse
    assert db.create_person("Nueva").person_id == 4
    assert db.next_photo_id() == 7
