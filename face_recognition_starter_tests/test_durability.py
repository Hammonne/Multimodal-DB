"""
Regresión: lo registrado debe persistir aunque el backend termine sin cerrar
chutaDB (terminal cerrada, proceso matado). Se simula con un subproceso que
registra una persona y sale con os._exit() sin storage.close().
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent

CHILD = textwrap.dedent(f"""
    import os, sys
    sys.path.insert(0, {str(ROOT)!r}); sys.path.insert(0, {str(TESTS)!r})
    from storage.storage_manager import StorageManager
    from face_recognition_app.db_schema import create_face_recognition_tables
    from face_recognition_app.db import FaceRecognitionDB
    from face_recognition_app.service import FaceService
    from conftest import FakeEngine, fake_image

    storage = StorageManager(wal_path="wal.log")
    create_face_recognition_tables(storage)
    service = FaceService(FaceRecognitionDB(storage), engine_factory=FakeEngine, data_dir="appdata")
    service.register_person("Ana", [("a.jpg", fake_image("ana/1")), ("b.jpg", fake_image("ana/2"))])
    service.register_person("Beto", [("c.jpg", fake_image("beto/1"))])
    service.delete_person(2)
    os._exit(0)          # sin storage.close()
""")


def test_writes_survive_unclean_exit(tmp_path, monkeypatch):
    subprocess.run([sys.executable, "-c", CHILD], cwd=tmp_path, check=True)

    monkeypatch.chdir(tmp_path)
    from storage.storage_manager import StorageManager

    from face_recognition_app.db import FaceRecognitionDB
    from face_recognition_app.db_schema import create_face_recognition_tables

    storage = StorageManager(wal_path=str(tmp_path / "wal.log"))
    create_face_recognition_tables(storage)
    db = FaceRecognitionDB(storage)
    try:
        assert [p.name for p in db.list_persons()] == ["Ana"]
        assert db.count_faces() == 4                     # 2 fotos x 2 modelos
        assert len(db.load_embeddings("arcface")[0]) == 2
    finally:
        storage.close()
