"""
Fixtures comunes: chutaDB en un directorio temporal (nunca la BD real) y
embeddings sintéticos "por identidad" para probar sin modelos de IA.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from storage.storage_manager import StorageManager  # noqa: E402

from face_recognition_app.db import FaceRecognitionDB  # noqa: E402
from face_recognition_app.db_schema import create_face_recognition_tables  # noqa: E402


class TempDB:
    """Abre/cierra/reabre un StorageManager dentro de un directorio temporal."""

    def __init__(self, path: Path):
        self.path = path
        self.storage: StorageManager | None = None
        self.db: FaceRecognitionDB | None = None

    def open(self) -> FaceRecognitionDB:
        self.storage = StorageManager(wal_path=str(self.path / "wal.log"))
        create_face_recognition_tables(self.storage)
        self.db = FaceRecognitionDB(self.storage)
        return self.db

    def close(self) -> None:
        if self.storage is not None:
            self.storage.close()
            self.storage = None
            self.db = None

    def reopen(self) -> FaceRecognitionDB:
        self.close()
        return self.open()


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    handle = TempDB(tmp_path)
    handle.open()
    yield handle
    handle.close()


class FakeEngine:
    """
    Motor de IA falso para probar servicio/API sin modelos.

    La "imagen" son bytes b"FAKE:<id>/<variante>,<id>/<variante>,...": una cara por
    elemento. Cada identidad tiene una dirección fija por modelo y cada variante
    agrega ruido determinista (como distintas fotos de la misma persona).
    b"FAKE:" = imagen sin rostros.
    """

    DIM = 32

    def __init__(self, noise: float = 0.2):
        self.noise = noise
        self.calls = 0

    def _vector(self, model: str, identity: str, variant: str) -> np.ndarray:
        import zlib

        base = np.random.default_rng(zlib.crc32(f"{model}|{identity}".encode())).normal(size=self.DIM)
        jitter = np.random.default_rng(zlib.crc32(f"{model}|{identity}|{variant}".encode())).normal(size=self.DIM)
        v = base + self.noise * jitter
        return (v / np.linalg.norm(v)).astype(np.float32)

    def analyze(self, source, models=("arcface", "facenet"), max_faces: int = 10):
        from face_recognition_app.embeddings import AnalyzedFace, DetectedFace, ImageAnalysis

        self.calls += 1
        if not isinstance(source, (bytes, bytearray)) or not bytes(source).startswith(b"FAKE:"):
            raise ValueError("El archivo no es una imagen válida.")
        spec = bytes(source)[5:].decode()
        faces = []
        for i, item in enumerate(x for x in spec.split(",") if x):
            identity, _, variant = item.partition("/")
            det = DetectedFace(bbox=np.array([10 + 60 * i, 10, 50 + 60 * i, 60], np.float32),
                               kps=np.zeros((5, 2), np.float32), det_score=0.99 - 0.01 * i)
            faces.append(AnalyzedFace(det, {m: self._vector(m, identity, variant) for m in models}))
        if max_faces:
            faces = faces[:max_faces]
        return ImageAnalysis(faces, (400, 300), 1.0, 1.0, {m: 1.0 for m in models})

    def warmup(self, models=()):
        pass


def fake_image(*faces: str) -> bytes:
    """fake_image("ana/1") -> una cara de 'ana', variante 1."""
    return ("FAKE:" + ",".join(faces)).encode()


def identity_embeddings(n_people: int, per_person: int, dim: int = 64,
                        noise: float = 0.25, seed: int = 0) -> np.ndarray:
    """
    (n_people, per_person, dim): cada identidad es una dirección aleatoria y sus
    "fotos" son esa dirección + ruido, normalizadas (imitan embeddings faciales).
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_people, 1, dim))
    samples = centers + noise * rng.normal(size=(n_people, per_person, dim))
    samples /= np.linalg.norm(samples, axis=2, keepdims=True)
    return samples.astype(np.float32)
