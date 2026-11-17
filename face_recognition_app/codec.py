"""
Conversión embedding <-> texto para guardarlo en una columna varchar de chutaDB.

chutaDB no tiene un tipo "vector": el embedding (float32) se guarda en base64.
Es exacto (sin pérdida) y solo usa caracteres ASCII, así que no choca con el
relleno de \\x00 de los varchar de tamaño fijo.
"""
from __future__ import annotations

import base64
import math

import numpy as np

from face_recognition_app import config

DTYPE = np.float32


def column_width(max_dim: int = config.MAX_EMBEDDING_DIM) -> int:
    """Ancho del varchar necesario para max_dim floats en base64."""
    raw_bytes = max_dim * np.dtype(DTYPE).itemsize
    return 4 * math.ceil(raw_bytes / 3)


EMBEDDING_COLUMN_WIDTH = column_width()


def normalize(vector) -> np.ndarray:
    """Devuelve el vector como float32 1-D con norma L2 = 1."""
    v = np.asarray(vector, dtype=DTYPE).reshape(-1)
    if v.size == 0:
        raise ValueError("El embedding está vacío.")
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("El embedding tiene norma cero o inválida.")
    return (v / norm).astype(DTYPE)


def encode(vector) -> str:
    v = np.asarray(vector, dtype=DTYPE).reshape(-1)
    if v.size > config.MAX_EMBEDDING_DIM:
        raise ValueError(
            f"El embedding tiene {v.size} dimensiones; la columna admite "
            f"hasta {config.MAX_EMBEDDING_DIM}."
        )
    return base64.b64encode(v.tobytes()).decode("ascii")


def decode(text, dim: int) -> np.ndarray:
    if isinstance(text, bytes):
        text = text.decode("ascii", "ignore")
    text = str(text).replace("\x00", "").strip()
    v = np.frombuffer(base64.b64decode(text), dtype=DTYPE)
    if v.size != dim:
        raise ValueError(
            f"Embedding corrupto: se esperaban {dim} valores y hay {v.size}."
        )
    return v.copy()
