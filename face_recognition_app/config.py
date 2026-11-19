"""
Configuración central de la app de reconocimiento facial.

Todo lo que se quiera cambiar más adelante (modelos activos, defaults de búsqueda,
rutas) se cambia aquí; el resto de módulos lee estos valores.
"""
from __future__ import annotations

import os
from pathlib import Path

# ------------------------------------------------------------------
# Rutas
# ------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("FACE_APP_DATA", APP_DIR / "data"))

PHOTOS_SUBDIR = "photos"
INDEXES_SUBDIR = "indexes"
LFW_SUBDIR = "lfw"
CALIBRATION_FILE = "calibration.json"
EVALUATION_SUBDIR = "evaluation"

# ------------------------------------------------------------------
# Modelos, índices y métricas
# ------------------------------------------------------------------

# Modelos de embeddings activos (el orden es el que se muestra en la UI).
# Cada nombre debe existir en embeddings.EMBEDDERS.
MODELS: tuple[str, ...] = ("arcface", "facenet")

MODEL_LABELS = {
    "arcface": "ArcFace",
    "facenet": "FaceNet",
}

# "exact" es la búsqueda secuencial (ground truth); el resto son índices del core.
INDEX_TYPES: tuple[str, ...] = ("exact", "hnsw", "ivf")

INDEX_LABELS = {
    "exact": "Exacto (secuencial)",
    "hnsw": "HNSW",
    "ivf": "IVF",
}

# Nombres canónicos de indexes/metricas.py.
METRICS: tuple[str, ...] = ("coseno", "euclidiana", "producto_punto")

METRIC_LABELS = {
    "coseno": "Coseno",
    "euclidiana": "Euclidiana (L2)",
    "producto_punto": "Producto punto",
}

# ------------------------------------------------------------------
# Defaults de búsqueda / construcción
# ------------------------------------------------------------------

DEFAULT_MODEL = "arcface"
DEFAULT_INDEX = "hnsw"
DEFAULT_METRIC = "coseno"
DEFAULT_K = 5
MAX_K = 50

# Parámetros HNSW
DEFAULT_M = 16
DEFAULT_EF_CONSTRUCTION = 100
DEFAULT_EF_SEARCH = 50

# Parámetros IVF (nlist=None => round(sqrt(N)))
DEFAULT_NLIST: int | None = None
DEFAULT_NPROBE = 8

# Umbral de similitud coseno (0..1) por modelo, usado si no hay calibración.
# Valores calibrados con LFW (prepare_lfw --calibrate, 874 consultas de validación);
# data/calibration.json, si existe, tiene prioridad.
DEFAULT_THRESHOLDS = {
    "arcface": 0.32,
    "facenet": 0.67,
}

# Máximo de rostros procesados por imagen de consulta.
MAX_FACES_PER_IMAGE = 10

# Máximo de fotos registradas por persona. Los vecinos (k) son fotos: sin tope,
# una persona con muchas fotos llenaría el top-k y ocultaría a otros candidatos.
MAX_PHOTOS_PER_PERSON = 10

# ------------------------------------------------------------------
# Almacenamiento del embedding en chutaDB
# ------------------------------------------------------------------

# Dimensión máxima que cabe en la columna FaceImage.embedding (float32 en base64).
MAX_EMBEDDING_DIM = 512
