"""
Extracción de embeddings faciales (fuera de chutaDB).

    imagen -> detección (SCRFD) -> alineación por 5 landmarks -> modelo -> vector L2-normalizado

- Un único detector para todos los modelos: la comparación entre modelos es justa.
- Modelos registrados en EMBEDDERS. Agregar uno = escribir una subclase de
  Embedder y registrarla; la API y el frontend lo toman de config.MODELS.

Los modelos se cargan de forma perezosa (la primera vez que se usan) y una sola
vez por proceso (get_engine()).
"""
from __future__ import annotations

import io
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from face_recognition_app import codec, config

# Imágenes más grandes se reducen antes de detectar (velocidad); los recortes
# se toman de la imagen reducida, que sigue teniendo resolución de sobra.
MAX_IMAGE_SIDE = 1600


# ----------------------------------------------------------------------
# Tipos
# ----------------------------------------------------------------------


@dataclass
class DetectedFace:
    bbox: np.ndarray          # [x1, y1, x2, y2] en píxeles de la imagen (reducida)
    kps: np.ndarray           # (5, 2) ojos, nariz, comisuras
    det_score: float

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))

    def to_dict(self, scale: float = 1.0) -> dict:
        """bbox en coordenadas de la imagen original (scale = original / reducida)."""
        x1, y1, x2, y2 = (float(v) * scale for v in self.bbox)
        return {
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "det_score": round(float(self.det_score), 4),
        }


@dataclass
class AnalyzedFace:
    face: DetectedFace
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class ImageAnalysis:
    faces: list[AnalyzedFace]
    image_size: tuple[int, int]      # (ancho, alto) originales
    scale: float                     # original / procesada
    detect_ms: float
    embed_ms: dict[str, float]


class NoFaceError(ValueError):
    pass


# ----------------------------------------------------------------------
# Lectura de imágenes
# ----------------------------------------------------------------------


def read_image(source) -> np.ndarray:
    """
    Devuelve la imagen en BGR (convención de OpenCV / InsightFace).

    source: ruta, bytes o np.ndarray (BGR). Respeta la orientación EXIF
    (fotos de celular / webcam).
    """
    if isinstance(source, np.ndarray):
        if source.size == 0:
            raise ValueError("La imagen está vacía.")
        return source

    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        if isinstance(source, (bytes, bytearray)):
            if not source:
                raise ValueError("La imagen está vacía.")
            pil = Image.open(io.BytesIO(source))
        else:
            path = Path(source)
            if not path.is_file():
                raise FileNotFoundError(f"No existe la imagen: {path}")
            pil = Image.open(path)
        pil = ImageOps.exif_transpose(pil).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("El archivo no es una imagen válida.") from exc

    rgb = np.asarray(pil)
    if rgb.size == 0:
        raise ValueError("La imagen está vacía.")
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _downscale(img: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    side = max(h, w)
    if side <= MAX_IMAGE_SIDE:
        return img, 1.0
    factor = MAX_IMAGE_SIDE / side
    small = cv2.resize(img, (round(w * factor), round(h * factor)), interpolation=cv2.INTER_AREA)
    return small, 1.0 / factor


# ----------------------------------------------------------------------
# Modelos de embeddings
# ----------------------------------------------------------------------


class Embedder(ABC):
    name: str
    dim: int

    @abstractmethod
    def embed(self, img: np.ndarray, faces: list[DetectedFace]) -> np.ndarray:
        """(n_faces, dim) float32 sin normalizar."""


class ArcFaceEmbedder(Embedder):
    """ArcFace (w600k_r50 de InsightFace buffalo_l), recorte alineado 112x112."""

    name = "arcface"
    dim = 512

    def __init__(self, recognition_model):
        self.model = recognition_model

    def embed(self, img, faces):
        from insightface.utils import face_align

        size = self.model.input_size[0]
        crops = [face_align.norm_crop(img, landmark=f.kps, image_size=size) for f in faces]
        return np.asarray(self.model.get_feat(crops), dtype=np.float32).reshape(len(faces), -1)


class FaceNetEmbedder(Embedder):
    """
    FaceNet (InceptionResnetV1 entrenado en VGGFace2, facenet-pytorch), 160x160.

    Recorte: misma alineación por 5 landmarks que ArcFace (plantilla escalada a
    160 px). Probado en 600 pares LFW: encuadres más amplios (x1.15, x1.3) dan la
    misma exactitud (99,3 %), así que se usa la plantilla tal cual.
    """

    name = "facenet"
    dim = 512
    input_size = 160
    # >1 = encuadre más amplio que la plantilla ArcFace
    crop_scale = 1.0

    def __init__(self):
        import torch
        from facenet_pytorch import InceptionResnetV1

        self.torch = torch
        self.model = InceptionResnetV1(pretrained="vggface2").eval()

    def _crop(self, img, face: DetectedFace) -> np.ndarray:
        from insightface.utils.face_align import arcface_dst

        size = self.input_size
        # Plantilla de landmarks de ArcFace (112 px) llevada a `size` px y
        # contraída hacia el centro según crop_scale (más contexto alrededor).
        dst = ((arcface_dst / 112.0 - 0.5) / self.crop_scale + 0.5) * size
        M, _ = cv2.estimateAffinePartial2D(face.kps.astype(np.float32), dst.astype(np.float32),
                                           method=cv2.LMEDS)
        return cv2.warpAffine(img, M, (size, size), borderValue=0.0)

    def embed(self, img, faces):
        crops = np.stack([self._crop(img, f) for f in faces])          # BGR uint8
        rgb = crops[..., ::-1].astype(np.float32)
        x = self.torch.from_numpy(np.ascontiguousarray(rgb)).permute(0, 3, 1, 2)
        x = (x - 127.5) / 128.0                                         # fixed_image_standardization
        with self.torch.no_grad():
            out = self.model(x)
        return out.numpy().astype(np.float32)


# Registro de modelos: nombre -> fábrica(engine) -> Embedder
EMBEDDERS = {
    "arcface": lambda engine: ArcFaceEmbedder(engine.analysis.models["recognition"]),
    "facenet": lambda engine: FaceNetEmbedder(),
}


# ----------------------------------------------------------------------
# Motor
# ----------------------------------------------------------------------


class FaceEngine:
    def __init__(self, det_size=(640, 640), det_thresh: float = 0.5):
        from insightface.app import FaceAnalysis

        self.analysis = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        self.analysis.prepare(ctx_id=-1, det_size=det_size, det_thresh=det_thresh)
        self._embedders: dict[str, Embedder] = {}
        self._lock = threading.RLock()

    # -- modelos ---------------------------------------------------------

    def embedder(self, model: str) -> Embedder:
        if model not in EMBEDDERS:
            raise ValueError(f"Modelo desconocido: {model!r} (usa {', '.join(EMBEDDERS)})")
        with self._lock:
            if model not in self._embedders:
                self._embedders[model] = EMBEDDERS[model](self)
            return self._embedders[model]

    def warmup(self, models=config.MODELS) -> None:
        for m in models:
            self.embedder(m)

    # -- pipeline --------------------------------------------------------

    def detect(self, img: np.ndarray, max_faces: int = 0) -> list[DetectedFace]:
        """Rostros detectados, del más grande al más pequeño."""
        bboxes, kpss = self.analysis.det_model.detect(img, max_num=0, metric="default")
        faces = [
            DetectedFace(bbox=bboxes[i, :4].astype(np.float32),
                         kps=kpss[i].astype(np.float32),
                         det_score=float(bboxes[i, 4]))
            for i in range(bboxes.shape[0])
        ]
        faces.sort(key=lambda f: f.area, reverse=True)
        return faces[:max_faces] if max_faces else faces

    def embed(self, img: np.ndarray, faces: list[DetectedFace], model: str) -> np.ndarray:
        """(n_faces, dim) embeddings L2-normalizados."""
        if not faces:
            return np.zeros((0, self.embedder(model).dim), dtype=np.float32)
        raw = self.embedder(model).embed(img, faces)
        return np.stack([codec.normalize(v) for v in raw])

    def analyze(self, source, models=config.MODELS,
                max_faces: int = config.MAX_FACES_PER_IMAGE) -> ImageAnalysis:
        """Detecta todas las caras (hasta max_faces) y calcula sus embeddings por modelo."""
        import time

        original = read_image(source)
        h, w = original.shape[:2]
        img, scale = _downscale(original)

        with self._lock:
            started = time.perf_counter()
            faces = self.detect(img, max_faces)
            detect_ms = (time.perf_counter() - started) * 1000

            analyzed = [AnalyzedFace(f) for f in faces]
            embed_ms: dict[str, float] = {}
            for model in models:
                started = time.perf_counter()
                E = self.embed(img, faces, model)
                embed_ms[model] = (time.perf_counter() - started) * 1000
                for a, v in zip(analyzed, E):
                    a.embeddings[model] = v

        return ImageAnalysis(analyzed, (w, h), scale, detect_ms, embed_ms)

    def embed_single(self, source, models=config.MODELS) -> tuple[AnalyzedFace, int, ImageAnalysis]:
        """
        Para registrar una foto: usa la cara más grande.
        Devuelve (cara, nº de caras detectadas, análisis). Lanza NoFaceError si no hay ninguna.
        """
        analysis = self.analyze(source, models, max_faces=0)
        if not analysis.faces:
            raise NoFaceError("No se detectó ningún rostro en la imagen.")
        return analysis.faces[0], len(analysis.faces), analysis


# ----------------------------------------------------------------------
# Singleton
# ----------------------------------------------------------------------

_engine: FaceEngine | None = None
_engine_lock = threading.Lock()


def get_engine() -> FaceEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = FaceEngine()
        return _engine
