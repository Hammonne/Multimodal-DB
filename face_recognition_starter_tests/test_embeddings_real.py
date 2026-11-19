"""
Fase 1: embeddings reales (ArcFace / FaceNet) con fotos de LFW.

Se salta si no están instalados los modelos o no se descargó LFW
(face_recognition_app/data/lfw).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("insightface")
pytest.importorskip("facenet_pytorch")

from face_recognition_app import config  # noqa: E402

LFW = config.DATA_DIR / config.LFW_SUBDIR / "lfw_home" / "lfw_funneled"
if not LFW.is_dir():
    pytest.skip("LFW no descargado", allow_module_level=True)

from face_recognition_app.embeddings import NoFaceError, get_engine, read_image  # noqa: E402

# Personas con muchas fotos en LFW
SAME = "George_W_Bush"
OTHERS = ["Colin_Powell", "Tony_Blair", "Serena_Williams"]


def _photos(person: str, n: int) -> list[Path]:
    return sorted((LFW / person).glob("*.jpg"))[:n]


@pytest.fixture(scope="module")
def engine():
    return get_engine()


@pytest.mark.parametrize("model", config.MODELS)
def test_embedding_shape_and_norm(engine, model):
    face, n_faces, analysis = engine.embed_single(_photos(SAME, 1)[0], models=[model])
    v = face.embeddings[model]
    assert v.dtype == np.float32 and v.ndim == 1 and v.size == 512
    assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-4)
    assert n_faces >= 1
    assert analysis.image_size == (250, 250)


@pytest.mark.parametrize("model", config.MODELS)
def test_same_person_more_similar_than_others(engine, model):
    same = [engine.embed_single(p, models=[model])[0].embeddings[model] for p in _photos(SAME, 3)]
    others = [engine.embed_single(_photos(o, 1)[0], models=[model])[0].embeddings[model]
              for o in OTHERS]
    sim_same = [same[0] @ same[1], same[0] @ same[2], same[1] @ same[2]]
    sim_diff = [s @ o for s in same for o in others]
    assert min(sim_same) > max(sim_diff)


def test_no_face_raises(engine):
    blank = np.full((200, 200, 3), 127, dtype=np.uint8)
    with pytest.raises(NoFaceError):
        engine.embed_single(blank)


def test_read_image_validation(tmp_path):
    bad = tmp_path / "x.jpg"
    bad.write_bytes(b"no soy una imagen")
    with pytest.raises(ValueError):
        read_image(bad)
    with pytest.raises(FileNotFoundError):
        read_image(tmp_path / "no_existe.jpg")
    with pytest.raises(ValueError):
        read_image(b"")
    # bytes de una imagen válida
    img = read_image(_photos(SAME, 1)[0].read_bytes())
    assert img.shape == (250, 250, 3)
