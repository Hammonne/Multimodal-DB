"""
Pipeline completo con modelos reales y fotos LFW (BD temporal):

    registrar 3 personas (3 fotos c/u) -> reconocer una foto NO registrada de cada una
    -> reconocer a alguien que no está registrado (debe salir "unknown")

Se salta si faltan los modelos o LFW.
"""
from __future__ import annotations

import pytest

pytest.importorskip("insightface")
pytest.importorskip("facenet_pytorch")

from face_recognition_app import config  # noqa: E402

LFW = config.DATA_DIR / config.LFW_SUBDIR / "lfw_home" / "lfw_funneled"
if not LFW.is_dir():
    pytest.skip("LFW no descargado", allow_module_level=True)

from face_recognition_app.embeddings import get_engine  # noqa: E402
from face_recognition_app.service import FaceService  # noqa: E402

PEOPLE = ["Gerhard_Schroeder", "Ariel_Sharon", "Hugo_Chavez"]
STRANGER = "Jennifer_Capriati"


def _photos(person, start, n):
    return [(p.name, p.read_bytes()) for p in sorted((LFW / person).glob("*.jpg"))[start:start + n]]


@pytest.fixture(scope="module")
def engine():
    return get_engine()


def test_register_and_recognize_real_faces(temp_db, tmp_path, engine):
    service = FaceService(temp_db.db, engine_factory=lambda: engine, data_dir=tmp_path / "app")
    ids = {}
    for person in PEOPLE:
        res = service.register_person(person.replace("_", " "), _photos(person, 0, 3))
        assert res["person"]["photo_count"] == 3
        ids[person] = res["person"]["person_id"]

    for person in PEOPLE:
        _, query = _photos(person, 5, 1)[0]                       # foto no registrada
        for model in config.MODELS:
            for index in ("exact", "hnsw", "ivf"):
                out = service.recognize(query, model=model, index=index, k=5)
                assert len(out["faces"]) >= 1
                decision = out["faces"][0]["result"]["decision"]
                assert decision["status"] == "known", (person, model, index, decision)
                assert decision["person_id"] == ids[person]

    _, query = _photos(STRANGER, 0, 1)[0]
    for model in config.MODELS:
        out = service.recognize(query, model=model, index="exact")
        assert out["faces"][0]["result"]["decision"]["status"] == "unknown", model

    # comparación de configuraciones sobre la misma consulta
    _, query = _photos(PEOPLE[0], 6, 1)[0]
    out = service.compare(query)
    rows = out["faces"][0]["configs"]
    assert len(rows) == len(config.MODELS) * 3
    assert all(r["decision"]["person_id"] == ids[PEOPLE[0]] for r in rows)
