"""
Fase 4: servicio + API REST con un motor de IA falso (rápido, sin modelos).

HTTP -> FastAPI -> FaceService -> FaceRecognitionDB / VectorStore -> chutaDB (dir temporal)
"""
from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from face_recognition_app.api import create_face_router
from face_recognition_app.service import FaceService

from conftest import FakeEngine, fake_image


@pytest.fixture
def client(temp_db, tmp_path):
    engine = FakeEngine()
    service = FaceService(temp_db.db, engine_factory=lambda: engine, data_dir=tmp_path / "appdata")
    app = FastAPI()
    app.include_router(create_face_router(temp_db.db, service=service))
    c = TestClient(app)
    c.service = service
    c.engine = engine
    return c


def _files(field: str, *images: bytes):
    return [(field, (f"foto{i}.jpg", img, "image/jpeg")) for i, img in enumerate(images)]


def _register(client, name, identity, n=3):
    images = [fake_image(f"{identity}/{v}") for v in range(n)]
    r = client.post("/api/faces/persons", data={"name": name}, files=_files("photos", *images))
    assert r.status_code == 200, r.text
    return r.json()


def _seed(client):
    people = {}
    for name, ident in [("Ana Torres", "ana"), ("Bruno Díaz", "bruno"), ("Carla Ruiz", "carla"),
                        ("Diego Paz", "diego"), ("Elena Vega", "elena")]:
        people[ident] = _register(client, name, ident)["person"]["person_id"]
    return people


# ----------------------------------------------------------------------


def test_health_and_config(client):
    r = client.get("/api/faces/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    cfg = client.get("/api/faces/config").json()
    assert [m["id"] for m in cfg["models"]] == ["arcface", "facenet"]
    assert [i["id"] for i in cfg["indexes"]] == ["exact", "hnsw", "ivf"]
    assert [m["id"] for m in cfg["metrics"]] == ["coseno", "euclidiana", "producto_punto"]
    assert set(cfg["thresholds"]) >= {"arcface", "facenet"}


def test_register_person_with_multiple_photos(client):
    data = _register(client, "Ana Torres", "ana", n=3)
    person = data["person"]
    assert person["name"] == "Ana Torres" and person["source"] == "own"
    assert person["photo_count"] == 3
    assert [p["status"] for p in data["photos"]] == ["ok"] * 3
    # una fila FaceImage por foto y modelo, con embedding en la tabla
    assert client.service.db.count_faces("arcface") == 3
    assert client.service.db.count_faces("facenet") == 3

    detail = client.get(f"/api/faces/persons/{person['person_id']}").json()
    assert len(detail["photos"]) == 3
    assert set(detail["photos"][0]["models"]) == {"arcface", "facenet"}

    img = client.get(f"/api/faces/images/{detail['photos'][0]['photo_id']}")
    assert img.status_code == 200 and img.content == fake_image("ana/0")


def test_register_reports_bad_photos(client):
    images = [fake_image("ana/0"), fake_image(), b"no es imagen", fake_image("ana/1", "x/1")]
    r = client.post("/api/faces/persons", data={"name": "Ana"}, files=_files("photos", *images))
    assert r.status_code == 200
    statuses = [p["status"] for p in r.json()["photos"]]
    assert statuses == ["ok", "error", "error", "ok"]
    assert "warning" in r.json()["photos"][3]          # 2 rostros: se usó el más grande
    assert r.json()["person"]["photo_count"] == 2

    # si ninguna foto sirve, no se crea la persona
    r = client.post("/api/faces/persons", data={"name": "Nadie"},
                    files=_files("photos", fake_image()))
    assert r.status_code == 400
    assert client.get("/api/faces/persons", params={"q": "nadie"}).json()["total"] == 0


@pytest.mark.parametrize("index", ["exact", "hnsw", "ivf"])
@pytest.mark.parametrize("metric", ["coseno", "euclidiana", "producto_punto"])
def test_recognize_known_person(client, index, metric):
    people = _seed(client)
    r = client.post("/api/faces/recognize",
                    data={"model": "arcface", "index": index, "metric": metric, "k": 5, "nprobe": 10},
                    files={"image": ("q.jpg", fake_image("carla/nueva"), "image/jpeg")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["faces"]) == 1
    result = body["faces"][0]["result"]
    assert result["index"] == index and result["metric"] == metric
    assert result["decision"]["status"] == "known"
    assert result["decision"]["person_id"] == people["carla"]
    assert result["persons"][0]["name"] == "Carla Ruiz"
    assert result["persons"][0]["votes"] == 3            # sus 3 fotos en el top-5
    assert 0 <= result["persons"][0]["similarity_pct"] <= 100
    pcts = [m["similarity_pct"] for m in result["matches"]]
    assert pcts == sorted(pcts, reverse=True)
    assert len(result["matches"]) == 5
    assert "detect_ms" in body["timings"] and "arcface" in body["timings"]["embed_ms"]


def test_recognize_unknown_and_threshold(client):
    _seed(client)
    img = fake_image("desconocido/1")
    r = client.post("/api/faces/recognize", data={"model": "facenet", "index": "exact"},
                    files={"image": ("q.jpg", img, "image/jpeg")})
    assert r.json()["faces"][0]["result"]["decision"]["status"] == "unknown"

    # umbral 0 => cualquier cosa se acepta como la más parecida
    r = client.post("/api/faces/recognize", data={"model": "facenet", "index": "exact", "threshold": 0},
                    files={"image": ("q.jpg", img, "image/jpeg")})
    assert r.json()["faces"][0]["result"]["decision"]["status"] == "known"


def test_recognize_multiple_faces_and_no_face(client):
    people = _seed(client)
    r = client.post("/api/faces/recognize", data={"index": "exact"},
                    files={"image": ("q.jpg", fake_image("ana/9", "bruno/9"), "image/jpeg")})
    faces = r.json()["faces"]
    assert [f["result"]["decision"]["person_id"] for f in faces] == [people["ana"], people["bruno"]]
    assert all(len(f["bbox"]) == 4 for f in faces)

    r = client.post("/api/faces/recognize", files={"image": ("q.jpg", fake_image(), "image/jpeg")})
    assert r.status_code == 200 and r.json()["faces"] == []


def test_compare_configurations(client):
    people = _seed(client)
    img = {"image": ("q.jpg", fake_image("elena/z"), "image/jpeg")}
    r = client.post("/api/faces/compare",
                    data={"models": "arcface,facenet", "indexes": "exact,hnsw,ivf",
                          "metrics": "coseno,euclidiana,producto_punto", "k": 3}, files=img)
    assert r.status_code == 200, r.text
    rows = r.json()["faces"][0]["configs"]
    assert len(rows) == 18                                 # 2 modelos x 3 índices x 3 métricas
    assert {(x["model"], x["index"], x["metric"]) for x in rows} == {
        (m, i, met) for m in ("arcface", "facenet") for i in ("exact", "hnsw", "ivf")
        for met in ("coseno", "euclidiana", "producto_punto")}
    for row in rows:
        assert row["decision"]["person_id"] == people["elena"]
        assert 0 <= row["recall_vs_exact"] <= 1
        assert len(row["in_exact"]) == len(row["matches"])
        assert len(row["in_exact_person"]) == len(row["persons"])
        if row["index"] == "exact":
            assert row["recall_vs_exact"] == 1 and all(row["in_exact"]) and all(row["in_exact_person"])

    # solo una dimensión (comparar índices)
    r = client.post("/api/faces/compare",
                    data={"models": "facenet", "indexes": "exact,ivf", "metrics": "coseno"}, files=img)
    assert [(x["model"], x["index"]) for x in r.json()["faces"][0]["configs"]] == [
        ("facenet", "exact"), ("facenet", "ivf")]
    r = client.post("/api/faces/compare", data={"models": "x"}, files=img)
    assert r.status_code == 400


def test_photo_limit_per_person(client, monkeypatch):
    from face_recognition_app import config

    monkeypatch.setattr(config, "MAX_PHOTOS_PER_PERSON", 3)
    pid = _register(client, "Ana", "ana", n=2)["person"]["person_id"]
    r = client.post(f"/api/faces/persons/{pid}/photos", files=_files("photos", fake_image("ana/8"), fake_image("ana/9")))
    assert r.status_code == 400 and "Máximo 3" in r.json()["detail"]
    r = client.post(f"/api/faces/persons/{pid}/photos", files=_files("photos", fake_image("ana/8")))
    assert r.status_code == 200 and r.json()["person"]["photo_count"] == 3
    r = client.post("/api/faces/persons", data={"name": "Beto"},
                    files=_files("photos", *[fake_image(f"beto/{i}") for i in range(4)]))
    assert r.status_code == 400
    assert client.get("/api/faces/config").json()["max_photos_per_person"] == 3


def test_persons_pagination_and_sort(client):
    _seed(client)
    _register(client, "Zoe Alba", "zoe", n=1)
    r = client.get("/api/faces/persons", params={"limit": 2, "offset": 0, "sort": "name_asc"}).json()
    assert r["total"] == 6 and [p["name"] for p in r["persons"]] == ["Ana Torres", "Bruno Díaz"]
    r = client.get("/api/faces/persons", params={"limit": 2, "offset": 4}).json()
    assert [p["name"] for p in r["persons"]] == ["Elena Vega", "Zoe Alba"]
    r = client.get("/api/faces/persons", params={"sort": "name_desc", "limit": 1}).json()
    assert r["persons"][0]["name"] == "Zoe Alba"
    r = client.get("/api/faces/persons", params={"sort": "photos_desc"}).json()
    assert r["persons"][-1]["name"] == "Zoe Alba"
    r = client.get("/api/faces/persons", params={"sort": "recent", "limit": 1}).json()
    assert r["persons"][0]["name"] == "Zoe Alba"
    assert client.get("/api/faces/persons", params={"sort": "x"}).status_code == 400


def test_health_and_stats(client):
    _seed(client)
    h = client.get("/api/faces/health").json()
    assert h["engine"] in ("ready", "idle") and isinstance(h["tasks"], list)
    s = client.get("/api/faces/stats").json()
    assert s["persons"] == {"total": 5, "own": 5, "lfw": 0}
    assert s["photos"] == 15 and s["face_rows"] == 30
    assert [m["embeddings"] for m in s["models"]] == [15, 15]
    assert len(s["indexes"]) == 12


def test_run_evaluation_from_api(client, tmp_path):
    import time

    import numpy as np

    people = _seed(client)
    engine = client.engine
    lfw = tmp_path / "appdata" / "lfw"
    lfw.mkdir(parents=True)
    queries = [{"person": n, "file": f"{i}/q.jpg", "known": True, "split": "test"}
               for n, i in [("Ana Torres", "ana"), ("Bruno Díaz", "bruno"), ("Carla Ruiz", "carla")]]
    queries += [{"person": "Nadie", "file": "nadie/q.jpg", "known": False, "split": "test"}]
    (lfw / "manifest.json").write_text(json.dumps({"gallery": [], "queries": queries}), encoding="utf-8")
    for m in ("arcface", "facenet"):
        files = [q["file"] for q in queries]
        vecs = [engine._vector(m, f.split("/")[0], "q") for f in files]
        np.savez(lfw / f"cache_{m}.npz", files=np.array(files), vectors=np.stack(vecs), failed=np.array([]))

    r = client.post("/api/faces/evaluation/run", json={"limit": None, "k": 5})
    assert r.status_code == 200, r.text
    for _ in range(100):
        status = client.get("/api/faces/evaluation/status").json()
        if status["state"] != "running":
            break
        time.sleep(0.1)
    assert status["state"] == "done", status
    ev = client.get("/api/faces/evaluation/last").json()
    assert ev["n_queries"] == 4 and len(ev["rows"]) == 18
    assert all(r["top1"] == 1.0 for r in ev["rows"])


def test_verify_photos(client):
    images = [fake_image("ana/1"), fake_image("ana/2"), fake_image("bruno/1")]
    r = client.post("/api/faces/verify", files=_files("images", *images))
    assert r.status_code == 200
    for model, res in r.json()["models"].items():
        m, same = res["similarity_pct"], res["same_person"]
        assert m[0][0] == 100.0
        assert m[0][1] > m[0][2]
        assert same[0][1] is True and same[0][2] is False

    r = client.post("/api/faces/verify", files=_files("images", fake_image("ana/1")))
    assert r.status_code == 400


def test_add_photos_and_delete_person(client):
    people = _seed(client)
    ana = people["ana"]
    # construir índices antes, para comprobar la actualización incremental
    for idx in ("hnsw", "ivf"):
        client.post("/api/faces/recognize", data={"index": idx},
                    files={"image": ("q.jpg", fake_image("ana/x"), "image/jpeg")})

    r = client.post(f"/api/faces/persons/{ana}/photos", files=_files("photos", fake_image("ana/7")))
    assert r.status_code == 200 and r.json()["person"]["photo_count"] == 4
    status = {i["name"]: i for i in client.get("/api/faces/indexes").json()["indexes"]}
    assert status["face_arcface_hnsw_coseno"]["size"] == 16

    r = client.delete(f"/api/faces/persons/{ana}")
    assert r.status_code == 200 and r.json()["faces_removed"] == 8
    r = client.post("/api/faces/recognize", data={"index": "hnsw", "k": 10},
                    files={"image": ("q.jpg", fake_image("ana/x"), "image/jpeg")})
    assert all(m["person_id"] != ana for m in r.json()["faces"][0]["result"]["matches"])
    assert client.get(f"/api/faces/persons/{ana}").status_code == 404
    assert client.delete(f"/api/faces/persons/{ana}").status_code == 404


def test_rebuild_index_and_validation_errors(client):
    _seed(client)
    r = client.post("/api/faces/indexes/rebuild",
                    json={"model": "arcface", "index": "ivf", "metric": "coseno", "nlist": 3})
    assert r.status_code == 200 and r.json()["nlist"] == 3

    img = {"image": ("q.jpg", fake_image("ana/1"), "image/jpeg")}
    assert client.post("/api/faces/recognize", data={"model": "vgg"}, files=img).status_code == 400
    assert client.post("/api/faces/recognize", data={"index": "kd"}, files=img).status_code == 400
    assert client.post("/api/faces/recognize", data={"metric": "l1"}, files=img).status_code == 400
    assert client.post("/api/faces/recognize", data={"k": 0}, files=img).status_code == 400
    assert client.post("/api/faces/recognize", data={"threshold": 2}, files=img).status_code == 400
    assert client.get("/api/faces/evaluation/last").status_code == 404


def test_frontend_is_served(client):
    r = client.get("/api/faces/ui/")
    assert r.status_code == 200 and "Sistema de Reconocimiento Facial" in r.text
    assert client.get("/api/faces/ui/app.js").status_code == 200
    assert client.get("/api/faces/ui/../api.py").status_code == 404
    assert client.get("/api/faces/ui/noexiste.js").status_code == 404


def test_calibrated_threshold_is_used(client, tmp_path):
    (tmp_path / "appdata" / "calibration.json").write_text(
        json.dumps({"arcface": {"threshold": 0.77}}), encoding="utf-8")
    assert client.get("/api/faces/config").json()["thresholds"]["arcface"] == 0.77
