"""Fase 6: métricas de la evaluación global con datos sintéticos."""
from __future__ import annotations

import numpy as np

from face_recognition_app.evaluation import evaluate, to_markdown
from face_recognition_app.service import FaceService

from conftest import identity_embeddings


def test_evaluate_metrics(temp_db, tmp_path):
    db = temp_db.db
    service = FaceService(db, engine_factory=lambda: None, data_dir=tmp_path / "app")
    E = identity_embeddings(12, 4, dim=48, noise=0.2, seed=5)
    caches = {m: {} for m in service.models}
    queries = []
    for p in range(12):
        name = f"P{p}"
        if p < 10:                                   # 10 registradas, 2 desconocidas
            person = db.create_person(name)
            faces, vecs = [], []
            for j in range(3):
                photo = db.next_photo_id()
                for m in service.models:
                    faces.append(db.add_face(person.person_id, photo, f"{p}_{j}.jpg", m, E[p, j]))
                    vecs.append(E[p, j])
            service.store.add(faces, vecs)
        key = f"{name}/q.jpg"
        for m in service.models:
            caches[m][key] = E[p, 3]
        queries.append({"person": name, "known": p < 10, "file": key})

    result = evaluate(service, queries, caches, k=5, metrics=["coseno"])
    rows = result["rows"]
    assert len(rows) == 2 * 3                         # modelos x índices
    assert result["n_known"] == 10 and result["n_unknown"] == 2
    for r in rows:
        assert r["top1"] == 1.0 and r["top5"] == 1.0
        assert 0 <= r["far"] <= 1 and 0 <= r["frr"] <= 1
        assert r["recall_at_k"] is not None and r["recall_at_k"] > 0.5
        if r["index"] == "exact":
            assert r["recall_at_k"] == 1.0 and r["candidates_mean"] == 30
    md = to_markdown(result)
    assert md.count("\n|") >= 2 + len(rows)
    assert np.isfinite([r["latency_mean_ms"] for r in rows]).all()
