"""
Evaluación global sobre LFW (consultas del split "test", que no están indexadas
ni se usaron para calibrar el umbral).

Para cada (modelo, índice, métrica):
  - Top-1 / Top-5: la persona correcta es la 1.ª / está entre las 5 primeras personas.
  - Decisión con el umbral calibrado del modelo:
      exactitud = (conocidas aceptadas con la identidad correcta + desconocidas rechazadas) / total
      FAR = desconocidas aceptadas / desconocidas
      FRR = conocidas rechazadas / conocidas
      misid = conocidas aceptadas con otra identidad / conocidas
  - Recall@k vs Exacto: fracción del top-k exacto que devuelve el índice.
  - Latencia de búsqueda (media, p50, p95), candidatos evaluados, construcción y disco.

Las consultas usan los embeddings cacheados por prepare_lfw (no se re-extraen):
se mide la base de datos, no el modelo.

Uso (backend apagado):
    python -m face_recognition_app.evaluation [--k 10] [--limit 300]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from face_recognition_app import config  # noqa: E402


def evaluate(service, queries: list[dict], caches: dict, k: int = 10,
             models=None, indexes=None, metrics=None, progress=None, verbose: bool = True) -> dict:
    """
    queries:  [{"person": nombre_en_BD, "known": bool, "file": clave_cache}]
    caches:   {modelo: {clave: embedding}}
    progress: callback(hechas, total) llamado tras cada configuración.
    """
    store, db = service.store, service.db
    models = models or service.models
    indexes = indexes or list(config.INDEX_TYPES)
    metrics = metrics or list(config.METRICS)
    thresholds = service.thresholds()
    rows = []
    total_configs = len(models) * len(indexes) * len(metrics)

    for model in models:
        qs = [q for q in queries if q["file"] in caches[model]]
        th = thresholds[model]
        for metric in metrics:
            exact_ids = {}
            for q in qs:
                res = store.search(model, caches[model][q["file"]], "exact", metric, k)
                exact_ids[q["file"]] = {h.face.face_id for h in res.hits}

            for index in indexes:
                if index != "exact":
                    store.ensure_index(model, index, metric)
                lat, cand, recall = [], [], []
                top1 = top5 = accepted_ok = rejected_unknown = 0
                far = frr = misid = 0
                n_known = sum(q["known"] for q in qs)
                n_unknown = len(qs) - n_known
                for q in qs:
                    res = store.search(model, caches[model][q["file"]], index, metric, k)
                    lat.append(res.search_ms)
                    cand.append(res.candidates)
                    got = {h.face.face_id for h in res.hits}
                    if exact_ids[q["file"]]:
                        recall.append(len(got & exact_ids[q["file"]]) / len(exact_ids[q["file"]]))

                    # personas ordenadas por su mejor similitud
                    best: dict[int, float] = {}
                    for h in res.hits:
                        best[h.face.person_id] = max(best.get(h.face.person_id, -1.0), h.similarity)
                    ranked = sorted(best.items(), key=lambda x: -x[1])
                    names = [db.get_person(pid).name for pid, _ in ranked]
                    top_sim = ranked[0][1] if ranked else -1.0
                    accepted = top_sim >= th

                    if q["known"]:
                        top1 += bool(names) and names[0] == q["person"]
                        top5 += q["person"] in names[:5]
                        if not accepted:
                            frr += 1
                        elif names[0] == q["person"]:
                            accepted_ok += 1
                        else:
                            misid += 1
                    else:
                        if accepted:
                            far += 1
                        else:
                            rejected_unknown += 1

                info = store.index_info(model, index, metric) if index != "exact" else {}
                rows.append({
                    "model": model,
                    "index": index,
                    "metric": metric,
                    "threshold": th,
                    "n_known": n_known,
                    "n_unknown": n_unknown,
                    "top1": top1 / max(n_known, 1),
                    "top5": top5 / max(n_known, 1),
                    "decision_accuracy": (accepted_ok + rejected_unknown) / max(len(qs), 1),
                    "far": far / max(n_unknown, 1),
                    "frr": frr / max(n_known, 1),
                    "misid": misid / max(n_known, 1),
                    "recall_at_k": float(np.mean(recall)) if recall else None,
                    "latency_mean_ms": float(np.mean(lat)),
                    "latency_p50_ms": float(np.percentile(lat, 50)),
                    "latency_p95_ms": float(np.percentile(lat, 95)),
                    "candidates_mean": float(np.mean(cand)),
                    "build_s": info.get("build_ms", 0) / 1000 if info else None,
                    "disk_mb": info.get("disk_bytes", 0) / 1024 / 1024 if info else None,
                    "params": ({"M": info.get("M"), "ef_construction": info.get("ef_construction"),
                                "ef_search": config.DEFAULT_EF_SEARCH} if index == "hnsw"
                               else {"nlist": info.get("nlist"), "nprobe": config.DEFAULT_NPROBE}
                               if index == "ivf" else {}),
                })
                r = rows[-1]
                if verbose:
                    print(f"{model:8s} {index:6s} {metric:15s} top1={r['top1']:.3f} "
                          f"acc={r['decision_accuracy']:.3f} FAR={r['far']:.3f} FRR={r['frr']:.3f} "
                          f"recall@{k}={r['recall_at_k'] or 0:.3f} lat={r['latency_mean_ms']:.2f}ms", flush=True)
                if progress:
                    progress(len(rows), total_configs)

    first = rows[0] if rows else {}
    return {
        "description": "Consultas LFW del split test (no indexadas, no usadas en la calibración).",
        "n_queries": len(queries),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "k": k,
        "gallery_size": store.gallery(models[0]).size if models else 0,
        "n_known": first.get("n_known", 0),
        "n_unknown": first.get("n_unknown", 0),
        "thresholds": thresholds,
        "rows": rows,
    }


def to_markdown(result: dict) -> str:
    lines = [
        "# Evaluación global (LFW)",
        "",
        f"{result['description']} Fecha: {result['created_at']}. Galería: {result['gallery_size']} fotos por modelo; "
        f"{result['n_known']} consultas conocidas y {result['n_unknown']} desconocidas; k={result['k']}.",
        f"Umbrales calibrados: " + ", ".join(f"{m} = {t:.2f}" for m, t in result["thresholds"].items()) + ".",
        "",
        f"| Modelo | Índice | Métrica | Top-1 | Top-5 | Exactitud decisión | FAR | FRR | Recall@{result['k']} vs exacto "
        "| Latencia media (ms) | p95 (ms) | Candidatos | Construcción (s) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    pct = lambda x: f"{x * 100:.1f}%"
    for r in result["rows"]:
        cells = [
            config.MODEL_LABELS.get(r["model"], r["model"]),
            config.INDEX_LABELS.get(r["index"], r["index"]),
            config.METRIC_LABELS.get(r["metric"], r["metric"]),
            pct(r["top1"]), pct(r["top5"]), pct(r["decision_accuracy"]), pct(r["far"]), pct(r["frr"]),
            pct(r["recall_at_k"]) if r["recall_at_k"] is not None else "–",
            f"{r['latency_mean_ms']:.2f}", f"{r['latency_p95_ms']:.2f}", f"{r['candidates_mean']:.0f}",
            f"{r['build_s']:.1f}" if r["build_s"] is not None else "–",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    from face_recognition_app.scripts.prepare_lfw import load_cache, load_manifest, open_db
    from face_recognition_app.service import FaceService

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None, help="máximo de consultas (para pruebas rápidas)")
    ap.add_argument("--split", default="test", choices=["test", "val", "all"])
    ap.add_argument("--metrics", default=",".join(config.METRICS))
    args = ap.parse_args()

    manifest = load_manifest()
    queries = [
        {"person": q["person"].replace("_", " "), "known": q["known"], "file": q["file"]}
        for q in manifest["queries"] if args.split == "all" or q["split"] == args.split
    ]
    if args.limit:
        queries = queries[: args.limit]

    storage, db = open_db()
    try:
        service = FaceService(db, engine_factory=lambda: None)
        caches = {m: load_cache(m) for m in service.models}
        result = evaluate(service, queries, caches, k=args.k,
                          metrics=[m.strip() for m in args.metrics.split(",") if m.strip()])
    finally:
        storage.close()

    out_dir = config.DATA_DIR / config.EVALUATION_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "last.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out_dir / "last.md").write_text(to_markdown(result), encoding="utf-8")
    print(f"-> {out_dir / 'last.json'}\n-> {out_dir / 'last.md'}")


if __name__ == "__main__":
    main()
