"""
Prepara LFW como dataset real de la app.

Pasos (cada uno reanudable):
  1. split   — manifest.json reproducible: galería / consultas conocidas / desconocidas.
  2. embed   — embeddings de todas las fotos del manifest con cada modelo
               (cache en data/lfw/cache_<modelo>.npz).
  3. seed    — registra la galería en chutaDB (Person + FaceImage con embedding),
               copia las fotos a data/photos y construye los índices (coseno).
  4. calibrate — umbral por modelo con las consultas de validación -> calibration.json.

Uso (desde Multimodal-DB, con el backend APAGADO porque comparte los archivos de la BD):
    python -m face_recognition_app.scripts.prepare_lfw --all
    python -m face_recognition_app.scripts.prepare_lfw --split --embed
    python -m face_recognition_app.scripts.prepare_lfw --seed --reset
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from face_recognition_app import config  # noqa: E402

LFW_DIR = config.DATA_DIR / config.LFW_SUBDIR
LFW_IMAGES = LFW_DIR / "lfw_home" / "lfw_funneled"
MANIFEST = LFW_DIR / "manifest.json"


def cache_path(model: str) -> Path:
    return LFW_DIR / f"cache_{model}.npz"


# ----------------------------------------------------------------------
# 1. split
# ----------------------------------------------------------------------


def make_split(min_photos: int, gallery_per_person: int, max_queries_per_person: int,
               n_unknown: int, max_people: int | None, seed: int) -> dict:
    if not LFW_IMAGES.is_dir():
        raise SystemExit(f"No está LFW en {LFW_IMAGES}. Descárgalo primero (ver README).")

    rng = random.Random(seed)
    people = {
        p.name: sorted(f.name for f in p.glob("*.jpg"))
        for p in sorted(LFW_IMAGES.iterdir()) if p.is_dir()
    }

    known = [p for p, files in people.items() if len(files) >= min_photos]
    rng.shuffle(known)
    if max_people:
        known = known[:max_people]
    known.sort()

    others = sorted(p for p, files in people.items() if 2 <= len(files) < min_photos)
    unknown = sorted(rng.sample(others, min(n_unknown, len(others))))

    gallery, queries = [], []
    for person in known:
        files = people[person][:]
        rng.shuffle(files)
        for f in sorted(files[:gallery_per_person]):
            gallery.append({"person": person, "file": f"{person}/{f}"})
        for f in sorted(files[gallery_per_person:gallery_per_person + max_queries_per_person]):
            queries.append({"person": person, "file": f"{person}/{f}", "known": True})
    for person in unknown:
        for f in people[person][:2]:
            queries.append({"person": person, "file": f"{person}/{f}", "known": False})

    # mitad de las identidades de consulta -> validación (calibrar umbral), resto -> test
    ids = sorted({q["person"] for q in queries})
    rng.shuffle(ids)
    val_ids = set(ids[: len(ids) // 2])
    for q in queries:
        q["split"] = "val" if q["person"] in val_ids else "test"

    manifest = {
        "seed": seed,
        "params": {
            "min_photos": min_photos,
            "gallery_per_person": gallery_per_person,
            "max_queries_per_person": max_queries_per_person,
            "n_unknown": n_unknown,
            "max_people": max_people,
        },
        "gallery": gallery,
        "queries": queries,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    n_known_q = sum(q["known"] for q in queries)
    print(f"[split] {len(known)} personas en galería, {len(gallery)} fotos de galería, "
          f"{n_known_q} consultas conocidas, {len(queries) - n_known_q} desconocidas "
          f"({len(unknown)} identidades) -> {MANIFEST}")
    return manifest


def load_manifest() -> dict:
    if not MANIFEST.exists():
        raise SystemExit("Falta manifest.json: corre primero --split.")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
# 2. embed
# ----------------------------------------------------------------------


def _central_face(faces, image_size):
    """En LFW la persona etiquetada es la cara más cercana al centro."""
    w, h = image_size
    c = np.array([w / 2, h / 2])
    return min(faces, key=lambda a: np.linalg.norm(
        (a.face.bbox[:2] + a.face.bbox[2:]) / 2 - c))


def load_cache(model: str) -> dict[str, np.ndarray]:
    path = cache_path(model)
    if not path.exists():
        return {}
    z = np.load(path, allow_pickle=False)
    return {str(f): v for f, v in zip(z["files"], z["vectors"])}


def save_cache(model: str, cache: dict[str, np.ndarray], failed: set[str]) -> None:
    files = sorted(cache)
    vectors = np.stack([cache[f] for f in files]) if files else np.zeros((0, 512), np.float32)
    np.savez(cache_path(model), files=np.array(files), vectors=vectors,
             failed=np.array(sorted(failed)))


def embed_all(models: list[str], det_size: int) -> None:
    from face_recognition_app.embeddings import FaceEngine

    manifest = load_manifest()
    files = sorted({x["file"] for x in manifest["gallery"] + manifest["queries"]})
    caches = {m: load_cache(m) for m in models}
    failed: set[str] = set()
    for m in models:
        if cache_path(m).exists():
            failed |= set(str(f) for f in np.load(cache_path(m))["failed"])

    todo = [f for f in files if f not in failed and any(f not in caches[m] for m in models)]
    print(f"[embed] {len(files)} fotos, {len(todo)} por procesar con {models}", flush=True)
    if not todo:
        return

    engine = FaceEngine(det_size=(det_size, det_size))
    started = time.perf_counter()
    for i, f in enumerate(todo, 1):
        analysis = engine.analyze(LFW_IMAGES / f, models=models, max_faces=0)
        if not analysis.faces:
            failed.add(f)
        else:
            face = _central_face(analysis.faces, analysis.image_size)
            for m in models:
                caches[m][f] = face.embeddings[m]
        if i % 100 == 0 or i == len(todo):
            for m in models:
                save_cache(m, caches[m], failed)
            rate = (time.perf_counter() - started) / i
            print(f"[embed] {i}/{len(todo)}  {rate * 1000:.0f} ms/foto  "
                  f"restan ~{rate * (len(todo) - i) / 60:.1f} min  sin rostro: {len(failed)}",
                  flush=True)


# ----------------------------------------------------------------------
# 3. seed
# ----------------------------------------------------------------------


def open_db():
    from storage.storage_manager import StorageManager

    from face_recognition_app.db import FaceRecognitionDB
    from face_recognition_app.db_schema import create_face_recognition_tables

    storage = StorageManager(wal_path=os.getenv("WAL_PATH", "chuta_wal.log"))
    create_face_recognition_tables(storage)
    return storage, FaceRecognitionDB(storage)


def seed(models: list[str], reset: bool, force: bool, build_indexes: bool) -> None:
    from face_recognition_app.db_schema import reset_face_recognition_tables
    from face_recognition_app.vector_store import VectorStore

    manifest = load_manifest()
    caches = {m: load_cache(m) for m in models}
    storage, db = open_db()
    try:
        own = [p for p in db.list_persons() if p.source == "own"]
        if reset:
            if own and not force:
                raise SystemExit(f"Hay {len(own)} personas propias registradas; usa --force para borrarlas.")
            # drop_table + create_table dejan las tablas nuevas en storage.tables;
            # FaceRecognitionDB las reutiliza. (No abrir una segunda instancia de
            # Table sobre el mismo archivo: el BufferManager compartido cerraría
            # también la vieja y su cabecera desactualizada pisaría la buena.)
            reset_face_recognition_tables(storage)
            from face_recognition_app.db import FaceRecognitionDB
            db = FaceRecognitionDB(storage)
            shutil.rmtree(config.DATA_DIR / config.INDEXES_SUBDIR, ignore_errors=True)
            shutil.rmtree(config.DATA_DIR / config.PHOTOS_SUBDIR, ignore_errors=True)

        existing = {p.name for p in db.list_persons() if p.source == "lfw"}
        by_person: dict[str, list[str]] = {}
        for g in manifest["gallery"]:
            if all(g["file"] in caches[m] for m in models):
                by_person.setdefault(g["person"], []).append(g["file"])

        store = VectorStore(db)
        for m in models:
            store.gallery(m)

        started = time.perf_counter()
        photos_dir = config.DATA_DIR / config.PHOTOS_SUBDIR
        n_faces = 0
        for person_name, files in sorted(by_person.items()):
            display = person_name.replace("_", " ")
            if display in existing:
                continue
            person = db.create_person(display, source="lfw")
            new_faces, new_vecs = [], []
            for f in files:
                photo_id = db.next_photo_id()
                dest = photos_dir / str(person.person_id) / f"{photo_id}.jpg"
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(LFW_IMAGES / f, dest)
                rel = dest.relative_to(config.DATA_DIR).as_posix()
                for m in models:
                    face = db.add_face(person.person_id, photo_id, rel, m, caches[m][f])
                    new_faces.append(face)
                    new_vecs.append(caches[m][f])
                    n_faces += 1
            store.add(new_faces, new_vecs)
        print(f"[seed] {db.count_persons()} personas, {db.count_faces()} filas FaceImage "
              f"(+{n_faces} nuevas en {time.perf_counter() - started:.1f}s)", flush=True)

        if build_indexes:
            for m in models:
                for index_type in ("hnsw", "ivf"):
                    info = store.ensure_index(m, index_type, config.DEFAULT_METRIC)
                    meta = store.index_info(m, index_type, config.DEFAULT_METRIC)
                    print(f"[seed] índice {meta['name']}: {meta['size']} vectores, "
                          f"construido en {meta['build_ms'] / 1000:.1f}s", flush=True)
    finally:
        storage.close()


# ----------------------------------------------------------------------
# 4. calibrate
# ----------------------------------------------------------------------


def best_threshold(top_sims: np.ndarray, correct_top1: np.ndarray, is_known: np.ndarray) -> tuple[float, float]:
    """
    Umbral que maximiza la exactitud de la decisión:
      conocida   -> acierto si top-1 es la persona correcta y sim >= umbral
      desconocida -> acierto si sim < umbral
    """
    best = (0.0, -1.0)
    for th in np.round(np.arange(0.0, 1.0001, 0.01), 2):
        accepted = top_sims >= th
        ok = np.where(is_known, accepted & correct_top1, ~accepted)
        acc = float(ok.mean())
        if acc > best[1]:
            best = (float(th), acc)
    return best


def calibrate(models: list[str]) -> None:
    from face_recognition_app.vector_store import VectorStore

    manifest = load_manifest()
    val = [q for q in manifest["queries"] if q["split"] == "val"]
    storage, db = open_db()
    try:
        store = VectorStore(db)
        result = {}
        for m in models:
            cache = load_cache(m)
            rows = [q for q in val if q["file"] in cache]
            sims, correct, known = [], [], []
            for q in rows:
                res = store.search(m, cache[q["file"]], "exact", "coseno", k=1)
                hit = res.hits[0]
                person = db.get_person(hit.face.person_id)
                sims.append(hit.similarity)
                correct.append(person.name == q["person"].replace("_", " "))
                known.append(q["known"])
            th, acc = best_threshold(np.array(sims), np.array(correct), np.array(known))
            result[m] = {"threshold": th, "val_accuracy": round(acc, 4), "n_val": len(rows)}
            print(f"[calibrate] {m}: umbral={th:.2f}  exactitud(val)={acc:.3f}  n={len(rows)}")
        path = config.DATA_DIR / config.CALIBRATION_FILE
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[calibrate] -> {path}")
    finally:
        storage.close()


# ----------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="split + embed + seed + calibrate")
    ap.add_argument("--split", action="store_true")
    ap.add_argument("--embed", action="store_true")
    ap.add_argument("--seed", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--reset", action="store_true", help="vaciar Person/FaceImage antes de sembrar")
    ap.add_argument("--force", action="store_true", help="permitir --reset aunque haya personas propias")
    ap.add_argument("--no-indexes", action="store_true", help="no construir índices al sembrar")
    ap.add_argument("--models", default=",".join(config.MODELS))
    ap.add_argument("--min-photos", type=int, default=4)
    ap.add_argument("--gallery-per-person", type=int, default=3)
    ap.add_argument("--max-queries-per-person", type=int, default=3)
    ap.add_argument("--unknown", type=int, default=200)
    ap.add_argument("--max-people", type=int, default=None)
    ap.add_argument("--det-size", type=int, default=320,
                    help="resolución del detector (LFW son 250x250: 320 basta)")
    ap.add_argument("--seed-value", type=int, default=42)
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    LFW_DIR.mkdir(parents=True, exist_ok=True)

    if args.all or args.split:
        make_split(args.min_photos, args.gallery_per_person, args.max_queries_per_person,
                   args.unknown, args.max_people, args.seed_value)
    if args.all or args.embed:
        embed_all(models, args.det_size)
    if args.all or args.seed:
        seed(models, args.reset or args.all, args.force, not args.no_indexes)
    if args.all or args.calibrate:
        calibrate(models)


if __name__ == "__main__":
    main()
