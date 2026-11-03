"""
Deja solo las fotos útiles para buscar por cara: el artista solo o, en las
bandas, una única foto de grupo; y fotos de cada integrante solo.

Las fotos extra (Wikimedia Commons, TheAudioDB) incluyen conciertos, público,
carteles y logos. Para cada artista o integrante se detectan las caras de todas
sus fotos y se toman como "suyas" las que se repiten en varias fotos (la
persona o los integrantes de la banda aparecen en muchas, los fans no).

  persona / integrante: se acepta una foto si tiene una cara suya de tamaño
      mínimo (--min-face) y ninguna cara de otra persona.
  banda: se acepta una sola foto de grupo: la principal de Deezer si tiene
      caras (si no, es un logo) o, si no, una extra con 2+ caras de la banda y
      ninguna ajena. Las fotos de cada integrante salen de fetch_members.py.
Si ninguna cara se repite, solo se acepta la foto principal (Deezer, o la
primera foto del integrante) si tiene exactamente una cara.

Necesita data/artist_types.csv (fetch_members.py) para saber qué artistas son
bandas; los artistas aún sin clasificar se dejan para la siguiente corrida.

Escribe data/artist_faces.csv y data/member_faces.csv. Con --delete además
borra del disco las fotos rechazadas y las quita de los CSV de imágenes. Es
reanudable: solo procesa artistas e integrantes nuevos.

Requiere face_recognition, numpy y Pillow (por ejemplo, el entorno del Lab 11):
  python scripts/filter_artist_faces.py
  python scripts/filter_artist_faces.py --delete
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

import face_recognition
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_members import MEMBER_FIELDS, MIMAGE_FIELDS  # noqa: E402
from fetch_music import ARTIST_FIELDS, IMAGE_FIELDS, load_csv, write_csv  # noqa: E402

MAX_SIDE = 1000
MAX_FACES = 12


def load(path):
    img = Image.open(path).convert("RGB")
    scale = min(1.0, MAX_SIDE / max(img.size))
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)))
    return np.asarray(img), scale


def faces_in(path, min_other):
    """[(encoding, alto relativo, (top, right, bottom, left))] con la caja en
    coordenadas de la imagen original; None si es una multitud."""
    arr, scale = load(path)
    boxes = [b for b in face_recognition.face_locations(arr, number_of_times_to_upsample=1)
             if (b[2] - b[0]) / arr.shape[0] >= min_other]
    if len(boxes) > MAX_FACES:
        return None
    encs = face_recognition.face_encodings(arr, boxes)
    return [(e, (b[2] - b[0]) / arr.shape[0], tuple(round(v / scale) for v in b))
            for e, b in zip(encs, boxes)]


def process(job):
    rows, data_dir, kind, tol, min_face, min_other = job
    per_image = []
    for r in rows:
        try:
            per_image.append(faces_in(os.path.join(data_dir, r["path"]), min_other))
        except Exception:
            per_image.append([])
    primary = 0 if kind == "integrante" else next(
        (i for i, r in enumerate(rows) if r["source"] == "deezer"), -1)

    # caras grandes y en cuántas otras fotos aparece alguien parecido
    big = [(i, f) for i, fs in enumerate(per_image) for f in fs or [] if f[1] >= min_face]
    support = []
    if big:
        encs = np.array([f[0] for _, f in big])
        for i, f in big:
            near = face_recognition.face_distance(encs, f[0]) < tol
            support.append(len({big[k][0] for k in np.flatnonzero(near)} - {i}))
    top = max(support, default=0)
    if top:
        min_support = max(1, round(0.25 * top))
        anchors = [f[0] for (_, f), s in zip(big, support) if s >= min_support]
    else:
        fs = per_image[primary] if primary >= 0 else None
        anchors = [fs[0][0]] if fs and len(fs) == 1 and fs[0][1] >= min_face else []
    anchors = np.array(anchors)

    result = []
    for fs in per_image:
        mine = []
        clean = False
        if fs and len(anchors):
            is_own = [face_recognition.face_distance(anchors, f[0]).min() < tol for f in fs]
            mine = [f for f, a in zip(fs, is_own) if a and f[1] >= min_face]
            clean = all(is_own)
        result.append((mine, clean))

    keep = [""] * len(rows)
    if kind == "banda":
        # la principal de Deezer es la foto oficial de la banda: basta con que
        # tenga caras (si no, es un logo); si no, una foto de grupo de las extras
        if primary >= 0 and per_image[primary]:
            keep[primary] = "banda"
        else:
            group = [i for i, (mine, clean) in enumerate(result) if clean and len(mine) >= 2]
            if group:
                keep[group[0]] = "banda"
    else:
        for i, (mine, clean) in enumerate(result):
            if mine and clean:
                keep[i] = "cara"

    out = []
    for r, fs, (mine, _), k in zip(rows, per_image, result, keep):
        box = ""
        if k == "cara":
            box = " ".join(map(str, max(mine, key=lambda f: f[1])[2]))
        out.append({"path": r["path"], "source": r["source"],
                    "n_faces": "multitud" if fs is None else len(fs),
                    "keep": bool(k), "kind": k, "face_box": box})
    return rows[0]["_owner"], out


def lock_alive(data_dir, name):
    lock = os.path.join(data_dir, name)
    if not os.path.exists(lock):
        return False
    try:
        os.kill(int(open(lock).read().strip()), 0)
        return True
    except (ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True


def delete_rejected(data_dir, faces_csv, images_csv, image_fields, owners_csv,
                    owner_fields, id_field, main_field=None):
    rejected = {r["path"] for r in load_csv(faces_csv) if r["keep"] != "True"}
    removed = 0
    for p in rejected:
        full = os.path.join(data_dir, p)
        if os.path.exists(full):
            os.remove(full)
            removed += 1
    images = [r for r in load_csv(images_csv) if r["path"] not in rejected]
    count = defaultdict(int)
    for r in images:
        count[r[id_field]] += 1
    owners = load_csv(owners_csv)
    for o in owners:
        o["n_images"] = count[o[id_field]]
        if main_field and o[main_field] in rejected:
            o[main_field] = ""
    write_csv(images_csv, image_fields, images)
    write_csv(owners_csv, owner_fields, owners)
    print(f"{os.path.basename(images_csv)}: borradas {removed} fotos, quedan {len(images)}")


def run_entity(args, pool, name, id_field, images, done, kinds):
    out_csv = os.path.join(args.data, f"{name}_faces.csv")
    fields = [id_field, "path", "source", "n_faces", "keep", "kind", "face_box"]
    groups = defaultdict(list)
    for r in images:
        oid = r[id_field]
        if oid in kinds and oid not in done:
            groups[oid].append(dict(r, _owner=oid))
    jobs = [(rows, args.data, kinds[oid], args.tolerance, args.min_face, args.min_other)
            for oid, rows in groups.items()]
    print(f"{name}: {len(done)} ya procesados, {len(jobs)} por procesar")
    if not jobs:
        return
    new = not os.path.exists(out_csv)
    kept = total = 0
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        for n, (oid, rows) in enumerate(pool.imap_unordered(process, jobs), 1):
            w.writerows([dict(r, **{id_field: oid}) for r in rows])
            f.flush()
            kept += sum(r["keep"] for r in rows)
            total += len(rows)
            if n % 25 == 0 or n == len(jobs):
                print(f"  [{n}/{len(jobs)}] {kept}/{total} fotos aceptadas")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--data", default=os.path.join(repo, "data"))
    ap.add_argument("--tolerance", type=float, default=0.55,
                    help="distancia máxima para considerar la misma persona")
    ap.add_argument("--min-face", type=float, default=0.08,
                    help="alto mínimo de la cara (fracción del alto de la imagen)")
    ap.add_argument("--min-other", type=float, default=0.03,
                    help="caras más pequeñas que esto se ignoran (fondo lejano)")
    ap.add_argument("--delete", action="store_true",
                    help="borrar las fotos rechazadas y quitarlas de los CSV")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    d = args.data

    if args.delete and (lock_alive(d, ".fetch.lock") or lock_alive(d, ".members.lock")):
        sys.exit("fetch_music.py o fetch_members.py está corriendo y reescribe los CSV: "
                 "espera a que terminen antes de usar --delete")

    types = {r["artist_id"]: r["type"] for r in load_csv(os.path.join(d, "artist_types.csv"))}
    artist_kinds = {a: ("banda" if t == "banda" else "persona") for a, t in types.items()}
    member_kinds = {r["member_id"]: "integrante"
                    for r in load_csv(os.path.join(d, "members.csv"))}
    pending = len({a["artist_id"] for a in load_csv(os.path.join(d, "artists.csv"))} - set(types))
    if pending:
        print(f"{pending} artistas sin clasificar: corre fetch_members.py y luego este script")

    with Pool(args.workers) as pool:
        run_entity(args, pool, "artist", "artist_id",
                   load_csv(os.path.join(d, "artist_images.csv")),
                   {r["artist_id"] for r in load_csv(os.path.join(d, "artist_faces.csv"))},
                   artist_kinds)
        run_entity(args, pool, "member", "member_id",
                   load_csv(os.path.join(d, "member_images.csv")),
                   {r["member_id"] for r in load_csv(os.path.join(d, "member_faces.csv"))},
                   member_kinds)

    if args.delete:
        delete_rejected(d, os.path.join(d, "artist_faces.csv"),
                        os.path.join(d, "artist_images.csv"), IMAGE_FIELDS,
                        os.path.join(d, "artists.csv"), ARTIST_FIELDS,
                        "artist_id", "main_image_path")
        if os.path.exists(os.path.join(d, "member_faces.csv")):
            delete_rejected(d, os.path.join(d, "member_faces.csv"),
                            os.path.join(d, "member_images.csv"), MIMAGE_FIELDS,
                            os.path.join(d, "members.csv"), MEMBER_FIELDS, "member_id")


if __name__ == "__main__":
    main()
