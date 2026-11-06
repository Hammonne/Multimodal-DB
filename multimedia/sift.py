# descriptores sift de las imágenes (parte 4), uno de 128 por punto clave
# se guardan en data/features/sift/, más info en multimedia/README.md
# uso: python -m multimedia.sift --source covers|artists|members|all

import argparse
import csv
import os
import time
from multiprocessing import Pool

import cv2
import numpy as np

MAX_SIDE = 512      # se achica la imagen: en 1000+ px SIFT es lento y no aporta más
N_FEATURES = 500    # máximo de puntos clave por imagen (los de mayor contraste)
ROOT_SIFT = True    # RootSIFT: normaliza L1 y saca raíz, mejora la distancia euclidiana

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")
OUT_DIR = os.path.join("features", "sift")
INDEX_FIELDS = ["path", "kind", "owner_id", "n_descriptors", "features_path"]


def leer_gris(path, max_side=MAX_SIDE):
    # en gris (sift no usa color) y con el lado mayor <= max_side
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"no se pudo leer {path}")
    h, w = img.shape
    escala = max_side / max(h, w)
    if escala < 1:
        img = cv2.resize(img, (round(w * escala), round(h * escala)),
                         interpolation=cv2.INTER_AREA)
    return img


def extraer_sift(path, n_features=N_FEATURES, root=ROOT_SIFT):
    # (n, 128), o (0, 128) si no encuentra puntos clave
    sift = cv2.SIFT_create(nfeatures=n_features)
    _, desc = sift.detectAndCompute(leer_gris(path), None)
    if desc is None:
        return np.zeros((0, 128), dtype=np.float32)
    if root:
        desc = desc / (desc.sum(axis=1, keepdims=True) + 1e-7)
        desc = np.sqrt(desc)
    return desc.astype(np.float32)


def leer_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def imagenes(source, data=DATA):
    # [(path, tipo, id del dueño)] de portadas, artistas o integrantes
    # en artistas e integrantes solo las que aceptó el filtro de caras
    out = []
    portadas = leer_csv(os.path.join(data, "covers.csv"))
    if source in ("covers", "all") and portadas:
        # covers.csv: las de las canciones y las de fetch_covers.py, sin las repetidas
        out += [(r["path"], "cover", r["album_id"]) for r in portadas if not r["duplicada_de"]]
    elif source in ("covers", "all"):
        vistos = set()
        for s in leer_csv(os.path.join(data, "songs.csv")):
            if s["cover_path"] and s["cover_path"] not in vistos:
                vistos.add(s["cover_path"])
                out.append((s["cover_path"], "cover", s["album_id"]))
    if source in ("artists", "all"):
        caras = leer_csv(os.path.join(data, "artist_faces.csv"))
        filas = [r for r in caras if r["keep"] == "True"] if caras else \
            leer_csv(os.path.join(data, "artist_images.csv"))
        out += [(r["path"], "artist", r["artist_id"]) for r in filas]
    if source in ("members", "all"):
        out += [(r["path"], "member", r["member_id"])
                for r in leer_csv(os.path.join(data, "member_faces.csv")) if r["keep"] == "True"]
    return out


def ruta_features(path):
    return os.path.join(OUT_DIR, os.path.splitext(path)[0] + ".npy")


def procesar(trabajo):
    path, kind, owner, data = trabajo
    destino = ruta_features(path)
    completo = os.path.join(data, destino)
    if os.path.exists(completo):
        n = len(np.load(completo, mmap_mode="r"))
    else:
        try:
            desc = extraer_sift(os.path.join(data, path))
        except Exception:
            return None
        os.makedirs(os.path.dirname(completo), exist_ok=True)
        np.save(completo, desc)
        n = len(desc)
    return {"path": path, "kind": kind, "owner_id": owner,
            "n_descriptors": n, "features_path": destino}


def main():
    ap = argparse.ArgumentParser(description="descriptores sift de las imágenes (parte 4)")
    ap.add_argument("--source", choices=["covers", "artists", "members", "all"], default="all")
    ap.add_argument("--limit", type=int, default=0, help="procesar solo las primeras N")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    lista = imagenes(args.source, args.data)
    if args.limit:
        lista = lista[:args.limit]
    print(f"{len(lista)} imágenes ({args.source})", flush=True)

    t0 = time.time()
    filas, fallos = [], 0
    with Pool(args.workers) as pool:
        trabajos = [(p, k, o, args.data) for p, k, o in lista]
        for i, r in enumerate(pool.imap(procesar, trabajos, chunksize=8), 1):
            if r is None:
                fallos += 1
            else:
                filas.append(r)
            if i % 500 == 0:
                print(f"  {i}/{len(lista)}", flush=True)

    indice = os.path.join(args.data, "features", f"sift_{args.source}.csv")
    os.makedirs(os.path.dirname(indice), exist_ok=True)
    with open(indice, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        w.writeheader()
        w.writerows(filas)

    n = [r["n_descriptors"] for r in filas]
    print(f"{len(filas)} imágenes en {time.time() - t0:.1f} s, {fallos} fallos; "
          f"descriptores por imagen: media {np.mean(n):.0f}, mín {min(n)}, máx {max(n)}; "
          f"total {sum(n)}\nÍndice: {indice}")


if __name__ == "__main__":
    main()
