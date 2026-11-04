# embeddings de cara (128 dims, resnet de dlib) como en el lab 11
# usa las fotos que dejó el filtro de caras, sale en data/features/faces.npz
# uso: python -m multimedia.face_embeddings

import argparse
import os
import time
from multiprocessing import Pool

import face_recognition
import numpy as np
from PIL import Image

from multimedia.sift import DATA, leer_csv

OUT = os.path.join("features", "faces.npz")
MAX_SIDE = 1000     # a resolución completa (hasta 3000 px) es ~10 veces más lento


def fotos_con_cara(data=DATA):
    # [(path, kind, owner_id, name, caja)]
    nombres = {r["artist_id"]: r["name"] for r in leer_csv(os.path.join(data, "artists.csv"))}
    nombres_m = {r["member_id"]: r["name"] for r in leer_csv(os.path.join(data, "members.csv"))}
    out = []
    for archivo, kind, campo, tabla in (("artist_faces.csv", "artist", "artist_id", nombres),
                                        ("member_faces.csv", "member", "member_id", nombres_m)):
        for r in leer_csv(os.path.join(data, archivo)):
            if r["kind"] == "cara" and r["face_box"]:
                caja = tuple(int(v) for v in r["face_box"].split())
                out.append((r["path"], kind, r[campo], tabla.get(r[campo], ""), caja))
    return out


def embedding(trabajo):
    path, caja, data = trabajo
    try:
        img = Image.open(os.path.join(data, path)).convert("RGB")
        escala = min(1.0, MAX_SIDE / max(img.size))
        if escala < 1:
            img = img.resize((round(img.width * escala), round(img.height * escala)))
            caja = tuple(round(v * escala) for v in caja)
        enc = face_recognition.face_encodings(np.asarray(img), [caja])
        return enc[0].astype(np.float32) if enc else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="embeddings de cara de artistas e integrantes")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    salida = os.path.join(args.data, OUT)

    previo = {"embedding": np.zeros((0, 128), np.float32), "path": [], "kind": [],
              "owner_id": [], "name": []}
    if os.path.exists(salida):
        with np.load(salida) as z:
            previo = {k: z[k] if k == "embedding" else z[k].tolist() for k in previo}
    hechos = set(previo["path"])
    lista = [f for f in fotos_con_cara(args.data) if f[0] not in hechos]
    print(f"{len(hechos)} ya calculados, {len(lista)} por calcular", flush=True)

    t0 = time.time()
    with Pool(args.workers) as pool:
        encs = pool.map(embedding, [(p, caja, args.data) for p, _, _, _, caja in lista],
                        chunksize=8)
    ok = [(f, e) for f, e in zip(lista, encs) if e is not None]

    nuevo = {
        "embedding": np.vstack([previo["embedding"]] + [e[None] for _, e in ok]),
        "path": previo["path"] + [f[0] for f, _ in ok],
        "kind": previo["kind"] + [f[1] for f, _ in ok],
        "owner_id": previo["owner_id"] + [f[2] for f, _ in ok],
        "name": previo["name"] + [f[3] for f, _ in ok],
    }
    os.makedirs(os.path.dirname(salida), exist_ok=True)
    np.savez(salida, **{k: np.asarray(v) for k, v in nuevo.items()})
    print(f"{len(ok)} embeddings nuevos en {time.time() - t0:.1f} s "
          f"({len(lista) - len(ok)} fallos); total {len(nuevo['path'])} en {salida}")


if __name__ == "__main__":
    main()
