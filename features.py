# extract_features (#5): imagen o audio -> histograma tf-idf de dimensión K con el codebook
# construir_base saca los vectores de toda la colección y el idf, eso es lo que indexan ivf y hnsw
# uso: python -m multimedia.features --tipo sift --k 1024
import argparse
import os
import time

import numpy as np

from multimedia.sift import DATA, extraer_sift, leer_csv

K_DEFECTO = 1024
IMAGEN = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
AUDIO = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}


def tipo_de(path):
    ext = os.path.splitext(str(path))[1].lower()
    if ext in IMAGEN:
        return "sift"
    if ext in AUDIO:
        return "mfcc"
    raise ValueError(f"no sé si {path!r} es imagen o audio")


def cargar_codebook(tipo, k=K_DEFECTO, data=DATA):
    z = np.load(os.path.join(data, "features", "codebook", f"{tipo}_k{k}.npz"))
    return {"centroides": z["centroides"].astype(np.float32), "media": z["media"], "desv": z["desv"]}


def descriptores(path, tipo):
    if tipo == "sift":
        return extraer_sift(path)
    from multimedia.mfcc import extraer_mfcc
    return extraer_mfcc(path)


def palabras(desc, codebook, lote=50_000):
    # cada descriptor a su centroide más cercano
    C = codebook["centroides"]
    if len(desc) == 0:
        return np.zeros(0, dtype=np.int64)
    D = (np.asarray(desc, np.float32) - codebook["media"]) / codebook["desv"]
    cc = (C * C).sum(1)
    out = []
    for i in range(0, len(D), lote):
        B = D[i:i + lote]
        out.append(np.argmin(cc[None] - 2 * B @ C.T, axis=1))
    return np.concatenate(out)


def histograma(etiquetas, k):
    # tf: fracción de descriptores que cae en cada palabra
    h = np.bincount(etiquetas, minlength=k).astype(np.float32)
    return h / max(h.sum(), 1)


def calcular_idf(tfs):
    n = len(tfs)
    df = (tfs > 0).sum(0)
    return np.log(n / np.maximum(df, 1)).astype(np.float32)


def pesar(tf, idf, normalizar=True):
    v = tf * idf
    if normalizar:
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        v = v / np.maximum(n, 1e-12)
    return v.astype(np.float32)


def objetos(tipo, data=DATA):
    # [(id, ruta del .npy de descriptores)] de la colección
    if tipo == "sift":
        return [(r["path"], r["features_path"])
                for r in leer_csv(os.path.join(data, "features", "sift_covers.csv"))]
    return [(r["track_id"], r["features_path"])
            for r in leer_csv(os.path.join(data, "features", "mfcc.csv"))]


def ruta_base(tipo, k, data=DATA):
    return os.path.join(data, "features", f"tfidf_{tipo}_k{k}.npz")


def construir_base(tipo, k=K_DEFECTO, data=DATA):
    cb = cargar_codebook(tipo, k, data)
    objs = objetos(tipo, data)
    tfs = np.zeros((len(objs), k), np.float32)
    for i, (_, f) in enumerate(objs):
        tfs[i] = histograma(palabras(np.load(os.path.join(data, f)), cb), k)
    idf = calcular_idf(tfs)
    np.savez(ruta_base(tipo, k, data), ids=np.array([o[0] for o in objs]), tf=tfs, idf=idf,
             vectores=pesar(tfs, idf), k=k)
    return ruta_base(tipo, k, data)


def cargar_base(tipo, k=K_DEFECTO, data=DATA):
    z = np.load(ruta_base(tipo, k, data))
    return {"ids": z["ids"], "tf": z["tf"], "idf": z["idf"], "vectores": z["vectores"], "k": int(z["k"])}


_cache = {}


def ExtractFeatures(objeto, k=K_DEFECTO, data=DATA, normalizar=True):
    # vector (K,) de una imagen o un audio, con el mismo codebook e idf que la base
    tipo = tipo_de(objeto)
    clave = (tipo, k, os.path.abspath(data))
    if clave not in _cache:
        _cache[clave] = (cargar_codebook(tipo, k, data), cargar_base(tipo, k, data)["idf"])
    cb, idf = _cache[clave]
    tf = histograma(palabras(descriptores(objeto, tipo), cb), k)
    return pesar(tf, idf, normalizar)


def main():
    ap = argparse.ArgumentParser(description="vectores tf-idf de toda la colección")
    ap.add_argument("--tipo", choices=["sift", "mfcc", "ambos"], default="ambos")
    ap.add_argument("--k", type=int, default=K_DEFECTO)
    ap.add_argument("--data", default=DATA)
    args = ap.parse_args()
    for t in (["sift", "mfcc"] if args.tipo == "ambos" else [args.tipo]):
        t0 = time.time()
        ruta = construir_base(t, args.k, args.data)
        b = cargar_base(t, args.k, args.data)
        usadas = (b["tf"] > 0).sum(1)
        print(f"{t}: {len(b['ids'])} vectores de {args.k} dims en {time.time() - t0:.1f} s, "
              f"palabras usadas por objeto: mediana {int(np.median(usadas))} -> {ruta}")


if __name__ == "__main__":
    main()
