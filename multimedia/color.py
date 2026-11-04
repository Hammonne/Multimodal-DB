# histograma de color de las portadas, para lo que sift no ve (color)
# 50 dims en hsv, casi sin brillo para que aguante portadas oscurecidas
# uso: python -m multimedia.color

import argparse
import os
import time
from multiprocessing import Pool

import cv2
import numpy as np

from multimedia.sift import DATA, imagenes

H_BINS, S_BINS, V_BINS = 12, 4, 1
GRIS_BINS = 2
MIN_S, MIN_V = 0.15, 0.04   # debajo de esto el píxel se cuenta como gris
LADO = 256                  # se achica la imagen: el histograma no necesita más
DIM = H_BINS * S_BINS * V_BINS + GRIS_BINS
OUT = os.path.join("features", "color_covers.npz")


def histograma_color(path):
    # vector de DIM con norma 1
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"no se pudo leer {path}")
    img = cv2.resize(img, (LADO, LADO), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    h = hsv[:, 0] / 180.0           # en OpenCV el tono va de 0 a 179
    s = hsv[:, 1] / 255.0
    v = hsv[:, 2] / 255.0

    gris = (s < MIN_S) | (v < MIN_V)
    hi = np.minimum((h * H_BINS).astype(int), H_BINS - 1)
    si = np.minimum(((s - MIN_S) / (1 - MIN_S) * S_BINS).astype(int), S_BINS - 1)
    vi = np.minimum(((v - MIN_V) / (1 - MIN_V) * V_BINS).astype(int), V_BINS - 1)
    gi = np.minimum((v * GRIS_BINS).astype(int), GRIS_BINS - 1)

    bins = np.where(gris, H_BINS * S_BINS * V_BINS + gi,
                    (hi * S_BINS + np.maximum(si, 0)) * V_BINS + np.maximum(vi, 0))
    hist = np.bincount(bins, minlength=DIM).astype(np.float32)
    return np.sqrt(hist / hist.sum())


def procesar(trabajo):
    path, data = trabajo
    try:
        return histograma_color(os.path.join(data, path))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="histograma de color de las portadas")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    lista = imagenes("covers", args.data)
    print(f"{len(lista)} portadas", flush=True)
    t0 = time.time()
    with Pool(args.workers) as pool:
        hists = pool.map(procesar, [(p, args.data) for p, _, _ in lista], chunksize=32)
    ok = [(f, h) for f, h in zip(lista, hists) if h is not None]

    salida = os.path.join(args.data, OUT)
    os.makedirs(os.path.dirname(salida), exist_ok=True)
    np.savez(salida, hist=np.vstack([h for _, h in ok]),
             path=np.array([f[0] for f, _ in ok]), album_id=np.array([f[2] for f, _ in ok]))
    print(f"{len(ok)} histogramas de {DIM} dimensiones en {time.time() - t0:.1f} s "
          f"({len(lista) - len(ok)} fallos) en {salida}")


if __name__ == "__main__":
    main()
