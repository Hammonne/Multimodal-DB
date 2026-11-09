# data augmentation de portadas: rotar, espejo, blanco y negro, otros colores, contraste, desenfoque...
# se toman portadas que no tienen foto real de discogs, se les aplica cada cambio y se busca la original
# entre las 20k: como sift es un descriptor local en gris, el color y la rotación deberían importar poco
# uso: python -m benchmark.multimedia.aumentadas [--n 100]
import argparse
import json
import os

import cv2
import numpy as np

from indexes.metricas import knn_exacto
from multimedia import features as F
from multimedia.sift import DATA, leer_csv

AQUI = os.path.dirname(os.path.abspath(__file__))
SALIDA = os.path.join(AQUI, "graficas")
CARPETA = "consultas_aumentadas"
K = 10


def rotar(img, grados):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), grados, 1)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    W, H = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += W / 2 - w / 2
    M[1, 2] += H / 2 - h / 2
    return cv2.warpAffine(img, M, (W, H), borderValue=(255, 255, 255))


def otros_colores(img):
    # gira el tono 90° en hsv: los rojos pasan a verdes, los azules a rojos...
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 0] = (hsv[..., 0].astype(int) + 45) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


CAMBIOS = {
    "rotar 90°": lambda img: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
    "rotar 25°": lambda img: rotar(img, 25),
    "espejo": lambda img: cv2.flip(img, 1),
    "blanco y negro": lambda img: cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR),
    "otros colores": otros_colores,
    "más contraste y brillo": lambda img: cv2.convertScaleAbs(img, alpha=1.6, beta=30),
    "desenfoque": lambda img: cv2.GaussianBlur(img, (0, 0), 3),
    "recorte 50 % + rotar 10°": lambda img: rotar(img[img.shape[0] // 4:3 * img.shape[0] // 4,
                                                     img.shape[1] // 4:3 * img.shape[1] // 4], 10),
}


def nombre_archivo(cambio):
    return {"rotar 90°": "rot90", "rotar 25°": "rot25", "espejo": "espejo",
            "blanco y negro": "byn", "otros colores": "colores",
            "más contraste y brillo": "contraste", "desenfoque": "desenfoque",
            "recorte 50 % + rotar 10°": "recorte_rot"}[cambio]


def generar(n, data=DATA, seed=0):
    # [(album_id, original, cambio, ruta)] con portadas que no tienen foto real
    con_foto = {f["album_id"] for f in leer_csv(os.path.join(data, "fotos_reales.csv"))}
    portadas = [r for r in leer_csv(os.path.join(data, "covers.csv"))
                if not r["duplicada_de"] and r["album_id"] not in con_foto]
    rng = np.random.default_rng(seed)
    elegidas = [portadas[i] for i in rng.choice(len(portadas), n, replace=False)]
    os.makedirs(os.path.join(data, CARPETA), exist_ok=True)
    filas = []
    for p in elegidas:
        img = cv2.imread(os.path.join(data, p["path"]))
        if img is None:
            continue
        for cambio, f in CAMBIOS.items():
            ruta = os.path.join(CARPETA, f"{p['album_id']}_{nombre_archivo(cambio)}.jpg")
            cv2.imwrite(os.path.join(data, ruta), f(img), [cv2.IMWRITE_JPEG_QUALITY, 90])
            filas.append({"album_id": p["album_id"], "original": p["path"], "album": p["album"],
                          "cambio": cambio, "path": ruta})
    import csv
    with open(os.path.join(data, "consultas_aumentadas.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    return filas


def evaluar(filas, data=DATA):
    base = F.cargar_base("sift", data=data)
    pos = {str(p): i for i, p in enumerate(base["ids"])}
    casos = []
    for f in filas:
        q = F.ExtractFeatures(os.path.join(data, f["path"]), data=data)
        ids, _ = knn_exacto(q, base["vectores"], K, "coseno")
        objetivo = pos[f["original"]]
        rank = int(np.flatnonzero(ids == objetivo)[0]) + 1 if objetivo in ids else 0
        casos.append({**f, "rank": rank, "resultados": [str(base["ids"][i]) for i in ids[:5]]})
    resumen = {}
    for cambio in CAMBIOS:
        cs = [c for c in casos if c["cambio"] == cambio]
        resumen[cambio] = {"n": len(cs), "top1": float(np.mean([c["rank"] == 1 for c in cs])),
                           "top10": float(np.mean([c["rank"] > 0 for c in cs]))}
    return casos, resumen


def grafica(resumen, casos, data=DATA):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 7.5))
    # arriba: una portada con todos sus cambios
    ejemplo = casos[0]["album_id"]
    fila = [c for c in casos if c["album_id"] == ejemplo]
    original = cv2.imread(os.path.join(data, fila[0]["original"]))
    imgs = [("original", original)] + [(c["cambio"], cv2.imread(os.path.join(data, c["path"]))) for c in fila]
    for j, (t, im) in enumerate(imgs):
        ax = fig.add_subplot(3, len(imgs), j + 1)
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(t, fontsize=7.5)
        ax.axis("off")
    # abajo: aciertos por cambio
    ax = fig.add_subplot(3, 1, (2, 3))
    nombres = list(resumen)
    x = np.arange(len(nombres))
    ax.bar(x - .2, [resumen[c]["top1"] for c in nombres], .4, color="#2a78d6", label="la original sale 1.°")
    ax.bar(x + .2, [resumen[c]["top10"] for c in nombres], .4, color="#eb6834", label="en el top 10")
    for xi, c in zip(x, nombres):
        ax.text(xi + .2, resumen[c]["top10"] + .01, f"{resumen[c]['top10']:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, [n.replace(" + ", "\n+ ").replace("y brillo", "\ny brillo") for n in nombres], fontsize=8.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("fracción de consultas")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Data augmentation: {resumen[nombres[0]]['n']} portadas sin foto real, cada una con "
                 f"{len(nombres)} cambios, buscadas entre las 20 005 (SIFT, coseno)", fontsize=11, x=.01,
                 ha="left")
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(os.path.join(SALIDA, "aumentadas.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="data augmentation de portadas")
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    filas = generar(args.n)
    casos, resumen = evaluar(filas)
    with open(os.path.join(SALIDA, "aumentadas.json"), "w", encoding="utf-8") as f:
        json.dump({"resumen": resumen, "casos": casos}, f, ensure_ascii=False, indent=1)
    grafica(resumen, casos)
    for c, v in resumen.items():
        print(f"{c:26s} top1 {v['top1']:.2f}  top10 {v['top10']:.2f}")


if __name__ == "__main__":
    main()
