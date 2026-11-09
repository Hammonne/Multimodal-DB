# fotos reales de cds y vinilos (discogs) contra las 20k portadas: ¿encuentra la portada digital?
# las fotos salen de scripts/fetch_fotos_reales.py; se usa la base tf-idf normalizada del motor (coseno)
# "principal" = la portada del objeto físico, "extra" = otras fotos (contraportada, disco, librillo)
# uso: python -m benchmark.multimedia.fotos_reales
import json
import os

import cv2
import numpy as np

from indexes.metricas import knn_exacto
from multimedia import features as F
from multimedia.sift import DATA, leer_csv

AQUI = os.path.dirname(os.path.abspath(__file__))
SALIDA = os.path.join(AQUI, "graficas")
K = 10


def parecido_pixel(a, b):
    # correlación de las dos imágenes en gris a 32 px: ~1 = escaneo casi idéntico a la portada digital
    ga = cv2.resize(cv2.imread(a, cv2.IMREAD_GRAYSCALE), (32, 32)).astype(np.float32).ravel()
    gb = cv2.resize(cv2.imread(b, cv2.IMREAD_GRAYSCALE), (32, 32)).astype(np.float32).ravel()
    return float(np.corrcoef(ga, gb)[0, 1])


def evaluar(data=DATA):
    base = F.cargar_base("sift", data=data)
    pos = {str(p): i for i, p in enumerate(base["ids"])}
    filas = [f for f in leer_csv(os.path.join(data, "fotos_reales.csv")) if f["cover_path"] in pos]
    casos = []
    for f in filas:
        foto = os.path.join(data, f["path"])
        if cv2.imread(foto) is None:
            continue
        q = F.ExtractFeatures(foto, data=data)
        ids, _ = knn_exacto(q, base["vectores"], K, "coseno")
        objetivo = pos[f["cover_path"]]
        rank = int(np.flatnonzero(ids == objetivo)[0]) + 1 if objetivo in ids else 0
        casos.append({"tipo": f["tipo"], "formato": f["formato"], "foto": f["path"],
                      "original": f["cover_path"], "rank": rank,
                      "parecido": parecido_pixel(foto, os.path.join(data, f["cover_path"])),
                      "resultados": [str(base["ids"][i]) for i in ids[:5]]})
    return casos


def resumen(casos):
    def tasa(cs):
        return {"n": len(cs), "top1": float(np.mean([c["rank"] == 1 for c in cs])) if cs else 0,
                "top10": float(np.mean([c["rank"] > 0 for c in cs])) if cs else 0}
    p = [c for c in casos if c["tipo"] == "principal"]
    return {"principal": tasa(p),
            "principal_casi_igual": tasa([c for c in p if c["parecido"] >= .9]),
            "principal_distinta": tasa([c for c in p if c["parecido"] < .9]),
            "principal_cd": tasa([c for c in p if c["formato"] == "CD"]),
            "principal_vinilo": tasa([c for c in p if c["formato"] == "Vinyl"]),
            "extra": tasa([c for c in casos if c["tipo"] == "extra"])}


def galeria(casos, archivo, data=DATA, n=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = [c for c in casos if c["tipo"] == "principal" and c["parecido"] < .9]
    filas = [c for c in p if c["rank"] == 1][:n] + [c for c in p if c["rank"] == 0][:n]
    if not filas:
        return
    fig, axes = plt.subplots(len(filas), 6, figsize=(12, 2.5 * len(filas)), squeeze=False)
    for fila, c in zip(axes, filas):
        fila[0].imshow(cv2.cvtColor(cv2.imread(os.path.join(data, c["foto"])), cv2.COLOR_BGR2RGB))
        fila[0].set_title(f"foto real ({c['formato']})\n{'acierto' if c['rank'] else 'error'}", fontsize=8)
        for j, r in enumerate(c["resultados"], 1):
            img = cv2.imread(os.path.join(data, r))
            fila[j].imshow(cv2.cvtColor(cv2.resize(img, (160, 160)), cv2.COLOR_BGR2RGB))
            ok = r == c["original"]
            for s in fila[j].spines.values():
                s.set_visible(True)
                s.set_color("#1baf7a" if ok else "#d03b3b")
                s.set_linewidth(4 if ok else 1.5)
            fila[j].set_title(f"#{j}" + (" ✓ original" if ok else ""), fontsize=8,
                              color="#1baf7a" if ok else "#62615c")
        for ax in fila:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Fotos reales de Discogs contra las 20 005 portadas (coseno); verde = la portada "
                 "digital", fontsize=11, x=.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 1 - .35 / len(filas)), h_pad=1.2)
    fig.savefig(os.path.join(SALIDA, archivo), dpi=110, pil_kwargs={"quality": 82})
    plt.close(fig)


def main():
    casos = evaluar()
    r = resumen(casos)
    with open(os.path.join(SALIDA, "fotos_reales.json"), "w", encoding="utf-8") as f:
        json.dump({"resumen": r, "casos": casos}, f, ensure_ascii=False, indent=1)
    galeria(casos, "galeria_fotos_reales.jpg")
    for k, v in r.items():
        print(f"{k:22s} n={v['n']:4d}  top1 {v['top1']:.2f}  top10 {v['top10']:.2f}")


if __name__ == "__main__":
    main()
