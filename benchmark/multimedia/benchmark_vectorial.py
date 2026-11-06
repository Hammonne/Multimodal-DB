# benchmark de la parte 4 (#11): ivf vs hnsw y euclidiana vs coseno, con k = 10
# mide construcción, consulta, recall@10 contra la búsqueda exacta y espacio, para varios tamaños
# también mira si encuentra la portada o canción original de una consulta modificada (galería)
# uso: python -m benchmark.vectorial.benchmark_vectorial [--data data/sim_1k] [--tamanos ...] [--solo-graficas]
# sin --tamanos elige solo según cuánta data haya: con 1k corre 250/500/750, con 10k 1000/2500/5000/10000
import argparse
import json
import os
import tempfile
import time

import cv2
import numpy as np

from indexes.hnsw import HNSW
from indexes.ivf import IVF
from indexes.metricas import knn_exacto
from multimedia import features as F
from multimedia.mfcc import cargar_audio, extraer_mfcc
from multimedia.sift import DATA, extraer_sift, leer_csv

AQUI = os.path.dirname(os.path.abspath(__file__))
SALIDA = os.path.join(AQUI, "graficas")
K, NQ = 10, 200
METRICAS = ["coseno", "euclidiana"]
INDICES = ["ivf", "hnsw"]
NPROBE, M, EFC, EF = 10, 16, 100, 50


# vectores

def objetos(tipo):
    # (id, ruta del archivo, ruta de los descriptores) en el orden de los csv
    if tipo == "sift":
        filas = leer_csv(os.path.join(DATA, "features", "sift_covers.csv"))
        return [(r["path"], r["path"], r["features_path"]) for r in filas]
    filas = leer_csv(os.path.join(DATA, "features", "mfcc.csv"))
    return [(r["track_id"], r["audio_path"], r["features_path"]) for r in filas]


def tfs(tipo, n):
    # histogramas tf de los primeros n objetos, se guardan para no recalcular
    cache = os.path.join(DATA, "features", "benchmark")
    os.makedirs(cache, exist_ok=True)
    ruta = os.path.join(cache, f"tf_{tipo}_k{F.K_DEFECTO}.npz")
    if os.path.exists(ruta):
        z = np.load(ruta)
        if len(z["tf"]) >= n:
            return z["tf"][:n]
    cb = F.cargar_codebook(tipo, data=DATA)
    objs = objetos(tipo)[:n]
    tf = np.zeros((len(objs), F.K_DEFECTO), np.float32)
    for i, (_, _, f) in enumerate(objs):
        tf[i] = F.histograma(F.palabras(np.load(os.path.join(DATA, f)), cb), F.K_DEFECTO)
    np.savez(ruta, tf=tf)
    return tf


def tf_de(desc, cb):
    return F.histograma(F.palabras(desc, cb), F.K_DEFECTO)


# índices

def construir(nombre, metrica, X):
    t0 = time.perf_counter()
    if nombre == "ivf":
        idx = IVF(nlist=max(1, int(round(np.sqrt(len(X))))), metrica=metrica).entrenar(X).agregar(X)
    else:
        idx = HNSW(M=M, ef_construction=EFC, metrica=metrica).agregar(X)
    return idx, time.perf_counter() - t0


def buscar(idx, q, k=K, param=None):
    if isinstance(idx, IVF):
        return idx.buscar(q, k, param or NPROBE)[0]
    return idx.buscar(q, k, param or EF)[0]


def espacio_kb(idx):
    # lo que ocupa guardado en disco: vectores + celdas o grafo
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "idx.npz")
        idx.guardar(p)
        return os.path.getsize(p) / 1024


def medir_consultas(idx, Q, exacto, param=None):
    t0 = time.perf_counter()
    res = [buscar(idx, q, K, param) for q in Q]
    ms = (time.perf_counter() - t0) / len(Q) * 1000
    recall = np.mean([len(set(r) & set(e)) / K for r, e in zip(res, exacto)])
    return ms, recall


# consultas modificadas, con respuesta conocida

def modificar_imagen(path, cambio):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    if cambio == "recortar 70 %":
        dh, dw = int(h * .15), int(w * .15)
        return img[dh:h - dh, dw:w - dw]
    if cambio == "oscurecer 60 %":
        return (img * .4).astype(np.uint8)
    small = cv2.resize(img, (200, 200 * h // w))
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 30])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def consultas_modificadas(tipo, n_base, n, rng):
    # [(cambio, posicion del original, descriptores)]
    objs = objetos(tipo)[:n_base]
    elegidos = rng.choice(n_base, n, replace=False)
    salida = []
    if tipo == "sift":
        with tempfile.TemporaryDirectory() as d:
            for i in elegidos:
                for cambio in ("recortar 70 %", "oscurecer 60 %", "jpeg + 200 px"):
                    p = os.path.join(d, "q.png")
                    cv2.imwrite(p, modificar_imagen(os.path.join(DATA, objs[i][1]), cambio))
                    salida.append((cambio, int(i), extraer_sift(p)))
        return salida
    for i in elegidos:
        ruta = os.path.join(DATA, objs[i][1])
        y = cargar_audio(ruta, 10.0, 5.0)
        salida.append(("fragmento 5 s", int(i), extraer_mfcc(y=y)))
        ruido = rng.normal(0, .02, len(y)).astype(np.float32)
        salida.append(("5 s + volumen y ruido", int(i), extraer_mfcc(y=(y * .5 + ruido))))
    return salida


# experimento

def experimento(tamanos):
    rng = np.random.default_rng(0)
    n_max = max(tamanos)
    filas, curvas, calidad, galeria = [], [], [], {}
    for tipo in ("sift", "mfcc"):
        tf_todo = tfs(tipo, n_max + NQ)
        tf_q = tf_todo[n_max:n_max + NQ]          # consultas que no están en ninguna base
        for n in tamanos:
            tf = tf_todo[:n]
            idf = F.calcular_idf(tf)
            X, Q = F.pesar(tf, idf, normalizar=False), F.pesar(tf_q, idf, normalizar=False)
            for metrica in METRICAS:
                t0 = time.perf_counter()
                exacto = [knn_exacto(q, X, K, metrica)[0] for q in Q]
                ms_lineal = (time.perf_counter() - t0) / NQ * 1000
                filas.append({"tipo": tipo, "n": n, "metrica": metrica, "indice": "lineal",
                              "construir_s": 0.0, "consulta_ms": ms_lineal, "recall10": 1.0,
                              "espacio_kb": X.nbytes / 1024})
                for nombre in INDICES:
                    idx, t_build = construir(nombre, metrica, X)
                    ms, rec = medir_consultas(idx, Q, exacto)
                    filas.append({"tipo": tipo, "n": n, "metrica": metrica, "indice": nombre,
                                  "construir_s": t_build, "consulta_ms": ms, "recall10": rec,
                                  "espacio_kb": espacio_kb(idx)})
                    print(f"{tipo} n={n} {metrica} {nombre}: {t_build:.1f} s, {ms:.2f} ms, "
                          f"recall {rec:.3f}", flush=True)
                    if n == n_max:
                        params = [1, 2, 5, 10, 20, 50] if nombre == "ivf" else [10, 20, 40, 80, 160]
                        for p in params:
                            ms_p, rec_p = medir_consultas(idx, Q, exacto, p)
                            curvas.append({"tipo": tipo, "metrica": metrica, "indice": nombre,
                                           "param": p, "consulta_ms": ms_p, "recall10": rec_p})
                        calidad_y_galeria(tipo, idx, nombre, metrica, idf, n, rng, calidad, galeria)
    return {"filas": filas, "curvas": curvas, "calidad": calidad, "galeria": galeria,
            "tamanos": tamanos, "k": K, "nq": NQ,
            "parametros": {"nprobe": NPROBE, "nlist": "sqrt(n)", "M": M,
                           "ef_construction": EFC, "ef_search": EF}}


_consultas = {}


def calidad_y_galeria(tipo, idx, nombre, metrica, idf, n, rng, calidad, galeria):
    # ¿sale el original entre los 10 primeros? lo mismo para las 4 combinaciones
    if tipo not in _consultas:
        _consultas[tipo] = consultas_modificadas(tipo, n, 100, np.random.default_rng(1))
    cb = F.cargar_codebook(tipo, data=DATA)
    por_cambio = {}
    casos = []
    for cambio, objetivo, desc in _consultas[tipo]:
        q = F.pesar(tf_de(desc, cb), idf, normalizar=False)
        res = buscar(idx, q, K)
        rank = int(np.flatnonzero(res == objetivo)[0]) + 1 if objetivo in res else 0
        por_cambio.setdefault(cambio, []).append(rank > 0)
        casos.append({"cambio": cambio, "objetivo": objetivo, "rank": rank,
                      "resultados": [int(r) for r in res[:5]]})
    for cambio, aciertos in por_cambio.items():
        calidad.append({"tipo": tipo, "indice": nombre, "metrica": metrica, "cambio": cambio,
                        "recall10_original": float(np.mean(aciertos))})
    galeria[f"{tipo}_{nombre}_{metrica}"] = casos


# gráficas (paleta categórica en orden fijo, la misma del codebook)

SERIES = {"lineal": "#62615c", "ivf": "#2a78d6", "hnsw": "#eb6834"}
TRAZO = {"coseno": "-", "euclidiana": "--"}
TINTA, TINTA_2, REJILLA, FONDO = "#1f1f1e", "#62615c", "#e4e3df", "#fcfcfb"
TITULO = {"sift": "Portadas (SIFT)", "mfcc": "Audio (MFCC)"}


def estilo(plt):
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": REJILLA, "axes.labelcolor": TINTA_2,
                         "xtick.color": TINTA_2, "ytick.color": TINTA_2, "axes.grid": True,
                         "grid.color": REJILLA, "grid.linewidth": .6, "figure.facecolor": FONDO,
                         "axes.facecolor": FONDO, "axes.spines.top": False,
                         "axes.spines.right": False, "lines.linewidth": 2, "lines.markersize": 6})


def leyenda_abajo(fig, axes):
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, 1))


def por_tamano(r, campo, archivo, titulo, eje_y, log=False, con_lineal=True):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, tipo in zip(axes, ("sift", "mfcc")):
        for metrica in METRICAS:
            for nombre in (["lineal"] if con_lineal else []) + INDICES:
                pts = [f for f in r["filas"] if f["tipo"] == tipo and f["metrica"] == metrica
                       and f["indice"] == nombre]
                ax.plot([f["n"] for f in pts], [f[campo] for f in pts], TRAZO[metrica],
                        marker="o", color=SERIES[nombre], label=f"{nombre.upper()} · {metrica}")
        ax.set_title(TITULO[tipo], color=TINTA, fontsize=11, loc="left")
        ax.set_xlabel("objetos en la base")
        ax.set_ylabel(eje_y)
        if log:
            from matplotlib.ticker import FuncFormatter
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    fig.suptitle(titulo, color=TINTA, fontsize=12, x=.01, ha="left")
    leyenda_abajo(fig, axes)
    fig.savefig(os.path.join(SALIDA, archivo), dpi=150)
    plt.close(fig)


def curvas(r):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    lineal = {(f["tipo"], f["metrica"]): f["consulta_ms"] for f in r["filas"]
              if f["indice"] == "lineal" and f["n"] == max(r["tamanos"])}
    for ax, tipo in zip(axes, ("sift", "mfcc")):
        for metrica in METRICAS:
            for nombre in INDICES:
                pts = [c for c in r["curvas"] if c["tipo"] == tipo and c["metrica"] == metrica
                       and c["indice"] == nombre]
                ax.plot([c["consulta_ms"] for c in pts], [c["recall10"] for c in pts],
                        TRAZO[metrica], marker="o", color=SERIES[nombre],
                        label=f"{nombre.upper()} · {metrica}")
            ax.axvline(lineal[(tipo, metrica)], color=SERIES["lineal"], ls=TRAZO[metrica], lw=1)
        from matplotlib.ticker import FuncFormatter
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_title(TITULO[tipo], color=TINTA, fontsize=11, loc="left")
        ax.set_xlabel("ms por consulta (log; línea gris = búsqueda lineal)")
        ax.set_ylabel("recall@10")
    fig.suptitle(f"Recall vs tiempo con {max(r['tamanos'])} objetos (nprobe 1–50, ef_search 10–160)",
                 color=TINTA, fontsize=12, x=.01, ha="left")
    leyenda_abajo(fig, axes)
    fig.savefig(os.path.join(SALIDA, "recall_vs_tiempo.png"), dpi=150)
    plt.close(fig)


def barras_calidad(r):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [3, 2]})
    combos = [(i, m) for i in INDICES for m in METRICAS]
    for ax, tipo in zip(axes, ("sift", "mfcc")):
        cambios = list(dict.fromkeys(c["cambio"] for c in r["calidad"] if c["tipo"] == tipo))
        ancho = .8 / len(combos)
        for j, (nombre, metrica) in enumerate(combos):
            vals = [next(c["recall10_original"] for c in r["calidad"] if c["tipo"] == tipo and
                         c["indice"] == nombre and c["metrica"] == metrica and c["cambio"] == ca)
                    for ca in cambios]
            x = np.arange(len(cambios)) + (j - (len(combos) - 1) / 2) * ancho
            barras = ax.bar(x, vals, ancho * .9, color=SERIES[nombre],
                            hatch="" if metrica == "coseno" else "///", edgecolor=FONDO,
                            label=f"{nombre.upper()} · {metrica}")
        ax.set_xticks(np.arange(len(cambios)), cambios, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_ylabel("¿sale el original en el top 10?")
        ax.set_title(TITULO[tipo], color=TINTA, fontsize=11, loc="left")
    fig.suptitle("Encontrar el original de una consulta modificada (100 por cambio)",
                 color=TINTA, fontsize=12, x=.01, ha="left")
    leyenda_abajo(fig, axes)
    fig.savefig(os.path.join(SALIDA, "calidad_original.png"), dpi=150)
    plt.close(fig)


def miniatura(path, lado=160):
    img = cv2.imread(path) if path and os.path.exists(path) else None
    if img is None:
        return np.full((lado, lado, 3), 235, np.uint8)
    return cv2.cvtColor(cv2.resize(img, (lado, lado)), cv2.COLOR_BGR2RGB)


def galeria(r, tipo, clave, archivo, n_casos=3):
    # filas: consulta modificada + top 5; borde verde = es el original, rojo = no
    import matplotlib.pyplot as plt
    objs = objetos(tipo)
    canciones = {}
    if tipo == "mfcc":
        for f in leer_csv(os.path.join(DATA, "songs.csv")):
            canciones[f["track_id"]] = f
    casos = r["galeria"][clave]
    exitos = [c for c in casos if c["rank"] == 1][:n_casos]
    errores = [c for c in casos if c["rank"] == 0][:n_casos]
    filas = [("acierto", c) for c in exitos] + [("error", c) for c in errores]
    if not filas:
        return
    fig, axes = plt.subplots(len(filas), 6, figsize=(12, 2.3 * len(filas)), squeeze=False)
    for fila, (clase, c) in zip(axes, filas):
        def portada_y_texto(pos):
            if tipo == "sift":
                return os.path.join(DATA, objs[pos][1]), os.path.basename(objs[pos][1])
            s = canciones.get(objs[pos][0], {})
            return (os.path.join(DATA, s.get("cover_path", "")),
                    f"{s.get('title', '?')[:18]}\n{s.get('artist', '?')[:18]}")
        ruta, texto = portada_y_texto(c["objetivo"])
        img = miniatura(ruta)
        if tipo == "sift":
            mod = modificar_imagen(ruta, c["cambio"])
            img = cv2.cvtColor(cv2.resize(mod, (160, 160)), cv2.COLOR_BGR2RGB)
        fila[0].imshow(img)
        fila[0].set_title(f"consulta: {c['cambio']}\n{texto}" if tipo == "mfcc"
                          else f"consulta ({clase})\n{c['cambio']}", fontsize=8, color=TINTA)
        for j, pos in enumerate(c["resultados"], 1):
            ruta, texto = portada_y_texto(pos)
            fila[j].imshow(miniatura(ruta))
            ok = pos == c["objetivo"]
            for s in fila[j].spines.values():
                s.set_visible(True)
                s.set_color("#1baf7a" if ok else "#d03b3b")
                s.set_linewidth(4 if ok else 1.5)
            fila[j].set_title(f"#{j}" + (" ✓ original" if ok else "") +
                              (f"\n{texto}" if tipo == "mfcc" else ""), fontsize=8,
                              color="#1baf7a" if ok else TINTA_2)
        for ax in fila:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
    indice, metrica = clave.split("_")[1:]
    fig.suptitle(f"{TITULO[tipo]}: {indice.upper()} con {metrica}, aciertos arriba y errores abajo "
                 f"(verde = el original)", color=TINTA, fontsize=11, x=.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(SALIDA, archivo), dpi=120)
    plt.close(fig)


def tabla_md(r):
    n = max(r["tamanos"])
    lineas = ["| tipo | índice | métrica | construir (s) | consulta (ms) | recall@10 | espacio (KB) |",
              "|---|---|---|---|---|---|---|"]
    for f in r["filas"]:
        if f["n"] == n:
            lineas.append(f"| {f['tipo']} | {f['indice']} | {f['metrica']} | {f['construir_s']:.1f} | "
                          f"{f['consulta_ms']:.2f} | {f['recall10']:.3f} | {f['espacio_kb']:.0f} |")
    return "\n".join(lineas)


def graficar(r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    estilo(plt)
    os.makedirs(SALIDA, exist_ok=True)
    por_tamano(r, "construir_s", "construccion_vs_n.png", "Tiempo de construcción del índice",
               "segundos", con_lineal=False)
    por_tamano(r, "consulta_ms", "consulta_vs_n.png", "Tiempo por consulta (k = 10)",
               "ms (log)", log=True)
    por_tamano(r, "recall10", "recall_vs_n.png", "Recall@10 contra la búsqueda exacta", "recall@10",
               con_lineal=False)
    por_tamano(r, "espacio_kb", "espacio_vs_n.png", "Espacio en disco del índice", "KB (log)",
               log=True)
    curvas(r)
    barras_calidad(r)
    for tipo in ("sift", "mfcc"):
        for nombre in INDICES:
            for metrica in METRICAS:
                galeria(r, tipo, f"{tipo}_{nombre}_{metrica}",
                        f"galeria_{tipo}_{nombre}_{metrica}.png")
    with open(os.path.join(SALIDA, "tabla.md"), "w", encoding="utf-8") as f:
        f.write(tabla_md(r) + "\n")


def tamanos_posibles():
    # lo que alcanza dejando NQ objetos aparte para las consultas
    disponibles = min(len(objetos("sift")), len(objetos("mfcc"))) - NQ
    if disponibles >= 10000:
        return [1000, 2500, 5000, 10000]
    n = disponibles // 250 * 250
    return [t for t in (n // 3 // 250 * 250, 2 * n // 3 // 250 * 250, n) if t > 0]


def main():
    global DATA, NQ, SALIDA
    ap = argparse.ArgumentParser(description="benchmark ivf vs hnsw y euclidiana vs coseno")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--tamanos", type=int, nargs="+")
    ap.add_argument("--salida", default=SALIDA, help="carpeta de las gráficas y resultados.json")
    ap.add_argument("--solo-graficas", action="store_true")
    args = ap.parse_args()
    DATA = os.path.abspath(args.data)
    if min(len(objetos("sift")), len(objetos("mfcc"))) < 2000:
        NQ = 100
    SALIDA = os.path.abspath(args.salida)
    os.makedirs(SALIDA, exist_ok=True)
    ruta = os.path.join(SALIDA, "resultados.json")
    if not args.solo_graficas:
        tamanos = sorted(args.tamanos or tamanos_posibles())
        print(f"data: {DATA}, tamaños: {tamanos}, consultas: {NQ}", flush=True)
        r = experimento(tamanos)
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=1)
    with open(ruta, encoding="utf-8") as f:
        r = json.load(f)
    graficar(r)
    print(f"gráficas en {SALIDA}")


if __name__ == "__main__":
    main()
