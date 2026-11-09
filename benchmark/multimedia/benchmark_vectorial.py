# benchmark de la parte 4 (#11): ivf vs hnsw y euclidiana vs coseno, con k = 10
# mide construcción, consulta, recall@10 contra la búsqueda exacta y espacio, para varios tamaños
# también mira si encuentra la portada o canción original de una consulta modificada (galería)
# uso: python -m benchmark.multimedia.benchmark_vectorial [--data data/sim_1k] [--tamanos ...] [--solo-graficas]
# sin --tamanos cada tipo elige según cuánta data tenga: 1k -> 250/500/750, 10k -> 1000...10000, 20k portadas -> hasta 20000
import argparse
import json
import os
import tempfile
import time
import zlib

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
# las mismas estructuras en librerías de c++, como referencia (1 hilo, igual que las nuestras)
LIBRERIAS = ["faiss_ivf", "faiss_hnsw", "hnswlib"]
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

class Libreria:
    # faiss y hnswlib detrás de la misma interfaz que IVF y HNSW: buscar(q, k, param) y guardar(p)
    def __init__(self, nombre, metrica, X):
        self.nombre, self.coseno = nombre, metrica == "coseno"
        X = self._prep(X)
        n, d = X.shape
        if nombre == "hnswlib":
            import hnswlib
            self.idx = hnswlib.Index(space="cosine" if self.coseno else "l2", dim=d)
            self.idx.init_index(max_elements=n, ef_construction=EFC, M=M)
            self.idx.set_num_threads(1)
            self.idx.add_items(X)
            return
        import faiss
        faiss.omp_set_num_threads(1)
        tipo = faiss.METRIC_INNER_PRODUCT if self.coseno else faiss.METRIC_L2
        if nombre == "faiss_hnsw":
            self.idx = faiss.IndexHNSWFlat(d, M, tipo)
            self.idx.hnsw.efConstruction = EFC
        else:
            cuant = faiss.IndexFlatIP(d) if self.coseno else faiss.IndexFlatL2(d)
            self.idx = faiss.IndexIVFFlat(cuant, d, max(1, int(round(np.sqrt(n)))), tipo)
            self.cuant = cuant          # que no lo borre el recolector de basura
            self.idx.train(X)
        self.idx.add(X)

    def _prep(self, X):
        # coseno = producto punto entre vectores normalizados
        X = np.ascontiguousarray(np.atleast_2d(X), dtype=np.float32)
        if self.coseno and self.nombre != "hnswlib":
            X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        return X

    def buscar(self, q, k, param=None):
        q = self._prep(q)
        if self.nombre == "hnswlib":
            self.idx.set_ef(max(param or EF, k))
            return self.idx.knn_query(q, k=k)[0][0]
        if self.nombre == "faiss_hnsw":
            self.idx.hnsw.efSearch = max(param or EF, k)
        else:
            self.idx.nprobe = param or NPROBE
        return self.idx.search(q, k)[1][0]

    def guardar(self, p):
        if self.nombre == "hnswlib":
            self.idx.save_index(p)
        else:
            import faiss
            faiss.write_index(self.idx, p)


def construir(nombre, metrica, X):
    t0 = time.perf_counter()
    if nombre in LIBRERIAS:
        idx = Libreria(nombre, metrica, X)
    elif nombre == "ivf":
        idx = IVF(nlist=max(1, int(round(np.sqrt(len(X))))), metrica=metrica).entrenar(X).agregar(X)
    else:
        idx = HNSW(M=M, ef_construction=EFC, metrica=metrica).agregar(X)
    return idx, time.perf_counter() - t0


def buscar(idx, q, k=K, param=None):
    if isinstance(idx, Libreria):
        return idx.buscar(q, k, param)
    if isinstance(idx, IVF):
        return idx.buscar(q, k, param or NPROBE)[0]
    return idx.buscar(q, k, param or EF)[0]


def espacio_kb(idx):
    # lo que ocupa guardado en disco: vectores + celdas o grafo
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "idx.bin" if isinstance(idx, Libreria) else "idx.npz")
        idx.guardar(p)
        return os.path.getsize(p) / 1024


def medir_consultas(idx, Q, exacto, param=None):
    t0 = time.perf_counter()
    res = [buscar(idx, q, K, param) for q in Q]
    ms = (time.perf_counter() - t0) / len(Q) * 1000
    recall = np.mean([len(set(r) & set(e)) / K for r, e in zip(res, exacto)])
    return ms, recall


# consultas modificadas, con respuesta conocida

CAMBIOS_IMAGEN = ("recortar 70 %", "oscurecer 60 %", "jpeg + 200 px", "foto simulada")

def foto_simulada(img, rng):
    # como si alguien le tomara una foto al cd o al póster: perspectiva, fondo, luz y cámara
    h, w = img.shape[:2]
    lado = 640
    img = cv2.resize(img, (lado, lado * h // w))
    h, w = img.shape[:2]
    fondo = np.zeros((int(h * 1.5), int(w * 1.5), 3), np.uint8)
    fondo[:] = rng.integers(40, 200, 3)
    fondo = cv2.add(fondo, rng.normal(0, 12, fondo.shape).clip(-30, 30).astype(np.int16).astype(np.uint8))
    H, W = fondo.shape[:2]
    ox, oy = (W - w) / 2, (H - h) / 2
    origen = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    destino = np.float32([[ox, oy], [ox + w, oy], [ox + w, oy + h], [ox, oy + h]])
    destino += rng.uniform(-.12, .12, (4, 2)).astype(np.float32) * [w, h]
    T = cv2.getPerspectiveTransform(origen, destino)
    foto = cv2.warpPerspective(img, T, (W, H), dst=fondo.copy(), borderMode=cv2.BORDER_TRANSPARENT)
    foto = foto.astype(np.float32)
    foto *= rng.uniform(.8, 1.2, 3)                 # tinte de color (luz cálida o fría)
    foto = foto * rng.uniform(.7, 1.1) + rng.uniform(-20, 20)
    yy, xx = np.mgrid[0:H, 0:W]                     # reflejo de luz en un punto
    cx, cy, r = rng.uniform(.2, .8) * W, rng.uniform(.2, .8) * H, rng.uniform(.15, .35) * W
    foto += 90 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2))[..., None]
    foto = cv2.GaussianBlur(foto.clip(0, 255).astype(np.uint8), (0, 0), rng.uniform(.8, 2))
    foto = (foto + rng.normal(0, 6, foto.shape)).clip(0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", foto, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def modificar_imagen(path, cambio):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    if cambio == "foto simulada":
        # misma foto para la misma portada, así la galería muestra la consulta que se usó
        return foto_simulada(img, np.random.default_rng(zlib.crc32(path.encode())))
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
                for cambio in CAMBIOS_IMAGEN:
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
        salida.append(("5 s grabado con celular", int(i), extraer_mfcc(y=grabado_celular(y, rng))))
    return salida


def grabado_celular(y, rng, sr=22050):
    # parlante y micrófono de celular: solo 300–3400 Hz, eco de la habitación y ruido de fondo
    from scipy.signal import butter, sosfilt
    sos = butter(4, [300, 3400], btype="band", fs=sr, output="sos")
    z = sosfilt(sos, y).astype(np.float32)
    eco = np.zeros_like(z)
    retraso = int(sr * rng.uniform(.03, .08))
    eco[retraso:] = z[:-retraso] * .4
    z = z + eco
    z = z / max(np.abs(z).max(), 1e-6) * .5
    return (z + rng.normal(0, .015, len(z))).astype(np.float32)


# experimento

def experimento(tamanos_por_tipo):
    rng = np.random.default_rng(0)
    filas, curvas, calidad, galeria = [], [], [], {}
    for tipo in ("sift", "mfcc"):
        tamanos = tamanos_por_tipo[tipo]
        n_max = max(tamanos)
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
                for nombre in INDICES + LIBRERIAS:
                    idx, t_build = construir(nombre, metrica, X)
                    ms, rec = medir_consultas(idx, Q, exacto)
                    filas.append({"tipo": tipo, "n": n, "metrica": metrica, "indice": nombre,
                                  "construir_s": t_build, "consulta_ms": ms, "recall10": rec,
                                  "espacio_kb": espacio_kb(idx)})
                    print(f"{tipo} n={n} {metrica} {nombre}: {t_build:.1f} s, {ms:.2f} ms, "
                          f"recall {rec:.3f}", flush=True)
                    if n == n_max and nombre in INDICES:
                        params = [1, 2, 5, 10, 20, 50] if nombre == "ivf" else [10, 20, 40, 80, 160]
                        for p in params:
                            ms_p, rec_p = medir_consultas(idx, Q, exacto, p)
                            curvas.append({"tipo": tipo, "metrica": metrica, "indice": nombre,
                                           "param": p, "consulta_ms": ms_p, "recall10": rec_p})
                        calidad_y_galeria(tipo, idx, nombre, metrica, idf, n, rng, calidad, galeria)
    return {"filas": filas, "curvas": curvas, "calidad": calidad, "galeria": galeria,
            "tamanos": tamanos_por_tipo, "k": K, "nq": NQ,
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
NOMBRE = {"lineal": "Búsqueda lineal (sin índice)", "ivf": "IVF", "hnsw": "HNSW"}
MARCA = {"lineal": "s", "ivf": "o", "hnsw": "D"}


def estilo(plt):
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": REJILLA, "axes.labelcolor": TINTA_2,
                         "xtick.color": TINTA_2, "ytick.color": TINTA_2, "axes.grid": True,
                         "grid.color": REJILLA, "grid.linewidth": .6, "figure.facecolor": FONDO,
                         "axes.facecolor": FONDO, "axes.spines.top": False,
                         "axes.spines.right": False, "lines.linewidth": 2, "lines.markersize": 6,
                         "legend.handlelength": 3.5})


def encabezado(fig, titulo, bajada):
    # título = la conclusión, bajada = qué se está midiendo
    fig.text(.01, .985, titulo, color=TINTA, fontsize=13, fontweight="bold", ha="left", va="top")
    fig.text(.01, .925, bajada, color=TINTA_2, fontsize=9.5, ha="left", va="top")


def leyenda_abajo(fig, handles, labels, arriba=.84):
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), frameon=False,
               fontsize=9)
    fig.tight_layout(rect=(0, .1 if len(labels) <= 4 else .14, 1, arriba))


def eje_log(ax, eje="y"):
    from matplotlib.ticker import FuncFormatter
    getattr(ax, f"set_{eje}scale")("log")
    fmt = getattr(ax, f"{eje}axis")
    fmt.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    fmt.set_minor_formatter(FuncFormatter(lambda v, _: ""))


def puntos(r, tipo, metrica, nombre):
    return [f for f in r["filas"] if f["tipo"] == tipo and f["metrica"] == metrica
            and f["indice"] == nombre]


def en_maximo(r, tipo, metrica, nombre, campo):
    return next(f[campo] for f in puntos(r, tipo, metrica, nombre) if f["n"] == maximo(r, tipo))


def linea(ax, r, tipo, nombre, campo, metrica="coseno", **kw):
    pts = puntos(r, tipo, metrica, nombre)
    return ax.plot([f["n"] for f in pts], [f[campo] for f in pts], TRAZO[metrica],
                   marker=MARCA[nombre], color=SERIES[nombre], label=NOMBRE[nombre], **kw)[0]


def fig_consulta(r, plt):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    veces = []
    for ax, tipo in zip(axes, ("sift", "mfcc")):
        for nombre in ["lineal"] + INDICES:
            linea(ax, r, tipo, nombre, "consulta_ms")
        lin = en_maximo(r, tipo, "coseno", "lineal", "consulta_ms")
        texto = [f"con N = {maximo(r, tipo):,}".replace(",", " ")]
        for nombre in INDICES:
            ms = en_maximo(r, tipo, "coseno", nombre, "consulta_ms")
            veces.append(lin / ms)
            texto.append(f"{NOMBRE[nombre]}: {lin / ms:.1f}× más rápido que la lineal")
        ax.text(.03, .96, "\n".join(texto), transform=ax.transAxes, va="top", fontsize=8.5,
                color=TINTA, bbox={"boxstyle": "round,pad=.4", "fc": "white", "ec": REJILLA})
        ax.set_title(TITULO[tipo], color=TINTA, fontsize=11, loc="left")
        ax.set_xlabel("objetos en la base (N)")
        ax.set_ylabel("ms por consulta (escala log)")
        eje_log(ax)
    encabezado(fig, f"Con índice, cada consulta es {min(veces):.0f}–{max(veces):.0f} veces más rápida "
                    "que revisar toda la base",
               "Tiempo promedio de una consulta k-NN (k = 10, coseno). La búsqueda lineal compara con "
               "los N objetos y crece con N;\nIVF revisa solo nprobe = 10 celdas y HNSW unos cientos "
               "de nodos del grafo, por eso crecen mucho menos. Con menos de ~2 500 objetos la lineal "
               "todavía gana.")
    h, l = axes[0].get_legend_handles_labels()
    leyenda_abajo(fig, h, l, arriba=.82)
    return fig


def fig_recall(r, plt):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    minimo = 1
    for ax, tipo in zip(axes, ("sift", "mfcc")):
        linea(ax, r, tipo, "lineal", "recall10")
        for nombre in INDICES:
            linea(ax, r, tipo, nombre, "recall10")
            minimo = min(minimo, min(f["recall10"] for f in puntos(r, tipo, "coseno", nombre)))
        ax.set_ylim(.85, 1.01)
        ax.set_title(TITULO[tipo], color=TINTA, fontsize=11, loc="left")
        ax.set_xlabel("objetos en la base (N)")
        ax.set_ylabel("recall@10")
    encabezado(fig, f"Los índices devuelven entre {minimo * 100:.0f} % y 99 % de los vecinos correctos",
               "Recall@10 = de los 10 vecinos que da la búsqueda lineal (la respuesta exacta), cuántos "
               "devuelve el índice. La lineal\nsiempre tiene 1.0; IVF y HNSW son aproximados y a cambio "
               "son más rápidos.")
    h, l = axes[0].get_legend_handles_labels()
    leyenda_abajo(fig, h, l)
    return fig


def fig_curvas(r, plt):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, tipo in zip(axes, ("sift", "mfcc")):
        for nombre in INDICES:
            pts = [c for c in r["curvas"] if c["tipo"] == tipo and c["metrica"] == "coseno"
                   and c["indice"] == nombre]
            ax.plot([c["consulta_ms"] for c in pts], [c["recall10"] for c in pts], "-",
                    marker=MARCA[nombre], color=SERIES[nombre], label=NOMBRE[nombre])
            for c, xy, ha in ((pts[0], (6, -4), "left"), (pts[-1], (-4, -14), "right")):
                ax.annotate(("nprobe " if nombre == "ivf" else "ef ") + str(c["param"]),
                            (c["consulta_ms"], c["recall10"]), xytext=xy, ha=ha,
                            textcoords="offset points", fontsize=7.5, color=SERIES[nombre])
        lin = en_maximo(r, tipo, "coseno", "lineal", "consulta_ms")
        ax.plot([lin], [1.0], MARCA["lineal"], ms=9, color=SERIES["lineal"], label=NOMBRE["lineal"])
        ax.axvline(lin, color=SERIES["lineal"], lw=1, ls=":")
        eje_log(ax, "x")
        ax.set_title(f"{TITULO[tipo]}, N = {maximo(r, tipo):,}".replace(",", " "), color=TINTA,
                     fontsize=11, loc="left")
        ax.set_xlabel("ms por consulta (escala log)")
        ax.set_ylabel("recall@10")
    encabezado(fig, "Todo lo que queda a la izquierda de la línea gris es más rápido que la lineal",
               "Cada punto es una configuración: IVF con nprobe 1–50 celdas abiertas, HNSW con "
               "ef_search 10–160 candidatos. Más arriba =\nmás preciso, más a la izquierda = más "
               "rápido. HNSW llega a ~0.99 sin acercarse al tiempo de la búsqueda lineal.")
    h, l = axes[0].get_legend_handles_labels()
    leyenda_abajo(fig, h, l, arriba=.82)
    return fig


def fig_construccion(r, plt):
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, tipo in zip(axes, ("sift", "mfcc")):
        for nombre in INDICES:
            linea(ax, r, tipo, nombre, "construir_s")
        ax.set_title(TITULO[tipo], color=TINTA, fontsize=11, loc="left")
        ax.set_xlabel("objetos en la base (N)")
        ax.set_ylabel("segundos (escala log)")
        eje_log(ax)
    t = {n: en_maximo(r, "sift", "coseno", n, "construir_s") for n in INDICES}
    encabezado(fig, f"El precio: HNSW tarda {t['hnsw'] / t['ivf']:.0f} veces más que IVF en construirse",
               "Tiempo para armar el índice desde cero (coseno). IVF corre K-Means con √N centroides; "
               "HNSW inserta los nodos uno por uno\nbuscando vecinos en el grafo (en Python puro). "
               "La búsqueda lineal no construye nada.")
    h, l = axes[0].get_legend_handles_labels()
    h.insert(0, Line2D([], [], color=SERIES["lineal"], marker=MARCA["lineal"], ls="none"))
    l.insert(0, "Búsqueda lineal: 0 s, no tiene índice")
    leyenda_abajo(fig, h, l)
    return fig


def fig_metricas(r, plt):
    # color = estructura, trazo = métrica; la leyenda las separa para que se lean las dos cosas
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for fila, tipo in zip(axes, ("sift", "mfcc")):
        for metrica in METRICAS:
            for nombre in ["lineal"] + INDICES:
                linea(fila[0], r, tipo, nombre, "consulta_ms", metrica)
            for nombre in INDICES:
                linea(fila[1], r, tipo, nombre, "recall10", metrica)
        eje_log(fila[0])
        fila[0].set_ylabel("ms por consulta (log)")
        fila[1].set_ylabel("recall@10")
        for ax in fila:
            ax.set_title(TITULO[tipo], color=TINTA, fontsize=11, loc="left")
            ax.set_xlabel("objetos en la base (N)")
    encabezado(fig, "Euclidiana sobre vectores sin normalizar: más lenta en el IVF y con peor recall",
               "Los mismos índices con coseno (línea continua) y euclidiana (línea punteada). Con "
               "euclidiana el K-Means del IVF agrupa por\ntamaño del vector y unas pocas celdas "
               "concentran casi toda la base.")
    h = [Line2D([], [], color=SERIES[n], marker=MARCA[n], lw=3) for n in ["lineal"] + INDICES]
    h += [Line2D([], [], color=TINTA_2, ls=TRAZO[m], lw=2) for m in METRICAS]
    l = [NOMBRE[n] for n in ["lineal"] + INDICES] + ["coseno (continua)", "euclidiana (punteada)"]
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .89))
    return fig


def fig_espacio(r, plt):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, tipo in zip(axes, ("sift", "mfcc")):
        for nombre in ["lineal"] + INDICES:
            pts = puntos(r, tipo, "coseno", nombre)
            ax.plot([f["n"] for f in pts], [f["espacio_kb"] / 1024 for f in pts], "-",
                    marker=MARCA[nombre], color=SERIES[nombre],
                    label="Búsqueda lineal (solo los vectores)" if nombre == "lineal" else NOMBRE[nombre])
        ax.set_title(TITULO[tipo], color=TINTA, fontsize=11, loc="left")
        ax.set_xlabel("objetos en la base (N)")
        ax.set_ylabel("MB en disco")
    encabezado(fig, "El espacio lo ponen los vectores: los índices agregan muy poco",
               "Tamaño guardado en disco. Cada objeto es un vector de 1 024 floats (4 KB); las celdas "
               "del IVF y el grafo del HNSW\nsuman unos pocos MB encima.")
    h, l = axes[0].get_legend_handles_labels()
    leyenda_abajo(fig, h, l)
    return fig


COLOR_FAMILIA = {"lineal": SERIES["lineal"], "ivf": SERIES["ivf"], "faiss_ivf": SERIES["ivf"],
                 "hnsw": SERIES["hnsw"], "faiss_hnsw": SERIES["hnsw"], "hnswlib": SERIES["hnsw"]}
TRAZO_IMPL = {"lineal": "-", "ivf": "-", "hnsw": "-", "faiss_ivf": "--", "faiss_hnsw": "--",
              "hnswlib": ":"}
MARCA_IMPL = {"lineal": "s", "ivf": "o", "hnsw": "D", "faiss_ivf": "o", "faiss_hnsw": "D",
              "hnswlib": "^"}
NOMBRE_IMPL = {"lineal": "Búsqueda lineal", "ivf": "IVF nuestro", "hnsw": "HNSW nuestro",
               "faiss_ivf": "Faiss IVFFlat", "faiss_hnsw": "Faiss HNSWFlat", "hnswlib": "hnswlib"}


def fig_librerias(r, plt):
    # solo coseno: color = estructura (ivf azul, hnsw naranja), trazo = implementación
    if not any(f["indice"] in LIBRERIAS for f in r["filas"]):
        return None
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    campos = [("construir_s", "construir (s, log)"), ("consulta_ms", "ms por consulta (log)"),
              ("recall10", "recall@10")]
    for fila, tipo in zip(axes, ("sift", "mfcc")):
        for ax, (campo, eje) in zip(fila, campos):
            nombres = INDICES + LIBRERIAS
            if campo == "consulta_ms":
                nombres = ["lineal"] + nombres
            for nombre in nombres:
                pts = puntos(r, tipo, "coseno", nombre)
                ax.plot([f["n"] for f in pts], [max(f[campo], 1e-3) for f in pts], TRAZO_IMPL[nombre],
                        marker=MARCA_IMPL[nombre], color=COLOR_FAMILIA[nombre],
                        label=NOMBRE_IMPL[nombre], lw=2.5 if nombre in ("ivf", "hnsw") else 1.8)
            if campo != "recall10":
                eje_log(ax)
            ax.set_ylabel(eje)
            ax.set_xlabel("objetos en la base (N)")
            ax.set_title(TITULO[tipo], color=TINTA, fontsize=10, loc="left")
    encabezado(fig, "Mismo recall que Faiss y hnswlib; la diferencia de tiempo es Python vs C++",
               "Coseno, 1 hilo, mismos M, ef, nlist y nprobe. Color = estructura (azul IVF, naranja "
               "HNSW); trazo = implementación (continua\nnuestra, rayada Faiss, punteada hnswlib).")
    h, l = axes[0][1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .89))
    return fig


def fig_calidad(r, plt):
    from matplotlib.patches import Patch
    cambios = {t: list(dict.fromkeys(c["cambio"] for c in r["calidad"] if c["tipo"] == t))
               for t in ("sift", "mfcc")}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8),
                             gridspec_kw={"width_ratios": [len(cambios["sift"]), len(cambios["mfcc"])]})
    combos = [(i, m) for i in INDICES for m in METRICAS]
    for ax, tipo in zip(axes, ("sift", "mfcc")):
        ancho = .8 / len(combos)
        for j, (nombre, metrica) in enumerate(combos):
            vals = [next(c["recall10_original"] for c in r["calidad"] if c["tipo"] == tipo and
                         c["indice"] == nombre and c["metrica"] == metrica and c["cambio"] == ca)
                    for ca in cambios[tipo]]
            x = np.arange(len(cambios[tipo])) + (j - (len(combos) - 1) / 2) * ancho
            ax.bar(x, vals, ancho * .9, color=SERIES[nombre],
                   hatch="" if metrica == "coseno" else "///", edgecolor=FONDO)
            for xi, v in zip(x, vals):
                if v == 0:          # sin esto una barra en cero no se ve
                    ax.text(xi, .01, "0", ha="center", va="bottom", fontsize=8, color=SERIES[nombre])
        ax.set_xticks(np.arange(len(cambios[tipo])),
                      [c.replace(" + ", "\n+ ").replace("grabado", "\ngrabado") for c in cambios[tipo]],
                      fontsize=8.5)
        ax.set_ylim(0, 1)
        ax.set_ylabel("¿sale el original en el top 10?")
        ax.set_title(TITULO[tipo], color=TINTA, fontsize=11, loc="left")
    encabezado(fig, "Encontrar el original a partir de una copia modificada",
               "100 consultas por cambio. Una barra alta = el índice encuentra la portada o canción "
               "original aunque la consulta esté\nrecortada, oscura, fotografiada o grabada con "
               "ruido.")
    h = [Patch(color=SERIES[n]) for n in INDICES] + [Patch(facecolor=TINTA_2, edgecolor=FONDO),
                                                     Patch(facecolor=TINTA_2, hatch="///", edgecolor=FONDO)]
    leyenda_abajo(fig, h, ["IVF", "HNSW", "coseno (lisa)", "euclidiana (rayada)"], arriba=.84)
    return fig


def miniatura(path, lado=160):
    img = cv2.imread(path) if path and os.path.exists(path) else None
    if img is None:
        return np.full((lado, lado, 3), 235, np.uint8)
    return cv2.cvtColor(cv2.resize(img, (lado, lado)), cv2.COLOR_BGR2RGB)


def galeria(r, tipo, clave, archivo, n_casos=3, cambio=None):
    # filas: consulta modificada + top 5; borde verde = es el original, rojo = no
    # con cambio solo se muestran las consultas de ese cambio (por ejemplo las fotos simuladas)
    import matplotlib.pyplot as plt
    objs = objetos(tipo)
    canciones = {}
    if tipo == "mfcc":
        for f in leer_csv(os.path.join(DATA, "songs.csv")):
            canciones[f["track_id"]] = f
    casos = [c for c in r["galeria"][clave] if cambio is None or c["cambio"] == cambio]
    exitos = [c for c in casos if c["rank"] == 1][:n_casos]
    errores = [c for c in casos if c["rank"] == 0][:n_casos]
    filas = [("acierto", c) for c in exitos] + [("error", c) for c in errores]
    if not filas:
        return
    alto = 2.9 if tipo == "mfcc" else 2.5       # en audio cada título ocupa 3 líneas
    fig, axes = plt.subplots(len(filas), 6, figsize=(12, alto * len(filas)), squeeze=False)
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
    que = f", solo '{cambio}'" if cambio else ""
    fig.suptitle(f"{TITULO[tipo]}: {indice.upper()} con {metrica}{que}, aciertos arriba y errores "
                 f"abajo (verde = el original)", color=TINTA, fontsize=11, x=.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 1 - .35 / len(filas)), h_pad=1.2)
    # son fotos de portadas: en jpg pesan ~10 veces menos que en png
    fig.savefig(os.path.join(SALIDA, archivo), dpi=110, pil_kwargs={"quality": 82})
    plt.close(fig)


def tabla_md(r):
    lineas = ["| tipo | índice | métrica | construir (s) | consulta (ms) | recall@10 | espacio (KB) |",
              "|---|---|---|---|---|---|---|"]
    for f in r["filas"]:
        if f["n"] == maximo(r, f["tipo"]):
            lineas.append(f"| {f['tipo']} | {f['indice']} | {f['metrica']} | {f['construir_s']:.1f} | "
                          f"{f['consulta_ms']:.2f} | {f['recall10']:.3f} | {f['espacio_kb']:.0f} |")
    return "\n".join(lineas)


def graficar(r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    estilo(plt)
    os.makedirs(SALIDA, exist_ok=True)
    figuras = {"consulta_vs_n.png": fig_consulta, "recall_vs_n.png": fig_recall,
               "recall_vs_tiempo.png": fig_curvas, "construccion_vs_n.png": fig_construccion,
               "coseno_vs_euclidiana.png": fig_metricas, "espacio_vs_n.png": fig_espacio,
               "nuestro_vs_librerias.png": fig_librerias, "calidad_original.png": fig_calidad}
    for archivo, hacer in figuras.items():
        fig = hacer(r, plt)
        if fig is not None:
            fig.savefig(os.path.join(SALIDA, archivo), dpi=150)
            plt.close(fig)
    for tipo in ("sift", "mfcc"):
        for nombre in INDICES:
            for metrica in METRICAS:
                galeria(r, tipo, f"{tipo}_{nombre}_{metrica}",
                        f"galeria_{tipo}_{nombre}_{metrica}.jpg")
    if "sift_hnsw_coseno" in r["galeria"]:
        galeria(r, "sift", "sift_hnsw_coseno", "galeria_foto_simulada.jpg", cambio="foto simulada")
    with open(os.path.join(SALIDA, "tabla.md"), "w", encoding="utf-8") as f:
        f.write(tabla_md(r) + "\n")


def maximo(r, tipo):
    t = r["tamanos"]
    return max(t[tipo] if isinstance(t, dict) else t)


def tamanos_posibles(tipo):
    # lo que alcanza dejando NQ objetos aparte para las consultas
    disponibles = len(objetos(tipo)) - NQ
    if disponibles >= 10000:
        fijos = [t for t in (1000, 2500, 5000, 10000, 20000, 50000, 100000) if t <= disponibles]
        # si sobra bastante se agrega todo lo que hay (20 005 portadas - 200 consultas = 19 800)
        resto = disponibles // 100 * 100
        return fijos + ([resto] if resto >= fijos[-1] * 1.5 else [])
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
        tamanos = {t: sorted(args.tamanos or tamanos_posibles(t)) for t in ("sift", "mfcc")}
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
