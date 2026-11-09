# datos para la página "cómo funciona" (web/app/como-funciona): cada paso del pipeline de la parte 4
# calculado sobre una portada o una canción real, para dibujarlo en el navegador
# sift: puntos clave, descriptor de 128 (4x4 celdas x 8 direcciones), palabra de cada punto, histograma tf-idf
# mfcc: onda, espectrograma, escala mel, 39 coeficientes por ventana, palabra de cada ventana, histograma
from __future__ import annotations

import csv
import os
import random
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from multimedia import features as F
from multimedia.sift import N_FEATURES, leer_gris

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
_cache: dict[str, Any] = {}


def _base(tipo: str) -> dict[str, Any]:
    if tipo not in _cache:
        b = F.cargar_base(tipo, F.K_DEFECTO, str(DATA))
        cb = F.cargar_codebook(tipo, F.K_DEFECTO, str(DATA))
        _cache[tipo] = {"ids": b["ids"].astype(str), "idf": b["idf"], "tf": b["tf"], "cb": cb}
    return _cache[tipo]


def resumen() -> dict[str, Any]:
    # los números de toda la colección
    def filas(nombre):
        p = DATA / "features" / nombre
        with p.open(newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    sift = filas("sift_covers.csv")
    mfcc = filas("mfcc.csv")
    caras = np.load(DATA / "features" / "faces.npz")
    bs, bm = _base("sift"), _base("mfcc")
    dueno = {f"{a}:{b}" for a, b in zip(caras["kind"], caras["owner_id"])}
    return {
        "portadas": len(sift), "descriptores_sift": sum(int(r["n_descriptors"]) for r in sift),
        "audios": len(mfcc), "ventanas_mfcc": sum(int(r["n_frames"]) for r in mfcc),
        "k": F.K_DEFECTO,
        # en la web el audio también tiene las 3 643 canciones que se sumaron después (ver multimedia_search)
        "vectores_sift": len(bs["ids"]), "vectores_mfcc": len(mfcc),
        "palabras_por_portada": int(np.median((bs["tf"] > 0).sum(1))),
        "palabras_por_audio": int(np.median((bm["tf"] > 0).sum(1))),
        "caras": int(len(caras["embedding"])), "personas": len(dueno),
    }


def ejemplos(tipo: str, n: int = 12, seed: int | None = None) -> list[dict[str, str]]:
    # portadas: su ruta; audio: canciones con título, artista y portada para mostrarlas
    ids = list(_base(tipo)["ids"])
    elegidos = random.Random(seed).sample(ids, min(n * 3 if tipo == "mfcc" else n, len(ids)))
    if tipo == "sift":
        return [{"id": i, "imagen": i} for i in elegidos]
    with (DATA / "songs.csv").open(newline="", encoding="utf-8") as fh:
        canciones = {r["track_id"]: r for r in csv.DictReader(fh)}
    salida = [{"id": i, "titulo": canciones[i]["title"], "artista": canciones[i]["artist"],
               "imagen": canciones[i]["cover_path"]} for i in elegidos if i in canciones]
    return salida[:n]


def _palabras_e_hist(desc: np.ndarray, tipo: str) -> tuple[np.ndarray, list[float], list[float]]:
    b = _base(tipo)
    palabras = F.palabras(desc, b["cb"])
    tf = F.histograma(palabras, F.K_DEFECTO)
    vec = F.pesar(tf, b["idf"])
    return palabras, tf.round(5).tolist(), vec.round(5).tolist()


def sift(path: str) -> dict[str, Any]:
    ruta = DATA / path
    if not ruta.is_file() or F.tipo_de(path) != "sift":
        raise ValueError("elige una portada de la colección")
    gris = leer_gris(str(ruta))
    alto, ancho = gris.shape[:2]
    det = cv2.SIFT_create(nfeatures=N_FEATURES)
    kps, desc = det.detectAndCompute(gris, None)
    if desc is None:
        raise ValueError("esta portada no tiene puntos clave (es muy plana)")
    crudo = desc.copy()
    # rootsift, igual que multimedia/sift.py
    desc = np.sqrt(desc / (desc.sum(axis=1, keepdims=True) + 1e-7)).astype(np.float32)
    palabras, tf, vec = _palabras_e_hist(desc, "sift")
    idf = _base("sift")["idf"]
    puntos = [{"x": round(k.pt[0], 1), "y": round(k.pt[1], 1), "tam": round(k.size, 1),
               "angulo": round(k.angle, 1), "palabra": int(p)}
              for k, p in zip(kps, palabras)]
    return {"path": path, "ancho": ancho, "alto": alto, "puntos": puntos,
            "descriptores": crudo.round(0).astype(int).tolist(),
            "tf": tf, "vector": vec, "idf": idf.round(4).tolist(),
            "palabras_distintas": int(len(set(palabras.tolist())))}


def mfcc(track_id: str) -> dict[str, Any]:
    import librosa
    from multimedia.mfcc import HOP, N_FFT, N_MELS, SR, extraer_mfcc
    with (DATA / "songs.csv").open(newline="", encoding="utf-8") as fh:
        fila = next((r for r in csv.DictReader(fh) if r["track_id"] == str(track_id)), None)
    if not fila or not fila.get("audio_path"):
        raise ValueError("elige una canción de la colección")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(str(DATA / fila["audio_path"]), sr=SR, mono=True)
    # para dibujar se reduce: la onda a 1 200 puntos y los espectrogramas a ~300 columnas
    paso_onda = max(1, len(y) // 1200)
    onda = np.abs(y[: len(y) // paso_onda * paso_onda]).reshape(-1, paso_onda).max(1)
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    espec = librosa.amplitude_to_db(S, ref=np.max)
    mel = librosa.power_to_db(librosa.feature.melspectrogram(S=S ** 2, sr=SR, n_mels=N_MELS), ref=np.max)
    desc = extraer_mfcc(y=y)                       # (ventanas, 39) como en la colección
    palabras, tf, vec = _palabras_e_hist(desc, "mfcc")
    cols = max(1, S.shape[1] // 300)

    def reducir(M, filas=None):
        # promedia columnas de a `cols` y, si hace falta, filas hasta quedar en `filas`
        M = M[:, : M.shape[1] // cols * cols].reshape(M.shape[0], -1, cols).mean(2)
        if filas and M.shape[0] > filas:
            f = M.shape[0] // filas
            M = M[: filas * f].reshape(filas, f, -1).mean(1)
        return M.round(1).tolist()

    paso_v = max(1, len(desc) // 300)
    return {"track_id": str(track_id), "titulo": fila["title"], "artista": fila["artist"],
            "audio": fila["audio_path"], "portada": fila.get("cover_path", ""),
            "sr": SR, "n_fft": N_FFT, "hop": HOP, "duracion": round(len(y) / SR, 2),
            "onda": onda.round(4).tolist(), "espectrograma": reducir(espec[:512], 128),
            "mel": reducir(mel), "n_ventanas": len(desc),
            "mfcc": desc[::paso_v].T.round(2).tolist(), "paso_ventanas": paso_v,
            "palabras": palabras[::paso_v].tolist(), "tf": tf, "vector": vec,
            "palabras_distintas": int(len(set(palabras.tolist())))}


# DoG: cómo encuentra SIFT los puntos clave

def _png(img: np.ndarray, lado: int = 220) -> str:
    import base64
    h, w = img.shape[:2]
    e = lado / max(h, w)
    chica = cv2.resize(img, (max(1, round(w * e)), max(1, round(h * e))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", chica, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def dog(path: str) -> dict[str, Any]:
    # una octava de la pirámide: la imagen borroneada con sigmas crecientes (sigma0 * 2^(i/3)) y la resta
    # de cada par vecino (diferencia de gaussianas); los puntos clave son máximos o mínimos de la DoG
    gris = leer_gris(str(DATA / path)).astype(np.float32)
    sigmas = [1.6 * 2 ** (i / 3) for i in range(5)]
    blur = [cv2.GaussianBlur(gris, (0, 0), s) for s in sigmas]
    dogs = [blur[i + 1] - blur[i] for i in range(4)]
    lim = max(float(np.abs(d).max()) for d in dogs) or 1
    visual = [np.clip(128 + d / lim * 400, 0, 255).astype(np.uint8) for d in dogs]
    # extremos de la segunda dog contra sus 26 vecinos (8 en su escala, 9 arriba y 9 abajo)
    d0, d1, d2 = dogs[0], dogs[1], dogs[2]
    maxf = cv2.dilate(d1, np.ones((3, 3)))
    minf = cv2.erode(d1, np.ones((3, 3)))
    vecinos_max = np.maximum(cv2.dilate(d0, np.ones((3, 3))), cv2.dilate(d2, np.ones((3, 3))))
    vecinos_min = np.minimum(cv2.erode(d0, np.ones((3, 3))), cv2.erode(d2, np.ones((3, 3))))
    umbral = 0.03 * 255
    es_max = (d1 >= maxf) & (d1 > vecinos_max) & (d1 > umbral)
    es_min = (d1 <= minf) & (d1 < vecinos_min) & (d1 < -umbral)
    ys, xs = np.nonzero(es_max | es_min)
    alto, ancho = gris.shape
    return {"ancho": ancho, "alto": alto, "sigmas": [round(s, 2) for s in sigmas],
            "gaussianas": [_png(b.astype(np.uint8)) for b in blur], "dogs": [_png(v) for v in visual],
            "extremos": [[int(x), int(y)] for x, y in list(zip(xs, ys))[:400]], "n_extremos": int(len(xs))}


# índices: una muestra de portadas reales proyectada en 2d para ver ivf y hnsw

def _muestra() -> dict[str, Any]:
    if "muestra" not in _cache:
        from sklearn.decomposition import PCA
        from indexes.hnsw import HNSW
        from indexes.ivf import IVF
        b = F.cargar_base("sift", F.K_DEFECTO, str(DATA))
        rng = np.random.default_rng(0)
        pos = np.sort(rng.choice(len(b["ids"]), 2000, replace=False))
        X = b["vectores"][pos].astype(np.float32)
        xy = PCA(2, random_state=0).fit_transform(X)
        xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))
        ivf = IVF(nlist=40, metrica="coseno").entrenar(X).agregar(X)
        celda = np.empty(len(X), int)
        for c, ids in enumerate(ivf.listas_ids):
            celda[ids] = c
        # hnsw chico (300 nodos, M = 4) para que el grafo se pueda dibujar
        chico = 300
        hnsw = HNSW(M=4, ef_construction=40, metrica="coseno", seed=1).agregar(X[:chico])
        _cache["muestra"] = {"ids": b["ids"][pos].astype(str), "X": X, "xy": xy, "ivf": ivf, "celda": celda,
                             "hnsw": hnsw, "chico": chico,
                             "centroides_xy": np.array([xy[celda == c].mean(0) if (celda == c).any() else [0.5, 0.5]
                                                        for c in range(ivf.nlist)])}
    return _cache["muestra"]


def ivf(consulta: int, nprobe: int) -> dict[str, Any]:
    from indexes.metricas import knn_exacto
    m = _muestra()
    X, ivf = m["X"], m["ivf"]
    consulta = int(consulta) % len(X)
    nprobe = max(1, min(int(nprobe), ivf.nlist))
    q = X[consulta]
    orden = np.argsort(ivf.metrica.distancia(q, ivf.centroides)).tolist()
    abiertas = orden[:nprobe]
    ids, _ = ivf.buscar(q, 11, nprobe)
    exactos, _ = knn_exacto(q, X, 11, "coseno")
    ids = [int(i) for i in ids if int(i) != consulta][:10]
    exactos = [int(i) for i in exactos if int(i) != consulta][:10]
    return {"puntos": [[round(float(x), 4), round(float(y), 4), int(c)] for (x, y), c in zip(m["xy"], m["celda"])],
            "centroides": m["centroides_xy"].round(4).tolist(), "nlist": ivf.nlist, "consulta": consulta,
            "orden_celdas": orden, "abiertas": abiertas, "revisados": int(ivf.candidatos),
            "resultado": ids, "exactos": exactos, "aciertos": len(set(ids) & set(exactos)),
            "imagenes": {str(i): m["ids"][i] for i in set(ids) | set(exactos) | {consulta}}}


def hnsw(consulta: int) -> dict[str, Any]:
    # la búsqueda del hnsw paso a paso: en cada nivel se avanza al vecino más cercano hasta no mejorar,
    # y en el nivel 0 se explora con ef_search candidatos
    import heapq
    m = _muestra()
    h, n = m["hnsw"], m["chico"]
    consulta = int(consulta) % len(m["X"])
    q = m["X"][consulta]
    dist = lambda nodos: h.metrica.distancia(q, h.vectores[nodos])
    ep = h.entrada
    niveles = []
    for l in range(h.nivel[h.entrada], 0, -1):
        camino, visitados, mejor = [ep], set([ep]), float(dist([ep])[0])
        while True:
            vec = h.vecinos[ep][l]
            visitados.update(vec)
            if not vec:
                break
            dv = dist(vec)
            i = int(np.argmin(dv))
            if dv[i] >= mejor:
                break
            ep, mejor = vec[i], float(dv[i])
            camino.append(ep)
        niveles.append({"nivel": l, "camino": camino, "visitados": sorted(visitados)})
    # nivel 0 con ef = 20
    ef = 20
    visto = {ep}
    d0 = float(dist([ep])[0])
    cand, mejores, orden_visita = [(d0, ep)], [(-d0, ep)], [ep]
    while cand:
        dc, c = heapq.heappop(cand)
        if dc > -mejores[0][0] and len(mejores) >= ef:
            break
        nuevos = [x for x in h.vecinos[c][0] if x not in visto]
        visto.update(nuevos)
        for dn, x in zip(dist(nuevos) if nuevos else [], nuevos):
            orden_visita.append(x)
            if len(mejores) < ef or dn < -mejores[0][0]:
                heapq.heappush(cand, (float(dn), x))
                heapq.heappush(mejores, (-float(dn), x))
                if len(mejores) > ef:
                    heapq.heappop(mejores)
    finales = [x for _, x in sorted((-d, x) for d, x in mejores)]
    finales = [x for x in finales if x != consulta][:10]
    aristas = {}
    for l in range(max(h.nivel) + 1):
        pares = set()
        for nodo in range(n):
            if h.nivel[nodo] >= l:
                for v in h.vecinos[nodo][l]:
                    pares.add((min(nodo, v), max(nodo, v)))
        aristas[l] = sorted(pares)
    xy = m["xy"][:n]
    return {"nodos": [[round(float(x), 4), round(float(y), 4), int(lv)] for (x, y), lv in zip(xy, h.nivel)],
            "aristas": {str(l): a for l, a in aristas.items()}, "entrada": int(h.entrada),
            "max_nivel": int(max(h.nivel)), "consulta": consulta, "consulta_xy": m["xy"][consulta].round(4).tolist(),
            "niveles": niveles, "nivel0_visitados": orden_visita, "resultado": finales,
            "por_nivel": [int(sum(1 for lv in h.nivel if lv >= l)) for l in range(max(h.nivel) + 1)],
            "imagenes": {str(i): m["ids"][i] for i in set(finales) | {consulta, int(h.entrada)}
                         | {x for nv in niveles for x in nv["camino"]}}}


# bolsa de palabras visuales en 2d: descriptores y centroides del codebook proyectados con pca

def _espacio_palabras() -> dict[str, Any]:
    if "palabras2d" not in _cache:
        from sklearn.decomposition import PCA
        b = _base("sift")
        cb = b["cb"]
        filas = list(csv.DictReader((DATA / "features" / "sift_covers.csv").open(newline="", encoding="utf-8")))
        rng = np.random.default_rng(0)
        muestra = []
        for i in rng.choice(len(filas), 60, replace=False):
            d = np.load(DATA / filas[i]["features_path"])
            if len(d):
                muestra.append(d[rng.choice(len(d), min(60, len(d)), replace=False)])
        D = (np.vstack(muestra) - cb["media"]) / cb["desv"]          # el mismo espacio donde vive el k-means
        pca = PCA(2, random_state=0).fit(np.vstack([D, cb["centroides"]]))
        fondo = pca.transform(D)
        cent = pca.transform(cb["centroides"])
        todo = np.vstack([fondo, cent])
        lo, hi = np.percentile(todo, 1, axis=0), np.percentile(todo, 99, axis=0)
        norm = lambda P: np.clip((P - lo) / (hi - lo), 0, 1)
        palabras_fondo = F.palabras(np.vstack(muestra), cb)
        _cache["palabras2d"] = {"pca": pca, "norm": norm, "cb": cb,
                                "fondo": [[round(float(x), 4), round(float(y), 4), int(p)]
                                          for (x, y), p in zip(norm(fondo), palabras_fondo)],
                                "centroides": norm(cent).round(4).tolist()}
    return _cache["palabras2d"]


def palabras2d(path: str) -> dict[str, Any]:
    e = _espacio_palabras()
    gris = leer_gris(str(DATA / path))
    _, desc = cv2.SIFT_create(nfeatures=N_FEATURES).detectAndCompute(gris, None)
    if desc is None:
        raise ValueError("esta portada no tiene puntos clave")
    desc = np.sqrt(desc / (desc.sum(axis=1, keepdims=True) + 1e-7)).astype(np.float32)
    pal = F.palabras(desc, e["cb"])
    P = e["norm"](e["pca"].transform((desc - e["cb"]["media"]) / e["cb"]["desv"]))
    return {"fondo": e["fondo"], "centroides": e["centroides"],
            "portada": [[round(float(x), 4), round(float(y), 4), int(p)] for (x, y), p in zip(P, pal)],
            "varianza": [round(float(v), 3) for v in e["pca"].explained_variance_ratio_]}


# caras: descriptor global, un vector de 128 por cara (face_recognition, como en el lab 11)

def _caras() -> dict[str, Any]:
    if "caras" not in _cache:
        from sklearn.decomposition import PCA
        from multimedia import caras as C
        datos, indice = C.abrir(str(DATA))
        cajas = {}
        for archivo in ("artist_faces.csv", "member_faces.csv"):
            with (DATA / archivo).open(newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    if r.get("kind") == "cara" and r.get("face_box"):
                        cajas[r["path"]] = [int(v) for v in r["face_box"].split()]
        E = datos["embedding"]
        xy = PCA(2, random_state=0).fit_transform(E)
        lo, hi = np.percentile(xy, 1, axis=0), np.percentile(xy, 99, axis=0)
        dueno = np.array([f"{a}:{b}" for a, b in zip(datos["kind"], datos["owner_id"])])
        _cache["caras"] = {"datos": datos, "indice": indice, "cajas": cajas, "dueno": dueno,
                           "xy": np.clip((xy - lo) / (hi - lo), 0, 1), "umbral": C.UMBRAL}
    return _cache["caras"]


def caras_ejemplos(n: int = 12) -> list[dict[str, str]]:
    # fotos de personas con 3 caras o más, así se ve que sus otras fotos quedan cerca
    c = _caras()
    _, inversa, cuenta = np.unique(c["dueno"], return_inverse=True, return_counts=True)
    candidatas = np.flatnonzero(cuenta[inversa] >= 3)
    elegidas = np.random.default_rng().choice(candidatas, min(n * 3, len(candidatas)), replace=False)
    vistos, salida = set(), []
    for i in elegidas:
        if c["dueno"][i] in vistos:
            continue
        vistos.add(c["dueno"][i])
        salida.append({"id": str(c["datos"]["path"][i]), "imagen": str(c["datos"]["path"][i]),
                       "titulo": str(c["datos"]["name"][i])})
        if len(salida) == n:
            break
    return salida


def caras(path: str) -> dict[str, Any]:
    from multimedia import caras as C
    c = _caras()
    datos = c["datos"]
    pos = np.flatnonzero(datos["path"] == path)
    if not len(pos) or path not in c["cajas"]:
        raise ValueError("elige una foto de cara de la colección")
    i = int(pos[0])
    emb = datos["embedding"][i]
    top, right, bottom, left = c["cajas"][path]
    puntos = {}
    try:
        # los 68 puntos de referencia con los que face_recognition alinea la cara antes de la red
        import face_recognition
        img = face_recognition.load_image_file(str(DATA / path))
        marcas = face_recognition.face_landmarks(img, [(top, right, bottom, left)])
        if marcas:
            puntos = {k: [[int(x), int(y)] for x, y in v] for k, v in marcas[0].items()}
        alto, ancho = img.shape[:2]
    except Exception:
        g = cv2.imread(str(DATA / path))
        alto, ancho = g.shape[:2]
    # distancia a todas las caras: la misma persona contra las demás
    d = np.linalg.norm(datos["embedding"] - emb, axis=1)
    misma = c["dueno"] == c["dueno"][i]
    misma[i] = False
    otras = ~(c["dueno"] == c["dueno"][i])
    bordes = np.linspace(0, 1.2, 61)
    hist_misma = np.histogram(d[misma], bordes)[0]
    hist_otras = np.histogram(d[otras], bordes)[0]
    # se busca en el hnsw y se descarta solo esta misma foto: las otras fotos de la persona sí cuentan
    pos_v, dist_v = c["indice"].buscar(emb, 60, 100)
    quedan = [(p, dd) for p, dd in zip(pos_v, dist_v) if int(p) != i]
    vecinos = C.personas(datos, [p for p, _ in quedan], [dd for _, dd in quedan], 10)
    return {"path": path, "nombre": str(datos["name"][i]), "ancho": int(ancho), "alto": int(alto),
            "caja": [top, right, bottom, left], "puntos": puntos, "embedding": emb.round(4).tolist(),
            "bordes": bordes.round(2).tolist(), "hist_misma": hist_misma.tolist(), "hist_otras": hist_otras.tolist(),
            "n_misma": int(misma.sum()), "n_otras": int(otras.sum()), "umbral": c["umbral"],
            "otras_fotos": [str(p) for p in datos["path"][misma][:8]],
            "mapa": [[round(float(x), 4), round(float(y), 4)] for x, y in c["xy"][::3]],
            "mapa_persona": [[round(float(x), 4), round(float(y), 4)] for x, y in c["xy"][misma | (np.arange(len(d)) == i)]],
            "mapa_yo": c["xy"][i].round(4).tolist(),
            "vecinos": [{"nombre": v["nombre"], "foto": v["foto"], "distancia": round(v["distancia"], 3),
                         "misma_persona": v["misma_persona"],
                         "xy": c["xy"][int(np.flatnonzero(datos["path"] == v["foto"])[0])].round(4).tolist()}
                        for v in vecinos],
            "revisados": int(c["indice"].candidatos), "total": int(len(d))}
