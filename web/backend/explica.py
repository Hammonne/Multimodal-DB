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
