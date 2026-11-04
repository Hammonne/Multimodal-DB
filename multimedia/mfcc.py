"""Descriptores locales MFCC de audio (Parte 4, §2.4.1)

Es el equivalente de SIFT para audio. Una canción no se describe con un solo
vector: se recorre con una ventana deslizante de ~93 ms que avanza ~23 ms (las
ventanas se solapan un 75 %) y cada ventana se convierte en un vector:

    señal -> ventana -> FFT -> 40 bandas mel -> log -> DCT -> 13 coeficientes

Los coeficientes describen el timbre de ese instante (qué instrumentos o voz
suenan y con qué color), no la melodía. Se descarta el coeficiente 0, que es
el volumen, para que la misma canción más fuerte o más suave dé lo mismo, y se
agregan sus deltas (cuánto cambia cada coeficiente entre ventanas) y
delta-deltas (la aceleración): 13 + 13 + 13 = 39 dimensiones.

Un preview de 30 s da ~1290 ventanas: matriz (n, 39) float32, igual que SIFT
da (n, 128) por imagen. En el siguiente paso (#4) se cuantizan con K-Means en
"palabras de audio" y (#5) cada canción pasa a ser un histograma TF-IDF.

La fila i corresponde al instante i * HOP / SR segundos, así que también sirve
para buscar por fragmentos (por ejemplo, segmentos de 5-10 s en la Opción C).

Se guardan en data/features/mfcc/<track_id>.npy más un índice CSV. Es
reanudable: no recalcula las canciones que ya tienen su .npy.

Uso:
    conda activate vision
    python -m multimedia.mfcc
    python -m multimedia.mfcc --limit 50
"""
import argparse
import csv
import os
import time
import warnings
from multiprocessing import Pool

import numpy as np

from multimedia.sift import DATA, leer_csv

SR = 22050          # frecuencia de muestreo: todos los audios se llevan a esta
N_FFT = 2048        # ventana de 2048 muestras = 93 ms
HOP = 512           # avance de 512 muestras = 23 ms (75 % de solapamiento)
N_MELS = 40         # bandas del banco de filtros mel
N_MFCC = 13         # coeficientes que se guardan (sin contar el 0)
DIM = 3 * N_MFCC    # MFCC + delta + delta-delta
OUT_DIR = os.path.join("features", "mfcc")
INDEX_FIELDS = ["track_id", "audio_path", "n_frames", "duration_s", "features_path"]


def cargar_audio(path, offset=0.0, duration=None):
    """Señal mono a SR Hz como float32 en [-1, 1]."""
    import librosa
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(path, sr=SR, mono=True, offset=offset, duration=duration)
    return y


def extraer_mfcc(path=None, y=None, offset=0.0, duration=None):
    """Descriptores MFCC de un audio: matriz (n, 39) float32, una fila por ventana.

    Se puede pasar la ruta (y opcionalmente un fragmento con offset/duration en
    segundos) o directamente la señal y. Si el audio es más corto que una
    ventana devuelve una matriz (0, 39)."""
    import librosa
    if y is None:
        y = cargar_audio(path, offset, duration)
    if len(y) < N_FFT:
        return np.zeros((0, DIM), dtype=np.float32)
    m = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC + 1, n_fft=N_FFT,
                             hop_length=HOP, n_mels=N_MELS)[1:]
    ancho = min(9, m.shape[1] - (1 - m.shape[1] % 2))   # deltas necesitan ancho impar <= n
    if ancho < 3:
        d1 = d2 = np.zeros_like(m)
    else:
        d1 = librosa.feature.delta(m, width=ancho, order=1)
        d2 = librosa.feature.delta(m, width=ancho, order=2)
    return np.vstack([m, d1, d2]).T.astype(np.float32)


def instante(fila):
    """Segundo del audio en que empieza la ventana número `fila`."""
    return fila * HOP / SR


def ruta_features(track_id):
    return os.path.join(OUT_DIR, f"{track_id}.npy")


def procesar(trabajo):
    track_id, audio_path, data = trabajo
    destino = ruta_features(track_id)
    completo = os.path.join(data, destino)
    if os.path.exists(completo):
        n = len(np.load(completo, mmap_mode="r"))
    else:
        try:
            desc = extraer_mfcc(os.path.join(data, audio_path))
        except Exception:
            return None
        os.makedirs(os.path.dirname(completo), exist_ok=True)
        np.save(completo, desc)
        n = len(desc)
    return {"track_id": track_id, "audio_path": audio_path, "n_frames": n,
            "duration_s": round(instante(n), 2), "features_path": destino}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="procesar solo las primeras N")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    lista = [(s["track_id"], s["audio_path"]) for s in leer_csv(os.path.join(args.data, "songs.csv"))
             if s["audio_path"]]
    if args.limit:
        lista = lista[:args.limit]
    print(f"{len(lista)} audios", flush=True)

    t0 = time.time()
    filas, fallos = [], 0
    with Pool(args.workers) as pool:
        trabajos = [(t, p, args.data) for t, p in lista]
        for i, r in enumerate(pool.imap(procesar, trabajos, chunksize=16), 1):
            if r is None:
                fallos += 1
            else:
                filas.append(r)
            if i % 1000 == 0:
                print(f"  {i}/{len(lista)}", flush=True)

    indice = os.path.join(args.data, "features", "mfcc.csv")
    os.makedirs(os.path.dirname(indice), exist_ok=True)
    with open(indice, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        w.writeheader()
        w.writerows(filas)

    n = [r["n_frames"] for r in filas]
    print(f"{len(filas)} audios en {time.time() - t0:.1f} s, {fallos} fallos; "
          f"ventanas por audio: media {np.mean(n):.0f}, mín {min(n)}, máx {max(n)}; "
          f"total {sum(n)}\nÍndice: {indice}")


if __name__ == "__main__":
    main()
