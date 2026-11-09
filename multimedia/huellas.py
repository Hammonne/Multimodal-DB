# prototipo de shazam (parte 5): huellas de audio con picos del espectrograma (wang, 2003)
# cada canción -> picos de energía -> pares de picos (f1, f2, dt) como hash -> índice invertido
# una consulta vota por (canción, desfase): si muchos hashes coinciden con el mismo desfase, es esa canción
# a diferencia de mfcc no describe la forma del espectro, solo dónde están los picos, por eso aguanta
# filtros de celular, eco y ruido
# uso: python -m multimedia.huellas --evaluar --n 1000
import argparse
import os
import time
import warnings

import numpy as np
from scipy.ndimage import maximum_filter

from multimedia.sift import DATA, leer_csv

SR = 11025              # alcanza hasta 5.5 kHz, ahí están los picos que importan
N_FFT, HOP = 1024, 256  # ventanas de ~93 ms cada ~23 ms
VECINDAD = (15, 11)     # un pico es el máximo en ±7 bandas y ±5 ventanas
FAN = 8                 # cada pico ancla se empareja con los 8 siguientes
DT_MAX = 64             # hasta ~1.5 s entre los dos picos del par


def cargar(path, offset=0.0, duration=None):
    import librosa
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(path, sr=SR, mono=True, offset=offset, duration=duration)
    return y


def picos(y):
    # [(ventana, banda)] de los máximos locales con energía por encima de la mediana
    import librosa
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    S = librosa.amplitude_to_db(S, ref=np.max)
    maximo = maximum_filter(S, size=VECINDAD) == S
    bandas, ventanas = np.nonzero(maximo & (S > np.median(S) + 10))
    orden = np.argsort(ventanas, kind="stable")
    return ventanas[orden], bandas[orden]


def hashes(y):
    # (hash, ventana del ancla): hash = f1 (9 bits) | f2 (9 bits) | dt (6 bits)
    t, f = picos(y)
    hs, ts = [], []
    for i in range(len(t)):
        for j in range(i + 1, min(i + 1 + FAN, len(t))):
            dt = t[j] - t[i]
            if 0 < dt < DT_MAX:
                hs.append((int(f[i]) & 511) << 15 | (int(f[j]) & 511) << 6 | int(dt))
                ts.append(int(t[i]))
    return np.array(hs, np.int64), np.array(ts, np.int32)


class IndiceHuellas:
    # índice invertido ordenado por hash: buscar un hash es una búsqueda binaria (searchsorted)

    def __init__(self):
        self.hash = self.cancion = self.tiempo = None

    def construir(self, canciones):
        # canciones: [(id numérico, señal)]
        H, C, T = [], [], []
        for cid, y in canciones:
            h, t = hashes(y)
            H.append(h)
            T.append(t)
            C.append(np.full(len(h), cid, np.int64))
        H, C, T = np.concatenate(H), np.concatenate(C), np.concatenate(T)
        orden = np.argsort(H, kind="stable")
        self.hash, self.cancion, self.tiempo = H[orden], C[orden], T[orden]
        return self

    def buscar(self, y, k=5):
        # [(canción, votos)] ordenados; votos = hashes que coinciden con el mismo desfase
        h, t = hashes(y)
        izq = np.searchsorted(self.hash, h, "left")
        der = np.searchsorted(self.hash, h, "right")
        largo = der - izq
        if not largo.sum():
            return []
        pos = np.repeat(izq, largo) + (np.arange(largo.sum()) - np.repeat(np.cumsum(largo) - largo, largo))
        canc = self.cancion[pos]
        desfase = self.tiempo[pos] - np.repeat(t, largo)
        pares, votos = np.unique(np.stack([canc, desfase]), axis=1, return_counts=True)
        mejor = {}
        for c, v in zip(pares[0], votos):
            mejor[c] = max(mejor.get(c, 0), v)
        return sorted(mejor.items(), key=lambda x: -x[1])[:k]

    def guardar(self, path):
        np.savez(path, hash=self.hash, cancion=self.cancion, tiempo=self.tiempo)

    @classmethod
    def cargar(cls, path):
        z = np.load(path)
        idx = cls()
        idx.hash, idx.cancion, idx.tiempo = z["hash"], z["cancion"], z["tiempo"]
        return idx


def evaluar(n=1000, n_consultas=100, data=DATA, seed=0):
    # mismas consultas que el benchmark de mfcc: 5 s, 5 s con ruido y 5 s grabado con celular
    from benchmark.multimedia.benchmark_vectorial import grabado_celular
    filas = [r for r in leer_csv(os.path.join(data, "songs.csv")) if r.get("audio_path")][:n]
    t0 = time.perf_counter()
    canciones = []
    for i, r in enumerate(filas):
        try:
            canciones.append((i, cargar(os.path.join(data, r["audio_path"]))))
        except Exception:
            pass
    t_carga = time.perf_counter() - t0
    t0 = time.perf_counter()
    idx = IndiceHuellas().construir(canciones)
    t_indice = time.perf_counter() - t0

    rng = np.random.default_rng(seed)
    largas = [(c, y) for c, y in canciones if len(y) > SR * 20]
    elegidas = [largas[i] for i in rng.choice(len(largas), min(n_consultas, len(largas)), replace=False)]
    cambios = {
        "fragmento 5 s": lambda y, r: y,
        "5 s + volumen y ruido": lambda y, r: y * .5 + r.normal(0, .02, len(y)).astype(np.float32),
        "5 s grabado con celular": lambda y, r: grabado_celular_sr(y, r, grabado_celular),
        "5 s + ruido fuerte": lambda y, r: y + r.normal(0, 3 * y.std(), len(y)).astype(np.float32),
        "3 s grabado con celular": lambda y, r: grabado_celular_sr(y[:SR * 3], r, grabado_celular),
    }
    res, ms = {}, []
    for nombre, f in cambios.items():
        top1 = top5 = 0
        for c, y in elegidas:
            frag = y[SR * 10:SR * 15]
            q = f(frag, np.random.default_rng(int(c)))
            t1 = time.perf_counter()
            r = idx.buscar(q, 5)
            ms.append((time.perf_counter() - t1) * 1000)
            top1 += bool(r) and r[0][0] == c
            top5 += any(x[0] == c for x in r)
        res[nombre] = {"top1": top1 / len(elegidas), "top5": top5 / len(elegidas)}
    return {"canciones": len(canciones), "hashes": int(len(idx.hash)), "consultas": len(elegidas),
            "carga_s": t_carga, "indice_s": t_indice, "ms_consulta": float(np.mean(ms)),
            "mb": idx.hash.nbytes * 2.5 / 2 ** 20, "resultados": res}


def grabado_celular_sr(y, rng, filtro):
    # el filtro del benchmark está pensado para 22 050 Hz; acá se sube, filtra y vuelve a bajar
    import librosa
    z = filtro(librosa.resample(y, orig_sr=SR, target_sr=22050), rng)
    return librosa.resample(z, orig_sr=22050, target_sr=SR)


def main():
    ap = argparse.ArgumentParser(description="huellas de audio tipo shazam")
    ap.add_argument("--evaluar", action="store_true")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--data", default=DATA)
    args = ap.parse_args()
    if not args.evaluar:
        ap.error("por ahora solo --evaluar")
    r = evaluar(args.n, data=args.data)
    print(f"{r['canciones']} canciones, {r['hashes']:,} hashes ({r['mb']:.0f} MB), "
          f"{r['consultas']} consultas\ncargar audio {r['carga_s']:.0f} s, armar índice "
          f"{r['indice_s']:.1f} s, consulta {r['ms_consulta']:.1f} ms")
    for nombre, v in r["resultados"].items():
        print(f"  {nombre:24s} top1 {v['top1']:.2f}  top5 {v['top5']:.2f}")


if __name__ == "__main__":
    main()
