# prototipo de "cantas y reconoce" (parte 5, en investigación)
# melodía de cada canción con pyin (notas midi por ventana) y dos formas de comparar un tarareo:
#   dtw: alinea las dos secuencias aunque vayan a distinta velocidad; no es métrica, no entra a ivf/hnsw
#   vector fijo: la melodía sin el tono (se resta la mediana) estirada a 64 puntos; con coseno sí es indexable
#   híbrido: el vector fijo elige 50 candidatos (eso sí puede ir en ivf/hnsw) y dtw solo los reordena
# el tarareo se simula desde la melodía de la misma canción (otro tono, otra velocidad, desafinado),
# así se prueba la comparación, no la extracción de la voz en una grabación real
# uso: python -m multimedia.tarareo --n 200
import argparse
import os
import time
import warnings

import numpy as np

from indexes.metricas import knn_exacto
from multimedia.sift import DATA, leer_csv

SR, HOP = 22050, 512
VENTANA, PASO = 5.0, 1.0        # cada canción se parte en tramos de 5 s cada 1 s
L = 64                          # largo del vector fijo


def melodia(path):
    # nota midi por ventana (nan donde no hay voz o tono claro)
    import librosa
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(path, sr=SR, mono=True)
        f0, _, _ = librosa.pyin(y, fmin=80, fmax=1000, sr=SR, hop_length=HOP)
    return librosa.hz_to_midi(f0)


def sin_tono(m):
    # sin huecos y relativa a su mediana: cantar en otro tono da la misma curva
    m = m[~np.isnan(m)]
    return m - np.median(m) if len(m) else m


def vector_fijo(m):
    m = sin_tono(m)
    if len(m) < 8:
        return None
    v = np.interp(np.linspace(0, len(m) - 1, L), np.arange(len(m)), m)
    v = v - v.mean()
    return (v / max(np.linalg.norm(v), 1e-9)).astype(np.float32)


def dtw(a, b):
    # costo de alinear a con b; o(len(a) * len(b))
    import librosa
    D, _ = librosa.sequence.dtw(a[None], b[None], metric="euclidean")
    return D[-1, -1] / (len(a) + len(b))


def tramos(m):
    por_tramo, paso = int(VENTANA * SR / HOP), int(PASO * SR / HOP)
    return [(i, m[i:i + por_tramo]) for i in range(0, max(1, len(m) - por_tramo), paso)]


def tarareo(m, rng):
    # otro tono (±5 semitonos), otra velocidad (±15 %), desafinado y con cortes
    m = m[~np.isnan(m)] + rng.uniform(-5, 5)
    largo = max(8, int(len(m) * rng.uniform(.85, 1.15)))
    m = np.interp(np.linspace(0, len(m) - 1, largo), np.arange(len(m)), m)
    m = m + rng.normal(0, .3, len(m))
    return m[rng.random(len(m)) > .1]


def evaluar(n=200, n_consultas=100, data=DATA, seed=0):
    filas = [r for r in leer_csv(os.path.join(data, "songs.csv")) if r.get("audio_path")][:n]
    # pyin tarda ~3 s por canción: las melodías se guardan para no recalcularlas
    cache = os.path.join(data, "features", f"melodias_{n}.npz")
    t0 = time.perf_counter()
    if os.path.exists(cache):
        z = np.load(cache)
        melodias = {int(k): z[k] for k in z.files}
    else:
        melodias = {}
        for i, r in enumerate(filas):
            try:
                melodias[i] = melodia(os.path.join(data, r["audio_path"]))
            except Exception:
                pass
        np.savez(cache, **{str(k): v for k, v in melodias.items()})
    t_mel = time.perf_counter() - t0

    # base: un vector y una secuencia por tramo
    vecs, seqs, duenos = [], [], []
    for c, m in melodias.items():
        for _, tr in tramos(m):
            v = vector_fijo(tr)
            if v is not None:
                vecs.append(v)
                seqs.append(sin_tono(tr))
                duenos.append(c)
    X, duenos = np.vstack(vecs), np.array(duenos)

    rng = np.random.default_rng(seed)
    con_voz = [c for c, m in melodias.items() if np.mean(~np.isnan(m)) > .3]
    elegidas = rng.choice(con_voz, min(n_consultas, len(con_voz)), replace=False)
    aciertos = {"vector fijo + coseno": [0, 0, 0.0], "dtw contra todos los tramos": [0, 0, 0.0],
                "híbrido: coseno 50 + dtw": [0, 0, 0.0]}
    for c in elegidas:
        m = melodias[c]
        inicio = int(10 * SR / HOP)
        q = tarareo(m[inicio:inicio + int(VENTANA * SR / HOP)], np.random.default_rng(int(c)))
        if len(q) < 8:
            continue
        t1 = time.perf_counter()
        ids, _ = knn_exacto(vector_fijo(q), X, 50, "coseno")
        orden = list(dict.fromkeys(duenos[ids]))
        a = aciertos["vector fijo + coseno"]
        a[0] += orden[:1] == [c]
        a[1] += c in orden[:5]
        a[2] += (time.perf_counter() - t1) * 1000
        t1 = time.perf_counter()
        qs = sin_tono(q)
        costos = np.array([dtw(qs, s) for s in seqs])
        mejor = {}
        for d, cost in zip(duenos, costos):
            mejor[d] = min(mejor.get(d, np.inf), cost)
        orden = sorted(mejor, key=mejor.get)
        a = aciertos["dtw contra todos los tramos"]
        a[0] += orden[:1] == [c]
        a[1] += c in orden[:5]
        a[2] += (time.perf_counter() - t1) * 1000
        t1 = time.perf_counter()
        ids, _ = knn_exacto(vector_fijo(q), X, 50, "coseno")
        mejor = {}
        for i in ids:
            mejor[duenos[i]] = min(mejor.get(duenos[i], np.inf), dtw(qs, seqs[i]))
        orden = sorted(mejor, key=mejor.get)
        a = aciertos["híbrido: coseno 50 + dtw"]
        a[0] += orden[:1] == [c]
        a[1] += c in orden[:5]
        a[2] += (time.perf_counter() - t1) * 1000
    nq = len(elegidas)
    return {"canciones": len(melodias), "tramos": len(X), "consultas": nq, "melodia_s": t_mel,
            "metodos": {k: {"top1": v[0] / nq, "top5": v[1] / nq, "ms": v[2] / nq}
                        for k, v in aciertos.items()}}


def main():
    ap = argparse.ArgumentParser(description="prototipo de búsqueda por tarareo")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--data", default=DATA)
    args = ap.parse_args()
    r = evaluar(args.n, data=args.data)
    print(f"{r['canciones']} canciones, {r['tramos']} tramos de {VENTANA:.0f} s, {r['consultas']} "
          f"tarareos simulados; melodías con pyin en {r['melodia_s']:.0f} s")
    for k, v in r["metodos"].items():
        print(f"  {k:28s} top1 {v['top1']:.2f}  top5 {v['top5']:.2f}  {v['ms']:.1f} ms por consulta")


if __name__ == "__main__":
    main()
