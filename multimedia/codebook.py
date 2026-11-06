# codebook con k-means (#4): prueba varios K para sift y mfcc y mide cuál recupera mejor
# guarda los centroides, una tabla de métricas y gráficas en data/features/codebook/
# uso: python -m multimedia.codebook --tipo sift --ks 128 256 512 1024 2048
import argparse
import csv
import os
import tempfile
import time

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

from multimedia.mfcc import cargar_audio, extraer_mfcc, instante
from multimedia.sift import DATA, extraer_sift, leer_csv

OUT = os.path.join("features", "codebook")
POR_OBJETO = 50         # máximo de descriptores por objeto en la muestra
TOTAL = 300_000         # tamaño de la muestra para entrenar
N_CONSULTAS = 300
SEED = 0


def objetos(tipo, data=DATA):
    # [(id, ruta del .npy)] en el orden de la base
    if tipo == "sift":
        filas = leer_csv(os.path.join(data, "features", "sift_covers.csv"))
        return [(r["path"], r["features_path"]) for r in filas]
    filas = leer_csv(os.path.join(data, "features", "mfcc.csv"))
    return [(r["track_id"], r["features_path"]) for r in filas]


def cargar_todo(objs, data=DATA):
    # todos los descriptores juntos + a qué objeto pertenece cada fila
    if not objs:
        raise ValueError(
            "no hay descriptores multimedia para entrenar el codebook. "
            "Ejecute primero `python -m multimedia.sift --source covers` "
            "y verifique que data/features/sift_covers.csv tenga filas."
        )
    mats = [np.load(os.path.join(data, f)) for _, f in objs]
    if not mats:
        raise ValueError("la colección multimedia no contiene archivos .npy")
    dueno = np.repeat(np.arange(len(mats)), [len(m) for m in mats])
    if not any(len(m) for m in mats):
        raise ValueError(
            "los descriptores SIFT están vacíos; revise que las imágenes "
            "existan y sean legibles."
        )
    return np.vstack(mats), dueno


def muestra(X, dueno, rng):
    # hasta POR_OBJETO filas al azar de cada objeto, así ninguno pesa más que otro
    orden = rng.permutation(len(X))
    orden = orden[np.argsort(dueno[orden], kind="stable")]
    inicio = np.searchsorted(dueno[orden], dueno[orden])
    pos = np.arange(len(orden)) - inicio
    elegidos = orden[pos < POR_OBJETO]
    if len(elegidos) > TOTAL:
        elegidos = rng.choice(elegidos, TOTAL, replace=False)
    return X[elegidos]


def consultas_sift(objs, n_desc, rng, data=DATA):
    # portadas modificadas, la respuesta correcta es la portada original
    def rotar(i, a):
        h, w = i.shape[:2]
        return cv2.warpAffine(i, cv2.getRotationMatrix2D((w / 2, h / 2), a, 1), (w, h),
                              borderMode=cv2.BORDER_REFLECT)
    cambios = {
        "rotar 15°": lambda i: rotar(i, 15),
        "recortar 70 %": lambda i: i[int(.15 * i.shape[0]):int(.85 * i.shape[0]),
                                     int(.15 * i.shape[1]):int(.85 * i.shape[1])],
        "oscurecer 60 %": lambda i: (i * .6).astype(np.uint8),
        "jpeg + 200 px": lambda i: cv2.resize(i, (200, 200)),
    }
    validos = np.flatnonzero(n_desc > 0)
    if not len(validos):
        return []
    elegidos = rng.choice(validos, min(N_CONSULTAS, len(validos)), replace=False)
    tmp = os.path.join(tempfile.gettempdir(), "consulta_sift.jpg")
    out = []
    for k in elegidos:
        img = cv2.imread(os.path.join(data, objs[k][0]))
        for nombre, f in cambios.items():
            calidad = 20 if "jpeg" in nombre else 90
            cv2.imwrite(tmp, f(img), [cv2.IMWRITE_JPEG_QUALITY, calidad])
            out.append((nombre, k, extraer_sift(tmp)))
    return out


def consultas_mfcc(objs, n_desc, rng, data=DATA):
    # fragmentos de 5 s, la respuesta correcta es la canción completa
    paths = {r["track_id"]: r["audio_path"] for r in leer_csv(os.path.join(data, "songs.csv"))}
    validos = np.flatnonzero(n_desc > 1000)     # al menos ~23 s de audio
    if not len(validos):
        return []
    elegidos = rng.choice(validos, min(N_CONSULTAS, len(validos)), replace=False)
    out = []
    for k in elegidos:
        dur = instante(n_desc[k])
        y = cargar_audio(os.path.join(data, paths[objs[k][0]]),
                         offset=float(rng.uniform(0, dur - 5)), duration=5)
        ruido = 0.5 * y + 0.01 * rng.standard_normal(len(y)).astype(np.float32)
        out.append(("fragmento 5 s", k, extraer_mfcc(y=y)))
        out.append(("5 s + volumen y ruido", k, extraer_mfcc(y=ruido)))
    return out


def tfidf(etiquetas, dueno, n_obj, K, idf=None):
    # histograma de palabras por objeto, tf normalizado, idf de la base y norma 1
    tf = np.bincount(dueno * K + etiquetas, minlength=n_obj * K).reshape(n_obj, K).astype(np.float32)
    tf /= np.maximum(tf.sum(1, keepdims=True), 1)
    if idf is None:
        idf = np.log(n_obj / np.maximum((tf > 0).sum(0), 1)).astype(np.float32)
    v = tf * idf
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    return v, idf


def asignar(km, X, lote=500_000):
    return np.concatenate([km.predict(X[i:i + lote]) for i in range(0, len(X), lote)])


def experimento(tipo, ks, data=DATA):
    rng = np.random.default_rng(SEED)
    out_dir = os.path.join(data, OUT)
    os.makedirs(out_dir, exist_ok=True)

    objs = objetos(tipo, data)
    t0 = time.time()
    X, dueno = cargar_todo(objs, data)
    n_desc = np.bincount(dueno, minlength=len(objs))
    print(f"{tipo}: {len(objs)} objetos, {len(X)} descriptores ({time.time() - t0:.0f} s)", flush=True)

    media, desv = np.zeros(X.shape[1], np.float32), np.ones(X.shape[1], np.float32)
    if tipo == "mfcc":  # cada dimensión tiene su escala, las deltas son mucho más chicas
        media, desv = X.mean(0), X.std(0) + 1e-6
        X -= media
        X /= desv
    S = muestra(X, dueno, rng)
    print(f"  muestra: {len(S)} descriptores", flush=True)

    cache = os.path.join(out_dir, f"consultas_{tipo}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        consultas = list(zip(z["nombre"], z["objetivo"], z["desc"]))
    else:
        t0 = time.time()
        consultas = (consultas_sift if tipo == "sift" else consultas_mfcc)(objs, n_desc, rng, data)
        desc = np.empty(len(consultas), dtype=object)   # lista de matrices, aunque midan lo mismo
        for i, c in enumerate(consultas):
            desc[i] = c[2]
        np.savez(cache, nombre=np.array([c[0] for c in consultas]),
                 objetivo=np.array([c[1] for c in consultas]), desc=desc)
        print(f"  {len(consultas)} consultas en {time.time() - t0:.0f} s", flush=True)
    consultas = [(n, int(o), (np.asarray(d, np.float32) - media) / desv) for n, o, d in consultas]
    tipos_consulta = list(dict.fromkeys(c[0] for c in consultas))

    filas = []
    for K in ks:
        t0 = time.time()
        km = MiniBatchKMeans(n_clusters=K, init="k-means++", batch_size=4096,
                             init_size=max(3 * 4096, 3 * K), n_init=3, max_iter=100,
                             random_state=SEED).fit(S)
        entrenar = time.time() - t0
        etiq_s = km.predict(S)
        inercia = float(((S - km.cluster_centers_[etiq_s]) ** 2).sum())
        tamanos = np.bincount(etiq_s, minlength=K)
        sub = rng.choice(len(S), min(20_000, len(S)), replace=False)
        db = davies_bouldin_score(S[sub], etiq_s[sub])
        sil = silhouette_score(S[sub[:min(10_000, len(sub))]], etiq_s[sub[:min(10_000, len(sub))]])

        t0 = time.time()
        base, idf = tfidf(asignar(km, X), dueno, len(objs), K)
        construir = time.time() - t0

        aciertos = {n: [] for n in tipos_consulta}
        t0 = time.time()
        for nombre, objetivo, d in consultas:
            if len(d) == 0:
                aciertos[nombre].append(10 ** 9)
                continue
            q, _ = tfidf(km.predict(d), np.zeros(len(d), int), 1, K, idf)
            sim = base @ q[0]
            aciertos[nombre].append(int((sim > sim[objetivo]).sum()) + 1)
        consulta_ms = (time.time() - t0) / len(consultas) * 1000

        fila = {"tipo": tipo, "K": K, "inercia": round(inercia, 1),
                "inercia_media": round(inercia / len(S), 4),
                "davies_bouldin": round(db, 3), "silhouette": round(sil, 4),
                "clusters_chicos": int((tamanos < 5).sum()),
                "tamano_min": int(tamanos.min()), "tamano_mediana": int(np.median(tamanos)),
                "tamano_max": int(tamanos.max()),
                "entrenar_s": round(entrenar, 1), "construir_s": round(construir, 1),
                "consulta_ms": round(consulta_ms, 2)}
        for n, r in aciertos.items():
            r = np.array(r)
            fila[f"r1 {n}"] = round(float(np.mean(r == 1)), 3)
            fila[f"r10 {n}"] = round(float(np.mean(r <= 10)), 3)
        filas.append(fila)
        np.savez(os.path.join(out_dir, f"{tipo}_k{K}.npz"), centroides=km.cluster_centers_,
                 media=media, desv=desv, tamanos=tamanos)
        print(f"  K={K}: inercia/desc {fila['inercia_media']}, "
              + ", ".join(f"r10 {n} {fila[f'r10 {n}']}" for n in tipos_consulta)
              + f" ({entrenar:.0f} s + {construir:.0f} s)", flush=True)

    with open(os.path.join(out_dir, f"metricas_{tipo}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)


SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # paleta categórica en orden fijo
TINTA, TINTA_2, REJILLA, FONDO = "#1f1f1e", "#62615c", "#e4e3df", "#fcfcfb"


def graficar(data=DATA):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = os.path.join(data, OUT)
    tablas = {t: leer_csv(os.path.join(out_dir, f"metricas_{t}.csv")) for t in ("sift", "mfcc")}
    tablas = {t: f for t, f in tablas.items() if f}
    titulos = {"sift": "SIFT (portadas)", "mfcc": "MFCC (audio)"}
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": REJILLA, "axes.labelcolor": TINTA_2,
                         "xtick.color": TINTA_2, "ytick.color": TINTA_2, "text.color": TINTA,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": FONDO, "axes.facecolor": FONDO})

    def ejes(ax, filas, ylabel):
        ks = [int(r["K"]) for r in filas]
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.minorticks_off()
        ax.grid(True, color=REJILLA, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("K (palabras)")
        ax.set_ylabel(ylabel)
        return ks

    def figura(nombre, ylabel, series_de, ylim=None):
        fig, axs = plt.subplots(1, len(tablas), figsize=(5.5 * len(tablas), 4), squeeze=False)
        for ax, (t, filas) in zip(axs[0], tablas.items()):
            ks = ejes(ax, filas, ylabel)
            for i, (etiqueta, ys) in enumerate(series_de(filas)):
                ax.plot(ks, ys, color=SERIES[i], linewidth=2, marker="o", markersize=7,
                        markeredgecolor=FONDO, markeredgewidth=2, label=etiqueta)
            ax.set_title(titulos[t], loc="left", fontsize=11, color=TINTA)
            if ylim:
                ax.set_ylim(*ylim)
            ax.set_xlim(ks[0] / 1.3, ks[-1] * 1.3)
            if len(ax.lines) > 1:
                ax.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=2,
                          bbox_to_anchor=(0.5, -0.18))
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, nombre), dpi=150)
        plt.close(fig)

    def recall(filas):
        cols = [c for c in filas[0] if c.startswith("r10 ")]
        return [(c[4:], [float(r[c]) for r in filas]) for c in cols]

    figura("recall10_vs_k.png", "recall@10", recall, (0, 1.02))
    figura("inercia_vs_k.png", "inercia por descriptor",
           lambda f: [("inercia", [float(r["inercia_media"]) for r in f])])
    figura("davies_bouldin_vs_k.png", "Davies–Bouldin (menor = mejor)",
           lambda f: [("davies–bouldin", [float(r["davies_bouldin"]) for r in f])])
    figura("consulta_vs_k.png", "ms por consulta",
           lambda f: [("consulta", [float(r["consulta_ms"]) for r in f])])
    figura("entrenar_vs_k.png", "segundos",
           lambda f: [("entrenar", [float(r["entrenar_s"]) for r in f]),
                      ("construir base", [float(r["construir_s"]) for r in f])])

    fig, axs = plt.subplots(1, len(tablas), figsize=(5.5 * len(tablas), 4), squeeze=False)
    for ax, (t, filas) in zip(axs[0], tablas.items()):
        ks = [int(r["K"]) for r in filas]
        tam = [np.load(os.path.join(out_dir, f"{t}_k{k}.npz"))["tamanos"] for k in ks]
        ax.boxplot(tam, tick_labels=[str(k) for k in ks], showfliers=False,
                   medianprops={"color": SERIES[0], "linewidth": 2},
                   boxprops={"color": TINTA_2}, whiskerprops={"color": TINTA_2},
                   capprops={"color": TINTA_2})
        ax.set_yscale("log")
        ax.grid(True, axis="y", color=REJILLA, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("K (palabras)")
        ax.set_ylabel("descriptores por palabra (muestra)")
        ax.set_title(titulos[t], loc="left", fontsize=11, color=TINTA)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "tamanos_vs_k.png"), dpi=150)
    plt.close(fig)
    print(f"gráficas en {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="codebook con k-means y barrido de K")
    ap.add_argument("--tipo", choices=["sift", "mfcc", "ambos"], default="ambos")
    ap.add_argument("--ks", type=int, nargs="+", default=[128, 256, 512, 1024, 2048])
    ap.add_argument("--solo-graficas", action="store_true")
    ap.add_argument("--data", default=DATA)
    args = ap.parse_args()
    if not args.solo_graficas:
        for t in (["sift", "mfcc"] if args.tipo == "ambos" else [args.tipo]):
            experimento(t, args.ks, args.data)
    graficar(args.data)


if __name__ == "__main__":
    main()
