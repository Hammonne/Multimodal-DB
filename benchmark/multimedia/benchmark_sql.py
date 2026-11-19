# benchmark de la parte 4 a través del SQL del motor, con CUALQUIER carpeta de imágenes o audios
# por cada tamaño: CREATE TABLE con una columna IMAGE o AUDIO, IMPORT INTO de la carpeta, CREATE INDEX IVF y HNSW
# (cada índice entrena su propio codebook con esos archivos) y consultas SIMILAR_TO k = 10 con y sin índice
# mide construcción, tiempo por consulta, recall@10 contra la búsqueda lineal y espacio en disco
# uso: python -m benchmark.multimedia.benchmark_sql --carpeta mis_fotos/ [--tamanos 100 250 500] [--consultas 30]
import argparse
import json
import os
import random
import shutil
import sys
import tempfile
import time

AQUI = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(AQUI))
sys.path.insert(0, os.path.join(REPO, "parser"))
sys.path.insert(1, REPO)

from executor import ExecuteVisitor  # noqa: E402
from parser import Parser  # noqa: E402
from scanner import Scanner  # noqa: E402
from multimedia import features as F  # noqa: E402
from storage.storage_manager import StorageManager  # noqa: E402

K = 10


def sql(v, texto):
    salida = None
    for salida in v.ejecutar(Parser(Scanner(texto)).parse_program()):
        pass
    return salida


def ids(resultado):
    return [fila[0] for fila in resultado.filas]


def medir_tamano(archivos, tipo_columna, consultas, trabajo):
    # una base nueva en una carpeta temporal: así cada tamaño parte de cero
    os.makedirs(trabajo, exist_ok=True)
    carpeta = os.path.join(trabajo, "archivos")
    os.makedirs(carpeta)
    for a in archivos:
        os.symlink(a, os.path.join(carpeta, os.path.basename(a)))
    anterior = os.getcwd()
    os.chdir(trabajo)
    try:
        v = ExecuteVisitor(StorageManager(wal_path="wal.log"))
        col = "archivo"
        sql(v, f"CREATE TABLE t (id INT PRIMARY KEY, nombre VARCHAR(80), {col} {tipo_columna}) USING HEAP;")
        t0 = time.perf_counter()
        sql(v, f"IMPORT INTO t FROM '{carpeta}';")
        fila = {"n": len(archivos), "importar_s": time.perf_counter() - t0}
        rutas_consulta = [os.path.join(carpeta, os.path.basename(c)) for c in consultas]

        def consultar(using):
            res, t0 = [], time.perf_counter()
            for r in rutas_consulta:
                res.append(ids(sql(v, f"SELECT id FROM t WHERE SIMILAR_TO('{r}', k={K}){using};")))
            return res, (time.perf_counter() - t0) / len(rutas_consulta) * 1000

        # lineal: la primera consulta entrena el modelo de la columna; se mide aparte
        t0 = time.perf_counter()
        sql(v, f"SELECT id FROM t WHERE SIMILAR_TO('{rutas_consulta[0]}', k={K}) USING SEQ;")
        fila["lineal_modelo_s"] = time.perf_counter() - t0
        exactos, fila["lineal_ms"] = consultar(" USING SEQ")
        for indice in ("ivf", "hnsw"):
            t0 = time.perf_counter()
            sql(v, f"CREATE INDEX ON t ({col}) USING {indice.upper()};")
            fila[f"{indice}_construir_s"] = time.perf_counter() - t0
            res, fila[f"{indice}_ms"] = consultar(f" USING {indice.upper()}")
            fila[f"{indice}_recall"] = sum(len(set(a) & set(b)) for a, b in zip(res, exactos)) / (K * len(res))
            base = f"idx_t_{col}_{indice}"
            fila[f"{indice}_kb"] = sum(os.path.getsize(f) for f in os.listdir(".")
                                       if f.startswith(base)) / 1024
            fila["k_codebook"] = int(next(i for idxs in v._abrir("t").secondary_indexes.values()
                                          for i in idxs if hasattr(i, "modelo") and i.modelo).modelo.k)
        return fila
    finally:
        os.chdir(anterior)


def graficar(filas, tipo, salida):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = [f["n"] for f in filas]
    colores = {"lineal": "#62615c", "ivf": "#2a78d6", "hnsw": "#eb6834"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for nombre in ("lineal", "ivf", "hnsw"):
        axes[0].plot(n, [f[f"{nombre}_ms"] for f in filas], "-o", color=colores[nombre],
                     label="Búsqueda lineal" if nombre == "lineal" else nombre.upper())
    axes[0].set_title("Tiempo por consulta SQL (ms)", loc="left")
    for nombre in ("ivf", "hnsw"):
        axes[1].plot(n, [f[f"{nombre}_recall"] for f in filas], "-o", color=colores[nombre], label=nombre.upper())
        axes[2].plot(n, [f[f"{nombre}_construir_s"] for f in filas], "-o", color=colores[nombre], label=nombre.upper())
    axes[1].axhline(1, color=colores["lineal"], label="Búsqueda lineal")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Recall@10 contra la búsqueda lineal", loc="left")
    axes[2].plot(n, [f["lineal_modelo_s"] for f in filas], "--o", color=colores["lineal"], label="modelo de la lineal")
    axes[2].set_title("CREATE INDEX (con su codebook) en s", loc="left")
    for ax in axes:
        ax.set_xlabel("filas en la tabla")
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Benchmark por SQL sobre una carpeta de {'imágenes' if tipo == 'IMAGE' else 'audios'} "
                 f"(cada índice entrena su propio codebook)", x=.01, ha="left")
    fig.tight_layout()
    fig.savefig(salida, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="benchmark por sql sobre cualquier carpeta de imágenes o audios")
    ap.add_argument("--carpeta", required=True)
    ap.add_argument("--tamanos", type=int, nargs="+")
    ap.add_argument("--consultas", type=int, default=30)
    ap.add_argument("--salida", default=os.path.join(AQUI, "sql"))
    args = ap.parse_args()
    archivos = sorted(os.path.abspath(os.path.join(args.carpeta, f)) for f in os.listdir(args.carpeta))
    imagenes = [a for a in archivos if os.path.splitext(a)[1].lower() in F.IMAGEN]
    audios = [a for a in archivos if os.path.splitext(a)[1].lower() in F.AUDIO]
    archivos, tipo = (imagenes, "IMAGE") if len(imagenes) >= len(audios) else (audios, "AUDIO")
    if not archivos:
        sys.exit("la carpeta no tiene imágenes ni audios")
    random.Random(0).shuffle(archivos)
    tamanos = sorted(t for t in (args.tamanos or [len(archivos)]) if t <= len(archivos))
    os.makedirs(args.salida, exist_ok=True)
    filas = []
    with tempfile.TemporaryDirectory() as tmp:
        for n in tamanos:
            base = archivos[:n]
            consultas = random.Random(n).sample(base, min(args.consultas, n))
            fila = medir_tamano(base, tipo, consultas, os.path.join(tmp, str(n)))
            filas.append(fila)
            print(f"n={n}: K={fila['k_codebook']} lineal {fila['lineal_ms']:.1f} ms · "
                  f"ivf {fila['ivf_construir_s']:.1f} s / {fila['ivf_ms']:.1f} ms / recall {fila['ivf_recall']:.3f} · "
                  f"hnsw {fila['hnsw_construir_s']:.1f} s / {fila['hnsw_ms']:.1f} ms / recall {fila['hnsw_recall']:.3f}",
                  flush=True)
    nombre = "imagenes" if tipo == "IMAGE" else "audios"
    with open(os.path.join(args.salida, f"resultados_{nombre}.json"), "w", encoding="utf-8") as fh:
        json.dump({"carpeta": os.path.abspath(args.carpeta), "tipo": tipo, "k": K, "filas": filas}, fh, indent=1)
    graficar(filas, tipo, os.path.join(args.salida, f"sql_{nombre}.png"))
    print(f"resultados y gráfica en {args.salida}")


if __name__ == "__main__":
    main()
