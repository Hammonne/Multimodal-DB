# búsqueda por cara: foto de una persona -> artistas o integrantes más parecidos
# descriptor global como en el lab 11: un embedding de 128 dims por cara (face_recognition), sin codebook
# los embeddings de faces.npz van directo a nuestro hnsw con euclidiana
# uso: python -m multimedia.caras foto.jpg [--k 5]   |   python -m multimedia.caras --evaluar
import argparse
import os
import time

import numpy as np

from indexes.hnsw import HNSW
from indexes.metricas import knn_exacto
from multimedia.sift import DATA

# face_recognition propone 0.6, pero con ~2 700 personas casi todo queda bajo 0.6 y acierta 79 %;
# con 0.45 responde en 70 % de las consultas y acierta 90 % (python -m multimedia.caras --evaluar)
UMBRAL = 0.45
M, EFC, EF = 16, 100, 50


def cargar_caras(data=DATA):
    z = np.load(os.path.join(data, "features", "faces.npz"))
    return {k: z[k] for k in z.files}


def ruta_indice(data=DATA):
    return os.path.join(data, "features", "faces_hnsw.npz")


def construir(caras, data=DATA, guardar=True):
    # ids = posición en faces.npz; el nombre y la foto salen de ahí
    indice = HNSW(M=M, ef_construction=EFC, metrica="euclidiana").agregar(caras["embedding"])
    if guardar:
        indice.guardar(ruta_indice(data))
    return indice


def abrir(data=DATA):
    caras = cargar_caras(data)
    ruta = ruta_indice(data)
    if os.path.exists(ruta):
        indice = HNSW.cargar(ruta)
        if len(indice) == len(caras["embedding"]):
            return caras, indice
    return caras, construir(caras, data)


def embeddings_de_foto(path):
    # [(caja, embedding)] de cada cara que aparezca en la foto
    import face_recognition
    from PIL import Image
    img = Image.open(path).convert("RGB")
    escala = min(1.0, 1000 / max(img.size))
    if escala < 1:
        img = img.resize((round(img.width * escala), round(img.height * escala)))
    arr = np.asarray(img)
    cajas = face_recognition.face_locations(arr)
    return list(zip(cajas, face_recognition.face_encodings(arr, cajas)))


def personas(caras, posiciones, distancias, k):
    # una persona puede tener varias fotos: se queda con su cara más cercana
    vistos, salida = set(), []
    for p, d in zip(posiciones, distancias):
        clave = (str(caras["kind"][p]), str(caras["owner_id"][p]))
        if clave in vistos:
            continue
        vistos.add(clave)
        salida.append({"nombre": str(caras["name"][p]), "tipo": clave[0], "id": clave[1],
                       "foto": str(caras["path"][p]), "distancia": float(d),
                       "misma_persona": bool(d < UMBRAL)})
        if len(salida) == k:
            break
    return salida


def buscar_embedding(caras, indice, emb, k=5):
    # se piden más caras que k porque varias pueden ser de la misma persona
    posiciones, distancias = indice.buscar(np.asarray(emb, np.float32), k * 5, max(EF, k * 5))
    return personas(caras, posiciones, distancias, k)


def buscar_foto(path, k=5, data=DATA):
    caras, indice = abrir(data)
    return [{"caja": caja, "resultados": buscar_embedding(caras, indice, emb, k)}
            for caja, emb in embeddings_de_foto(path)]


def evaluar(data=DATA, n_consultas=300, seed=0):
    # cada consulta es una cara de alguien con 2+ fotos, sacada de la base:
    # acierta si la persona más cercana es la misma (con otra foto)
    caras = cargar_caras(data)
    X = caras["embedding"]
    dueno = np.array([f"{a}:{b}" for a, b in zip(caras["kind"], caras["owner_id"])])
    _, inversa, cuenta = np.unique(dueno, return_inverse=True, return_counts=True)
    con_varias = np.flatnonzero(cuenta[inversa] >= 2)
    rng = np.random.default_rng(seed)
    q = rng.choice(con_varias, min(n_consultas, len(con_varias)), replace=False)
    base = np.setdiff1d(np.arange(len(X)), q)

    t0 = time.perf_counter()
    indice = HNSW(M=M, ef_construction=EFC, metrica="euclidiana").agregar(X[base])
    t_construir = time.perf_counter() - t0

    t0 = time.perf_counter()
    exactos = [base[knn_exacto(X[i], X[base], 10, "euclidiana")[0]] for i in q]
    ms_lineal = (time.perf_counter() - t0) / len(q) * 1000
    t0 = time.perf_counter()
    res = [base[indice.buscar(X[i], 10, EF)[0]] for i in q]
    ms_hnsw = (time.perf_counter() - t0) / len(q) * 1000

    recall = np.mean([len(set(r) & set(e)) / 10 for r, e in zip(res, exactos)])
    top1 = np.mean([dueno[r[0]] == dueno[i] for r, i in zip(res, q)])
    top5 = np.mean([dueno[i] in set(dueno[r[:5]]) for r, i in zip(res, q)])
    d1 = np.array([np.linalg.norm(X[r[0]] - X[i]) for r, i in zip(res, q)])
    misma = np.array([dueno[r[0]] == dueno[i] for r, i in zip(res, q)])
    bajo = d1 < UMBRAL
    return {"caras": int(len(X)), "personas": int(len(cuenta)), "consultas": int(len(q)),
            "construir_s": t_construir, "ms_lineal": ms_lineal, "ms_hnsw": ms_hnsw,
            "recall10": float(recall), "top1_persona": float(top1), "top5_persona": float(top5),
            "bajo_umbral": float(bajo.mean()),
            "acierto_si_bajo_umbral": float(misma[bajo].mean()) if bajo.any() else 0.0}


def main():
    ap = argparse.ArgumentParser(description="buscar artistas por cara con hnsw")
    ap.add_argument("foto", nargs="?")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--evaluar", action="store_true")
    ap.add_argument("--reconstruir", action="store_true")
    args = ap.parse_args()
    if args.evaluar:
        r = evaluar(args.data)
        print(f"{r['caras']} caras de {r['personas']} personas, {r['consultas']} consultas\n"
              f"construir hnsw: {r['construir_s']:.1f} s\n"
              f"consulta: hnsw {r['ms_hnsw']:.2f} ms vs lineal {r['ms_lineal']:.2f} ms "
              f"({r['ms_lineal'] / r['ms_hnsw']:.1f}x)\n"
              f"recall@10 contra la lineal: {r['recall10']:.3f}\n"
              f"la persona correcta es la 1.a: {r['top1_persona']:.1%}, está en el top 5 de caras: "
              f"{r['top5_persona']:.1%}\n"
              f"distancia < {UMBRAL}: {r['bajo_umbral']:.1%} de las consultas, "
              f"y de esas acierta {r['acierto_si_bajo_umbral']:.1%}")
        return
    if args.reconstruir:
        construir(cargar_caras(args.data), args.data)
    if not args.foto:
        ap.error("falta la foto (o --evaluar)")
    encontradas = buscar_foto(args.foto, args.k, args.data)
    if not encontradas:
        print("no se encontró ninguna cara en la foto")
    for j, c in enumerate(encontradas, 1):
        print(f"cara {j} {c['caja']}:")
        for r in c["resultados"]:
            marca = "misma persona" if r["misma_persona"] else ""
            print(f"  {r['distancia']:.3f}  {r['nombre']} ({r['tipo']})  {r['foto']}  {marca}")


if __name__ == "__main__":
    main()
