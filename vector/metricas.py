# métricas de similitud para ivf y hnsw: euclidiana, producto punto y coseno
# todas devuelven una distancia donde menor = más parecido, así se ordenan igual
# uso: m = get_metrica("coseno"); m.distancia(q, X)
import numpy as np


def _matriz(a):
    a = np.asarray(a, dtype=np.float32)
    return a[None] if a.ndim == 1 else a


def _salida(d, q):
    return d[0] if np.ndim(q) == 1 else d


def producto_punto(q, X):
    return _salida(_matriz(q) @ _matriz(X).T, q)


def coseno(q, X):
    Q, X = _matriz(q), _matriz(X)
    nq = np.linalg.norm(Q, axis=1, keepdims=True)
    nx = np.linalg.norm(X, axis=1)
    sim = (Q @ X.T) / np.maximum(nq * nx, 1e-12)   # un vector en cero queda con similitud 0
    return _salida(sim, q)


def euclidiana(q, X):
    # ||q - x||² = ||q||² + ||x||² - 2 q·x, en una sola multiplicación de matrices
    # en float64 porque en float32 la resta pierde precisión cuando q ≈ x
    Q, X = _matriz(q).astype(np.float64), _matriz(X).astype(np.float64)
    d2 = (Q * Q).sum(1)[:, None] + (X * X).sum(1)[None] - 2 * Q @ X.T
    return _salida(np.sqrt(np.maximum(d2, 0)).astype(np.float32), q)


class Metrica:
    def __init__(self, nombre, distancia, similitud):
        self.nombre = nombre
        self.distancia = distancia
        self.similitud = similitud

    def __repr__(self):
        return f"Metrica({self.nombre})"


METRICAS = {
    "euclidiana": Metrica("euclidiana", euclidiana, lambda q, X: -euclidiana(q, X)),
    "producto_punto": Metrica("producto_punto", lambda q, X: -producto_punto(q, X), producto_punto),
    "coseno": Metrica("coseno", lambda q, X: 1 - coseno(q, X), coseno),
}
ALIAS = {"l2": "euclidiana", "euclidean": "euclidiana", "euclidiana": "euclidiana",
         "dot": "producto_punto", "ip": "producto_punto", "inner_product": "producto_punto",
         "producto_punto": "producto_punto", "cosine": "coseno", "cos": "coseno", "coseno": "coseno"}


def get_metrica(nombre):
    if isinstance(nombre, Metrica):
        return nombre
    clave = ALIAS.get(str(nombre).strip().lower().replace(" ", "_").replace("-", "_"))
    if clave is None:
        raise ValueError(f"métrica desconocida: {nombre!r} (usa {', '.join(METRICAS)})")
    return METRICAS[clave]


def knn_exacto(q, X, k=10, metrica="coseno"):
    # búsqueda lineal: es la respuesta exacta contra la que se mide el recall de ivf y hnsw
    d = get_metrica(metrica).distancia(q, X)
    D = d[None] if d.ndim == 1 else d
    k = min(k, D.shape[1])
    top = np.argpartition(D, k - 1, axis=1)[:, :k]
    orden = np.take_along_axis(D, top, axis=1).argsort(axis=1, kind="stable")
    ids = np.take_along_axis(top, orden, axis=1)
    dist = np.take_along_axis(D, ids, axis=1)
    return (ids[0], dist[0]) if d.ndim == 1 else (ids, dist)
