# índice ivf (#7): k-means parte los vectores en nlist celdas y solo se buscan las nprobe más cercanas
# igual que ivfflat de pgvector (lists, probes) pero propio, con las tres métricas de metricas.py
# uso: ivf = IVF(nlist=100, metrica="coseno").entrenar(X); ivf.agregar(X); ivf.buscar(q, k=10, nprobe=10)
import numpy as np
from sklearn.cluster import KMeans

from indexes.metricas import get_metrica


class IVF:
    def __init__(self, nlist=100, metrica="coseno", seed=0):
        self.nlist = nlist
        self.metrica = get_metrica(metrica)
        self.seed = seed
        self.centroides = None
        self.listas_vec = []
        self.listas_ids = []
        self.candidatos = 0

    def _celda(self, X):
        return np.argmin(self.metrica.distancia(X, self.centroides), axis=1)

    def entrenar(self, X):
        X = np.asarray(X, np.float32)
        if len(X) < self.nlist:
            raise ValueError(f"se necesitan al menos {self.nlist} vectores para {self.nlist} celdas")
        if self.metrica.nombre == "coseno":    # k-means esférico: agrupa por dirección
            X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        km = KMeans(n_clusters=self.nlist, init="k-means++", n_init=1, random_state=self.seed).fit(X)
        self.centroides = km.cluster_centers_.astype(np.float32)
        self.listas_vec = [np.zeros((0, X.shape[1]), np.float32) for _ in range(self.nlist)]
        self.listas_ids = [np.zeros(0, np.int64) for _ in range(self.nlist)]
        return self

    def agregar(self, X, ids=None):
        if self.centroides is None:
            raise RuntimeError("primero entrenar()")
        X = np.asarray(X, np.float32)
        if X.ndim == 1:
            X = X[None]
        inicio = sum(len(l) for l in self.listas_ids)
        ids = np.arange(inicio, inicio + len(X)) if ids is None else np.asarray(ids, np.int64)
        celdas = self._celda(X)
        for c in np.unique(celdas):
            m = celdas == c
            self.listas_vec[c] = np.vstack([self.listas_vec[c], X[m]])
            self.listas_ids[c] = np.concatenate([self.listas_ids[c], ids[m]])
        return self

    def _buscar_uno(self, q, k, nprobe):
        d_c = self.metrica.distancia(q, self.centroides)
        abiertas = np.argsort(d_c)[:min(nprobe, self.nlist)]
        V = np.vstack([self.listas_vec[c] for c in abiertas])
        I = np.concatenate([self.listas_ids[c] for c in abiertas])
        self.candidatos += len(I)
        if len(I) == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.float32)
        d = self.metrica.distancia(q, V)
        k = min(k, len(I))
        top = np.argpartition(d, k - 1)[:k]
        top = top[np.argsort(d[top], kind="stable")]
        return I[top], d[top]

    def buscar(self, q, k=10, nprobe=10):
        # (ids, distancias) ordenados de más a menos parecido, menor distancia = más parecido
        if self.centroides is None:
            raise RuntimeError("primero entrenar() y agregar()")
        q = np.asarray(q, np.float32)
        self.candidatos = 0
        if q.ndim == 1:
            return self._buscar_uno(q, k, nprobe)
        res = [self._buscar_uno(x, k, nprobe) for x in q]
        return [r[0] for r in res], [r[1] for r in res]

    def __len__(self):
        return sum(len(l) for l in self.listas_ids)

    def tamanos(self):
        return np.array([len(l) for l in self.listas_ids])

    def guardar(self, path):
        orden = np.concatenate([[0], np.cumsum(self.tamanos())])
        np.savez(path, centroides=self.centroides, metrica=self.metrica.nombre,
                 nlist=self.nlist, seed=self.seed, inicios=orden,
                 vectores=np.vstack(self.listas_vec), ids=np.concatenate(self.listas_ids))

    @classmethod
    def cargar(cls, path):
        z = np.load(path)
        ivf = cls(int(z["nlist"]), str(z["metrica"]), int(z["seed"]))
        ivf.centroides = z["centroides"]
        a = z["inicios"]
        # cada z["..."] vuelve a leer el arreglo entero del archivo: se lee una vez y se corta
        V, I = z["vectores"], z["ids"]
        ivf.listas_vec = [V[a[i]:a[i + 1]] for i in range(ivf.nlist)]
        ivf.listas_ids = [I[a[i]:a[i + 1]] for i in range(ivf.nlist)]
        return ivf
