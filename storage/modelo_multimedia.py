# modelo de una columna IMAGE o AUDIO: su propio codebook e idf, entrenados con los archivos de esa tabla
# así el motor funciona con cualquier dataset, no solo con el nuestro (que tiene su codebook en data/features)
# uso: m = ModeloMultimedia("sift").entrenar(rutas); v = m.vector("consulta.jpg"); m.guardar("x.npz")
import os
import time

import numpy as np

from multimedia import features as F

K_MAXIMO = 1024
K_MINIMO = 8
MUESTRA_KMEANS = 200_000      # descriptores para entrenar k-means; con más casi no cambia y tarda mucho


_CACHE = {}                   # (ruta, fecha, tipo) -> descriptores: la lineal, el ivf y el hnsw no los repiten
CACHE_MAXIMO = 5000


def k_sugerido(n_descriptores):
    # regla práctica: K ~ raíz del total de descriptores, entre 8 y 1024
    # (20 000 portadas x ~450 descriptores -> 1024; 50 portadas -> ~150)
    return int(max(K_MINIMO, min(K_MAXIMO, round(np.sqrt(n_descriptores)))))


class ModeloMultimedia:

    def __init__(self, tipo, k=None):
        if tipo not in ("sift", "mfcc"):
            raise ValueError("el tipo del modelo es sift (imagen) o mfcc (audio)")
        self.tipo = tipo
        self.k = k
        self.codebook = None          # {"centroides", "media", "desv"} como en multimedia/features.py
        self.idf = None
        self.entrenado_con = 0
        self.segundos = 0.0

    def descriptores(self, ruta):
        # sift: (n, 128) por imagen; mfcc: (n, 39) por audio. Una imagen plana puede dar 0 filas
        clave = (os.path.abspath(ruta), os.path.getmtime(ruta), self.tipo)
        if clave not in _CACHE:
            if len(_CACHE) >= CACHE_MAXIMO:
                _CACHE.pop(next(iter(_CACHE)))
            d = F.descriptores(ruta, self.tipo)
            _CACHE[clave] = np.asarray(d, np.float32).reshape(-1, 128 if self.tipo == "sift" else 39)
        return _CACHE[clave]

    def entrenar(self, rutas, descriptores=None):
        # rutas: archivos de la tabla; devuelve self. descriptores: los ya calculados, para no repetir
        from sklearn.cluster import MiniBatchKMeans
        t0 = time.perf_counter()
        descs = descriptores if descriptores is not None else [self.descriptores(r) for r in rutas]
        con_datos = [d for d in descs if len(d)]
        if not con_datos:
            raise ValueError("ningún archivo de la columna tiene descriptores (¿están vacíos o no se pueden leer?)")
        todos = np.vstack(con_datos)
        media, desv = todos.mean(0), todos.std(0) + 1e-6
        k = min(self.k or k_sugerido(len(todos)), len(todos))
        rng = np.random.default_rng(0)
        muestra = todos[rng.choice(len(todos), min(MUESTRA_KMEANS, len(todos)), replace=False)]
        km = MiniBatchKMeans(n_clusters=k, batch_size=4096, n_init=3, random_state=0)
        km.fit((muestra - media) / desv)
        self.k = k
        self.codebook = {"centroides": km.cluster_centers_.astype(np.float32),
                         "media": media.astype(np.float32), "desv": desv.astype(np.float32)}
        tfs = np.vstack([self._tf(d) for d in descs])
        self.idf = F.calcular_idf(tfs)
        self.entrenado_con = len(descs)
        self.segundos = time.perf_counter() - t0
        self._tfs = tfs
        return self

    def _tf(self, desc):
        return F.histograma(F.palabras(desc, self.codebook), self.k)

    def vectores_entrenamiento(self):
        # los vectores tf-idf de los archivos con los que se entrenó (sin volver a sacar descriptores)
        return F.pesar(self._tfs, self.idf)

    def vector_de_descriptores(self, desc):
        return F.pesar(self._tf(desc), self.idf)

    def vector(self, ruta):
        if self.codebook is None:
            raise RuntimeError("el modelo todavía no está entrenado")
        if not os.path.exists(ruta):
            raise FileNotFoundError(f"no existe el archivo '{ruta}'")
        return self.vector_de_descriptores(self.descriptores(ruta))

    def guardar(self, ruta):
        np.savez(ruta, tipo=self.tipo, k=self.k, centroides=self.codebook["centroides"],
                 media=self.codebook["media"], desv=self.codebook["desv"], idf=self.idf,
                 entrenado_con=self.entrenado_con, segundos=self.segundos)

    @classmethod
    def cargar(cls, ruta):
        z = np.load(ruta)
        m = cls(str(z["tipo"]), int(z["k"]))
        m.codebook = {"centroides": z["centroides"], "media": z["media"], "desv": z["desv"]}
        m.idf, m.entrenado_con, m.segundos = z["idf"], int(z["entrenado_con"]), float(z["segundos"])
        return m
