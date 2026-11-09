# índice vectorial (#10): ivf o hnsw sobre una columna que guarda rutas de imágenes o audios
# cada ruta se vuelve un vector tf-idf con ExtractFeatures y el índice guarda (vector, rid)
# se guarda en <nombre>.npz y se reconstruye solo si la tabla cambió con el índice cerrado
import os

import numpy as np

from indexes.hnsw import HNSW
from indexes.ivf import IVF
from indexes.metricas import get_metrica
from multimedia import features as F
from storage.formats.data_types import MEDIA_DATA_TYPES
from storage.modelo_multimedia import ModeloMultimedia
from storage.rid import RID

# los vectores van normalizados, así coseno, euclidiana y producto punto dan el mismo
# ranking y un solo índice sirve para las tres métricas del WITH METRIC
METRICA_INDICE = "coseno"


def limpiar(valor):
    # los varchar vuelven con relleno de \x00 o espacios
    if isinstance(valor, bytes):
        valor = valor.decode("utf-8", "ignore")
    return str(valor).replace("\x00", "").strip()


def ruta_archivo(valor, data=F.DATA):
    # ruta tal cual si existe, si no relativa a data/ (así se guardan en songs.csv)
    ruta = limpiar(valor)
    if os.path.isabs(ruta) or os.path.exists(ruta):
        return os.path.abspath(ruta)
    return os.path.abspath(os.path.join(data, ruta))


class Vectorizador:
    # usa lo ya calculado cuando se puede: el vector de la base tf-idf o los descriptores en .npy
    # para un archivo nuevo (la consulta) recién corre sift o mfcc

    def __init__(self, data=F.DATA, k=F.K_DEFECTO):
        self.data = os.path.abspath(data)
        self.k = k
        self._bases = {}
        self._codebooks = {}

    def _base(self, tipo):
        if tipo not in self._bases:
            try:
                b = F.cargar_base(tipo, self.k, self.data)
                self._bases[tipo] = (dict(zip(b["ids"].tolist(), range(len(b["ids"])))),
                                     b["vectores"], b["idf"])
            except FileNotFoundError:
                self._bases[tipo] = ({}, None, None)
        return self._bases[tipo]

    def _codebook(self, tipo):
        if tipo not in self._codebooks:
            self._codebooks[tipo] = F.cargar_codebook(tipo, self.k, self.data)
        return self._codebooks[tipo]

    def vector(self, ruta):
        tipo = F.tipo_de(ruta)
        rel = os.path.relpath(ruta, self.data)
        stem = os.path.splitext(os.path.basename(ruta))[0]
        ids, vectores, idf = self._base(tipo)

        clave = rel if tipo == "sift" else (stem if rel.startswith("audio" + os.sep) else None)
        if clave in ids:
            return vectores[ids[clave]]

        if idf is not None and not rel.startswith(".."):
            carpeta = os.path.join("sift", os.path.dirname(rel)) if tipo == "sift" else "mfcc"
            npy = os.path.join(self.data, "features", carpeta, stem + ".npy")
            if os.path.exists(npy):
                tf = F.histograma(F.palabras(np.load(npy), self._codebook(tipo)), self.k)
                return F.pesar(tf, idf)

        if not os.path.exists(ruta):
            raise FileNotFoundError(f"no existe el archivo '{ruta}'")
        return F.ExtractFeatures(ruta, k=self.k, data=self.data)


class VectorSecondaryIndex:

    def __init__(self, filename, tipo_indice, vectorizador=None):
        self.filename = filename
        self.tipo_indice = tipo_indice          # "ivf" | "hnsw"
        self.vectorizador = vectorizador or Vectorizador()
        self.tipo = None                        # "sift" | "mfcc", sale de la primera fila
        self.indice = None
        self.rids = []                          # posición en el índice -> RID
        self.borrados = set()
        self.pendientes = []                    # filas que llegaron antes de tener índice
        self.candidatos = 0
        # columnas IMAGE / AUDIO: modelo propio (codebook + idf) entrenado con los archivos de la tabla
        self.modelo = None
        self.pendientes_media = []              # (ruta, rid) que esperan a que se entrene el modelo

    # construcción

    def construir(self, table, column_index, nlist=100, M=16, ef_construction=100):
        self.nlist, self.M, self.ef_construction = nlist, M, ef_construction
        tipo_columna = table.schema[column_index].strip().lower()
        if tipo_columna in MEDIA_DATA_TYPES:
            return self._construir_media(table, column_index, MEDIA_DATA_TYPES[tipo_columna])
        vectores, rids = [], []
        for rid, registro in table.scan():
            v = self._vector_de(registro[column_index])
            if v is not None:
                vectores.append(v)
                rids.append(rid)
        self.rids = rids
        self.borrados = set()
        if not vectores:
            return self
        X = np.vstack(vectores).astype(np.float32)
        if self.tipo_indice == "ivf":
            self.indice = IVF(nlist=min(nlist, len(X)), metrica=METRICA_INDICE)
            self.indice.entrenar(X).agregar(X)
        else:
            self.indice = HNSW(M=M, ef_construction=ef_construction, metrica=METRICA_INDICE)
            self.indice.agregar(X)
        self.guardar()
        return self

    def _construir_media(self, table, column_index, tipo):
        # columna IMAGE o AUDIO: se entrena un modelo con los archivos de esta tabla (no con nuestro dataset)
        self.tipo = tipo
        self.modelo = ModeloMultimedia(tipo)
        self.pendientes_media = []
        for rid, registro in table.scan():
            if limpiar(registro[column_index]):
                self.pendientes_media.append((ruta_archivo(registro[column_index], self.vectorizador.data), rid))
        self.rids, self.borrados, self.indice = [], set(), None
        self._entrenar_media()
        return self

    def _entrenar_media(self):
        # entrena el codebook con todo lo pendiente y arma el ivf o el hnsw; si no hay filas espera a la primera
        if not self.pendientes_media:
            return
        faltan = [r for r, _ in self.pendientes_media if not os.path.exists(r)]
        if faltan:
            raise FileNotFoundError(f"no existe el archivo '{faltan[0]}' (y {len(faltan) - 1} más)"
                                    if len(faltan) > 1 else f"no existe el archivo '{faltan[0]}'")
        rutas = [r for r, _ in self.pendientes_media] + getattr(self, "_rutas", [])
        rids = [rid for _, rid in self.pendientes_media] + [r for i, r in enumerate(self.rids)
                                                          if i not in self.borrados]
        self.modelo = ModeloMultimedia(self.tipo).entrenar(rutas)
        X = self.modelo.vectores_entrenamiento()
        self._armar(X)
        self.rids, self.borrados, self._rutas = rids, set(), rutas
        self.pendientes_media = []
        self.guardar()

    def _armar(self, X):
        nlist, M, efc = getattr(self, "nlist", 100), getattr(self, "M", 16), getattr(self, "ef_construction", 100)
        if self.tipo_indice == "ivf":
            self.indice = IVF(nlist=max(1, min(nlist, len(X))), metrica=METRICA_INDICE)
            self.indice.entrenar(X).agregar(X)
        else:
            self.indice = HNSW(M=M, ef_construction=efc, metrica=METRICA_INDICE)
            self.indice.agregar(X)

    def vector_consulta(self, ruta):
        # el vector de un archivo de consulta con el mismo modelo que la columna
        if self.modelo is not None:
            self._entrenar_media()
            return self.modelo.vector(ruta)
        return self.vectorizador.vector(ruta)

    def _vector_de(self, valor):
        if not limpiar(valor):
            return None
        ruta = ruta_archivo(valor, self.vectorizador.data)
        tipo = F.tipo_de(ruta)
        if self.tipo is None:
            self.tipo = tipo
        elif tipo != self.tipo:
            raise ValueError(f"la columna mezcla imágenes y audios ('{limpiar(valor)}')")
        return self.vectorizador.vector(ruta)

    # mantenimiento desde Table.insert / delete / update

    def _insert_ref(self, key, rid):
        if self.modelo is not None:
            # columna IMAGE / AUDIO: con el modelo ya entrenado se agrega directo; si no, se espera
            if not limpiar(key):
                return
            ruta = ruta_archivo(key, self.vectorizador.data)
            if self.modelo.codebook is None or self.indice is None:
                self.pendientes_media.append((ruta, rid))
                return
            self.indice.agregar(self.modelo.vector(ruta)[None])
            self.rids.append(rid)
            self._rutas = getattr(self, "_rutas", []) + [ruta]
            return
        v = self._vector_de(key)
        if v is None:
            return
        if self.indice is None:
            self.pendientes.append((v, rid))
            if len(self.pendientes) >= 2:
                X = np.vstack([p[0] for p in self.pendientes])
                self.indice = (IVF(nlist=1, metrica=METRICA_INDICE).entrenar(X)
                               if self.tipo_indice == "ivf"
                               else HNSW(metrica=METRICA_INDICE))
                self.indice.agregar(X)
                self.rids = [p[1] for p in self.pendientes]
                self.pendientes = []
            return
        self.indice.agregar(v[None])
        self.rids.append(rid)

    def delete_ref(self, key, rid):
        # el ivf y el hnsw no borran: se marca y se salta al buscar
        for i, r in enumerate(self.rids):
            if r == rid and i not in self.borrados:
                self.borrados.add(i)
                return

    # búsqueda

    def __len__(self):
        return len(self.rids) - len(self.borrados) + len(self.pendientes) + len(self.pendientes_media)

    def knn(self, q, k, nprobe=10, ef_search=50):
        # [(rid, vector)] de los k más parecidos, del más al menos parecido
        if self.modelo is not None and self.pendientes_media:
            self._entrenar_media()
        self.candidatos = 0
        if self.indice is None:
            if not self.pendientes:
                return []
            X = np.vstack([p[0] for p in self.pendientes])
            orden = np.argsort(get_metrica(METRICA_INDICE).distancia(q, X), kind="stable")
            self.candidatos = len(X)
            return [(self.pendientes[i][1], self.pendientes[i][0]) for i in orden[:k]]
        pedir = k + len(self.borrados)
        if self.tipo_indice == "ivf":
            ids, _ = self.indice.buscar(q, pedir, nprobe)
        else:
            ids, _ = self.indice.buscar(q, pedir, max(ef_search, pedir))
        self.candidatos = self.indice.candidatos
        salida = []
        for i in ids:
            i = int(i)
            if i in self.borrados:
                continue
            salida.append((self.rids[i], self.vector(i)))
            if len(salida) == k:
                break
        return salida

    def vector(self, i):
        if self.tipo_indice == "ivf":
            for vs, ids in zip(self.indice.listas_vec, self.indice.listas_ids):
                pos = np.flatnonzero(ids == i)
                if len(pos):
                    return vs[pos[0]]
        return self.indice.vectores[i]

    def todos(self):
        # [(rid, vector)] de todo lo vigente, para la búsqueda lineal exacta
        if self.modelo is not None and self.pendientes_media:
            self._entrenar_media()
        if self.indice is None:
            return [(r, v) for v, r in self.pendientes]
        return [(r, self.vector(i)) for i, r in enumerate(self.rids) if i not in self.borrados]

    # disco: <nombre>.npz con los rid y <nombre>_indice.npz con el ivf o el hnsw

    def _ruta_indice(self):
        base = self.filename[:-4] if self.filename.endswith(".npz") else self.filename
        return base + "_indice.npz"

    def _ruta_modelo(self):
        base = self.filename[:-4] if self.filename.endswith(".npz") else self.filename
        return base + "_modelo.npz"

    def guardar(self):
        if self.indice is None:
            return
        if self.modelo is not None and self.modelo.codebook is not None:
            self.modelo.guardar(self._ruta_modelo())
        self.indice.guardar(self._ruta_indice())
        np.savez(self.filename, rids=np.array(self.rids, np.int64).reshape(-1, 2),
                 borrados=np.array(sorted(self.borrados), np.int64),
                 tipo=self.tipo or "", tipo_indice=self.tipo_indice)

    def cargar(self, n_filas):
        # True si lo de disco sirve (misma cantidad de filas vigentes), si no hay que reconstruir
        ruta_indice = self._ruta_indice()
        if not (os.path.exists(self.filename) and os.path.exists(ruta_indice)):
            return False
        z = np.load(self.filename)
        rids = [RID(int(a), int(b)) for a, b in z["rids"]]
        borrados = set(z["borrados"].tolist())
        if len(rids) - len(borrados) != n_filas:
            return False
        self.rids, self.borrados, self.tipo = rids, borrados, str(z["tipo"]) or None
        self.indice = (IVF if self.tipo_indice == "ivf" else HNSW).cargar(ruta_indice)
        if os.path.exists(self._ruta_modelo()):
            self.modelo = ModeloMultimedia.cargar(self._ruta_modelo())
        return True

    def close(self):
        self.guardar()

    def eliminar_archivos(self):
        for f in (self.filename, self._ruta_indice(), self._ruta_modelo()):
            if os.path.exists(f):
                os.remove(f)


def score(metrica, q, v):
    # SIMILARITY_SCORE(): mayor = más parecido, así ORDER BY score DESC sirve con las tres
    m = get_metrica(metrica)
    if m.nombre == "euclidiana":
        return float(1.0 / (1.0 + m.distancia(q, v[None])[0]))
    return float(m.similitud(q, v[None])[0])
