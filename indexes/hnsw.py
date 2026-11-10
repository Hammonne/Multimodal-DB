# índice hnsw (#8): grafo por niveles, arriba pocos nodos con saltos largos y abajo todos
# la búsqueda baja nivel por nivel acercándose a la consulta (malkov y yashunin, 2018)
# uso: h = HNSW(M=16, ef_construction=100, metrica="coseno").agregar(X); h.buscar(q, k=10, ef_search=50)
import heapq
import math

import numpy as np

from indexes.metricas import get_metrica


class HNSW:
    def __init__(self, M=16, ef_construction=100, metrica="coseno", heuristica=True, seed=0):
        self.M = M
        self.M0 = 2 * M                 # en el nivel 0 se permiten el doble de vecinos
        self.ef_construction = ef_construction
        self.metrica = get_metrica(metrica)
        self.heuristica = heuristica
        self.mL = 1 / math.log(M)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.vectores = None
        self.ids = np.zeros(0, np.int64)
        self.nivel = []                 # nivel más alto de cada nodo
        self.vecinos = []               # vecinos[nodo][l] = lista de nodos en el nivel l
        self.entrada = None
        self.candidatos = 0

    def __len__(self):
        return len(self.nivel)

    def _dist(self, q, nodos):
        self.candidatos += len(nodos)
        return self.metrica.distancia(q, self.vectores[nodos])

    def _buscar_nivel(self, q, entradas, ef, l, permitido=None):
        # búsqueda voraz en un nivel: candidatos por explorar (min-heap) y los ef mejores (max-heap)
        # permitido (set de ids o None): búsqueda filtrada como hnswlib / qdrant. El grafo se recorre
        # entero (los nodos que no cumplen sirven de puente), pero solo los permitidos entran al resultado
        ok = (lambda n: True) if permitido is None else (lambda n: int(self.ids[n]) in permitido)
        d = self._dist(q, entradas)
        visto = set(entradas)
        cand = [(float(x), n) for x, n in zip(d, entradas)]
        heapq.heapify(cand)
        mejores = [(-float(x), n) for x, n in zip(d, entradas) if ok(n)]
        heapq.heapify(mejores)
        while len(mejores) > ef:
            heapq.heappop(mejores)
        while cand:
            dc, c = heapq.heappop(cand)
            if len(mejores) >= ef and dc > -mejores[0][0]:
                break
            nuevos = [n for n in self.vecinos[c][l] if n not in visto]
            if not nuevos:
                continue
            visto.update(nuevos)
            for dn, n in zip(self._dist(q, nuevos), nuevos):
                dn = float(dn)
                if len(mejores) < ef or dn < -mejores[0][0]:
                    heapq.heappush(cand, (dn, n))
                    if ok(n):
                        heapq.heappush(mejores, (-dn, n))
                        if len(mejores) > ef:
                            heapq.heappop(mejores)
        return sorted((-d, n) for d, n in mejores)

    def _elegir_vecinos(self, q_vec, candidatos, m):
        # heurística: un candidato entra solo si está más cerca de q que de los ya elegidos,
        # así los vecinos apuntan en direcciones distintas y el grafo no se queda en un grupo
        if not self.heuristica or len(candidatos) <= m:
            return [n for _, n in candidatos[:m]]
        elegidos, descartados = [], []
        for d, n in candidatos:
            if len(elegidos) >= m:
                break
            if not elegidos or d < self.metrica.distancia(self.vectores[n], self.vectores[elegidos]).min():
                elegidos.append(n)
            else:
                descartados.append(n)
        for n in descartados:           # si sobran lugares se completan con los descartados
            if len(elegidos) >= m:
                break
            elegidos.append(n)
        return elegidos

    def _insertar(self, nodo):
        q = self.vectores[nodo]
        l_nodo = int(-math.log(1 - self.rng.random()) * self.mL)
        self.nivel.append(l_nodo)
        self.vecinos.append([[] for _ in range(l_nodo + 1)])
        if self.entrada is None:
            self.entrada = nodo
            return
        ep = [self.entrada]
        l_max = self.nivel[self.entrada]
        for l in range(l_max, l_nodo, -1):
            ep = [self._buscar_nivel(q, ep, 1, l)[0][1]]
        for l in range(min(l_max, l_nodo), -1, -1):
            w = self._buscar_nivel(q, ep, self.ef_construction, l)
            m_max = self.M0 if l == 0 else self.M
            elegidos = self._elegir_vecinos(q, w, self.M)
            self.vecinos[nodo][l] = elegidos
            for n in elegidos:
                lista = self.vecinos[n][l]
                lista.append(nodo)
                if len(lista) > m_max:
                    d = self.metrica.distancia(self.vectores[n], self.vectores[lista])
                    orden = sorted(zip(d.tolist(), lista))
                    self.vecinos[n][l] = self._elegir_vecinos(self.vectores[n], orden, m_max)
            ep = [n for _, n in w]
        if l_nodo > l_max:
            self.entrada = nodo

    def agregar(self, X, ids=None):
        X = np.asarray(X, np.float32)
        if X.ndim == 1:
            X = X[None]
        inicio = len(self)
        ids = np.arange(inicio, inicio + len(X)) if ids is None else np.asarray(ids, np.int64)
        self.vectores = X.copy() if self.vectores is None else np.vstack([self.vectores, X])
        self.ids = np.concatenate([self.ids, ids])
        for nodo in range(inicio, inicio + len(X)):
            self._insertar(nodo)
        return self

    def _buscar_uno(self, q, k, ef_search, permitido=None):
        ep = [self.entrada]
        for l in range(self.nivel[self.entrada], 0, -1):
            ep = [self._buscar_nivel(q, ep, 1, l)[0][1]]
        # el filtro solo se aplica en el nivel 0: arriba solo se busca un buen punto de entrada
        w = self._buscar_nivel(q, ep, max(ef_search, k), 0, permitido)[:k]
        return self.ids[[n for _, n in w]], np.array([d for d, _ in w], np.float32)

    def buscar(self, q, k=10, ef_search=50, permitido=None):
        # (ids, distancias) ordenados de más a menos parecido, menor distancia = más parecido
        # permitido: set de ids que pueden salir en el resultado (búsqueda con filtro); None = todos
        if self.entrada is None:
            raise RuntimeError("el índice está vacío, primero agregar()")
        q = np.asarray(q, np.float32)
        self.candidatos = 0
        if q.ndim == 1:
            return self._buscar_uno(q, k, ef_search, permitido)
        res = [self._buscar_uno(x, k, ef_search, permitido) for x in q]
        return [r[0] for r in res], [r[1] for r in res]

    def niveles(self):
        # cuántos nodos llegan a cada nivel
        return np.bincount(self.nivel)[::-1].cumsum()[::-1]

    def guardar(self, path):
        planos, inicios = [], [0]
        for nodo in range(len(self)):
            for l in range(self.nivel[nodo] + 1):
                planos.extend(self.vecinos[nodo][l])
                inicios.append(len(planos))
        np.savez(path, vectores=self.vectores, ids=self.ids, nivel=np.array(self.nivel),
                 vecinos=np.array(planos, np.int64), inicios=np.array(inicios, np.int64),
                 entrada=-1 if self.entrada is None else self.entrada,
                 params=np.array([self.M, self.ef_construction, int(self.heuristica), self.seed]),
                 metrica=self.metrica.nombre)

    @classmethod
    def cargar(cls, path):
        z = np.load(path)
        M, efc, heur, seed = (int(x) for x in z["params"])
        h = cls(M, efc, str(z["metrica"]), bool(heur), seed)
        h.vectores, h.ids, h.nivel = z["vectores"], z["ids"], z["nivel"].tolist()
        planos, inicios = z["vecinos"].tolist(), z["inicios"]
        i = 0
        for nodo in range(len(h.nivel)):
            listas = []
            for _ in range(h.nivel[nodo] + 1):
                listas.append(planos[inicios[i]:inicios[i + 1]])
                i += 1
            h.vecinos.append(listas)
        h.entrada = None if int(z["entrada"]) < 0 else int(z["entrada"])
        return h
