import os
import tempfile
import unittest

import numpy as np

from indexes.hnsw import HNSW
from indexes.metricas import knn_exacto

rng = np.random.default_rng(0)
centros = rng.standard_normal((20, 32)) * 4
X = (centros[rng.integers(0, 20, 2000)] + rng.standard_normal((2000, 32))).astype(np.float32)
Q = (centros[rng.integers(0, 20, 50)] + rng.standard_normal((50, 32))).astype(np.float32)
METRICAS = ["euclidiana", "producto_punto", "coseno"]


def recall(h, metrica, ef, k=10):
    ok = 0
    for q in Q:
        exacto, _ = knn_exacto(q, X, k, metrica)
        aprox, _ = h.buscar(q, k, ef)
        ok += len(set(exacto) & set(aprox))
    return ok / (k * len(Q))


class TestHNSW(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.idx = {m: HNSW(M=12, ef_construction=80, metrica=m).agregar(X) for m in METRICAS}

    def test_recall_alto(self):
        for m, h in self.idx.items():
            self.assertGreaterEqual(recall(h, m, 100), 0.95, m)

    def test_recall_sube_con_ef(self):
        h = self.idx["euclidiana"]
        r = [recall(h, "euclidiana", ef) for ef in (10, 40, 160)]
        self.assertLessEqual(r[0], r[1] + 0.02)
        self.assertLessEqual(r[1], r[2] + 0.02)
        self.assertGreater(r[2], r[0])

    def test_revisa_menos_que_todo(self):
        h = self.idx["coseno"]
        h.buscar(Q[0], 10, 50)
        self.assertLess(h.candidatos, len(X) / 2)

    def test_grado_maximo(self):
        for h in self.idx.values():
            for nodo in range(len(h)):
                for l, vec in enumerate(h.vecinos[nodo]):
                    self.assertLessEqual(len(vec), h.M0 if l == 0 else h.M)
                    self.assertNotIn(nodo, vec)

    def test_niveles(self):
        n = self.idx["coseno"].niveles()
        self.assertEqual(n[0], len(X))
        self.assertGreater(len(n), 1)
        self.assertTrue((np.diff(n) <= 0).all())
        self.assertLess(n[1], len(X) / 4)

    def test_ordenado_por_distancia(self):
        for h in self.idx.values():
            _, d = h.buscar(Q[0], 10, 50)
            self.assertTrue((np.diff(d) >= 0).all())

    def test_encuentra_el_mismo_vector(self):
        h = self.idx["euclidiana"]
        for i in (0, 500, 1999):
            ids, d = h.buscar(X[i], 1, 50)
            self.assertEqual(ids[0], i)
            self.assertAlmostEqual(float(d[0]), 0, places=3)

    def test_varias_consultas(self):
        ids, dist = self.idx["coseno"].buscar(Q[:3], 5, 30)
        self.assertEqual(len(ids), 3)
        self.assertEqual(len(ids[0]), 5)

    def test_ids_propios_y_agregar_despues(self):
        h = HNSW(M=8, ef_construction=40).agregar(X[:300], ids=np.arange(300) + 1000)
        h.agregar(X[1999], ids=[7])
        self.assertEqual(h.buscar(X[10], 1, 30)[0][0], 1010)
        self.assertEqual(h.buscar(X[1999], 1, 30)[0][0], 7)

    def test_k_mayor_que_n(self):
        h = HNSW(M=4).agregar(X[:15])
        ids, _ = h.buscar(X[0], 50, 50)
        self.assertEqual(len(ids), 15)

    def test_sin_heuristica(self):
        h = HNSW(M=12, ef_construction=80, metrica="euclidiana", heuristica=False).agregar(X[:800])
        exacto, _ = knn_exacto(Q[0], X[:800], 10, "euclidiana")
        aprox, _ = h.buscar(Q[0], 10, 100)
        self.assertGreaterEqual(len(set(exacto) & set(aprox)), 8)

    def test_guardar_y_cargar(self):
        h = self.idx["coseno"]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "hnsw.npz")
            h.guardar(p)
            otro = HNSW.cargar(p)
        self.assertEqual(otro.metrica.nombre, "coseno")
        self.assertEqual(otro.entrada, h.entrada)
        for q in Q[:5]:
            a, da = h.buscar(q, 10, 50)
            b, db = otro.buscar(q, 10, 50)
            np.testing.assert_array_equal(a, b)
            np.testing.assert_allclose(da, db)

    def test_vacio(self):
        with self.assertRaises(RuntimeError):
            HNSW().buscar(Q[0])


if __name__ == "__main__":
    unittest.main()
