import os
import tempfile
import unittest

import numpy as np

from indexes.ivf import IVF
from indexes.metricas import knn_exacto

rng = np.random.default_rng(0)
centros = rng.standard_normal((20, 32)) * 4
X = (centros[rng.integers(0, 20, 3000)] + rng.standard_normal((3000, 32))).astype(np.float32)
Q = (centros[rng.integers(0, 20, 50)] + rng.standard_normal((50, 32))).astype(np.float32)
METRICAS = ["euclidiana", "producto_punto", "coseno"]


def recall(ivf, metrica, nprobe, k=10):
    ok = 0
    for q in Q:
        exacto, _ = knn_exacto(q, X, k, metrica)
        aprox, _ = ivf.buscar(q, k, nprobe)
        ok += len(set(exacto) & set(aprox))
    return ok / (k * len(Q))


class TestIVF(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.idx = {m: IVF(nlist=32, metrica=m).entrenar(X).agregar(X) for m in METRICAS}

    def test_cada_vector_en_una_lista(self):
        for ivf in self.idx.values():
            self.assertEqual(len(ivf), len(X))
            ids = np.concatenate(ivf.listas_ids)
            np.testing.assert_array_equal(np.sort(ids), np.arange(len(X)))

    def test_todas_las_listas_es_exacto(self):
        for m, ivf in self.idx.items():
            for q in Q[:10]:
                exacto, d_e = knn_exacto(q, X, 10, m)
                aprox, d_a = ivf.buscar(q, 10, nprobe=32)
                np.testing.assert_allclose(d_a, d_e, rtol=1e-4, atol=1e-4)
                self.assertEqual(set(exacto), set(aprox))

    def test_recall_sube_con_nprobe(self):
        for m, ivf in self.idx.items():
            r = [recall(ivf, m, p) for p in (1, 4, 16)]
            self.assertLessEqual(r[0], r[1] + 1e-9)
            self.assertLessEqual(r[1], r[2] + 1e-9)
            self.assertGreaterEqual(r[2], 0.9, m)

    def test_revisa_menos_que_todo(self):
        ivf = self.idx["coseno"]
        ivf.buscar(Q[0], 10, nprobe=2)
        self.assertLess(ivf.candidatos, len(X) / 4)

    def test_ordenado_por_distancia(self):
        for ivf in self.idx.values():
            _, d = ivf.buscar(Q[0], 10, nprobe=8)
            self.assertTrue((np.diff(d) >= 0).all())

    def test_varias_consultas(self):
        ids, dist = self.idx["euclidiana"].buscar(Q[:3], 5, nprobe=4)
        self.assertEqual(len(ids), 3)
        self.assertEqual(len(ids[0]), 5)

    def test_ids_propios(self):
        ivf = IVF(nlist=8, metrica="l2").entrenar(X[:500])
        ivf.agregar(X[:500], ids=np.arange(500) + 1000)
        ids, _ = ivf.buscar(X[7], 1, nprobe=8)
        self.assertEqual(ids[0], 1007)

    def test_agregar_despues(self):
        ivf = IVF(nlist=8).entrenar(X[:500]).agregar(X[:500])
        ivf.agregar(X[2999])
        ids, _ = ivf.buscar(X[2999], 1, nprobe=8)
        self.assertEqual(ids[0], 500)

    def test_k_mayor_que_candidatos(self):
        ivf = IVF(nlist=4).entrenar(X[:20]).agregar(X[:20])
        ids, _ = ivf.buscar(X[0], 50, nprobe=4)
        self.assertEqual(len(ids), 20)

    def test_guardar_y_cargar(self):
        ivf = self.idx["coseno"]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "ivf.npz")
            ivf.guardar(p)
            otro = IVF.cargar(p)
        self.assertEqual(otro.metrica.nombre, "coseno")
        np.testing.assert_array_equal(otro.tamanos(), ivf.tamanos())
        for q in Q[:5]:
            a, da = ivf.buscar(q, 10, 4)
            b, db = otro.buscar(q, 10, 4)
            np.testing.assert_array_equal(a, b)
            np.testing.assert_allclose(da, db)

    def test_errores(self):
        with self.assertRaises(RuntimeError):
            IVF().buscar(Q[0])
        with self.assertRaises(RuntimeError):
            IVF().agregar(X)
        with self.assertRaises(ValueError):
            IVF(nlist=100).entrenar(X[:10])


if __name__ == "__main__":
    unittest.main()
