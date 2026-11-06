import unittest

import numpy as np

from indexes.metricas import (METRICAS, coseno, euclidiana, get_metrica, knn_exacto,
                              producto_punto)

rng = np.random.default_rng(0)
X = rng.standard_normal((200, 16)).astype(np.float32)
q = rng.standard_normal(16).astype(np.float32)


class TestMetricas(unittest.TestCase):

    def test_contra_formula_directa(self):
        np.testing.assert_allclose(euclidiana(q, X), np.sqrt(((X - q) ** 2).sum(1)), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(producto_punto(q, X), X @ q, rtol=1e-5, atol=1e-5)
        cos = X @ q / (np.linalg.norm(X, axis=1) * np.linalg.norm(q))
        np.testing.assert_allclose(coseno(q, X), cos, rtol=1e-5, atol=1e-5)

    def test_formas(self):
        for m in METRICAS.values():
            self.assertEqual(m.distancia(q, X).shape, (200,))
            self.assertEqual(m.distancia(X[:5], X).shape, (5, 200))

    def test_mismo_vector(self):
        self.assertAlmostEqual(float(euclidiana(q, q[None])[0]), 0, places=3)
        self.assertAlmostEqual(float(coseno(q, q[None])[0]), 1, places=5)

    def test_coseno_ignora_magnitud(self):
        np.testing.assert_allclose(coseno(q, X), coseno(10 * q, 3 * X), rtol=1e-5, atol=1e-5)

    def test_vector_cero(self):
        d = coseno(np.zeros(16), X)
        self.assertFalse(np.isnan(d).any())
        np.testing.assert_array_equal(d, 0)

    def test_menor_es_mas_parecido(self):
        for m in METRICAS.values():
            np.testing.assert_array_equal(np.argsort(m.distancia(q, X)),
                                          np.argsort(-m.similitud(q, X)))

    def test_normalizados_mismo_ranking(self):
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        qn = q / np.linalg.norm(q)
        orden = [np.argsort(get_metrica(m).distancia(qn, Xn))[:10] for m in METRICAS]
        np.testing.assert_array_equal(orden[0], orden[1])
        np.testing.assert_array_equal(orden[1], orden[2])

    def test_alias(self):
        self.assertIs(get_metrica("L2"), METRICAS["euclidiana"])
        self.assertIs(get_metrica("cosine"), METRICAS["coseno"])
        self.assertIs(get_metrica("producto punto"), METRICAS["producto_punto"])
        self.assertIs(get_metrica(METRICAS["coseno"]), METRICAS["coseno"])
        with self.assertRaises(ValueError):
            get_metrica("manhattan")

    def test_knn_exacto(self):
        for nombre, m in METRICAS.items():
            ids, dist = knn_exacto(q, X, k=10, metrica=nombre)
            np.testing.assert_array_equal(ids, np.argsort(m.distancia(q, X), kind="stable")[:10])
            self.assertTrue((np.diff(dist) >= 0).all())

    def test_knn_varias_consultas_y_k_grande(self):
        ids, dist = knn_exacto(X[:3], X, k=5, metrica="euclidiana")
        self.assertEqual(ids.shape, (3, 5))
        np.testing.assert_array_equal(ids[:, 0], [0, 1, 2])
        ids, _ = knn_exacto(q, X[:4], k=10)
        self.assertEqual(len(ids), 4)


if __name__ == "__main__":
    unittest.main()
