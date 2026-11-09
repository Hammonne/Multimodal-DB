import unittest

import numpy as np

from multimedia import huellas as H
from multimedia import tarareo as T


def cancion_falsa(seed, segundos=12):
    # una nota distinta cada 0.25 s, con un poco de ruido
    rng = np.random.default_rng(seed)
    t = np.arange(int(H.SR * .25)) / H.SR
    notas = [np.sin(2 * np.pi * rng.uniform(200, 2000) * t) for _ in range(segundos * 4)]
    y = np.concatenate(notas).astype(np.float32)
    return y + rng.normal(0, .01, len(y)).astype(np.float32)


class TestHuellas(unittest.TestCase):

    def test_encuentra_el_fragmento(self):
        canciones = [(i, cancion_falsa(i)) for i in range(3)]
        idx = H.IndiceHuellas().construir(canciones)
        frag = canciones[1][1][H.SR * 4:H.SR * 8]
        self.assertEqual(idx.buscar(frag, 1)[0][0], 1)

    def test_aguanta_ruido(self):
        canciones = [(i, cancion_falsa(i)) for i in range(3)]
        idx = H.IndiceHuellas().construir(canciones)
        frag = canciones[2][1][H.SR * 3:H.SR * 7]
        frag = frag + np.random.default_rng(9).normal(0, .2, len(frag)).astype(np.float32)
        self.assertEqual(idx.buscar(frag, 1)[0][0], 2)


class TestTarareo(unittest.TestCase):

    def test_vector_fijo_ignora_el_tono(self):
        m = np.array([60, 62, 64, 65, 67, 65, 64, 62] * 4, float)
        a, b = T.vector_fijo(m), T.vector_fijo(m + 5)
        self.assertTrue(np.allclose(a, b, atol=1e-5))

    def test_vector_fijo_ignora_la_velocidad(self):
        m = np.repeat([60, 64, 67, 72], 20).astype(float)
        lento = np.repeat([60, 64, 67, 72], 30).astype(float)
        self.assertGreater(float(T.vector_fijo(m) @ T.vector_fijo(lento)), .95)

    def test_salta_huecos(self):
        m = np.array([60, np.nan, 62, 64, np.nan, 65, 67, 69, 71, 72], float)
        self.assertEqual(len(T.vector_fijo(m)), T.L)


if __name__ == "__main__":
    unittest.main()
