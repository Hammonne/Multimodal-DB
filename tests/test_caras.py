import unittest

import numpy as np

from multimedia import caras as C


def caras_falsas():
    # 3 personas, 3 fotos cada una, cada persona alrededor de su propio centro
    rng = np.random.default_rng(0)
    centros = rng.normal(0, 1, (3, 128)).astype(np.float32)
    emb = np.vstack([c + rng.normal(0, .01, (3, 128)) for c in centros]).astype(np.float32)
    duenos = np.repeat(["1", "2", "3"], 3)
    return {"embedding": emb, "kind": np.array(["artist"] * 9), "owner_id": duenos,
            "name": np.array([f"persona {d}" for d in duenos]),
            "path": np.array([f"artists/{i}.jpg" for i in range(9)])}, centros


class TestCaras(unittest.TestCase):

    def test_persona_correcta_primero(self):
        caras, centros = caras_falsas()
        indice = C.construir(caras, guardar=False)
        res = C.buscar_embedding(caras, indice, centros[1], k=3)
        self.assertEqual(res[0]["id"], "2")
        self.assertTrue(res[0]["misma_persona"])

    def test_una_fila_por_persona(self):
        caras, centros = caras_falsas()
        indice = C.construir(caras, guardar=False)
        res = C.buscar_embedding(caras, indice, centros[0], k=3)
        self.assertEqual(len({r["id"] for r in res}), len(res))
        self.assertEqual(len(res), 3)

    def test_lejos_no_es_misma_persona(self):
        caras, centros = caras_falsas()
        indice = C.construir(caras, guardar=False)
        res = C.buscar_embedding(caras, indice, centros[0] + 5, k=1)
        self.assertFalse(res[0]["misma_persona"])


if __name__ == "__main__":
    unittest.main()
