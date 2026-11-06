import os
import shutil
import tempfile
import unittest

import cv2
import numpy as np

from multimedia import features as F

rng = np.random.default_rng(0)


def codebook_falso(k, dim):
    return {"centroides": rng.standard_normal((k, dim)).astype(np.float32),
            "media": np.zeros(dim, np.float32), "desv": np.ones(dim, np.float32)}


def imagen_con_textura(path, seed):
    r = np.random.default_rng(seed)
    img = (r.random((256, 256)) * 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    for _ in range(30):
        x, y = r.integers(0, 230, 2)
        cv2.rectangle(img, (int(x), int(y)), (int(x) + 20, int(y) + 12), int(r.integers(0, 255)), -1)
    cv2.imwrite(path, img)


class TestPiezas(unittest.TestCase):

    def test_tipo_de(self):
        self.assertEqual(F.tipo_de("a/b/portada.JPG"), "sift")
        self.assertEqual(F.tipo_de("cancion.mp3"), "mfcc")
        with self.assertRaises(ValueError):
            F.tipo_de("letra.txt")

    def test_palabras_es_el_centroide_mas_cercano(self):
        cb = codebook_falso(8, 4)
        desc = cb["centroides"][[3, 3, 5, 0]] + 1e-3
        np.testing.assert_array_equal(F.palabras(desc, cb), [3, 3, 5, 0])

    def test_palabras_por_lotes_igual(self):
        cb = codebook_falso(16, 6)
        desc = rng.standard_normal((1000, 6)).astype(np.float32)
        np.testing.assert_array_equal(F.palabras(desc, cb, lote=7), F.palabras(desc, cb))

    def test_palabras_usa_media_y_desv(self):
        cb = codebook_falso(4, 3)
        cb["media"], cb["desv"] = np.full(3, 10, np.float32), np.full(3, 2, np.float32)
        desc = cb["centroides"][[2]] * 2 + 10
        self.assertEqual(F.palabras(desc, cb)[0], 2)

    def test_histograma(self):
        h = F.histograma(np.array([0, 0, 2]), 4)
        np.testing.assert_allclose(h, [2 / 3, 0, 1 / 3, 0])
        np.testing.assert_array_equal(F.histograma(np.zeros(0, int), 4), 0)

    def test_idf(self):
        tfs = np.array([[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 0, 0]], np.float32)
        np.testing.assert_allclose(F.calcular_idf(tfs), np.log([1, 4, 4]), rtol=1e-6)

    def test_pesar_normaliza(self):
        tf = rng.random((5, 10)).astype(np.float32)
        v = F.pesar(tf, np.ones(10, np.float32))
        np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1, rtol=1e-5)
        sin = F.pesar(tf, np.ones(10, np.float32), normalizar=False)
        np.testing.assert_allclose(sin, tf, rtol=1e-6)

    def test_palabra_en_todos_pesa_cero(self):
        tfs = np.array([[0.5, 0.5, 0], [0.5, 0, 0.5]], np.float32)
        v = F.pesar(tfs, F.calcular_idf(tfs))
        np.testing.assert_allclose(v[:, 0], 0)

    def test_vector_vacio_sin_nan(self):
        v = F.pesar(np.zeros(6, np.float32), np.ones(6, np.float32))
        self.assertFalse(np.isnan(v).any())


class TestExtremoAExtremo(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from multimedia.sift import extraer_sift
        cls.dir = tempfile.mkdtemp()
        d = cls.dir
        os.makedirs(os.path.join(d, "covers"))
        os.makedirs(os.path.join(d, "features", "sift", "covers"))
        os.makedirs(os.path.join(d, "features", "codebook"))
        filas, todos = [], []
        for i in range(6):
            img = os.path.join("covers", f"{i}.jpg")
            imagen_con_textura(os.path.join(d, img), i)
            desc = extraer_sift(os.path.join(d, img))
            todos.append(desc)
            npy = os.path.join("features", "sift", "covers", f"{i}.npy")
            np.save(os.path.join(d, npy), desc)
            filas.append(f"{img},cover,{i},{len(desc)},{npy}")
        cv2.imwrite(os.path.join(d, "covers", "plana.jpg"), np.full((200, 200), 128, np.uint8))
        with open(os.path.join(d, "features", "sift_covers.csv"), "w") as f:
            f.write("path,kind,owner_id,n_descriptors,features_path\n" + "\n".join(filas) + "\n")
        r = np.random.default_rng(1)
        cent = np.vstack([t[r.choice(len(t), 5, replace=False)] for t in todos] + [todos[0][:2]])
        np.savez(os.path.join(d, "features", "codebook", "sift_k32.npz"), centroides=cent,
                 media=np.zeros(128, np.float32), desv=np.ones(128, np.float32))
        F._cache.clear()
        F.construir_base("sift", 32, d)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir)
        F._cache.clear()

    def test_base(self):
        b = F.cargar_base("sift", 32, self.dir)
        self.assertEqual(b["vectores"].shape, (6, 32))
        np.testing.assert_allclose(np.linalg.norm(b["vectores"], axis=1), 1, rtol=1e-5)

    def test_misma_imagen_mismo_vector(self):
        b = F.cargar_base("sift", 32, self.dir)
        for i in range(6):
            v = F.ExtractFeatures(os.path.join(self.dir, "covers", f"{i}.jpg"), 32, self.dir)
            self.assertEqual(v.shape, (32,))
            np.testing.assert_allclose(v, b["vectores"][i], rtol=1e-4, atol=1e-5)
            self.assertEqual(int(np.argmax(b["vectores"] @ v)), i)

    def test_imagen_plana(self):
        v = F.ExtractFeatures(os.path.join(self.dir, "covers", "plana.jpg"), 32, self.dir)
        self.assertFalse(np.isnan(v).any())
        np.testing.assert_array_equal(v, 0)


if __name__ == "__main__":
    unittest.main()
