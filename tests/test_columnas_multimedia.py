import csv
import os
import shutil
import sys
import tempfile
import unittest

import cv2
import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "parser"))
sys.path.insert(1, RAIZ)

from executor import ExecuteVisitor, ExecutionError  # noqa: E402
from parser import Parser  # noqa: E402
from scanner import Scanner  # noqa: E402
from storage.modelo_multimedia import ModeloMultimedia, k_sugerido  # noqa: E402
from storage.storage_manager import StorageManager  # noqa: E402


def imagen_falsa(ruta, semilla):
    # figuras al azar: tienen esquinas y bordes, así SIFT encuentra puntos (nada de nuestro dataset)
    rng = np.random.default_rng(semilla)
    img = np.full((320, 320, 3), 255, np.uint8)
    for _ in range(25):
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        x, y = (int(v) for v in rng.integers(10, 300, 2))
        if rng.random() < .5:
            cv2.rectangle(img, (x, y), (x + int(rng.integers(10, 60)), y + int(rng.integers(10, 60))), color, -1)
        else:
            cv2.circle(img, (x, y), int(rng.integers(5, 35)), color, -1)
    cv2.imwrite(ruta, img)


def audio_falso(ruta, semilla, sr=22050, segundos=3):
    # acordes de tonos al azar
    from scipy.io import wavfile
    rng = np.random.default_rng(semilla)
    t = np.arange(sr * segundos) / sr
    y = sum(np.sin(2 * np.pi * f * t) for f in rng.uniform(150, 3000, 4))
    y = y * (0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(.5, 3) * t))
    wavfile.write(ruta, sr, (y / np.abs(y).max() * 20000).astype(np.int16))


class TestColumnasMultimedia(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.antes = os.getcwd()
        cls.dir = tempfile.mkdtemp()
        os.chdir(cls.dir)
        cls.fotos = os.path.join(cls.dir, "fotos")
        cls.audios = os.path.join(cls.dir, "audios")
        os.makedirs(cls.fotos)
        os.makedirs(cls.audios)
        for i in range(12):
            imagen_falsa(os.path.join(cls.fotos, f"f{i:02d}.png"), i)
        for i in range(8):
            audio_falso(os.path.join(cls.audios, f"a{i}.wav"), i)
        # una versión rotada de f03, que no está en la tabla
        img = cv2.imread(os.path.join(cls.fotos, "f03.png"))
        cls.rotada = os.path.join(cls.dir, "rotada.png")
        cv2.imwrite(cls.rotada, cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
        cls.sm = StorageManager(wal_path="wal.log")
        cls.v = ExecuteVisitor(cls.sm)

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.antes)
        shutil.rmtree(cls.dir, ignore_errors=True)

    def sql(self, texto, v=None):
        salida = None
        for salida in (v or self.v).ejecutar(Parser(Scanner(texto)).parse_program()):
            pass
        return salida

    def test_1_tipos_e_import_de_carpeta(self):
        self.sql("CREATE TABLE galeria (id INT PRIMARY KEY, nombre VARCHAR(30), foto IMAGE) USING HEAP;")
        self.sql(f"IMPORT INTO galeria FROM '{self.fotos}';")
        filas = self.sql("SELECT id, nombre, foto FROM galeria;").filas
        self.assertEqual(len(filas), 12)
        self.assertTrue(filas[0][2].endswith(".png"))

    def test_2_lineal_sin_indice(self):
        # sin índice el motor entrena un modelo con los archivos de la columna
        filas = self.sql(f"SELECT nombre FROM galeria WHERE SIMILAR_TO('{self.rotada}', k=3);").filas
        self.assertEqual(filas[0][0], "f03")

    def test_3_indices_ivf_y_hnsw(self):
        self.sql("CREATE INDEX ON galeria (foto) USING HNSW;")
        self.sql("CREATE INDEX ON galeria (foto) USING IVF;")
        for indice in ("HNSW", "IVF"):
            filas = self.sql(f"SELECT nombre FROM galeria WHERE SIMILAR_TO('{self.rotada}', k=3) "
                             f"USING {indice};").filas
            self.assertEqual(filas[0][0], "f03", indice)
        self.assertTrue(os.path.exists("idx_galeria_foto_hnsw_modelo.npz"))

    def test_4_insert_despues_del_indice(self):
        self.sql(f"INSERT INTO galeria VALUES (99, 'rotada', '{self.rotada}');")
        filas = self.sql(f"SELECT id FROM galeria WHERE SIMILAR_TO('{self.rotada}', k=1) USING HNSW;").filas
        self.assertEqual(filas[0][0], 99)

    def test_5_reabrir_conserva_el_modelo(self):
        self.sm.close()
        sm = StorageManager(wal_path="wal.log")
        v = ExecuteVisitor(sm)
        try:
            filas = self.sql(f"SELECT nombre FROM galeria WHERE SIMILAR_TO('{self.rotada}', k=2) USING HNSW;", v).filas
            self.assertIn(filas[0][0], ("rotada", "f03"))
        finally:
            sm.close()
            type(self).sm = StorageManager(wal_path="wal.log")
            type(self).v = ExecuteVisitor(self.sm)

    def test_6_tabla_completa_con_audio_join_y_group_by(self):
        ruta_csv = os.path.join(self.audios, "canciones.csv")
        with open(ruta_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id_cancion", "nombre_cancion", "id_artista", "audio_cancion"])
            for i in range(8):
                w.writerow([i, f"tema {i}", i % 3, f"a{i}.wav"])     # rutas relativas al csv
        self.sql("CREATE TABLE artistas (id INT PRIMARY KEY, nombre VARCHAR(20)) USING HEAP;")
        for i, n in enumerate(["Ana", "Beto", "Cata"]):
            self.sql(f"INSERT INTO artistas VALUES ({i}, '{n}');")
        self.sql("CREATE TABLE canciones (id_cancion INT PRIMARY KEY, nombre_cancion VARCHAR(20), "
                 "id_artista INT, audio_cancion AUDIO) USING HEAP;")
        self.sql(f"IMPORT INTO canciones FROM '{ruta_csv}';")
        self.sql("CREATE INDEX ON canciones (id_artista) USING BTREE;")
        self.sql("CREATE INDEX ON canciones (audio_cancion) USING IVF;")
        grupos = self.sql("SELECT artistas.nombre, COUNT(*) AS n FROM canciones JOIN artistas "
                          "ON canciones.id_artista = artistas.id GROUP BY artistas.nombre;").filas
        self.assertEqual(sorted(g[1] for g in grupos), [2, 3, 3])
        filtro = self.sql("SELECT nombre_cancion FROM canciones WHERE id_artista = 1;").filas
        self.assertEqual(len(filtro), 3)
        parecida = self.sql(f"SELECT id_cancion FROM canciones WHERE SIMILAR_TO('{self.audios}/a5.wav', k=1) "
                            "USING IVF;").filas
        self.assertEqual(parecida[0][0], 5)

    def test_7_indice_vectorial_solo_en_columnas_multimedia(self):
        with self.assertRaises(ExecutionError):
            self.sql("CREATE INDEX ON artistas (id) USING HNSW;")

    def test_8_k_del_codebook_se_adapta(self):
        self.assertEqual(k_sugerido(100), 10)
        self.assertEqual(k_sugerido(9_000_000), 1024)
        m = ModeloMultimedia("sift").entrenar([os.path.join(self.fotos, f) for f in sorted(os.listdir(self.fotos))])
        self.assertLessEqual(m.k, 1024)
        self.assertEqual(m.vector(self.rotada).shape, (m.k,))

    # búsqueda con filtro, parámetros, JOIN y GROUP BY (usan las tablas de los tests 1-6)

    def ids(self, texto):
        return [f[0] for f in self.sql(texto).filas]

    def test_9a_filtro_antes_de_buscar(self):
        import executor as modulo
        exactos = self.ids(f"SELECT id FROM galeria WHERE id < 6 AND SIMILAR_TO('{self.rotada}', k=3) USING SEQ;")
        self.assertEqual(len(exactos), 3)
        self.assertTrue(all(i < 6 for i in exactos))
        antes = modulo.UMBRAL_PREFILTRO
        try:
            for umbral, estrategia in ((antes, "pre-filtro exacto"), (0, "filtro dentro del indice")):
                modulo.UMBRAL_PREFILTRO = umbral
                for indice in ("HNSW", "IVF"):
                    consulta = (f"SELECT id FROM galeria WHERE id < 6 AND SIMILAR_TO('{self.rotada}', k=3) "
                                f"USING {indice} WITH NPROBE=100" if indice == "IVF" else
                                f"SELECT id FROM galeria WHERE id < 6 AND SIMILAR_TO('{self.rotada}', k=3) "
                                f"USING {indice}")
                    self.assertEqual(self.ids(consulta + ";"), exactos, (umbral, indice))
                    plan = self.sql(f"EXPLAIN ANALYZE {consulta};").explain
                    self.assertIn(estrategia, plan, (umbral, indice))
        finally:
            modulo.UMBRAL_PREFILTRO = antes

    def test_9b_nprobe_y_ef_search(self):
        plan = self.sql(f"EXPLAIN SELECT id FROM galeria WHERE SIMILAR_TO('{self.rotada}', k=3) "
                        "USING IVF WITH NPROBE=7;").explain
        self.assertIn("nprobe=7", plan)
        plan = self.sql(f"EXPLAIN SELECT id FROM galeria WHERE SIMILAR_TO('{self.rotada}', k=3) "
                        "WITH EF_SEARCH=120;").explain
        self.assertIn("ef_search=120", plan)
        exactos = self.ids(f"SELECT id FROM galeria WHERE SIMILAR_TO('{self.rotada}', k=5) USING SEQ;")
        todo = self.ids(f"SELECT id FROM galeria WHERE SIMILAR_TO('{self.rotada}', k=5) USING IVF WITH NPROBE=100;")
        self.assertEqual(todo, exactos)

    def test_9c_similar_to_con_join(self):
        a5 = f"{self.audios}/a5.wav"
        filas = self.sql("SELECT canciones.nombre_cancion, artistas.nombre, SIMILARITY_SCORE() AS s "
                         "FROM canciones JOIN artistas ON canciones.id_artista = artistas.id "
                         f"WHERE SIMILAR_TO('{a5}', k=2);").filas
        self.assertEqual(len(filas), 2)
        self.assertEqual(filas[0][:2], ["tema 5", "Cata"])
        # filtro sobre la otra tabla: siguen saliendo k filas, todas de Ana
        filas = self.sql("SELECT canciones.id_cancion, artistas.nombre FROM canciones JOIN artistas "
                         "ON canciones.id_artista = artistas.id "
                         f"WHERE artistas.nombre = 'Ana' AND SIMILAR_TO('{a5}', k=2);").filas
        self.assertEqual(len(filas), 2)
        self.assertTrue(all(f[1] == "Ana" for f in filas))
        # la columna de audio puede estar en la tabla de la derecha
        filas = self.sql("SELECT canciones.id_cancion FROM artistas JOIN canciones "
                         "ON artistas.id = canciones.id_artista "
                         f"WHERE SIMILAR_TO('{a5}', k=1);").filas
        self.assertEqual(filas, [[5]])

    def test_9d_similar_to_con_group_by(self):
        a5 = f"{self.audios}/a5.wav"
        grupos = self.sql("SELECT artistas.nombre, COUNT(*) AS n FROM canciones JOIN artistas "
                          "ON canciones.id_artista = artistas.id "
                          f"WHERE SIMILAR_TO('{a5}', k=6) GROUP BY artistas.nombre;").filas
        self.assertEqual(sum(g[1] for g in grupos), 6)
        grupos = self.sql(f"SELECT id_artista, COUNT(*) AS n FROM canciones WHERE SIMILAR_TO('{a5}', k=4) "
                          "GROUP BY id_artista;").filas
        self.assertEqual(sum(g[1] for g in grupos), 4)


if __name__ == "__main__":
    unittest.main()
