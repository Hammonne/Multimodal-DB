import contextlib
import csv
import io
import os
import shutil
import tempfile
import unittest

from parser import Parser
from scanner import Scanner
from executor import ExecuteVisitor, ExecutionError
from storage.storage_manager import StorageManager
from storage.vector_index import VectorSecondaryIndex
from multimedia.features import DATA

N = 40
HAY_DATA = all(os.path.exists(os.path.join(DATA, f)) for f in
               ("songs.csv", "features/codebook/sift_k1024.npz", "features/codebook/mfcc_k1024.npz"))


@unittest.skipUnless(HAY_DATA, "hace falta data/ con el dataset y los codebooks")
class TestExecutorSimilitud(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.antes = os.getcwd()
        cls.dir = tempfile.mkdtemp()
        os.chdir(cls.dir)
        cls.sm = StorageManager()
        cls.ex = ExecuteVisitor(cls.sm)
        with open(os.path.join(DATA, "songs.csv"), newline="", encoding="utf-8") as f:
            cls.filas = list(csv.DictReader(f))[:N]
        cls.sql("CREATE TABLE canciones (id INT PRIMARY KEY, titulo VARCHAR(80), genero VARCHAR(20), "
                "audio VARCHAR(40), portada VARCHAR(40)) USING HEAP")
        for i, r in enumerate(cls.filas):
            titulo = r["title"].replace("'", "''")[:78]
            cls.sql(f"INSERT INTO canciones VALUES ({i}, '{titulo}', '{r['genre'][:18]}', "
                    f"'{r['audio_path']}', '{r['cover_path']}')")
        cls.sql("CREATE INDEX ON canciones (portada) USING HNSW")
        cls.sql("CREATE INDEX ON canciones (audio) USING IVF")
        cls.img = cls.filas[3]["cover_path"]
        cls.aud = cls.filas[4]["audio_path"]

    @classmethod
    def tearDownClass(cls):
        cls.sm.close()
        os.chdir(cls.antes)
        shutil.rmtree(cls.dir, ignore_errors=True)

    @classmethod
    def sql(cls, texto):
        with contextlib.redirect_stdout(io.StringIO()):
            prog = Parser(Scanner(texto)).parse_program()
        return cls.ex.ejecutar(prog)[-1]

    def ids(self, sql):
        return [f[0] for f in self.sql(sql).filas]

    def test_imagen_se_encuentra_a_si_misma(self):
        r = self.sql(f"SELECT id, SIMILARITY_SCORE() AS s FROM canciones "
                     f"WHERE SIMILAR_TO('{self.img}', k=5) USING HNSW WITH METRIC=cosine")
        self.assertEqual(r.columnas, ["id", "s"])
        self.assertEqual(len(r.filas), 5)
        self.assertEqual(self.filas[r.filas[0][0]]["cover_path"], self.img)
        self.assertAlmostEqual(r.filas[0][1], 1.0, places=4)
        scores = [f[1] for f in r.filas]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_ejemplo_del_enunciado_audio(self):
        r = self.sql(f"SELECT titulo, SIMILARITY_SCORE() as score FROM canciones "
                     f"WHERE SIMILAR_TO('{self.aud}', k=5) USING IVF WITH METRIC=euclidean "
                     f"ORDER BY score DESC")
        self.assertEqual(r.filas[0][0], self.filas[4]["title"].replace("''", "'")[:78])
        self.assertTrue(all(0 < f[1] <= 1 for f in r.filas))

    def test_order_by_asc_invierte(self):
        desc = self.ids(f"SELECT id FROM canciones WHERE SIMILAR_TO('{self.img}', k=5) USING HNSW "
                        f"ORDER BY SIMILARITY_SCORE() DESC")
        asc = self.ids(f"SELECT id FROM canciones WHERE SIMILAR_TO('{self.img}', k=5) USING HNSW "
                       f"ORDER BY SIMILARITY_SCORE() ASC")
        self.assertEqual(desc, asc[::-1])

    def test_indice_igual_a_lineal(self):
        # con 40 filas el hnsw tiene que dar lo mismo que la búsqueda exacta
        con = self.ids(f"SELECT id FROM canciones WHERE SIMILAR_TO('{self.img}', k=5) USING HNSW")
        sin = self.ids(f"SELECT id FROM canciones WHERE SIMILAR_TO('{self.img}', k=5) USING SEQ")
        self.assertEqual(con, sin)

    def test_sin_using_usa_el_indice(self):
        # sin USING se usa el índice de la tabla; USING SEQ fuerza la búsqueda lineal
        plan = self.sql(f"EXPLAIN SELECT id FROM canciones WHERE SIMILAR_TO('{self.img}', k=3)").explain
        self.assertIn("Index Scan using hnsw_canciones_portada", plan)
        plan = self.sql(f"EXPLAIN SELECT id FROM canciones WHERE SIMILAR_TO('{self.aud}', k=3)").explain
        self.assertIn("Index Scan using ivf_canciones_audio", plan)
        plan = self.sql(f"EXPLAIN SELECT id FROM canciones WHERE SIMILAR_TO('{self.img}', k=3) USING SEQ").explain
        self.assertIn("Seq Scan on canciones", plan)

    def test_filtro_y_limit(self):
        genero = self.filas[3]["genre"][:18]
        r = self.sql(f"SELECT id, genero FROM canciones WHERE genero = '{genero}' AND "
                     f"SIMILAR_TO('{self.img}', k=4) USING HNSW LIMIT 2")
        self.assertLessEqual(len(r.filas), 2)
        self.assertTrue(all(f[1] == genero for f in r.filas))

    def test_select_todo(self):
        r = self.sql(f"SELECT * FROM canciones WHERE SIMILAR_TO('{self.aud}', k=3) USING IVF")
        self.assertEqual(r.columnas, ["id", "titulo", "genero", "audio", "portada"])
        self.assertEqual(len(r.filas), 3)

    def test_explain(self):
        plan = self.sql(f"EXPLAIN SELECT id FROM canciones WHERE SIMILAR_TO('{self.img}', k=3) "
                        f"USING HNSW").explain
        self.assertIn("Index Scan using hnsw_canciones_portada", plan)
        plan = self.sql(f"EXPLAIN ANALYZE SELECT id FROM canciones WHERE SIMILAR_TO('{self.aud}', "
                        f"k=3) USING SEQ").explain
        self.assertIn("Seq Scan on canciones", plan)
        self.assertIn("rows=3", plan)

    def test_insert_y_delete(self):
        self.sql(f"INSERT INTO canciones VALUES (500, 'copia', 'x', '{self.aud}', '{self.img}')")
        self.assertIn(500, self.ids(f"SELECT id FROM canciones WHERE SIMILAR_TO('{self.img}', k=2) USING HNSW"))
        self.sql("DELETE FROM canciones WHERE id = 500")
        self.assertNotIn(500, self.ids(f"SELECT id FROM canciones WHERE SIMILAR_TO('{self.img}', k=5) USING HNSW"))

    def test_guardar_y_cargar(self):
        idx = self.sm.index_manager.open_indexes["idx_canciones_portada_hnsw"]
        idx.guardar()
        otro = VectorSecondaryIndex("idx_canciones_portada_hnsw.npz", "hnsw")
        self.assertTrue(otro.cargar(len(idx)))
        self.assertFalse(VectorSecondaryIndex("idx_canciones_portada_hnsw.npz", "hnsw").cargar(len(idx) + 1))
        self.assertEqual(otro.tipo, "sift")

    def test_errores(self):
        malos = [
            (f"SELECT * FROM canciones WHERE SIMILAR_TO('no_existe.jpg', k=3) USING HNSW", "no existe"),
            (f"SELECT * FROM canciones WHERE SIMILAR_TO('{self.img}', k=3) USING IVF", "no tiene un indice IVF"),
            ("SELECT * FROM canciones WHERE SIMILAR_TO('a.txt', k=3)", "imagen o audio"),
            ("CREATE INDEX ON canciones (id) USING HNSW", "VARCHAR"),
            (f"SELECT genero, SIMILARITY_SCORE() FROM canciones WHERE SIMILAR_TO('{self.img}', k=3) "
             "GROUP BY genero", "SIMILARITY_SCORE() no se puede usar con GROUP BY"),
        ]
        for sql, mensaje in malos:
            with self.assertRaises(ExecutionError, msg=sql) as ctx:
                self.sql(sql)
            self.assertIn(mensaje, str(ctx.exception), sql)


if __name__ == "__main__":
    unittest.main()
