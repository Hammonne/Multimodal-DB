import contextlib
import io
import unittest

from parser import Parser
from scanner import Scanner
from parser.parser import SqlError
from ast_sql import (AndCond, CompareCond, ExplainStmt, SelectStmt, SimilarityScoreExpr,
                     SimilarToExpr, SortDir, VectorIndex)
from visitor import PrintVisitor


def parsear(sql):
    with contextlib.redirect_stdout(io.StringIO()):
        return Parser(Scanner(sql)).parse_program().slist


def uno(sql):
    return parsear(sql)[0]


def imprimir(sql):
    salida = io.StringIO()
    with contextlib.redirect_stdout(salida):
        PrintVisitor().imprimir(Parser(Scanner(sql)).parse_program())
    return salida.getvalue().strip().splitlines()[-1]


class TestSimilarTo(unittest.TestCase):

    def test_ejemplo_imagenes(self):
        s = uno("SELECT * FROM imagenes WHERE SIMILAR_TO('foto_consulta.jpg', k=10) "
                "USING HNSW WITH METRIC=cosine;")
        self.assertIsInstance(s, SelectStmt)
        self.assertTrue(s.select_all)
        self.assertIsInstance(s.condicion, SimilarToExpr)
        self.assertEqual(s.condicion.archivo, "foto_consulta.jpg")
        self.assertEqual(s.condicion.k, 10)
        self.assertEqual(s.indice_vectorial, VectorIndex.HNSW)
        self.assertEqual(s.metrica_vectorial, "coseno")

    def test_ejemplo_canciones(self):
        s = uno("SELECT nombre, artista, SIMILARITY_SCORE() as score FROM canciones "
                "WHERE SIMILAR_TO('audio_query.mp3', k=5) USING IVF WITH METRIC=euclidean "
                "ORDER BY score DESC;")
        self.assertEqual([i.alias for i in s.proyeccion], ["", "", "score"])
        self.assertIsInstance(s.proyeccion[2].columna, SimilarityScoreExpr)
        self.assertEqual(s.condicion.k, 5)
        self.assertEqual(s.indice_vectorial, VectorIndex.IVF)
        self.assertEqual(s.metrica_vectorial, "euclidiana")
        self.assertEqual(s.order_by.columna, "score")
        self.assertEqual(s.direccion, SortDir.DESC_DIR)

    def test_metricas(self):
        casos = {"euclidean": "euclidiana", "L2": "euclidiana", "cosine": "coseno",
                 "COSENO": "coseno", "dot": "producto_punto", "ip": "producto_punto",
                 "'cosine'": "coseno"}
        for nombre, esperado in casos.items():
            s = uno(f"SELECT * FROM t WHERE SIMILAR_TO('a.jpg', k=3) USING IVF WITH METRIC={nombre}")
            self.assertEqual(s.metrica_vectorial, esperado, nombre)

    def test_sin_with_es_coseno(self):
        s = uno("SELECT * FROM t WHERE SIMILAR_TO('a.mp3', k=3) USING HNSW")
        self.assertEqual(s.metrica_vectorial, "coseno")

    def test_sin_using(self):
        # sin USING el executor elige: el índice de la tabla o, si no hay, búsqueda lineal
        s = uno("SELECT * FROM t WHERE SIMILAR_TO('a.mp3', k=3)")
        self.assertEqual(s.indice_vectorial, VectorIndex.NINGUNO)

    def test_using_seq_es_lineal(self):
        for palabra in ("SEQ", "seq", "LINEAL", "SEQUENTIAL"):
            s = uno(f"SELECT * FROM t WHERE SIMILAR_TO('a.jpg', k=3) USING {palabra} WITH METRIC=dot")
            self.assertEqual(s.indice_vectorial, VectorIndex.LINEAL)
            self.assertEqual(s.metrica_vectorial, "producto_punto")

    def test_k_sin_nombre(self):
        self.assertEqual(uno("SELECT * FROM t WHERE SIMILAR_TO('a.jpg', 7)").condicion.k, 7)

    def test_minusculas(self):
        s = uno("select * from t where similar_to('a.jpg', K=2) using hnsw with metric=dot")
        self.assertEqual(s.condicion.k, 2)
        self.assertEqual(s.indice_vectorial, VectorIndex.HNSW)

    def test_con_otra_condicion(self):
        s = uno("SELECT * FROM t WHERE genero = 'Rock' AND SIMILAR_TO('a.jpg', k=5) USING IVF")
        self.assertIsInstance(s.condicion, AndCond)
        self.assertIsInstance(s.condicion.condiciones[0], CompareCond)
        self.assertIsInstance(s.condicion.condiciones[1], SimilarToExpr)

    def test_order_by_score_y_limit(self):
        s = uno("SELECT titulo FROM t WHERE SIMILAR_TO('a.jpg', k=10) USING HNSW "
                "ORDER BY SIMILARITY_SCORE() DESC LIMIT 3")
        self.assertIsInstance(s.order_by, SimilarityScoreExpr)
        self.assertEqual(s.limite, 3)

    def test_explain(self):
        s = uno("EXPLAIN SELECT * FROM t WHERE SIMILAR_TO('a.jpg', k=10) USING IVF")
        self.assertIsInstance(s, ExplainStmt)
        self.assertEqual(s.sentencia.indice_vectorial, VectorIndex.IVF)

    def test_alias_en_columna_normal(self):
        self.assertEqual(uno("SELECT titulo AS t FROM canciones").proyeccion[0].alias, "t")

    def test_imprimir(self):
        self.assertEqual(
            imprimir("SELECT titulo, SIMILARITY_SCORE() AS s FROM t "
                     "WHERE SIMILAR_TO('a.jpg', k=4) USING HNSW WITH METRIC=dot"),
            "SELECT titulo, SIMILARITY_SCORE() AS s FROM t WHERE SIMILAR_TO('a.jpg', k=4) "
            "USING HNSW WITH METRIC=producto_punto;")

    def test_lo_anterior_sigue_igual(self):
        s = parsear("CREATE INDEX ON t (c) USING BTREE; SELECT * FROM t WHERE id = 3 ORDER BY id")
        self.assertEqual(len(s), 2)
        self.assertEqual(s[1].indice_vectorial, VectorIndex.NINGUNO)

    def test_create_index_vectorial(self):
        from ast_sql import IndexKind
        self.assertEqual(uno("CREATE INDEX ON canciones (portada) USING HNSW").tipo, IndexKind.HNSW_IDX)
        self.assertEqual(uno("CREATE INDEX ON canciones (audio) USING ivf").tipo, IndexKind.IVF_IDX)
        with self.assertRaises(SqlError):
            parsear("CREATE INDEX ON canciones (audio) USING HNSW CLUSTERED")

    def test_errores(self):
        malos = {
            "SELECT * FROM t WHERE SIMILAR_TO('a.jpg', k=3) USING HNSW WITH METRIC=manhattan": "metrica desconocida",
            "SELECT * FROM t WHERE SIMILAR_TO('a.jpg', k=0)": "mayor que 0",
            "SELECT * FROM t WHERE SIMILAR_TO('a.jpg', k=2.5)": "mayor que 0",
            "SELECT * FROM t WHERE SIMILAR_TO(a.jpg, k=3)": "entre comillas",
            "SELECT * FROM t WHERE SIMILAR_TO('a.jpg')": "el k",
            "SELECT * FROM t WHERE SIMILAR_TO('a.jpg', n=3)": "k=",
            "SELECT * FROM t WHERE SIMILAR_TO('a.jpg', k=3) USING BTREE": "IVF, HNSW o SEQ",
            "SELECT * FROM t WHERE SIMILAR_TO('a.jpg', k=3) USING IVF WITH cosine": "METRIC",
            "SELECT * FROM t WHERE id = 1 USING HNSW": "solo se usa con SIMILAR_TO",
            "SELECT SIMILARITY_SCORE() FROM t": "solo se usa con SIMILAR_TO",
            "SELECT * FROM t ORDER BY SIMILARITY_SCORE()": "solo se usa con SIMILAR_TO",
            "SELECT SIMILARITY_SCORE(3) FROM t WHERE SIMILAR_TO('a.jpg', k=3)": "no recibe argumentos",
            "SELECT * FROM t WHERE SIMILAR_TO('a.jpg', k=3) AND SIMILAR_TO('b.jpg', k=3)": "un SIMILAR_TO",
            "SELECT * FROM t WHERE id = 1 OR SIMILAR_TO('a.jpg', k=3)": "dentro de un OR",
        }
        for sql, mensaje in malos.items():
            with self.assertRaises(SqlError, msg=sql) as ctx:
                parsear(sql)
            self.assertIn(mensaje, ctx.exception.mensaje, sql)


if __name__ == "__main__":
    unittest.main()
