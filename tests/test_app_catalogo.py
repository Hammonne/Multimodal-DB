import json
import tempfile
import unittest
from pathlib import Path

from web.backend.app_catalogo import Catalogo, _lrc


class TestAppCatalogo(unittest.TestCase):

    def test_lrc(self):
        lineas = _lrc("[00:12.50]hola\n[01:02.00] mundo \nsin tiempo\n[02:00.1]")
        self.assertEqual([l["t"] for l in lineas], [12.5, 62.0, 120.1])
        self.assertEqual([l["texto"] for l in lineas], ["hola", "mundo", ""])

    def test_registrar_y_leer_actividad(self):
        with tempfile.TemporaryDirectory() as d:
            raiz = Path(d)
            (raiz / "data").mkdir()
            c = Catalogo(raiz)
            c.registrar("portadas parecidas", indice="HNSW", revisados=10, total=100)
            c.registrar("buscar por letra", indice="SPIMI")
            act = c.actividad()
            self.assertEqual(act[0]["accion"], "buscar por letra")
            guardado = [json.loads(l) for l in (raiz / "data" / "app_log.jsonl").read_text().splitlines()]
            self.assertEqual(len(guardado), 2)
            # al reiniciar se vuelve a leer del archivo
            self.assertEqual(Catalogo(raiz).actividad()[1]["indice"], "HNSW")


if __name__ == "__main__":
    unittest.main()
