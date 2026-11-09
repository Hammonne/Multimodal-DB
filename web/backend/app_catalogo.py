# backend de la app (la vista "bonita" tipo reproductor de música)
# el catálogo (artistas, álbumes, canciones) sale de los csv de data/ porque es solo mostrar;
# las búsquedas sí usan nuestros índices: hnsw con sift (portadas), hnsw con mfcc (audio) y hnsw con caras
# cada búsqueda deja un registro en data/app_log.jsonl que la consola técnica muestra en "actividad de la app"
from __future__ import annotations

import csv
import json
import re
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from web.backend.multimedia_search import multimedia_search


def _leer(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


EDICION = re.compile(r"\b(deluxe|remaster(ed)?|expanded|anniversary|edition|edición|version|versión|"
                     r"live|en vivo|acoustic|collection|the encore|bonus|special|explicit|clean)\b", re.I)


def titulo_base(titulo: str) -> str:
    # 'The Life of a Showgirl: The Encore' y 'The Life of a Showgirl + Acoustic Collection' -> mismo álbum
    t = re.sub(r"\(.*?\)|\[.*?\]", " ", titulo or "")
    t = re.split(r"\s[:+\-–]\s|:\s|\s\+", t)[0]
    t = EDICION.sub(" ", t)
    return re.sub(r"[^\w]+", " ", t.casefold()).strip()


def _lrc(texto: str) -> list[dict[str, Any]]:
    # [mm:ss.xx] línea -> {t: segundos, texto}
    salida = []
    for linea in texto.splitlines():
        m = re.match(r"\[(\d+):(\d+(?:\.\d+)?)\](.*)", linea)
        if m:
            salida.append({"t": int(m.group(1)) * 60 + float(m.group(2)), "texto": m.group(3).strip()})
    return salida


class Catalogo:
    def __init__(self, root: Path) -> None:
        self.data = root / "data"
        self.log_path = self.data / "app_log.jsonl"
        self._log: deque[dict[str, Any]] = deque(maxlen=300)
        self._log_lock = threading.Lock()
        self._listo = False
        self._lock = threading.Lock()

    # carga

    def _cargar(self) -> None:
        with self._lock:
            if self._listo:
                return
            d = self.data
            self.canciones = {r["track_id"]: r for r in _leer(d / "songs.csv")}
            perfiles = {r["artist_id"]: r["profile_path"] for r in _leer(d / "artist_profiles.csv")}
            tipos = {r["artist_id"]: r["type"] for r in _leer(d / "artist_types.csv")}
            self.artistas = {}
            for r in _leer(d / "artists.csv"):
                foto = perfiles.get(r["artist_id"]) or r.get("main_image_path") or ""
                if foto and not (d / foto).is_file():
                    foto = r.get("main_image_path") or ""
                self.artistas[r["artist_id"]] = {"id": r["artist_id"], "nombre": r["name"], "genero": r["genre"],
                                                 "foto": foto, "tipo": tipos.get(r["artist_id"], "")}
            # las canciones se bajaron como "top de cada artista": en una colaboración el nombre puede ser
            # de otro (I Love It -> Kanye West) aunque el artist_id sea el del artista buscado (Lil Pump)
            self.artista_por_nombre = {a["nombre"].casefold(): a["id"] for a in self.artistas.values()}
            self.albumes = {}
            for r in _leer(d / "covers.csv"):
                if r["duplicada_de"]:
                    continue
                self.albumes[r["album_id"]] = {"id": r["album_id"], "titulo": r["album"], "artista_id": r["artist_id"],
                                               "portada": r["path"], "fecha": r.get("release_date", "")}
            self.album_de_portada = {a["portada"]: a["id"] for a in self.albumes.values()}
            self.canciones_de_album = defaultdict(list)
            self.canciones_de_artista = defaultdict(list)
            for c in self.canciones.values():
                self.canciones_de_album[c["album_id"]].append(c["track_id"])
                self.canciones_de_artista[c["artist_id"]].append(c["track_id"])
                if c["album_id"] not in self.albumes and c.get("cover_path"):
                    self.albumes[c["album_id"]] = {"id": c["album_id"], "titulo": c["album"],
                                                   "artista_id": c["artist_id"], "portada": c["cover_path"], "fecha": ""}
                    self.album_de_portada[c["cover_path"]] = c["album_id"]
            self.albumes_de_artista = defaultdict(list)
            for a in self.albumes.values():
                self.albumes_de_artista[a["artista_id"]].append(a["id"])
            self.fotos_artista = defaultdict(list)
            for r in _leer(d / "artist_images.csv"):
                if (d / r["path"]).is_file():
                    self.fotos_artista[r["artist_id"]].append(r["path"])
            self.integrantes = defaultdict(list)
            fotos_m = {}
            for r in _leer(d / "member_images.csv"):
                fotos_m.setdefault(r["member_id"], r["path"])
            self.banda_de = {}
            for r in _leer(d / "band_members.csv"):
                self.banda_de.setdefault(r["member_id"], r["artist_id"])
                self.integrantes[r["artist_id"]].append({
                    "nombre": r["member"], "rol": r["roles"] or "integrante",
                    "actual": r["current"] == "True", "fundador": r["founder"] == "True",
                    "foto": fotos_m.get(r["member_id"], "")})
            self._listo = True

    # registro de actividad

    def registrar(self, accion: str, **detalle: Any) -> dict[str, Any]:
        entrada = {"hora": time.strftime("%H:%M:%S"), "ts": time.time(), "accion": accion, **detalle}
        with self._log_lock:
            self._log.appendleft(entrada)
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entrada, ensure_ascii=False) + "\n")
        return entrada

    def actividad(self, limite: int = 100) -> list[dict[str, Any]]:
        with self._log_lock:
            if not self._log and self.log_path.is_file():
                for linea in self.log_path.read_text(encoding="utf-8").splitlines()[-300:]:
                    try:
                        self._log.appendleft(json.loads(linea))
                    except json.JSONDecodeError:
                        pass
            return list(self._log)[:limite]

    # piezas para las respuestas

    def _cancion(self, tid: str) -> dict[str, Any] | None:
        c = self.canciones.get(tid)
        if not c:
            return None
        dueno = self.artistas.get(c["artist_id"], {})
        if dueno.get("nombre", "").casefold() == c["artist"].casefold():
            artista_id, con = c["artist_id"], None
        else:
            # el nombre es de otro artista: se enlaza a su perfil si existe y se muestra con quién aparece
            artista_id = self.artista_por_nombre.get(c["artist"].casefold(), "")
            con = {"id": c["artist_id"], "nombre": dueno.get("nombre", "")} if dueno else None
        return {"id": tid, "titulo": c["title"], "artista": c["artist"], "artista_id": artista_id, "con": con,
                "album": c["album"], "album_id": c["album_id"], "portada": c["cover_path"],
                "audio": c["audio_path"], "duracion": int(float(c["duration_s"] or 0)),
                "genero": c["genre"], "explicita": c["explicit"] == "True",
                "letra": c["has_lyrics"] == "True", "rank": int(c["deezer_rank"] or 0)}

    def _album(self, aid: str) -> dict[str, Any] | None:
        a = self.albumes.get(aid)
        if not a:
            return None
        art = self.artistas.get(a["artista_id"], {})
        return {**a, "artista": art.get("nombre", ""), "n_canciones": len(self.canciones_de_album.get(aid, []))}

    def _artista(self, arid: str) -> dict[str, Any] | None:
        a = self.artistas.get(arid)
        return dict(a) if a else None

    def _sin_ediciones(self, ids: list[str]) -> list[dict[str, Any]]:
        # una tarjeta por álbum: la edición con más canciones, y cuántas ediciones más hay
        grupos: dict[tuple[str, str], list[str]] = {}
        for aid in ids:
            a = self.albumes.get(aid)
            if a:
                grupos.setdefault((a["artista_id"], titulo_base(a["titulo"]) or a["titulo"]), []).append(aid)
        salida = []
        for ediciones in grupos.values():
            ediciones.sort(key=lambda x: (-len(self.canciones_de_album.get(x, [])), len(self.albumes[x]["titulo"])))
            album = self._album(ediciones[0])
            album["ediciones"] = len(ediciones) - 1
            salida.append(album)
        return salida

    # pantallas

    def inicio(self) -> dict[str, Any]:
        self._cargar()
        por_genero = defaultdict(lambda: defaultdict(int))
        for c in self.canciones.values():
            por_genero[c["genre"]][c["artist_id"]] += int(c["deezer_rank"] or 0)
        generos = sorted(por_genero, key=lambda g: -sum(por_genero[g].values()))
        secciones = []
        for g in generos[:8]:
            top = sorted(por_genero[g], key=lambda a: -por_genero[g][a])
            artistas = [self._artista(a) for a in top if self.artistas.get(a, {}).get("foto")][:10]
            secciones.append({"genero": g, "artistas": artistas})
        populares = sorted(self.canciones, key=lambda t: -int(self.canciones[t]["deezer_rank"] or 0))
        albumes = self._sin_ediciones(list(dict.fromkeys(self.canciones[t]["album_id"] for t in populares[:400])))[:12]
        return {"secciones": secciones, "albumes": albumes,
                "canciones": [self._cancion(t) for t in populares[:10]],
                "totales": {"canciones": len(self.canciones), "artistas": len(self.artistas),
                            "albumes": len(self.albumes)}}

    def artista(self, arid: str) -> dict[str, Any]:
        self._cargar()
        a = self._artista(arid)
        if not a:
            raise KeyError("artista no encontrado")
        canciones = sorted(self.canciones_de_artista.get(arid, []),
                           key=lambda t: -int(self.canciones[t]["deezer_rank"] or 0))
        albumes = self._sin_ediciones(self.albumes_de_artista.get(arid, []))
        albumes.sort(key=lambda x: (-x["n_canciones"], x["titulo"]))
        self.registrar("abrir artista", titulo=a["nombre"], indice="ninguno (catálogo en csv)",
                       sql=f"SELECT * FROM artistas WHERE id = {arid};", resultados=len(albumes))
        fotos = list(dict.fromkeys(([a["foto"]] if a["foto"] else []) + self.fotos_artista.get(arid, [])))
        return {**a, "canciones": [self._cancion(t) for t in canciones[:20]],
                "albumes": albumes, "integrantes": self.integrantes.get(arid, []), "fotos": fotos}

    def se_parece(self, arid: str, k: int = 8) -> dict[str, Any]:
        # artistas con caras parecidas: los embeddings de 128 dims de faces.npz (face_recognition, ya
        # calculados) en nuestro hnsw con euclidiana; no hace falta volver a detectar caras
        import numpy as np
        from multimedia import caras
        self._cargar()
        a = self._artista(arid)
        if not a:
            raise KeyError("artista no encontrado")
        if not hasattr(self, "_caras"):
            self._caras = caras.abrir(str(self.data))
        datos, indice = self._caras
        mias = np.flatnonzero((datos["kind"] == "artist") & (datos["owner_id"] == arid))
        if not len(mias):
            return {"artista": a, "parecidos": [], "tecnico": None}
        q = datos["embedding"][mias].mean(axis=0)
        t0 = time.perf_counter()
        pos, dist = indice.buscar(q, 60, 100)
        ms = (time.perf_counter() - t0) * 1000
        vistos, salida = {arid}, []
        for p, dd in zip(pos, dist):
            p = int(p)
            if datos["kind"][p] != "artist" or str(datos["owner_id"][p]) in vistos:
                continue
            otro = self._artista(str(datos["owner_id"][p]))
            if not otro:
                continue
            vistos.add(otro["id"])
            salida.append({**otro, "foto_cara": str(datos["path"][p]), "distancia": float(dd),
                           "misma_persona": bool(dd < caras.UMBRAL)})
            if len(salida) >= k:
                break
        self.registrar("se parece a", titulo=a["nombre"], indice="HNSW (M=16, ef_search=100)",
                       metrica="euclidiana", descriptor="embedding de cara de 128 dims (ResNet de dlib)",
                       sql=f"SELECT * FROM artistas WHERE SIMILAR_TO(cara('{a['nombre']}'), k={k}) USING HNSW;",
                       revisados=int(indice.candidatos), total=len(datos["embedding"]), ms=round(ms, 2),
                       resultados=len(salida))
        return {"artista": a, "parecidos": salida,
                "tecnico": {"indice": "HNSW", "revisados": int(indice.candidatos),
                            "total": len(datos["embedding"]), "ms": ms}}

    def album(self, aid: str) -> dict[str, Any]:
        self._cargar()
        a = self._album(aid)
        if not a:
            raise KeyError("álbum no encontrado")
        canciones = [self._cancion(t) for t in self.canciones_de_album.get(aid, [])]
        return {**a, "canciones": canciones}

    def cancion(self, tid: str) -> dict[str, Any]:
        self._cargar()
        c = self._cancion(tid)
        if not c:
            raise KeyError("canción no encontrada")
        fila = self.canciones[tid]
        letra, lrc = "", []
        if fila.get("lrc_path") and (self.data / fila["lrc_path"]).is_file():
            lrc = _lrc((self.data / fila["lrc_path"]).read_text(encoding="utf-8", errors="replace"))
        if fila.get("lyrics_path") and (self.data / fila["lyrics_path"]).is_file():
            letra = (self.data / fila["lyrics_path"]).read_text(encoding="utf-8", errors="replace")
        return {**c, "texto": letra, "lrc": lrc}

    # búsquedas con nuestros índices

    def portadas_parecidas(self, aid: str, k: int = 12) -> dict[str, Any]:
        self._cargar()
        a = self._album(aid)
        if not a:
            raise KeyError("álbum no encontrado")
        r = multimedia_search.search("sift", "hnsw", "coseno", a["portada"], k + 40)
        # fotos de este mismo álbum (discogs o versiones modificadas) que están en la colección,
        # con el puesto en que las encontró el buscador (0 = no salió entre los primeros)
        base = multimedia_search._bases.get("sift", {})
        puesto = {x["id"]: n for n, x in enumerate(r["results"], 1)}
        fotos = []
        if base:
            for img, etiqueta, original in zip(base["ids"], base["etiqueta"], base["original"]):
                if etiqueta != "portada" and str(original) == a["portada"]:
                    fotos.append({"imagen": str(img), "que_es": str(etiqueta), "puesto": puesto.get(str(img), 0)})
        fotos.sort(key=lambda f: (f["puesto"] == 0, f["puesto"]))
        vistos, salida = {aid}, []
        for x in r["results"]:
            if x["is_query"]:
                continue
            # una foto extra de otro álbum cuenta como ese álbum
            otro = self.album_de_portada.get(x.get("original") or x["id"])
            if otro and otro not in vistos:
                vistos.add(otro)
                salida.append({**self._album(otro), "distancia": x["distance"]})
            if len(salida) >= k:
                break
        self.registrar("portadas parecidas", titulo=f"{a['titulo']} ({a['artista']})",
                       indice="HNSW (M=16, ef_search=50)", metrica="coseno",
                       descriptor="SIFT + bolsa de palabras visuales (K=1024) + TF-IDF",
                       sql=f"SELECT * FROM albumes WHERE SIMILAR_TO('{a['portada']}', k={k}) USING HNSW;",
                       revisados=r["candidates"], total=r["documents"], ms=round(r["duration_ms"], 2),
                       resultados=len(salida))
        return {"album": a, "parecidos": salida, "fotos": fotos,
                "tecnico": {"indice": "HNSW", "revisados": r["candidates"], "total": r["documents"],
                            "ms": r["duration_ms"]}}

    def suena_parecido(self, tid: str, k: int = 10) -> dict[str, Any]:
        self._cargar()
        c = self._cancion(tid)
        if not c:
            raise KeyError("canción no encontrada")
        r = multimedia_search.search("mfcc", "hnsw", "coseno", tid, k + 1)
        salida = [{**self._cancion(x["id"]), "distancia": x["distance"]} for x in r["results"]
                  if not x["is_query"] and self._cancion(x["id"])][:k]
        self.registrar("suena parecido", titulo=f"{c['titulo']} ({c['artista']})",
                       indice="HNSW (M=16, ef_search=50)", metrica="coseno",
                       descriptor="MFCC (39 por ventana) + bolsa de palabras de audio (K=1024) + TF-IDF",
                       sql=f"SELECT * FROM canciones WHERE SIMILAR_TO('{c['audio']}', k={k}) USING HNSW;",
                       revisados=r["candidates"], total=r["documents"], ms=round(r["duration_ms"], 2),
                       resultados=len(salida))
        return {"cancion": c, "parecidas": salida,
                "tecnico": {"indice": "HNSW", "revisados": r["candidates"], "total": r["documents"],
                            "ms": r["duration_ms"]}}

    def _caras_indice(self):
        from multimedia import caras
        if not hasattr(self, "_caras"):
            self._caras = caras.abrir(str(self.data))
        return self._caras

    def reconocer_caras(self, contenido: bytes, en_vivo: bool = False) -> dict[str, Any]:
        # como en el lab 11: se detectan las caras de la foto (o del cuadro de la cámara), cada una se
        # vuelve un embedding de 128 dims y se busca la más cercana en el hnsw de caras de artistas
        import cv2
        import face_recognition
        import numpy as np
        from multimedia import caras
        self._cargar()
        img = cv2.imdecode(np.frombuffer(contenido, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("no se pudo leer la imagen")
        alto, ancho = img.shape[:2]
        escala = min(1.0, 640 / max(alto, ancho))
        chica = cv2.resize(img, (round(ancho * escala), round(alto * escala))) if escala < 1 else img
        rgb = cv2.cvtColor(chica, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        cajas = face_recognition.face_locations(rgb)
        embs = face_recognition.face_encodings(rgb, cajas)
        t_cara = (time.perf_counter() - t0) * 1000
        datos, indice = self._caras_indice()
        t0 = time.perf_counter()
        salida, revisados = [], 0
        for (top, right, bottom, left), emb in zip(cajas, embs):
            res = caras.buscar_embedding(datos, indice, emb, k=10)
            revisados += int(indice.candidatos)
            candidatos = []
            for r in res:
                artista_id = r["id"] if r["tipo"] == "artist" else self.banda_de.get(r["id"], "")
                art = self.artistas.get(artista_id, {})
                candidatos.append({"nombre": r["nombre"], "artista_id": artista_id,
                                   "banda": art.get("nombre", "") if r["tipo"] == "member" else "",
                                   "foto": r["foto"], "distancia": r["distancia"],
                                   "misma_persona": r["misma_persona"]})
            salida.append({"caja": [round(v / escala) for v in (top, right, bottom, left)],
                           "candidatos": candidatos})
        t_busqueda = (time.perf_counter() - t0) * 1000
        nombres = tuple(c["candidatos"][0]["nombre"] if c["candidatos"] and c["candidatos"][0]["misma_persona"]
                        else "desconocido" for c in salida)
        # en vivo llegan ~1 cuadro por segundo: solo se registra cuando cambia quién aparece
        if not en_vivo or nombres != getattr(self, "_ultimas_caras", None):
            self._ultimas_caras = nombres
            if salida:
                self.registrar("reconocer cara" + (" en vivo" if en_vivo else ""),
                               titulo=", ".join(nombres) or "sin caras", indice="HNSW (M=16, ef_search=50)",
                               metrica=f"euclidiana (misma persona si < {caras.UMBRAL})",
                               descriptor="embedding de cara de 128 dims (ResNet de dlib, face_recognition)",
                               sql="SELECT nombre FROM artistas WHERE SIMILAR_TO(cara(foto), k=10) USING HNSW;",
                               revisados=revisados, total=len(datos["embedding"]),
                               ms=round(t_cara + t_busqueda, 1), resultados=len(salida))
        return {"caras": salida, "ancho": ancho, "alto": alto,
                "tecnico": {"indice": "HNSW", "revisados": revisados, "total": len(datos["embedding"]),
                            "ms": t_busqueda, "extraccion_ms": t_cara}}

    def reconocer_portada_bytes(self, contenido: bytes, en_vivo: bool = False) -> dict[str, Any]:
        # un cuadro de la cámara: se guarda en un archivo temporal para sacarle los sift y se borra
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as fh:
            fh.write(contenido)
            ruta = Path(fh.name)
        try:
            return self.reconocer(ruta, registrar=not en_vivo, en_vivo=en_vivo)
        finally:
            ruta.unlink(missing_ok=True)

    def reconocer(self, path: Path, registrar: bool = True, en_vivo: bool = False) -> dict[str, Any]:
        # foto de un cd o vinilo -> álbum, fragmento de audio -> canción
        self._cargar()
        r = multimedia_search.search_file(path, "hnsw", "coseno", 12)
        if r["kind"] == "sift":
            vistos, salida = set(), []
            for x in r["results"]:
                aid = self.album_de_portada.get(x.get("original") or x["id"])
                if aid and aid not in vistos:
                    vistos.add(aid)
                    salida.append({**self._album(aid), "distancia": x["distance"]})
            tipo, desc = "álbum", "SIFT + bolsa de palabras visuales (K=1024) + TF-IDF"
        else:
            salida = [{**self._cancion(x["id"]), "distancia": x["distance"]}
                      for x in r["results"] if self._cancion(x["id"])]
            tipo, desc = "canción", "MFCC + bolsa de palabras de audio (K=1024) + TF-IDF"
        primero = salida[0]["id"] if salida else None
        if en_vivo and primero != getattr(self, "_ultima_portada", None):
            self._ultima_portada, registrar = primero, True
        if registrar:
            self.registrar(f"reconocer {tipo}" + (" en vivo" if en_vivo else ""),
                       titulo=(salida[0]["titulo"] if en_vivo and salida else path.name.split("_", 1)[-1]),
                       indice="HNSW (M=16, ef_search=50)",
                       metrica="coseno", descriptor=desc,
                       sql=f"SELECT * FROM ... WHERE SIMILAR_TO('{path.name}', k=12) USING HNSW;",
                       revisados=r["candidates"], total=r["documents"],
                       ms=round(r["duration_ms"] + r.get("extract_ms", 0), 2), resultados=len(salida))
        return {"tipo": tipo, "resultados": salida[:8],
                "tecnico": {"indice": "HNSW", "revisados": r["candidates"], "total": r["documents"],
                            "ms": r["duration_ms"], "extraccion_ms": r.get("extract_ms", 0)}}


catalogo = Catalogo(Path(__file__).resolve().parents[2])
