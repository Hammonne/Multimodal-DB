from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from indexes.hnsw import HNSW
from indexes.ivf import IVF
from multimedia import features


class MultimediaSearchManager:
    def __init__(self, root: Path) -> None:
        self.data = Path(os.getenv("MULTIMEDIA_DATA", root / "data"))
        self._indexes: dict[tuple[str, str, str], tuple[Any, np.ndarray, np.ndarray]] = {}
        self._locks: dict[tuple[str, str, str], threading.Lock] = {}
        self._locks_lock = threading.Lock()
        self._bases: dict[str, dict[str, Any]] = {}

    def _load_base(self, kind: str) -> dict[str, np.ndarray]:
        requested = int(os.getenv("MULTIMEDIA_K", features.K_DEFECTO))
        path = Path(features.ruta_base(kind, requested, str(self.data)))
        if not path.is_file():
            candidates = list(
                (self.data / "features").glob(f"tfidf_{kind}_k*.npz")
            )
            if candidates:
                candidates.sort(
                    key=lambda candidate: int(candidate.stem.rsplit("_k", 1)[1]),
                    reverse=True,
                )
                path = candidates[0]
                requested = int(path.stem.rsplit("_k", 1)[1])
        if not path.is_file():
            raise FileNotFoundError(
                f"no existe la base multimedia {path}. "
                f"Ejecute primero el pipeline: python -m multimedia.sift --source covers; "
                f"python -m multimedia.codebook --tipo {kind} --ks 1024; "
                f"python -m multimedia.features --tipo {kind}."
            )
        base = features.cargar_base(kind, requested, str(self.data))
        return self._con_extras(kind, base)

    def _ruta_extras(self, kind: str, k: int) -> Path:
        return self.data / "features" / f"extras_{kind}_k{k}.npz"

    def _con_extras(self, kind: str, base: dict[str, Any]) -> dict[str, Any]:
        # solo para la web: a las portadas se suman las fotos reales de discogs y las versiones
        # modificadas (data augmentation), así al buscar un álbum aparecen también sus fotos
        if kind in self._bases:
            return self._bases[kind]
        n = len(base["ids"])
        base["etiqueta"] = np.array(["portada"] * n, dtype=object)
        base["original"] = np.array([str(i) for i in base["ids"]], dtype=object)
        extras = self._ruta_extras(kind, int(base["k"]))
        if extras.is_file() and os.getenv("MULTIMEDIA_EXTRAS", "1") != "0":
            z = np.load(extras, allow_pickle=False)
            base["ids"] = np.concatenate([base["ids"].astype(str), z["ids"].astype(str)])
            base["vectores"] = np.vstack([base["vectores"], z["vectores"]])
            base["etiqueta"] = np.concatenate([base["etiqueta"], z["etiqueta"].astype(object)])
            base["original"] = np.concatenate([base["original"], z["original"].astype(object)])
        self._bases[kind] = base
        return base

    def construir_extras_audio(self) -> Path:
        # el codebook y la base tf-idf de audio se hicieron con las primeras 10 000 canciones; las que se
        # sumaron después (con letra) tienen sus mfcc en data/features/mfcc pero no su vector: se calcula
        # con el mismo codebook e idf para que también se puedan buscar
        base = features.cargar_base("mfcc", features.K_DEFECTO, str(self.data))
        cb = features.cargar_codebook("mfcc", int(base["k"]), str(self.data))
        ya = set(base["ids"].astype(str))
        ids, vecs = [], []
        for tid, npy in features.objetos("mfcc", str(self.data)):
            if str(tid) in ya:
                continue
            ruta = self.data / npy
            if not ruta.is_file():
                continue
            tf = features.histograma(features.palabras(np.load(ruta), cb), int(base["k"]))
            ids.append(str(tid))
            vecs.append(features.pesar(tf, base["idf"]))
        salida = self._ruta_extras("mfcc", int(base["k"]))
        np.savez(salida, ids=np.array(ids), vectores=np.vstack(vecs).astype(np.float32),
                 etiqueta=np.array(["canción"] * len(ids)), original=np.array(ids))
        return salida

    def construir_extras(self, kind: str = "sift") -> Path:
        # vectores tf-idf de data/fotos_reales.csv y data/consultas_aumentadas.csv con el mismo
        # codebook e idf que las portadas; se corre una vez: python -m web.backend.multimedia_search
        import csv
        base = features.cargar_base(kind, features.K_DEFECTO, str(self.data))
        filas = []
        reales = self.data / "fotos_reales.csv"
        if reales.is_file():
            with reales.open(newline="", encoding="utf-8") as fh:
                for f in csv.DictReader(fh):
                    nombre = {"principal": f"foto real ({f['formato']})",
                              "edicion": f"foto real ({f['formato']}, otra edición)"}.get(
                                  f["tipo"], f"foto extra ({f['formato']})")
                    filas.append((f["path"], nombre, f["cover_path"]))
        for archivo in ("consultas_aumentadas.csv", "aumentadas_web.csv"):
            aumentadas = self.data / archivo
            if aumentadas.is_file():
                with aumentadas.open(newline="", encoding="utf-8") as fh:
                    for f in csv.DictReader(fh):
                        filas.append((f["path"], f"aumentada: {f['cambio']}", f["original"]))
        ids, vecs, etiquetas, originales = [], [], [], []
        for path, etiqueta, original in filas:
            try:
                v = features.ExtractFeatures(str(self.data / path), k=int(base["k"]), data=str(self.data))
            except Exception:
                continue
            ids.append(path)
            vecs.append(v)
            etiquetas.append(etiqueta)
            originales.append(original)
        salida = self._ruta_extras(kind, int(base["k"]))
        np.savez(salida, ids=np.array(ids), vectores=np.vstack(vecs).astype(np.float32),
                 etiqueta=np.array(etiquetas), original=np.array(originales))
        return salida

    def _ruta_cache(self, kind: str, index_name: str, metric: str, k: int, n: int) -> Path:
        # el hnsw de 10k audios tarda ~1.5 min y el de 20k portadas ~3 min en python:
        # se guarda la primera vez y después se carga en ~1 s
        return self.data / "features" / "indices_web" / f"{kind}_{index_name}_{metric}_k{k}_n{n}.npz"

    def _build(self, kind: str, index_name: str, metric: str) -> tuple[Any, np.ndarray, np.ndarray]:
        key = (kind, index_name, metric)
        with self._locks_lock:
            lock = self._locks.setdefault(key, threading.Lock())
        # si llegan dos búsquedas a la vez, la segunda espera al mismo índice en vez de armar otro
        with lock:
            if key in self._indexes:
                return self._indexes[key]
            base = self._load_base(kind)
            X = np.asarray(base["vectores"], dtype=np.float32)
            if len(X) == 0:
                raise ValueError("la base multimedia está vacía")
            if index_name not in {"ivf", "hnsw"}:
                raise ValueError("índice inválido: use ivf o hnsw")
            clase = IVF if index_name == "ivf" else HNSW
            cache = self._ruta_cache(kind, index_name, metric, int(base["k"]), len(X))
            started = time.perf_counter()
            if cache.is_file():
                index = clase.cargar(str(cache))
                meta = cache.with_suffix(".json")
                index.build_ms = json.loads(meta.read_text())["build_ms"] if meta.is_file() else 0.0
            else:
                if index_name == "ivf":
                    nlist = max(1, min(int(round(np.sqrt(len(X)))), len(X)))
                    index = IVF(nlist=nlist, metrica=metric).entrenar(X).agregar(X)
                else:
                    index = HNSW(M=16, ef_construction=100, metrica=metric).agregar(X)
                index.build_ms = (time.perf_counter() - started) * 1000
                cache.parent.mkdir(parents=True, exist_ok=True)
                index.guardar(str(cache))
                cache.with_suffix(".json").write_text(json.dumps({"build_ms": index.build_ms}))
            self._indexes[key] = (index, base["ids"], X)
            return self._indexes[key]

    def precalentar(self) -> None:
        # arma (o carga) de fondo los índices más usados, así la primera búsqueda no espera
        def trabajo() -> None:
            for kind in ("sift", "mfcc"):
                for index_name in ("ivf", "hnsw"):
                    try:
                        self._build(kind, index_name, "coseno")
                    except (FileNotFoundError, ValueError):
                        pass
        threading.Thread(target=trabajo, daemon=True).start()

    @staticmethod
    def _validar(kind: str, index_name: str, metric: str, top_k: int) -> tuple[str, str, str]:
        kind, index_name, metric = kind.lower(), index_name.lower(), metric.lower()
        if kind not in {"sift", "mfcc"}:
            raise ValueError("tipo inválido: use sift o mfcc")
        if index_name not in {"ivf", "hnsw"}:
            raise ValueError("índice inválido: use ivf o hnsw")
        if metric not in {"coseno", "euclidiana", "producto_punto"}:
            raise ValueError("métrica inválida")
        if not 1 <= top_k <= 100:
            raise ValueError("top_k debe estar entre 1 y 100")
        return kind, index_name, metric

    def _vecinos(self, kind: str, index_name: str, metric: str, query: np.ndarray, top_k: int,
                 query_position: int | None, query_id: str) -> dict[str, Any]:
        key = (kind, index_name, metric)
        index, ids, vectors = self._indexes.get(key) or self._build(kind, index_name, metric)
        started = time.perf_counter()
        if index_name == "ivf":
            result_ids, distances = index.buscar(query, top_k, nprobe=min(10, index.nlist))
        else:
            result_ids, distances = index.buscar(query, top_k, ef_search=max(50, top_k))
        # el índice devuelve posiciones en la base, el id real (número o ruta) sale de ids
        base = self._bases.get(kind, {})
        etiquetas, originales = base.get("etiqueta"), base.get("original")
        results = [{
            "rank": rank,
            "id": str(ids[int(position)]),
            "distance": float(distance),
            "is_query": int(position) == query_position,
            "label": str(etiquetas[int(position)]) if etiquetas is not None else "",
            "original": str(originales[int(position)]) if originales is not None else "",
        } for rank, (position, distance) in enumerate(zip(result_ids, distances), start=1)]
        return {
            "kind": kind,
            "index": index_name,
            "metric": metric,
            "query_id": query_id,
            "results": results,
            "documents": len(ids),
            "dimension": int(vectors.shape[1]),
            "build_ms": float(getattr(index, "build_ms", 0)),
            "duration_ms": (time.perf_counter() - started) * 1000,
            "candidates": int(getattr(index, "candidatos", 0)),
        }

    def search(self, kind: str, index_name: str, metric: str, query_id: str, top_k: int) -> dict[str, Any]:
        # consulta = un objeto que ya está en la base
        kind, index_name, metric = self._validar(kind, index_name, metric, top_k)
        _, ids, vectors = self._indexes.get((kind, index_name, metric)) or self._build(kind, index_name, metric)
        matches = np.flatnonzero(ids.astype(str) == str(query_id))
        if len(matches) == 0:
            raise ValueError(f"no existe el objeto multimedia {query_id!r}")
        position = int(matches[0])
        return self._vecinos(kind, index_name, metric, vectors[position], top_k, position, str(query_id))

    def search_file(self, path: Path, index_name: str, metric: str, top_k: int) -> dict[str, Any]:
        # consulta = un archivo subido que no está en la base: se le saca el vector con el mismo
        # codebook e idf que la colección
        kind = features.tipo_de(path)
        kind, index_name, metric = self._validar(kind, index_name, metric, top_k)
        k = int(self._load_base(kind)["k"])
        started = time.perf_counter()
        query = features.ExtractFeatures(str(path), k=k, data=str(self.data))
        extract_ms = (time.perf_counter() - started) * 1000
        result = self._vecinos(kind, index_name, metric, query, top_k, None, path.name)
        result["extract_ms"] = extract_ms
        return result

    def _ruta_grupos(self) -> Path:
        return self.data / "features" / "ejemplos_web.json"

    def armar_grupos(self, minimo: int = 5, top: int = 10) -> dict[str, list[str]]:
        # cada portada con fotos extra se busca en el hnsw y se cuenta cuántas de sus propias fotos
        # salen entre los top primeros; en la web solo se ofrecen las que llegan a minimo
        base = self._load_base("sift")
        index, ids, vectors = self._build("sift", "hnsw", "coseno")
        pos = {str(i): n for n, i in enumerate(ids)}
        tipos: dict[str, set[str]] = {}
        for o, e in zip(base["original"], base["etiqueta"]):
            if e == "portada":
                continue
            tipos.setdefault(str(o), set()).add("aumentada" if e.startswith("aumentada") else "foto")
        grupos: dict[str, list[tuple[int, str]]] = {"fotos": [], "aumentadas": [], "ambos": []}
        for original, t in tipos.items():
            if original not in pos:
                continue
            res, _ = index.buscar(vectors[pos[original]], top + 1, max(50, top + 1))
            propias = sum(1 for p in res if str(base["original"][int(p)]) == original
                          and int(p) != pos[original])
            if propias < minimo:
                continue
            grupo = "ambos" if t == {"foto", "aumentada"} else ("fotos" if t == {"foto"} else "aumentadas")
            grupos[grupo].append((propias, original))
        salida = {g: [o for _, o in sorted(v, reverse=True)] for g, v in grupos.items()}
        con_extras = set(tipos)
        salida["ninguno"] = [str(i) for i, e in zip(base["ids"], base["etiqueta"])
                             if e == "portada" and str(i) not in con_extras][:40]
        self._ruta_grupos().write_text(json.dumps(salida, ensure_ascii=False))
        return salida

    def grupos(self) -> dict[str, list[str]]:
        ruta = self._ruta_grupos()
        return json.loads(ruta.read_text()) if ruta.is_file() else {}

    def todos_los_ejemplos(self, kind: str, grupo: str = "") -> list[str]:
        # lista completa para paginar en la consola; siempre portadas originales (nunca fotos reales
        # ni versiones modificadas) o, en audio, canciones
        base = self._load_base(kind)
        if kind != "sift":
            return [str(v) for v in base["ids"]]
        if grupo in ("fotos", "aumentadas", "ambos"):
            return self.grupos().get(grupo, [])
        portadas = [str(i) for i, e in zip(base["ids"], base["etiqueta"]) if e == "portada"]
        if grupo == "ninguno":
            con_extras = {str(o) for o, e in zip(base["original"], base["etiqueta"]) if e != "portada"}
            return [p for p in portadas if p not in con_extras]
        return portadas

    def examples(self, kind: str, limit: int = 16, grupo: str = "", offset: int = 0) -> dict[str, Any]:
        todos = self.todos_los_ejemplos(kind, grupo)
        return {"ids": todos[offset:offset + limit], "total": len(todos), "offset": offset, "limit": limit}

multimedia_search = MultimediaSearchManager(Path(__file__).resolve().parents[2])


if __name__ == "__main__":
    # 1) vectores de las fotos extra, 2) índices con todo, 3) qué portadas llegan a 5 fotos propias
    print(multimedia_search.construir_extras_audio())
    for indice in ("ivf", "hnsw"):
        multimedia_search._build("mfcc", indice, "coseno")
    print(multimedia_search.construir_extras("sift"))
    for indice in ("ivf", "hnsw"):
        multimedia_search._build("sift", indice, "coseno")
    print({g: len(v) for g, v in multimedia_search.armar_grupos().items()})
