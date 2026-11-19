"""
Caché de personas frecuentes.

El reconocimiento tiene dos costos: extraer el embedding del rostro (red
neuronal, decenas de milisegundos) y buscarlo en el índice vectorial. El
embedding no se puede cachear —cada foto que entra es distinta, aunque sea de la
misma persona—, pero la búsqueda sí.

La idea es mantener en Redis los embeddings de las caras que más veces se
reconocieron. Antes de ir al índice se comparan contra ellas: si alguna supera
el umbral con margen suficiente, se devuelve ese resultado y la búsqueda no
llega a tocar el índice.

Es la misma lógica que un caché de CPU: un conjunto chico de entradas calientes
delante de una estructura grande y más lenta.

Decisiones que conviene conocer:

- *El caché nunca puede devolver una persona distinta de la que devolvería el
  índice.* Para eso no alcanza con superar el umbral de reconocimiento: se
  exige además un margen (`CACHE_MARGIN`). Una cara ambigua no se resuelve por
  caché, se manda al índice.
- *Se cachea por (modelo, versión de la galería).* Si se registra o se borra una
  persona, `FaceRecognitionDB.version(model)` cambia y las entradas viejas
  quedan huérfanas: no se leen más.
- *Si Redis no está disponible, se degrada a un caché en memoria del proceso.*
  La aplicación no depende de Redis para funcionar; Redis le agrega que el caché
  sobreviva al reinicio y se comparta entre procesos.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass

import numpy as np

from face_recognition_app import config


# ----------------------------------------------------------------------
# Resultado de una consulta al caché
# ----------------------------------------------------------------------


@dataclass
class CacheHit:
    face_id: int
    similarity: float
    lookup_ms: float


# ----------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------


class MemoryBackend:
    """Caché en el proceso. Es el modo de respaldo cuando no hay Redis."""

    name = "memoria"

    def __init__(self) -> None:
        self._emb: dict[str, bytes] = {}
        self._freq: dict[str, float] = {}
        self._gen = 0
        self._lock = threading.Lock()

    def generation(self, ns: str) -> int:
        return self._gen

    def snapshot(self, ns: str) -> dict[str, bytes]:
        with self._lock:
            return {k[len(ns) + 1:]: v for k, v in self._emb.items()
                    if k.startswith(ns + "|")}

    def touch(self, ns: str, key: str) -> None:
        with self._lock:
            self._freq[f"{ns}|{key}"] = self._freq.get(f"{ns}|{key}", 0) + 1

    def store(self, ns: str, key: str, blob: bytes, max_hot: int) -> None:
        with self._lock:
            full = f"{ns}|{key}"
            nuevo = full not in self._emb
            self._emb[full] = blob
            self._freq[full] = self._freq.get(full, 0) + 1
            if nuevo:
                self._gen += 1
            # desalojo: se quedan las max_hot más frecuentes
            propias = [k for k in self._emb if k.startswith(ns + "|")]
            if len(propias) > max_hot:
                sobran = sorted(propias, key=lambda k: self._freq.get(k, 0))
                for k in sobran[:len(propias) - max_hot]:
                    self._emb.pop(k, None)
                    self._freq.pop(k, None)
                self._gen += 1

    def clear(self) -> None:
        with self._lock:
            self._emb.clear()
            self._freq.clear()
            self._gen += 1

    def ping(self) -> bool:
        return True


class RedisBackend:
    """
    Caché compartido en Redis.

    Claves:
      <ns>:emb   HASH  face_id -> embedding (float32 crudo)
      <ns>:freq  ZSET  face_id -> cantidad de reconocimientos
      <ns>:gen   INT   cambia cuando cambia la MEMBRESÍA del conjunto caliente

    `gen` existe para no traer el hash completo en cada consulta: el cliente
    guarda una copia local y solo la refresca cuando entra o sale una cara. Las
    actualizaciones de frecuencia no invalidan la copia, porque no cambian qué
    embeddings hay.
    """

    name = "redis"

    def __init__(self, client) -> None:
        self.client = client

    def generation(self, ns: str) -> int:
        valor = self.client.get(f"{ns}:gen")
        return int(valor) if valor else 0

    def snapshot(self, ns: str) -> dict[str, bytes]:
        crudo = self.client.hgetall(f"{ns}:emb")
        return {(k.decode() if isinstance(k, bytes) else k): v for k, v in crudo.items()}

    def touch(self, ns: str, key: str) -> None:
        self.client.zincrby(f"{ns}:freq", 1, key)

    def store(self, ns: str, key: str, blob: bytes, max_hot: int) -> None:
        pipe = self.client.pipeline()
        pipe.hsetnx(f"{ns}:emb", key, blob)
        pipe.zincrby(f"{ns}:freq", 1, key)
        pipe.zcard(f"{ns}:freq")
        nuevo, _freq, total = pipe.execute()

        if nuevo:
            self.client.incr(f"{ns}:gen")

        if total > max_hot:
            # se descartan las menos frecuentes
            sobran = self.client.zrange(f"{ns}:freq", 0, total - max_hot - 1)
            if sobran:
                pipe = self.client.pipeline()
                pipe.hdel(f"{ns}:emb", *sobran)
                pipe.zrem(f"{ns}:freq", *sobran)
                pipe.incr(f"{ns}:gen")
                pipe.execute()

    def clear(self) -> None:
        for clave in self.client.scan_iter(match=f"{config.CACHE_PREFIX}:*"):
            self.client.delete(clave)

    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except Exception:
            return False


def _abrir_backend(url: str | None):
    """
    Devuelve (backend, detalle). Si Redis no responde, cae a memoria: el
    reconocimiento tiene que seguir funcionando sin infraestructura extra.
    """
    if not url:
        return MemoryBackend(), "sin REDIS_URL"
    try:
        import redis  # import local: la dependencia es opcional
    except ImportError:
        return MemoryBackend(), "el paquete redis no está instalado"

    try:
        cliente = redis.from_url(url, socket_connect_timeout=1, socket_timeout=1)
        cliente.ping()
        return RedisBackend(cliente), url
    except Exception as exc:
        return MemoryBackend(), f"Redis no responde ({type(exc).__name__})"


# ----------------------------------------------------------------------
# Caché
# ----------------------------------------------------------------------


class FaceCache:
    def __init__(
        self,
        url: str | None = None,
        enabled: bool | None = None,
        max_hot: int | None = None,
        margin: float | None = None,
    ) -> None:
        self.enabled = config.CACHE_ENABLED if enabled is None else bool(enabled)
        self.max_hot = config.CACHE_MAX_HOT if max_hot is None else int(max_hot)
        self.margin = config.CACHE_MARGIN if margin is None else float(margin)

        if self.enabled:
            self.backend, self.detalle = _abrir_backend(
                config.REDIS_URL if url is None else url)
        else:
            self.backend, self.detalle = MemoryBackend(), "deshabilitado"

        # copia local del conjunto caliente, por namespace
        self._local: dict[str, tuple[int, list[int], np.ndarray]] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    # ---------------- interno ----------------

    @staticmethod
    def _ns(model: str, version: int) -> str:
        return f"{config.CACHE_PREFIX}:{model}:{version}"

    def _matriz(self, ns: str) -> tuple[list[int], np.ndarray] | None:
        """
        Copia local del conjunto caliente como matriz, para comparar de una
        sola vez con un producto matriz-vector en lugar de cara por cara.
        """
        gen = self.backend.generation(ns)
        with self._lock:
            cacheado = self._local.get(ns)
            if cacheado is not None and cacheado[0] == gen:
                return (cacheado[1], cacheado[2]) if cacheado[1] else None

        crudo = self.backend.snapshot(ns)
        if not crudo:
            with self._lock:
                self._local[ns] = (gen, [], np.zeros((0, 0), dtype=np.float32))
            return None

        ids = [int(k) for k in crudo]
        matriz = np.stack([np.frombuffer(crudo[str(i)], dtype=np.float32) for i in ids])
        with self._lock:
            self._local[ns] = (gen, ids, matriz)
        return ids, matriz

    # ---------------- API ----------------

    def lookup(self, model: str, version: int, embedding: np.ndarray,
               threshold: float) -> CacheHit | None:
        """
        Busca el embedding entre las caras calientes.

        Devuelve un acierto solo si supera `threshold + margin`. El margen evita
        que el caché resuelva casos ambiguos que el índice resolvería mejor:
        ante la duda, se devuelve None y la consulta sigue su camino normal.
        """
        if not self.enabled:
            return None

        inicio = time.perf_counter()
        datos = self._matriz(self._ns(model, version))
        if datos is None:
            self.misses += 1
            return None

        ids, matriz = datos
        if matriz.shape[1] != embedding.shape[0]:
            self.misses += 1
            return None

        # los embeddings se guardan normalizados: el producto punto es el coseno
        sims = matriz @ embedding.astype(np.float32)
        mejor = int(np.argmax(sims))
        similitud = float(sims[mejor])

        if similitud < threshold + self.margin:
            self.misses += 1
            return None

        self.backend.touch(self._ns(model, version), str(ids[mejor]))
        self.hits += 1
        return CacheHit(
            face_id=ids[mejor],
            similarity=similitud,
            lookup_ms=(time.perf_counter() - inicio) * 1000,
        )

    def record(self, model: str, version: int, face_id: int,
               embedding: np.ndarray) -> None:
        """Promueve una cara al conjunto caliente tras un reconocimiento bueno."""
        if not self.enabled:
            return
        vector = np.asarray(embedding, dtype=np.float32)
        norma = float(np.linalg.norm(vector))
        if norma > 0:
            vector = vector / norma
        self.backend.store(self._ns(model, version), str(face_id),
                           vector.tobytes(), self.max_hot)

    def invalidate(self) -> None:
        self.backend.clear()
        with self._lock:
            self._local.clear()

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "enabled": self.enabled,
            "backend": self.backend.name,
            "detail": self.detalle,
            "max_hot": self.max_hot,
            "margin": self.margin,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }
