"""
Caché de personas frecuentes.

Se prueban las dos mitades por separado:

1. El caché en sí (`FaceCache`) contra su backend de memoria y, si hay un Redis
   levantado, también contra Redis. Los tests marcados con `redis` se saltean
   solos cuando no hay servidor.
2. Su integración en `FaceService.recognize`: que la segunda vez que se
   reconoce la misma cara el resultado salga del caché, y —lo más importante—
   que el resultado cacheado coincida con el que devolvería el índice.
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from face_recognition_app.api import create_face_router
from face_recognition_app.cache import FaceCache, MemoryBackend, RedisBackend
from face_recognition_app.service import FaceService

from conftest import FakeEngine, fake_image


# ----------------------------------------------------------------------
# utilidades
# ----------------------------------------------------------------------


def _vector(semilla: int, dim: int = 64) -> np.ndarray:
    rng = np.random.default_rng(semilla)
    v = rng.normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _redis_disponible() -> bool:
    try:
        import redis
        cliente = redis.from_url("redis://localhost:6379/15", socket_connect_timeout=0.3,
                                 socket_timeout=0.3)
        cliente.ping()
        return True
    except Exception:
        return False


def _fakeredis():
    """Cliente Redis simulado, para ejercitar la lógica sin un servidor."""
    try:
        import fakeredis
        return fakeredis.FakeStrictRedis()
    except ImportError:
        return None


def _cache_redis(max_hot: int) -> FaceCache | None:
    """
    Caché sobre Redis: el servidor real si está disponible, y si no uno
    simulado. Devuelve None solo si no hay ninguno de los dos.
    """
    if _redis_disponible():
        cache = FaceCache(url="redis://localhost:6379/15", max_hot=max_hot)
        cache.invalidate()
        return cache
    cliente = _fakeredis()
    if cliente is None:
        return None
    cache = FaceCache(url="", max_hot=max_hot)
    cache.backend = RedisBackend(cliente)
    return cache


necesita_redis = pytest.mark.skipif(
    not _redis_disponible() and _fakeredis() is None,
    reason="no hay Redis local ni fakeredis instalado")


# ----------------------------------------------------------------------
# 1. el caché por sí solo
# ----------------------------------------------------------------------


def test_cache_vacio_no_acierta():
    cache = FaceCache(url="", max_hot=8)
    assert cache.lookup("arcface", 1, _vector(1), threshold=0.3) is None
    assert cache.stats()["hits"] == 0
    assert cache.stats()["misses"] == 1


def test_cache_acierta_con_el_mismo_vector():
    cache = FaceCache(url="", max_hot=8, margin=0.05)
    v = _vector(1)
    cache.record("arcface", 1, face_id=42, embedding=v)

    hit = cache.lookup("arcface", 1, v, threshold=0.3)
    assert hit is not None
    assert hit.face_id == 42
    assert hit.similarity == pytest.approx(1.0, abs=1e-5)


def test_cache_no_acierta_con_un_vector_distinto():
    cache = FaceCache(url="", max_hot=8)
    cache.record("arcface", 1, face_id=42, embedding=_vector(1))
    # otro vector al azar: en 64 dimensiones su coseno ronda cero
    assert cache.lookup("arcface", 1, _vector(999), threshold=0.3) is None


def test_el_margen_evita_los_casos_ambiguos():
    """
    Una cara que supera el umbral pero no el margen tiene que ir al índice.
    Es la garantía de que el caché no resuelve lo que podría resolver mal.
    """
    cache = FaceCache(url="", max_hot=8, margin=0.10)
    base = _vector(1)
    cache.record("arcface", 1, face_id=7, embedding=base)

    # vector a ~0.88 de coseno del original: supera 0.80 pero no 0.80 + 0.10
    ruido = _vector(2)
    mezcla = 0.88 * base + np.sqrt(1 - 0.88 ** 2) * (ruido - ruido @ base * base)
    mezcla = (mezcla / np.linalg.norm(mezcla)).astype(np.float32)
    similitud = float(mezcla @ base)
    assert 0.80 < similitud < 0.90, similitud

    assert cache.lookup("arcface", 1, mezcla, threshold=0.80) is None
    # con un umbral más bajo, el mismo vector sí entra
    assert cache.lookup("arcface", 1, mezcla, threshold=0.50) is not None


def test_versiones_distintas_no_se_mezclan():
    """Si cambia la galería, las entradas viejas dejan de usarse."""
    cache = FaceCache(url="", max_hot=8)
    v = _vector(1)
    cache.record("arcface", version=1, face_id=42, embedding=v)

    assert cache.lookup("arcface", 1, v, threshold=0.3) is not None
    assert cache.lookup("arcface", 2, v, threshold=0.3) is None   # otra versión
    assert cache.lookup("facenet", 1, v, threshold=0.3) is None   # otro modelo


def test_desalojo_por_frecuencia():
    """Con el caché lleno, se conservan las caras más reconocidas."""
    cache = FaceCache(url="", max_hot=3)
    vectores = {i: _vector(i) for i in range(5)}

    for i in range(3):
        cache.record("arcface", 1, face_id=i, embedding=vectores[i])
    # la 0 se usa mucho; la 1 y la 2 no
    for _ in range(5):
        cache.lookup("arcface", 1, vectores[0], threshold=0.3)

    # entran dos caras nuevas: el caché tiene que descartar las menos usadas
    cache.record("arcface", 1, face_id=3, embedding=vectores[3])
    cache.record("arcface", 1, face_id=4, embedding=vectores[4])

    assert cache.lookup("arcface", 1, vectores[0], threshold=0.3) is not None, \
        "la cara más frecuente no debería desalojarse"
    assert cache.lookup("arcface", 1, vectores[4], threshold=0.3) is not None


def test_invalidate_vacia_el_cache():
    cache = FaceCache(url="", max_hot=8)
    v = _vector(1)
    cache.record("arcface", 1, face_id=42, embedding=v)
    assert cache.lookup("arcface", 1, v, threshold=0.3) is not None

    cache.invalidate()
    assert cache.lookup("arcface", 1, v, threshold=0.3) is None


def test_deshabilitado_nunca_acierta():
    cache = FaceCache(url="", enabled=False)
    v = _vector(1)
    cache.record("arcface", 1, face_id=42, embedding=v)
    assert cache.lookup("arcface", 1, v, threshold=0.3) is None
    assert cache.stats()["enabled"] is False


def test_url_invalida_cae_a_memoria():
    """Sin Redis la aplicación tiene que seguir funcionando."""
    cache = FaceCache(url="redis://localhost:1/0")
    assert isinstance(cache.backend, MemoryBackend)
    v = _vector(1)
    cache.record("arcface", 1, face_id=42, embedding=v)
    assert cache.lookup("arcface", 1, v, threshold=0.3) is not None


# ----------------------------------------------------------------------
# 2. el mismo comportamiento, contra Redis de verdad
# ----------------------------------------------------------------------


@necesita_redis
def test_redis_backend():
    cache = _cache_redis(max_hot=4)
    assert isinstance(cache.backend, RedisBackend)

    v = _vector(1)
    cache.record("arcface", 1, face_id=42, embedding=v)

    hit = cache.lookup("arcface", 1, v, threshold=0.3)
    assert hit is not None and hit.face_id == 42

    # otra instancia sobre el mismo Redis tiene que ver lo guardado:
    # es lo que Redis agrega frente al caché en memoria
    otro = FaceCache(url="", max_hot=4)
    otro.backend = RedisBackend(cache.backend.client)
    assert otro.lookup("arcface", 1, v, threshold=0.3) is not None

    cache.invalidate()
    assert otro.lookup("arcface", 1, v, threshold=0.3) is None


@necesita_redis
def test_redis_aisla_por_version_y_modelo():
    cache = _cache_redis(max_hot=4)
    v = _vector(1)
    cache.record("arcface", version=1, face_id=42, embedding=v)
    assert cache.lookup("arcface", 1, v, threshold=0.3) is not None
    assert cache.lookup("arcface", 2, v, threshold=0.3) is None
    assert cache.lookup("facenet", 1, v, threshold=0.3) is None
    cache.invalidate()


@necesita_redis
def test_redis_desaloja():
    cache = _cache_redis(max_hot=2)
    for i in range(6):
        cache.record("arcface", 1, face_id=i, embedding=_vector(i))
    # nunca más de max_hot entradas
    assert len(cache.backend.snapshot(cache._ns("arcface", 1))) <= 2
    cache.invalidate()


# ----------------------------------------------------------------------
# 3. integración con el reconocimiento
# ----------------------------------------------------------------------


@pytest.fixture
def client(temp_db, tmp_path):
    engine = FakeEngine()
    service = FaceService(temp_db.db, engine_factory=lambda: engine,
                          data_dir=tmp_path / "appdata",
                          cache=FaceCache(url="", max_hot=16))
    app = FastAPI()
    app.include_router(create_face_router(temp_db.db, service=service))
    c = TestClient(app)
    c.service = service
    return c


def _registrar(client, nombre, identidad, n=3):
    fotos = [(("photos", (f"f{i}.jpg", fake_image(f"{identidad}/{i}"), "image/jpeg")))
             for i in range(n)]
    r = client.post("/api/faces/persons", data={"name": nombre}, files=fotos)
    assert r.status_code == 200, r.text
    return r.json()


def _reconocer(client, identidad):
    r = client.post("/api/faces/recognize",
                    files={"image": ("q.jpg", fake_image(identidad), "image/jpeg")})
    assert r.status_code == 200, r.text
    return r.json()["faces"][0]["result"]


def test_el_segundo_reconocimiento_sale_del_cache(client):
    _registrar(client, "Ana Torres", "ana")
    _registrar(client, "Bruno Díaz", "bruno")

    primero = _reconocer(client, "ana/0")
    assert primero["decision"]["status"] == "known"
    assert primero.get("source") != "cache", "la primera vez no puede haber cache"

    segundo = _reconocer(client, "ana/0")
    assert segundo["source"] == "cache"
    assert segundo["search_ms"] == 0.0, "un acierto de cache no toca el indice"

    # y lo esencial: el cache devuelve la MISMA persona que el indice
    assert segundo["decision"]["person_id"] == primero["decision"]["person_id"]
    assert segundo["decision"]["name"] == primero["decision"]["name"]

    stats = client.get("/api/faces/cache").json()
    assert stats["hits"] >= 1


def test_el_cache_no_cambia_la_respuesta(client):
    """Para varias personas, con y sin caché el resultado tiene que coincidir."""
    for nombre, ident in [("Ana Torres", "ana"), ("Bruno Díaz", "bruno"),
                          ("Carla Ruiz", "carla")]:
        _registrar(client, nombre, ident)

    esperado = {}
    for ident in ("ana/0", "bruno/0", "carla/0"):
        esperado[ident] = _reconocer(client, ident)["decision"]["person_id"]

    # segunda vuelta: ahora varias deberían resolverse por caché
    for ident, person_id in esperado.items():
        res = _reconocer(client, ident)
        assert res["decision"]["person_id"] == person_id, ident


def test_un_desconocido_no_se_cachea(client):
    _registrar(client, "Ana Torres", "ana")

    antes = client.get("/api/faces/cache").json()["hits"]
    r1 = _reconocer(client, "zzz/0")
    r2 = _reconocer(client, "zzz/0")
    despues = client.get("/api/faces/cache").json()["hits"]

    # si no se reconoció a nadie, no hay nada que promover al conjunto caliente
    assert r1["decision"]["status"] == r2["decision"]["status"]
    if r1["decision"]["status"] == "unknown":
        assert despues == antes
        assert r2.get("source") != "cache"


def test_endpoint_de_cache(client):
    _registrar(client, "Ana Torres", "ana")
    _reconocer(client, "ana/0")
    _reconocer(client, "ana/0")

    stats = client.get("/api/faces/cache").json()
    assert stats["enabled"] is True
    assert stats["backend"] in ("memoria", "redis")
    assert 0.0 <= stats["hit_rate"] <= 1.0

    vacio = client.post("/api/faces/cache/clear").json()
    assert vacio["enabled"] is True
    # tras vaciarlo, el siguiente reconocimiento vuelve a pasar por el índice
    assert _reconocer(client, "ana/0").get("source") != "cache"
