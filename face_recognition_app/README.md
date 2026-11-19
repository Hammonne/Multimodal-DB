# Reconocimiento facial sobre chutaDB

Aplicación web que registra personas con varias fotos, reconoce rostros en fotos subidas o tomadas con la
cámara, y compara modelos (ArcFace / FaceNet), índices (Exacto / HNSW / IVF) y métricas (coseno / euclidiana /
producto punto) usando las estructuras vectoriales propias de chutaDB.

```text
navegador ──HTTP──> /api/faces (api.py) ──> FaceService (service.py)
                                              ├─ FaceEngine (embeddings.py)   detección SCRFD + ArcFace/FaceNet   [fuera de la BD]
                                              ├─ VectorStore (vector_store.py) Exacto / HNSW / IVF del core
                                              └─ FaceRecognitionDB (db.py)     tablas Person / FaceImage(+embedding)
```

- Los **embeddings los calcula la app** (inferencia de redes neuronales) y se **guardan en chutaDB**, en la
  columna `FaceImage.embedding` (float32 en base64). La tabla es la fuente de verdad.
- La **búsqueda** la hace chutaDB: secuencial (`knn_exacto`) o con los `HNSW` / `IVF` del core
  (`VectorSecondaryIndex`), construidos desde esa columna. Los archivos de índice (`data/indexes`) son
  derivados: si faltan o no coinciden con la tabla se reconstruyen.
- Resultado: `RID → FaceImage → Person`.

## Instalación (WSL / Linux, Python 3.14)

Ver `requirements.txt`. Los pesos de los modelos se descargan solos la primera vez:
InsightFace `buffalo_l` (~280 MB, en `~/.insightface`) y FaceNet VGGFace2 (~107 MB, en la caché de torch).

## Preparar los datos (una vez, con el backend apagado)

```bash
cd Multimodal-DB
# 1. LFW (~200 MB)
mkdir -p face_recognition_app/data/lfw
python -c "from sklearn.datasets import fetch_lfw_people as f; f(data_home='face_recognition_app/data/lfw', min_faces_per_person=0, resize=0.1)"
# 2. split + embeddings (~30-40 min en CPU, reanudable) + siembra en chutaDB + índices + calibración del umbral
python -m face_recognition_app.scripts.prepare_lfw --all
# 3. evaluación global (tabla para el informe; se ve también en la pestaña "Índices y evaluación")
python -m face_recognition_app.evaluation
```

`prepare_lfw --all` **vacía** `Person` / `FaceImage` antes de sembrar; se niega si hay personas propias
registradas (usar `--force`). Los pasos sueltos: `--split`, `--embed`, `--seed [--reset]`, `--calibrate`.

## Ejecutar

```bash
cd Multimodal-DB
python -m uvicorn web.backend.main:app --port 8000
# abrir http://localhost:8000/api/faces/ui/
```

La primera carga de modelos tarda ~40 s (se hace en segundo plano al arrancar; la barra superior lo indica).
`FACE_APP_WARMUP=0` desactiva esa precarga. `FACE_APP_DATA` cambia la carpeta de datos.

## Uso

| Pestaña | Qué hace |
|---|---|
| **Reconocer** | Sube o toma una foto. Parámetros: modelo, índice, métrica, k, umbral (%), ef_search / nprobe. Muestra cada rostro con su decisión (conocido / desconocido), el ranking de personas con % de similitud, las top-k fotos y los tiempos. Con "Comparar configuraciones" agrega una tabla con la misma consulta en modelos × índices (× métricas): mejor persona, %, decisión, Recall@k vs exacto, latencia y candidatos. |
| **Registrar persona** | Nombre + varias fotos (archivos o cámara). Cada foto genera una fila `FaceImage` por modelo y se agrega a los índices de forma incremental. |
| **Verificar fotos** | 2–6 fotos → matriz de similitud por modelo y "misma persona" según el umbral. |
| **Personas** | Buscar, ver fotos, agregar fotos, eliminar. |
| **Índices y evaluación** | Estado de los 12 índices, reconstrucción con otros parámetros (M, ef_construction, nlist) y la tabla de la evaluación global. |

**Porcentaje de similitud** = similitud coseno × 100 (los embeddings están normalizados, así que es la misma
escala para las tres métricas). La **distancia** mostrada es la de la métrica elegida (la que usa el índice).
Con vectores normalizados las tres métricas producen el mismo ranking (‖a−b‖² = 2 − 2·cos); difieren en la
escala de la distancia. El **umbral** por defecto está calibrado por modelo con LFW (`data/calibration.json`).

## API (`/api/faces`)

| Método | Ruta | |
|---|---|---|
| GET | `/health` | estado del motor (`idle/loading/ready/error`) y tareas en curso |
| GET | `/config` · `/stats` | modelos/índices/métricas/defaults/umbrales · contenido de chutaDB |
| GET | `/persons?q=&source=&sort=&offset=&limit=` · `/persons/{id}` | listar (paginado) / detalle |
| POST | `/persons` (multipart `name`, `photos[]`) | registrar persona |
| POST | `/persons/{id}/photos` | agregar fotos |
| DELETE | `/persons/{id}` | eliminar |
| POST | `/recognize` (multipart `image`, `model`, `index`, `metric`, `k`, `threshold`, `ef_search`, `nprobe`) | reconocer |
| POST | `/compare` (multipart `image`, `models`, `indexes`, `metrics` como listas separadas por comas, `k`, `threshold`, `threshold_model`, `ef_search`, `nprobe`) | misma consulta con varias configuraciones |
| POST | `/verify` (multipart `images[]`, `threshold`) | foto vs foto (matriz de similitud) |
| GET / POST | `/indexes` · `/indexes/rebuild` | estado / reconstruir |
| GET / POST | `/evaluation/last` · `/evaluation/status` · `/evaluation/run` | evaluación con consultas LFW |
| GET | `/images/{photo_id}` · `/ui/` | fotos · frontend |

Cada alta/baja hace `FaceRecognitionDB.flush()`: chutaDB solo escribe sus páginas a disco al cerrar, así que
sin esto lo registrado se perdería si el backend termina sin cierre ordenado.

## Tests

```bash
python -m pytest face_recognition_starter_tests -q
```

Los tests usan chutaDB en un directorio temporal. Los de servicio/API usan un motor de IA falso (rápidos);
`test_embeddings_real.py` y `test_pipeline_real.py` usan los modelos reales y LFW (se saltan si faltan).

## Decisiones de diseño relevantes

- **Índices fuera de `sys_indexes`**: `IndexManager._load_embedding_index` falla al abrir la tabla si el
  índice no tiene tantas filas como `FaceImage` (con dos modelos nunca coincide). La app gestiona sus
  `VectorSecondaryIndex` y los registra en `index_manager.open_indexes`, no en el catálogo.
- **PK del heap**: el insert de chutaDB verifica la PK con un scan completo (cuadrático al sembrar). La app
  asigna IDs con mapas en memoria y desactiva esa verificación solo en sus dos tablas.
- **Extensible**: un modelo nuevo = una clase en `embeddings.EMBEDDERS` + `config.MODELS`; el frontend
  arma los selectores desde `/config`.

## Caché de personas frecuentes (Redis)

El reconocimiento tiene dos costos: extraer el embedding (red neuronal) y buscarlo
en el índice. El embedding no se puede cachear, porque cada foto que entra es
distinta aunque sea de la misma persona. La búsqueda sí.

`cache.py` mantiene en Redis los embeddings de las caras que más veces se
reconocieron. Antes de ir al índice se comparan contra ese conjunto caliente con
un solo producto matriz-vector; si alguna supera el umbral **con margen**, se
devuelve ese resultado y la búsqueda nunca toca el índice.

```text
embedding ──> conjunto caliente (Redis)  ──acierto──> resultado   (search_ms = 0)
                     │ fallo
                     └──────> VectorStore (Exacto / HNSW / IVF) ──> resultado
                                                                        │
                                            se promueve al conjunto ◄───┘
```

Tres garantías de corrección:

- **El caché nunca devuelve una persona distinta de la que devolvería el índice.**
  No alcanza con superar el umbral de reconocimiento: se exige además
  `CACHE_MARGIN`. Una cara ambigua se manda al índice.
- **Las entradas se aíslan por (modelo, versión de galería).** Al registrar o
  borrar una persona cambia `FaceRecognitionDB.version(model)` y las entradas
  viejas dejan de leerse.
- **Si la cara cacheada ya no existe en la base, el acierto se descarta** y la
  consulta sigue por el camino normal.

Si Redis no está disponible, el caché degrada a memoria del proceso: la
aplicación funciona igual, pero el caché no sobrevive al reinicio ni se comparte
entre procesos.

### Configuración

| Variable | Por defecto | Qué hace |
|---|---|---|
| `FACE_APP_REDIS_URL` | `redis://localhost:6379/0` | vacía o inalcanzable → caché en memoria |
| `FACE_APP_CACHE` | `1` | `0` lo desactiva |
| `FACE_APP_CACHE_MAX_HOT` | `256` | caras calientes por modelo y versión |
| `FACE_APP_CACHE_MARGIN` | `0.05` | margen sobre el umbral para resolver por caché |

`docker-compose.yml` ya levanta el servicio `redis` y le pasa la URL al backend.

### Endpoints

```bash
GET  /api/faces/cache         # backend, aciertos, fallos, hit rate
POST /api/faces/cache/clear   # vaciar (p. ej. para medir en frío)
```

En la respuesta de `/recognize`, un resultado servido por caché trae
`"source": "cache"`, `"search_ms": 0.0` y `"cache_ms"` con lo que costó la
consulta al conjunto caliente.

### Pruebas

```bash
python -m pytest face_recognition_starter_tests/test_cache.py -q
```

16 pruebas: el caché por sí solo, el backend de Redis —contra un servidor real
si lo hay, y si no contra `fakeredis`— y la integración con `recognize`, que
verifica que el resultado cacheado coincida con el del índice.
