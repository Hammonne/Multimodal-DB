# Plan de demostración web y dockerización

Este documento convierte los requisitos de las partes 2.3 y 2.4 en una
demostración reproducible para la presentación. La meta no es mostrar
solamente una tabla de resultados: la web debe permitir ejecutar una búsqueda,
ver el algoritmo usado, inspeccionar sus resultados y relacionarlos con las
mediciones del benchmark.

## 1. Estado actual

El repositorio ya contiene los componentes principales:

| Requisito | Implementación actual | Evidencia |
|---|---|---|
| Índice invertido SPIMI | `text_based/spimi.py` | bloques en memoria, runs temporales y merge |
| TF-IDF + coseno | `text_based/tfidf.py` | ranking `top_k` sobre postings |
| BM25 | `text_based/bm25.py` | `k1`, `b`, longitud promedio e IDF |
| SIFT y MFCC | `multimedia/sift.py`, `multimedia/mfcc.py` | descriptores locales |
| IVF, HNSW y métricas | `indexes/ivf.py`, `indexes/hnsw.py`, `indexes/metricas.py` | búsqueda k-NN y recall |
| Benchmark multimedia | `benchmark/multimedia/benchmark_vectorial.py` | tiempos, recall, espacio y galerías |
| Consola web | `web/app/` y `web/backend/main.py` | SQL, planes, resultados y transacciones |

El benchmark documentado actualmente llega a 10 000 objetos. El requisito de
100 000 debe ejecutarse con un dataset de ese tamaño o, si no se dispone de
datos reales, con un dataset sintético claramente etiquetado como tal. No se
deben presentar resultados de 10 000 como si fueran de 100 000.

## 2. Qué debe verse en la web

Se recomienda agregar dos vistas al frontend actual, sin quitar la consola SQL.

### 2.1 Vista “Búsqueda de texto”

Controles:

- caja de consulta;
- selector de técnica: `TF-IDF + Coseno` o `BM25`;
- `k` de resultados;
- dataset/tamaño: 1 000, 10 000 o 100 000;
- botón **Construir índice** y botón **Buscar**.

Resultados que deben aparecer:

1. tabla con `id`, título, fragmento del documento y score;
2. técnica, cantidad de documentos, cantidad de términos y fecha de
   construcción;
3. tiempo de construcción, tiempo de consulta y memoria/espacio;
4. Precision@10 y Recall@10 cuando la consulta tenga conjunto de relevancia
   conocido;
5. una sección “Cómo se calculó” con TF, IDF, normalización, `k1` y `b`.

Consultas mínimas para la demo:

```sql
-- 1 palabra
buscar: vectorial

-- 3 palabras
buscar: base de datos

-- 5 o más palabras
buscar: búsqueda eficiente en documentos de texto y datos vectoriales
```

La sintaxis SQL solicitada debe aceptarse como una extensión del parser:

```sql
SELECT * FROM documentos
WHERE MATCH(contenido, 'base de datos vectorial')
USING TF_IDF LIMIT 10;

SELECT *, SCORE() AS relevancia
FROM articulos
WHERE MATCH(texto, 'machine learning')
USING BM25
ORDER BY relevancia DESC;
```

La implementación debe validar que la técnica y la tabla existan y devolver un
error explícito si no están disponibles. No conviene convertir una consulta
inválida en una búsqueda vacía, porque eso ocultaría errores del parser.

### 2.2 Vista “Búsqueda multimedia”

Controles:

- carga o selección de una imagen/audio de consulta;
- selector de índice: `IVF` o `HNSW`;
- selector de métrica: `cosine`, `euclidean` o `dot`;
- `k`, inicialmente 10;
- selector de tipo de descriptor: `SIFT` o `MFCC`.

Cada resultado debe mostrar:

- miniatura o reproductor;
- id, nombre y metadatos;
- score/distancia;
- posición en el ranking;
- tiempo de consulta.

La pantalla debe incluir una galería separada de aciertos y errores. Un borde
verde identifica el objeto original cuando la consulta es una imagen recortada,
oscurecida, comprimida o un audio recortado/con ruido. Esto hace visible
`Recall@10`, no solamente una cifra.

### 2.3 Vista “Comparación”

La comparación debe leer los `resultados.json` generados por los benchmarks y
mostrar:

- construcción vs. tamaño del dataset;
- consulta vs. tamaño;
- Recall@10 vs. tamaño;
- espacio/memoria vs. tamaño;
- tabla filtrable por técnica, métrica y tipo de dato.

Para texto, agregar una tercera serie para PostgreSQL GIN. GIN no es parte del
motor propio: debe ejecutarse en un contenedor PostgreSQL separado, usando
`tsvector`, `websearch_to_tsquery` o `plainto_tsquery`, y el mismo conjunto de
documentos y consultas. La tabla debe indicar que GIN es un baseline externo.

## 3. API web propuesta

Mantener `/api/query` para SQL general y agregar endpoints especializados:

```text
GET  /api/search/health
POST /api/text/index
POST /api/text/search
GET  /api/text/benchmarks
POST /api/multimedia/index
POST /api/multimedia/search
GET  /api/multimedia/benchmarks
GET  /api/multimedia/gallery/{run_id}
```

La primera vertical web ya disponible expone `POST /api/text/index` y
`POST /api/text/search` para SPIMI, BM25 y TF-IDF. También está disponible
`POST /api/multimedia/search`, que consulta una base multimedia TF-IDF
precalculada usando IVF o HNSW, SIFT o MFCC, y las métricas coseno,
euclidiana o producto punto. `GET /api/multimedia/benchmarks` expone las
filas de `benchmark/multimedia/graficas/resultados.json` y
`GET /api/multimedia/benchmarks/assets/{filename}` sirve sus gráficas y
galerías. Las vistas `Texto`, `Multimedia` y `Benchmarks` de la consola
consumen estos endpoints y muestran tiempos, candidatos, scores, distancias,
Recall@10 y espacio en disco. Además,
`GET /api/multimedia/benchmarks/gallery/{gallery_key}` devuelve los casos
individuales de cada configuración (transformación, id original, posición del
original y top 5 recuperado). Si la corrida no generó el archivo de imagen de
la galería, la web conserva la tabla de casos y lo indica sin inventar una
imagen.

El baseline GIN se ejecuta con `python -m benchmark.text_gin`. Usa el mismo
`lyrics.jsonl`, las mismas cantidades de documentos y consultas de 1, 3 y 5+
palabras, y guarda `benchmark/text/results_gin.json`. La instancia PostgreSQL
de Docker se inicia con `docker compose up postgres`; su DSN por defecto es
`postgresql://postgres:postgres@localhost:5432/multimodal`. La API expone el
resultado mediante `GET /api/text/benchmarks`. Si el benchmark todavía no se
ha ejecutado, la web devuelve un estado explícito de “no disponible” y no
mezcla valores de GIN con los resultados propios.

Contratos mínimos:

```json
{
  "query": "machine learning",
  "algorithm": "bm25",
  "top_k": 10,
  "dataset": "1k"
}
```

La respuesta debe incluir `results`, `duration_ms`, `index_build_ms`,
`index_size_bytes`, `dataset_size` y, cuando exista ground truth,
`precision_at_10` y `recall_at_10`.

Los índices y benchmarks deben identificarse por una clave que incluya
dataset, descriptor, algoritmo y métrica. Así se evita servir por accidente un
índice SIFT cuando se pidió MFCC o mezclar tamaños durante la comparación.

## 4. Metodología experimental

### Texto

1. Preparar exactamente los mismos documentos para SPIMI, TF-IDF, BM25 y GIN.
2. Ejecutar tres tamaños: 1 000, 10 000 y 100 000.
3. Ejecutar consultas de 1, 3 y 5+ palabras, con varias repeticiones.
4. Separar el primer resultado de las repeticiones para no confundir carga en
   frío con caché.
5. Medir construcción, consulta promedio/p95, memoria máxima y tamaño en disco.
6. Definir los documentos relevantes antes de consultar y calcular
   Precision@10 y Recall@10 con la misma función para las tres técnicas.
7. Guardar parámetros, commit, fecha, hardware y resultados en JSON.

### Multimedia

1. Usar el mismo codebook y la misma normalización para IVF y HNSW.
2. Ejecutar SIFT y MFCC en 1 000, 10 000 y 100 000 objetos.
3. Comparar IVF/HNSW con coseno, euclidiana y producto punto.
4. Usar búsqueda lineal exacta como ground truth.
5. Medir construcción, consulta promedio/p95, Recall@10 y memoria.
6. Guardar galerías de aciertos/errores y los parámetros `nlist`, `nprobe`,
   `M`, `ef_construction` y `ef_search`.

Para cada gráfica se deben mostrar unidades y tamaño del dataset. Los resultados
de Python propio y de Faiss/hnswlib deben estar en series distintas; las
librerías externas sirven como referencia de rendimiento, no reemplazan la
implementación solicitada.

## 5. Ventajas y desventajas que debe mostrar la tabla

| Técnica | Ventajas | Desventajas | Recomendación |
|---|---|---|---|
| TF-IDF + coseno | simple, interpretable y útil para documentos cortos | sensible a longitud y no modela tan bien la saturación | baseline explicable |
| BM25 | normaliza longitud y suele rankear mejor texto heterogéneo | requiere calibrar `k1` y `b` | ranking textual principal |
| PostgreSQL GIN | integrado, maduro y fácil de operar | no es comparable como índice propio y requiere PostgreSQL | baseline de producción |
| IVF | construcción rápida y menos costo de consulta | pierde recall si `nprobe` es bajo | datasets grandes y actualizaciones |
| HNSW | buen recall y consultas rápidas | construcción y memoria mayores | muchas consultas y pocos cambios |

## 6. Dockerización

La solución recomendada usa dos servicios:

- `backend`: Python + FastAPI + motor, puerto 8000;
- `frontend`: Next.js, puerto 3000.

`data/` y el WAL deben ser volúmenes o bind mounts. Los datasets grandes,
features, índices generados y `node_modules` no deben copiarse a la imagen.
Esto mantiene las imágenes pequeñas y permite regenerar índices sin reconstruir
el frontend.

### Comandos

```bash
docker compose up --build
```

Abrir <http://localhost:3000>. Verificar el backend con:

```bash
curl http://localhost:8000/api/health
```

Para detener los servicios:

```bash
docker compose down
```

No usar `docker compose down -v` durante una demo si se quieren conservar los
índices y datos persistidos.

### Checklist de aceptación

- [ ] `docker compose up --build` inicia ambos servicios.
- [ ] `/api/health` responde `{"status":"ok"}`.
- [ ] la web carga sin depender de un `.venv` del host.
- [ ] una consulta SQL sigue mostrando resultados y plan.
- [ ] se puede construir y consultar al menos un índice de texto y uno
  multimedia.
- [ ] un reinicio conserva `data/` y el WAL.
- [ ] las gráficas y galerías se pueden abrir desde la vista de comparación.
- [ ] los resultados indican si son reales, sintéticos o pendientes.

## 7. Orden de implementación

1. Congelar el formato JSON de benchmarks y generar los tamaños faltantes.
2. Implementar los servicios de búsqueda de texto y multimedia.
3. Agregar la sintaxis `MATCH ... USING` y `SIMILAR_TO ... USING`.
4. Agregar endpoints y pruebas de API.
5. Construir las vistas web con estados de carga y errores explícitos.
6. Ejecutar GIN en PostgreSQL y añadirlo al mismo formato de resultados.
7. Ejecutar el benchmark 100k y generar gráficas finales.
8. Validar Docker desde una máquina limpia.
9. Ensayar la presentación con una consulta corta, una larga, una imagen
   modificada y un audio recortado.
