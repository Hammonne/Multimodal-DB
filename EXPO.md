# EXPO.md — chutaDB: texto, multimedia e IA

Resumen en español de las Partes 3, 4 y 5 del proyecto **chutaDB**, con énfasis en:

- qué índices se usan y cómo funcionan a alto nivel;
- ventajas, debilidades y comparaciones entre índices;
- resultados experimentales y recomendaciones;
- qué hace PostgreSQL para poder sacarnos “años luz” de ventaja, más allá de que Python sea más lento.

> Idea central para la exposición: **no estamos comparando solo algoritmos, estamos comparando un motor educativo en Python contra un RDBMS maduro escrito en C, con planificador, caché, concurrencia, estadísticas y extensiones nativas.** El algoritmo puede ser el mismo; la diferencia está en la ingeniería alrededor.

---

## 0. Mapa general de la solución

| Parte | Problema | Índice / técnica principal | Baseline externo |
|---|---|---|---|
| 2.3 | Buscar dentro de texto | SPIMI (índice invertido) + TF-IDF/coseno + BM25 | GIN de PostgreSQL |
| 2.4 | Buscar imágenes/audio parecidos | SIFT/MFCC + codebook K-Means + IVF / HNSW | Búsqueda lineal exacta, Faiss/hnswlib |
| 2.5 | Aplicación real | RAG híbrido: BM25 + HNSW + RRF + LLM | PostgreSQL/pgvector como referencia conceptual |

El motor base también usa índices relacionales y espaciales: **B+ tree**, **hash extensible**, **bitmap** y **R-Tree**. No son el foco de esta exposición, pero sostienen filtros, joins y el mapa de la aplicación.

---

## 1. Parte 3 — Base de datos textual con búsqueda full-text

### 1.1 Pipeline de búsqueda textual

```text
documentos (JSONL / tabla SQL)
        │
        ▼
tokenización y conteo de términos
        │
        ▼
índice invertido SPIMI  ──►  término → [(doc_id, frecuencia), ...]
        │
        ├──► TF-IDF + similitud del coseno
        └──► BM25
        │
        ▼
top-k documentos ordenados por SCORE()
```

Cada documento es una unidad indexable. En el proyecto se usan letras de canciones y, en el RAG, chunks de papers.

### 1.2 Índice usado: SPIMI

**SPIMI** significa *Single-Pass In-Memory Indexing*. La idea es construir un índice invertido leyendo cada documento una sola vez, manteniendo un diccionario temporal en RAM y volcándolo a disco cuando se supera un presupuesto de memoria.

```text
diccionario temporal en RAM:
{
  "gato": {1: 2, 5: 1},
  "perro": {1: 1}
}
```

- **Presupuesto:** por defecto 64 MiB estimados en bytes.
- **Bloques o runs:** cuando se llena la memoria, se ordenan los términos y se escribe un bloque binario temporal.
- **Merge progresivo:** los bloques se fusionan en grupos de como máximo `MAX_OPEN_RUNS` (16 por defecto), para no abrir cientos de archivos a la vez.
- **Salida final:** un `.bin` con los términos y postings, y un `.json` con metadata y offsets.
- **Consulta:** se busca el término en el diccionario de offsets, se hace `seek()` al registro y se leen solo sus postings.

Formato simplificado de un registro:

```text
uint32  longitud_del_término
bytes   término UTF-8
uint32  document_frequency (df)
repetido df veces:
    int64   doc_id
    uint32  term_frequency (tf)
```

**Complejidad aproximada:**

| Operación | Coste |
|---|---|
| Tokenizar y contar | O(T), T = tokens totales |
| Ordenar y escribir un bloque | O(P_b log P_b), P_b = postings del bloque |
| Merge progresivo | O(P log B), P = postings, B = bloques |
| `get_postings(term)` | O(df del término) |

**Ventajas de SPIMI:**

- Lee cada documento una sola vez.
- La RAM está acotada por un presupuesto.
- Se puede volcar en medio de un documento.
- El índice final es binario y compacto.
- Al cargar solo se lee la metadata; consultar un término no recorre todo el índice.
- Bueno para un motor embebido, sin servidor y fácil de explicar.

**Debilidades de SPIMI:**

- **No guarda posiciones**, así que no puede hacer búsqueda de frases exactas.
- **No aplica stemming ni elimina stopwords** en la implementación actual.
- El límite de memoria es una estimación, no el RSS exacto del proceso.
- El merge final junta en RAM todos los postings de un término.
- Los `doc_id` deben ser enteros.
- **No se actualiza incrementalmente**: si cambian los documentos, se reconstruye.
- No está integrado con transacciones ni locks.
- El ranking se calcula después, en una pasada sobre los postings candidatos.

### 1.3 Ranking: TF-IDF + coseno vs BM25

**TF-IDF + coseno:**

```text
idf(t) = log10(N / df(t))
peso(t, d) = tf(t, d) · idf(t)

similitud(q, d) =
    Σ peso(t, q) · peso(t, d)
    ─────────────────────────
        ||q|| · ||d||
```

- Favorece términos raros.
- El coseno mide el ángulo entre vectores, no la magnitud.
- Es simple y explicable, pero **no modela bien la cobertura parcial** de consultas con varias palabras.

**BM25:**

```text
idf(t) = log(1 + (N − df(t) + 0.5) / (df(t) + 0.5))
norm(d) = 1 − b + b · |d| / avgdl
tf_bm25(t, d) = f(t, d) · (k1 + 1) / (f(t, d) + k1 · norm(d))

puntaje(q, d) = Σ idf(t) · tf_bm25(t, d)
```

Parámetros por defecto: `k1 = 1.2`, `b = 0.75`.

- **Satura la frecuencia**: repetir una palabra 20 veces no vale 20 veces más.
- **Normaliza por longitud**: un documento largo no gana solo por ser largo.
- **Mantiene el idf positivo** para términos frecuentes gracias al `1 +`.
- En el benchmark fue el mejor ranker para consultas de 3 y 5+ palabras.

### 1.4 Extensión SQL

El parser reconoce `MATCH`, `SCORE()` y los tipos de índice textuales:

```sql
SELECT *
FROM documentos
WHERE MATCH(contenido, 'base de datos vectorial')
USING TF_IDF
LIMIT 10;
```

```sql
SELECT *, SCORE() AS relevancia
FROM articulos
WHERE MATCH(texto, 'machine learning')
USING BM25
ORDER BY relevancia DESC;
```

También se puede registrar el índice:

```sql
CREATE INDEX ON documentos (contenido) USING SPIMI;
-- o USING TF_IDF / USING BM25, que reutilizan el mismo índice invertido
```

Puntos importantes:

- `MATCH` tokeniza la consulta, busca cada término y genera candidatos.
- `SCORE()` expone el puntaje del ranker.
- Sin índice registrado, el motor puede construir uno automático sobre la tabla.
- `BM25` y `TF_IDF` **no crean índices distintos**: comparten el índice SPIMI y cambian solo la fórmula de ranking.

### 1.5 Baseline: GIN de PostgreSQL

GIN (*Generalized Inverted Index*) es el índice de texto de PostgreSQL. A alto nivel:

- Convierte el texto a `tsvector`.
- Guarda un **entry tree** de términos y **posting lists / posting trees** de documentos.
- Soporta `tsquery`, `plainto_tsquery`, `to_tsvector`, diccionarios, stemming, stopwords y búsqueda de frases cuando el `tsvector` conserva posiciones.
- Puede usar **pending list** y `fastupdate` para acelerar inserciones.
- Una consulta GIN suele producir un **bitmap index scan**: primero obtiene un bitmap de filas candidatas, luego accede al heap y hace *recheck*.
- El ranking no lo hace GIN: lo hace una función aparte como `ts_rank_cd`.

En el benchmark se usó PostgreSQL 18 con:

```sql
CREATE TABLE gin_documents (
    id BIGINT PRIMARY KEY,
    content TEXT NOT NULL,
    search_vector TSVECTOR GENERATED ALWAYS AS
        (to_tsvector('simple', content)) STORED
);

CREATE INDEX gin_documents_search_idx
ON gin_documents USING GIN (search_vector);
```

Consulta:

```sql
SELECT id
FROM gin_documents
WHERE search_vector @@ plainto_tsquery('simple', '...')
LIMIT 10;
```

### 1.6 Resultados experimentales de texto

**Metodología:**

- Corpus: letras de `data/lyrics.jsonl`, hasta 10 000 documentos reales.
- Consultas: 1 palabra, 3 palabras y 5+ palabras.
- Ground truth: un documento es relevante si contiene al menos 1, 2 o 3 términos según la longitud de la consulta.
- Métricas: tiempo de construcción, tiempo de consulta, Precision@10, Recall@10, espacio y memoria.
- PostgreSQL GIN usa `ts_rank_cd` en el benchmark completo.

**Construcción y espacio a N = 10 000:**

| Técnica | Construcción | Espacio en disco |
|---|---:|---:|
| TF-IDF + coseno | 7.7 s | 14.9 MB (índice SPIMI) |
| BM25 | 7.2 s | 14.9 MB (mismo índice SPIMI) |
| GIN / PostgreSQL | **5.9 s** | 46.6 MB (índice + tabla + `tsvector`) |

GIN construye más rápido y su índice solo pesa menos, pero al sumar la tabla y el `tsvector` ocupa aproximadamente **3.1 veces más** que SPIMI a 10 000 documentos.

**Consulta promedio a N = 10 000:**

| Técnica | Consulta media | Precision@10 | Recall@10 |
|---|---:|---:|---:|
| TF-IDF + coseno | 8.22 ms | 0.680 | 0.036 |
| **BM25** | **6.34 ms** | **0.970** | **0.066** |
| GIN / PostgreSQL | 58.48 ms | 0.590 | 0.033 |

**Consulta de 5+ palabras a N = 10 000:**

| Técnica | Consulta | Precision@10 |
|---|---:|---:|
| TF-IDF + coseno | 18.40 ms | 0.467 |
| **BM25** | **13.44 ms** | **0.967** |
| GIN / PostgreSQL | 146.36 ms | 0.333 |

Lectura:

- Con **una palabra**, las tres técnicas son rápidas y precisas.
- Con **varias palabras**, BM25 es claramente mejor en ranking y más rápido que GIN.
- TF-IDF + coseno es una línea base simple, pero castiga la cobertura parcial.
- GIN es un índice excelente para *encontrar* candidatos, pero `ts_rank_cd` no está optimizado para esta noción de relevancia y el costo de bitmap + heap + recheck + ordenamiento se nota con consultas multivocablo.
- **No es una comparación pura de índices:** TF-IDF/BM25 corren embebidos en Python, sin red; GIN incluye parseo SQL, planificación, servidor y ida/vuelta.

### 1.7 Ventajas y debilidades: texto

| Técnica | Fuerte en | Débil en | Cuándo usarla |
|---|---|---|---|
| **SPIMI + TF-IDF** | Simple, embebido, explicable, bueno con términos raros y una palabra | Peor ranking multivocablo; sin posiciones, stemming ni stopwords; reconstrucción completa | Línea base educativa o índice embebido pequeño |
| **SPIMI + BM25** | Mejor ranking general, saturación y normalización por longitud, consulta rápida | Misma construcción que SPIMI; sin concurrencia ni actualización incremental | Recomendada dentro de chutaDB para texto |
| **GIN / PostgreSQL** | Construcción rápida, persistencia, transacciones, concurrencia, diccionarios y frase | Ranking con `ts_rank_cd` flojo para esta métrica; consultas multivocablo más lentas; más espacio total | Cuando ya hay PostgreSQL y pesa más la operación que la latencia o el ranking |

---

## 2. Parte 4 — Base de datos multimedia: imágenes y audio

### 2.1 Pipeline de características

```text
imagen ──► SIFT ──► descriptores locales de 128 dimensiones
audio  ──► MFCC ──► coeficientes acústicos (39 por ventana)

descriptores ──► K-Means (codebook) ──► K “palabras” visuales/auditivas
        │
        ▼
histograma de palabras + TF-IDF ──► vector de dimensión K
        │
        ▼
IVF / HNSW / búsqueda lineal
```

**SIFT** detecta puntos clave invariantes a escala y rotación y describe la zona alrededor de cada punto. Funciona en gris, así que el color no le importa.

**MFCC** resume la forma del espectro de audio en ventanas cortas. Es útil para similitud acústica, pero es frágil si se pierden frecuencias, por ejemplo al grabar con un celular.

**Codebook:**

- Se agrupan los descriptores de toda la colección con K-Means.
- Cada centroide es una “palabra visual” o “palabra auditiva”.
- Cada objeto se convierte en un histograma de frecuencia de palabras.
- Se aplica TF-IDF para dar más peso a palabras discriminativas.
- El resultado es un vector de dimensión `K` (en el dataset principal, K = 1024).
- En columnas `IMAGE`/`AUDIO` del motor, cada índice entrena su propio codebook con `K = raíz(total de descriptores)`, entre 8 y 1024.

### 2.2 Índices vectoriales usados

#### Búsqueda lineal o secuencial (`SEQ`)

- Compara la consulta contra **todos** los vectores.
- Es la respuesta exacta y sirve como *ground truth* para medir recall.
- Coste de consulta O(N).
- No necesita construcción.
- Con menos de ~2 500 objetos suele ser la mejor opción.

#### IVF (*Inverted File Index*)

```text
entrenar: K-Means sobre los vectores ──► nlist centroides (celdas)
agregar:  cada vector va a la lista de su centroide más cercano
buscar:   q ──► distancia a los nlist centroides
               ──► abrir las nprobe celdas más cercanas
               ──► comparar q solo con los vectores de esas listas
               ──► top k
```

- **`nlist`**: número de celdas, no cantidad de vectores por celda.
- **`nprobe`**: cuántas celdas se abren por consulta. Más `nprobe` = más recall y más tiempo.
- Se revisa aproximadamente `nprobe / nlist` de la base, más el costo de comparar contra los centroides.
- Con coseno se normaliza antes de entrenar: K-Means esférico agrupa por dirección.
- Ventajas: construcción rápida, actualización barata (agregar a una lista), poca memoria extra.
- Debilidades: es aproximado; si el vecino real cae en una celda no abierta, se pierde. Las celdas no se reentrenan automáticamente. Con euclidiana sobre vectores sin normalizar, K-Means puede agrupar por tamaño y degenerar.

#### HNSW (*Hierarchical Navigable Small World*)

```text
nivel 2:  A ──────────────────────── F        pocos nodos, saltos largos
nivel 1:  A ─────── C ───────── E ── F
nivel 0:  A ── B ── C ── D ── E ── F ── G ── H   todos los nodos
```

- Cada vector es un nodo de un grafo conectado con vecinos cercanos.
- Hay varios niveles: arriba pocos nodos y saltos largos; abajo todos los nodos.
- La búsqueda empieza en el nivel superior, baja haciendo búsqueda voraz y en el nivel 0 explora `ef_search` candidatos.
- **`M`**: conexiones por nodo.
- **`ef_construction`**: candidatos al insertar.
- **`ef_search`**: candidatos al consultar. Más alto = más recall y más tiempo.
- Ventajas: muy buen recall, consulta casi logarítmica, el más rápido en el benchmark.
- Debilidades: construcción cara, actualización/borrado costosos, consume más memoria por el grafo, es aproximado y requiere ajustar `ef_search`.
- En Python puro, construir 20 000 portadas tarda ~174 s; consultar ~1.5 ms.

### 2.3 Métricas de similitud

| Métrica | Similitud | Distancia usada por el índice | Alias SQL |
|---|---|---|---|
| Euclidiana | — | `||q − x||` | `euclidean`, `l2` |
| Producto punto | `q · x` | `−q · x` | `dot`, `ip`, `inner_product` |
| Coseno | `q · x / (||q|| ||x||)` | `1 − coseno` | `cosine`, `cos` |

Detalles importantes:

- Los índices ordenan como **distancia**: menor = más parecido.
- Si los vectores están normalizados, las tres métricas dan el mismo ranking:
  `||q − x||² = 2 − 2·cos`, y `q · x = cos`.
- Por eso el motor guarda los vectores normalizados y un solo índice sirve para las tres métricas; la métrica solo cambia el valor mostrado por `SIMILARITY_SCORE()`.
- En el benchmark se usan vectores **sin normalizar** para comparar coseno contra euclidiana. Ahí se ve que la euclidiana empeora: el “vecino más cercano” depende del tamaño del vector, no solo de su contenido.

### 2.4 Extensión SQL multimedia

```sql
SELECT *
FROM imagenes
WHERE SIMILAR_TO('foto_consulta.jpg', k=10)
USING HNSW WITH METRIC=cosine;
```

```sql
SELECT nombre, artista, SIMILARITY_SCORE() AS score
FROM canciones
WHERE SIMILAR_TO('audio_query.mp3', k=5)
USING IVF WITH METRIC=euclidean
ORDER BY score DESC;
```

También se puede crear el índice:

```sql
CREATE INDEX ON imagenes (archivo) USING HNSW;
CREATE INDEX ON canciones (audio) USING IVF;
```

Sintaxis soportada:

- `SIMILAR_TO('ruta', k=N)`.
- `USING IVF | HNSW | SEQ`.
- `WITH METRIC = cosine | euclidean | dot`.
- `WITH NPROBE = n` para IVF.
- `WITH EF_SEARCH = n` para HNSW.
- `SIMILARITY_SCORE()` en `SELECT` u `ORDER BY`.
- `JOIN` y `GROUP BY` sobre las k filas más parecidas.
- Filtros adicionales en el `AND`:
  - si cumplen **2 000 filas o menos**, se hace **pre-filtro exacto**: se comparan todas esas filas;
  - si cumplen más, se hace **búsqueda filtrada dentro del índice**, usando los nodos permitidos como resultados y los no permitidos como puentes en HNSW, o abriendo más celdas en IVF.

### 2.5 Resultados experimentales multimedia

**Metodología:**

- 200 consultas que no están en la base.
- k = 10.
- Recall@10 contra búsqueda lineal exacta.
- Vectores TF-IDF de portadas (SIFT) y audio (MFCC).
- IVF: `nlist = √N`, `nprobe = 10`.
- HNSW: `M = 16`, `ef_construction = 100`, `ef_search = 50`.

**Resultado principal con coseno:**

| Coseno, k = 10 | Construir | Consulta | Más rápido que lineal | Recall@10 |
|---|---:|---:|---:|---:|
| **Portadas, N = 19 800** | | | | |
| Búsqueda lineal | — | 14.4 ms | — | 1.000 |
| IVF | 11 s | 2.6 ms | 5.5× | 0.941 |
| HNSW | 174 s | **1.5 ms** | **9.5×** | **0.981** |
| **Audio, N = 10 000** | | | | |
| Búsqueda lineal | — | 7.7 ms | — | 1.000 |
| IVF | 3.9 s | 1.8 ms | 4.3× | 0.922 |
| HNSW | 83 s | **1.5 ms** | **5.2×** | **0.987** |

Conclusiones:

- Con índice, cada consulta es **4–10 veces más rápida** que la lineal.
- La ventaja crece con N: la lineal crece proporcionalmente a N; IVF y HNSW mucho menos.
- HNSW es el más rápido y preciso, pero tarda **15–20 veces más** en construirse que IVF.
- Con menos de ~2 500 objetos, la búsqueda lineal todavía gana: no hay que construir nada y tarda menos de 2 ms.
- El espacio extra es pequeño frente a los vectores: ~0.7 MB para celdas IVF y ~4 MB para el grafo HNSW con 20 000 portadas; los vectores pesan ~77 MB.

**Coseno vs euclidiana (vectores sin normalizar):**

| Portadas 19 800 | Recall coseno | Recall euclidiana |
|---|---:|---:|
| IVF | 0.941 | 0.839 |
| HNSW | 0.981 | 0.801 |

La euclidiana sobre TF-IDF sin normalizar empeora con N porque las normas varían mucho. En audio, el IVF con euclidiana puede mostrar recall alto, pero a costa de revisar casi toda la base: no es una victoria real de calidad.

**Contra Faiss y hnswlib (mismos parámetros, 1 hilo):**

| | Construir 20k portadas | Consulta | Recall@10 portadas |
|---|---:|---:|---:|
| Búsqueda lineal nuestra | — | 14.4 ms | 1.000 |
| IVF nuestro | 11 s | 2.6 ms | 0.941 |
| Faiss IVFFlat | 0.3 s | 0.19 ms | 0.910 |
| HNSW nuestro | 174 s | 1.5 ms | 0.981 |
| Faiss HNSWFlat | 6.0 s | 0.19 ms | 0.970 |
| hnswlib | 38.5 s | 0.96 ms | 0.966 |

El recall es el mismo o mejor: **el algoritmo está bien implementado**. La diferencia de tiempo es de lenguaje y de ingeniería: C++ con SIMD y estructuras compactas contra Python puro.

### 2.6 Galería de aciertos y errores

- **Data augmentation:** rotar 90°, blanco y negro y cambiar colores casi no afectan (0.99–1.00), porque SIFT trabaja en gris y es invariante a rotación.
- **Recorte fuerte o desenfoque:** bajan el acierto porque se pierden detalles finos.
- **Fotos reales de CDs/vinilos:** el 66 % encuentra su portada digital en el top 10; casi siempre sale primera cuando acierta.
- **Audio grabado con celular:** no acierta nunca, porque MFCC depende de frecuencias que el micrófono y el parlante eliminan.
- **Conclusión:** en las consultas difíciles, el límite suele ser el **descriptor y el codebook**, no el índice. Un mejor embedding (CLIP, DINOv2, VGGish, PANNs) ayudaría más que cambiar IVF por HNSW.

### 2.7 Comparación y recomendaciones: IVF vs HNSW

| | IVF | HNSW |
|---|---|---|
| Estructura | Celdas de K-Means + listas | Grafo de vecinos por niveles |
| Construcción 20k portadas | ~10 s | ~175 s |
| Consulta 20k | 2.6 ms, recall 0.94 | 1.8–1.5 ms, recall 0.98 |
| Crecimiento con N | Depende de N / nlist | Casi logarítmico |
| Agregar después | Barato, pero no reentrena celdas | Caro: cada nodo busca vecinos |
| Perilla principal | `nprobe` | `ef_search` |
| Memoria extra | ~0.7 MB | ~4 MB |
| Ideal para | Reconstruir seguido o cargar mucha data nueva | Consultas frecuentes con base estable |

**Recomendación:**

| Si… | Usar |
|---|---|
| La base es chica (< 2 500) | Búsqueda lineal (`SEQ`) |
| Se consulta mucho y la base cambia poco | HNSW + coseno (o vectores normalizados) |
| Se reconstruye seguido o se carga mucha data | IVF + coseno, con `nprobe` ≈ 20 % de `nlist` |
| Se quiere euclidiana | Normalizar antes; si no, IVF puede degenerar |
| Se necesita velocidad de producción | Faiss o hnswlib detrás de la misma interfaz |

---

## 3. Parte 5 — Aplicación de inteligencia artificial

### 3.1 Aplicación elegida: asistente RAG sobre papers

La aplicación principal es un **chatbot RAG** sobre un catálogo de papers de visión por computador, integrado en la consola web de chutaDB.

```text
PDF
 │
 ▼
PyMuPDF: extracción de texto por página
 │
 ▼
limpieza + chunking (200–500 palabras, solapamiento ~50)
 │
 ▼
chunks.jsonl ──► SPIMI + BM25
 │
 ├──► embeddings multilingual-e5-small ──► HNSW
 │
 ▼
consulta del usuario
 │
 ├──► BM25: top 20 chunks
 ├──► HNSW: top 20 chunks
 │
 ▼
Reciprocal Rank Fusion (RRF)
 │
 ▼
top 5 chunks (máximo 2 por paper)
 │
 ▼
prompt restringido ──► LLM ──► respuesta con citas y fuentes
```

Componentes:

- **Ingesta:** PDF nativo, sin OCR en la primera versión.
- **Chunking:** respeta páginas y oraciones; guarda `paper_id`, `chunk_id`, páginas y sección cuando se puede.
- **Texto:** SPIMI + BM25, reutilizados de la Parte 3.
- **Vectorial:** embeddings de `intfloat/multilingual-e5-small`, normalizados, indexados con HNSW (`M=16`, `ef_construction=200`, `ef_search=50`).
- **Fusión:** RRF con `k=60`; no necesita normalizar scores entre BM25 y coseno.
- **LLM:** proveedor configurable por variables de entorno: Ollama local, Gemini u otra API compatible con OpenAI.
- **Citas:** la respuesta devuelve chunk, paper, autores, año y páginas.

### 3.2 Tipos de datos integrados

La aplicación integra más de dos tipos:

| Tipo | Uso en la app |
|---|---|
| **Relacional** | Tablas de papers, chunks, canciones, artistas, personas, transacciones |
| **Texto** | BM25/SPIMI sobre chunks y letras |
| **Multimedia** | Búsqueda por imagen/audio con SIFT/MFCC + IVF/HNSW |
| **Espacial** | Mapa de puntos con R-Tree y consultas de distancia |

### 3.3 API REST y UI

El backend es **FastAPI**; el frontend es **Next.js/React**. No se usa GraphQL: la integración es REST/JSON.

Endpoints relevantes:

| Endpoint | Función |
|---|---|
| `POST /api/query` | Ejecutar SQL en el motor |
| `GET /api/tables` | Catálogo de tablas |
| `POST /api/text/index` | Construir índice SPIMI |
| `POST /api/text/search` | Buscar con BM25 o TF-IDF |
| `POST /api/multimedia/search` | Buscar por ID de objeto |
| `POST /api/multimedia/search-file` | Subir imagen/audio y buscar |
| `POST /api/rag/ingest` | Ingerir un PDF |
| `POST /api/rag/build` | Construir índices textual y vectorial |
| `POST /api/rag/ask` | Preguntar al corpus |
| `GET /api/rag/papers` | Listar papers y estado |
| `GET /api/rag/health` | Verificar corpus, índices y LLM |
| `GET /api/spatial/points` | Datos para el mapa |

La UI incluye pestañas de **SQL**, **Texto**, **Multimedia**, **RAG**, **Benchmarks** y el mapa espacial. Desde la interfaz se puede:

- ejecutar SQL y ver el plan `EXPLAIN`;
- arrastrar CSV/JSON/JSONL o un ZIP con imágenes/audio;
- construir el índice SPIMI y comparar BM25 vs TF-IDF;
- buscar imágenes o audio similar;
- cargar resultados de benchmark;
- preguntar a los papers y revisar las fuentes.

### 3.4 Demo sugerida de la Parte 5

1. Ingerir un PDF o usar el corpus ya procesado.
2. Pulsar **Construir índices** (SPIMI + embeddings + HNSW).
3. Hacer una pregunta en modo `hybrid`.
4. Mostrar la respuesta y las fuentes: paper, páginas y chunk.
5. Cambiar a modo `bm25` y `vector` para comparar recuperación.
6. Mostrar que la app también usa datos relacionales, texto y multimedia.

### 3.5 Aplicación adicional: reconocimiento facial

El repositorio incluye además una app de reconocimiento facial que:

- calcula embeddings con ArcFace/FaceNet fuera de la base;
- los guarda en chutaDB;
- busca con búsqueda exacta, IVF o HNSW;
- compara modelos, índices y métricas desde la interfaz.

Sirve como segunda demostración de que los índices vectoriales del motor se pueden reutilizar en una aplicación real.

---

## 4. ¿Qué haría que PostgreSQL nos gane por años luz?

Esta es la parte clave de la comparación. **No es solo Python vs C.** PostgreSQL acumula décadas de optimización en capas que un motor educativo en Python no tiene.

### 4.1 Ejecución nativa, SIMD y localidad de memoria

- PostgreSQL y sus extensiones (GIN, pgvector, PostGIS) están escritos en C/C++.
- Los bucles de comparación de vectores y de recorrido de índices usan estructuras compactas, acceso secuencial y, en varios casos, SIMD.
- Python paga por cada operación: objetos, despacho dinámico, GIL, menos localidad de caché.
- Ejemplo concreto del proyecto: nuestro HNSW iguala el recall de Faiss/hnswlib, pero es **8–30 veces más lento**. La diferencia no es el algoritmo, es el lenguaje y la implementación.

### 4.2 Optimizador basado en costos

PostgreSQL no ejecuta la consulta “tal como se escribe”. La parsea, reescribe, estima costos y elige un plan:

- **Estadísticas** (`ANALYZE`): histogramas, valores más comunes (MCV), `n_distinct`, correlación física.
- **Selectividad y cardinalidad:** estima cuántas filas devuelve cada predicado.
- **Elección de acceso:** sequential scan vs index scan vs index-only scan vs bitmap scan.
- **Join order** y elección de algoritmo: nested loop, hash join, merge join.
- **Ordenamiento:** sort en memoria, external sort, hash aggregate.
- **LIMIT:** puede detenerse temprano si el plan lo permite.
- **Índices parciales, de expresión y covering indexes:** reducen trabajo antes de ejecutar.
- **Paralelismo:** parallel seq scan, parallel join, parallel index build.
- **JIT:** compila expresiones y predicados en algunos planes.

Nuestro motor tiene `EXPLAIN` y reglas, pero no un optimizador maduro con estimaciones estadísticas y búsqueda de planes.

### 4.3 Almacenamiento, caché y E/S

- Páginas de 8 KB, `shared_buffers`, buffer manager y caché del sistema operativo.
- Prefetch, `effective_io_concurrency`, lectura secuencial y reutilización de páginas.
- **Visibility map** e **index-only scans**: evitan ir al heap cuando el índice ya tiene lo necesario.
- **HOT updates** y free space map: reducen trabajo en actualizaciones.
- **TOAST** y compresión para valores grandes.
- **WAL, checkpoints y recuperación**: durabilidad y crash recovery.
- **Autovacuum**: limpia versiones muertas y mantiene estadísticas.

Nuestro motor tiene páginas, buffer pool y WAL, pero no la madurez, el paralelismo ni la integración de PostgreSQL.

### 4.4 Índices nativos y especializados

| Necesidad | PostgreSQL | Nuestro equivalente |
|---|---|---|
| Igualdad/rango ordenado | B-tree | B+ tree propio |
| Igualdad exacta | Hash index | Hash extensible |
| Baja cardinalidad | Bitmap index scan | Bitmap propio |
| Texto | GIN + `tsvector` + diccionarios | SPIMI + postings |
| Espacial | GiST/SP-GiST + PostGIS | R-Tree propio |
| Vectores | pgvector: `ivfflat` y `hnsw` | IVF y HNSW propios |

Lo importante no es solo que PostgreSQL tenga el índice, sino que:

- el índice está integrado con el planificador;
- puede combinarse con otros índices en un mismo plan (bitmap AND/OR);
- soporta concurrencia, transacciones y recuperación;
- tiene mantenimiento (`VACUUM`, `REINDEX`, `ANALYZE`);
- tiene extensiones maduras y años de pruebas.

### 4.5 GIN vs SPIMI en texto

**Ventajas de GIN:**

- Construcción muy rápida y mantenimiento incremental.
- Diccionarios, stemming, stopwords, sinónimos y búsqueda de frases con posiciones.
- Pending list y `fastupdate` para inserciones.
- Posting trees y compresión.
- Planificador que decide bitmap scan, recheck y orden de joins.
- Transacciones, MVCC, concurrencia y recuperación.

**Dónde nuestro enfoque compite:**

- BM25 produce mejor ranking que `ts_rank_cd` para la métrica usada.
- Al estar embebido, evitamos red, parseo SQL y planificación; por eso ganamos en consultas multivocablo en este benchmark.
- El índice SPIMI es más compacto si solo se compara el índice, aunque GIN gana en funcionalidad.

**Lo que nos falta frente a GIN:**

- Posiciones para frases exactas.
- Stemming y stopwords.
- Actualización incremental real.
- Concurrencia y transacciones.
- Un planificador que estime selectividad y combine índices.

### 4.6 pgvector vs nuestro IVF/HNSW

**Ventajas de pgvector:**

- Implementación nativa en C con SIMD.
- HNSW e IVFFlat integrados con SQL, transacciones y `VACUUM`.
- Construcción paralela de índices.
- Filtros y *iterative index scans*.
- Parámetros tipo `hnsw.ef_search` e `ivfflat.probes`.
- Mantenimiento y ecosistema de PostgreSQL.

**Dónde nuestro enfoque compite:**

- El recall de nuestro HNSW es comparable al de Faiss/hnswlib.
- Entendemos y controlamos el algoritmo completo.
- La interfaz SQL es didáctica y muestra el plan vectorial.

**Lo que nos falta frente a pgvector:**

- Velocidad nativa y SIMD.
- Paralelismo.
- Borrados y compactación eficientes.
- Concurrencia, transacciones y recuperación.
- Filtrado avanzado y mantenimiento automático.

### 4.7 Concurrencia, transacciones y operación

Esta es la ventaja más “de años luz” en producción:

- **MVCC:** lectores y escritores no se bloquean entre sí.
- **Locks** finos y detección de deadlocks.
- **WAL** y recuperación ante caídas.
- **Autovacuum** y mantenimiento continuo.
- **Replicación** y alta disponibilidad.
- **Particionamiento** y planes paralelos.
- **Prepared statements**, pooling y seguridad.
- **Extensiones**: pgvector, PostGIS, pg_trgm, full-text search, etc.
- **Observabilidad**: `pg_stat_statements`, `EXPLAIN (ANALYZE, BUFFERS)`, métricas.

Nuestro motor tiene WAL, locks y transacciones básicas, pero no la madurez operativa de PostgreSQL.

### 4.8 Resumen: por qué PostgreSQL nos gana

| Estrategia de PostgreSQL | Qué aporta | Impacto frente a chutaDB |
|---|---|---|
| C/C++ + SIMD | Bucles rápidos y compactos | 8–30× en vectores, más en texto |
| Planificador por costos | Elige el mejor plan | Menos trabajo innecesario |
| Estadísticas | Estima selectividad y cardinalidad | Mejores decisiones de acceso y join |
| Buffer pool + caché SO | Evita E/S repetida | Consultas repetidas mucho más rápidas |
| Índices nativos integrados | B-tree, GIN, GiST, pgvector | Combinación y mantenimiento reales |
| Paralelismo | Usa varios núcleos | Construcción y escaneos más rápidos |
| MVCC + WAL | Concurrencia y durabilidad | Uso real multiusuario |
| Autovacuum + REINDEX | Mantenimiento continuo | Índices sanos a largo plazo |
| Extensiones | PostGIS, pgvector, pg_trgm | Capacidades listas para producción |

**Nuestra conclusión honesta:** en calidad de algoritmo y recall estamos cerca; en velocidad nativa, planificación, concurrencia y operación, PostgreSQL está en otra liga. Lo correcto en producción no es reimplementar PostgreSQL en Python, sino usar PostgreSQL/pgvector/Faiss/hnswlib detrás de una interfaz como la nuestra.

---

## 5. Tabla comparativa final de índices

| Índice | Estructura | Consulta | Construcción | Memoria | Fuerte en | Débil en |
|---|---|---|---|---|---|---|
| **Lineal / SEQ** | Lista de vectores | O(N) | Nada | Solo vectores | Exactitud, N chico | No escala |
| **SPIMI** | Diccionario + postings | O(df) por término | O(T + P log B) | Acotada por presupuesto | Texto embebido, compacto | Sin posiciones, sin updates |
| **GIN** | Entry tree + posting trees | Bitmap + heap + recheck | Rápida | Media | Texto rico, concurrencia | Ranking flojo, espacio total |
| **IVF** | Celdas K-Means + listas | O(nprobe · N/nlist) | Rápida | Baja | Actualización, reconstrucción | Aproximado, celdas viejas |
| **HNSW** | Grafo multicapa | ~O(log N) | Cara | Media-alta | Recall y latencia | Construcción, updates, borrados |
| **B+ tree** | Árbol ordenado | O(log N) | Media | Media | Rangos, orden, igualdad | No sirve para texto/vectores |
| **Hash** | Tabla hash | O(1) promedio | Rápida | Media | Igualdad exacta | Sin rangos |
| **Bitmap** | Bitmaps por valor | Rápido con AND/OR | Rápida | Baja si cardinalidad baja | Columnas categóricas | Cardinalidad alta |
| **R-Tree** | Árbol de bounding boxes | O(log N) aproximado | Media | Media | Datos espaciales | No para alta dimensión |

---

## 6. Alcance real de los experimentos

Para ser transparentes en la exposición:

- **Texto:** el corpus real disponible tiene ~10 000 documentos. El benchmark reportado usa 1 000–10 000. Existe un generador de corpus sintético de 100 000 para probar la interfaz, pero no se usa para afirmar un benchmark real de 100 000.
- **Multimedia:** el dataset principal tiene ~20 000 portadas y ~10 000 audios. Se evaluó hasta ~20k/10k; el profesor aceptó ese alcance. Los scripts permiten descargar más portadas si se quiere llegar a 50k/100k.
- **Comparación GIN:** no es una comparación pura de índices, porque GIN incluye servidor, parseo, planificación y heap. Aun así, es la comparación realista entre un índice embebido y un RDBMS.
- **Memoria:** `tracemalloc` mide asignaciones de Python; no mide el RSS completo de PostgreSQL ni memoria de C fuera del allocator de Python.
- **Una sola máquina:** los tiempos sirven para comparar órdenes de magnitud, no como benchmark absoluto.

---

## 7. Guion corto para la exposición

1. **Problema:** chutaDB debe buscar texto, imágenes y audio, y además usarlos desde una app real.
2. **Texto:** SPIMI construye el índice invertido; TF-IDF y BM25 lo reutilizan. BM25 gana en ranking; GIN construye más rápido pero ordena peor y ocupa más en total.
3. **Multimedia:** SIFT/MFCC → codebook K-Means → histograma TF-IDF → IVF/HNSW. HNSW gana en latencia y recall; IVF gana en construcción y actualización.
4. **Métricas:** coseno es la mejor opción sobre TF-IDF sin normalizar; con vectores normalizados las tres métricas dan el mismo ranking.
5. **App:** FastAPI + Next.js, RAG híbrido BM25 + HNSW + RRF + LLM, con fuentes verificables; también SQL, texto, multimedia, mapa y benchmarks.
6. **PostgreSQL:** no nos gana solo por C; nos gana por planificador, estadísticas, caché, índices nativos, paralelismo, MVCC, WAL, autovacuum y extensiones. Nuestro valor es entender y demostrar los algoritmos; en producción, usaríamos PostgreSQL/pgvector/Faiss por debajo.

---

## 8. Frases clave para cerrar

- “Nuestro HNSW iguala el recall de Faiss; la diferencia es de ingeniería, no de algoritmo.”
- “GIN encuentra rápido, pero `ts_rank_cd` no rankea como BM25.”
- “Con menos de 2 500 objetos, el mejor índice es no tener índice: búsqueda lineal.”
- “El límite de las consultas difíciles es el descriptor, no el índice.”
- “PostgreSQL no es solo un índice: es un planificador, una caché, un motor transaccional y un ecosistema.”
- “Lo valioso no es reimplementar PostgreSQL en Python, sino entender qué hace y cuándo conviene usarlo.”
