# Plan de implementación: RAG para documentos académicos

Este plan adapta los requisitos de [`data.md`](./data.md) a la arquitectura real de
chutaDB: motor de base de datos propio en Python, índices SPIMI/BM25/TF-IDF/HNSW,
backend FastAPI y frontend Next.js.

## 1. Objetivo y alcance

Construir un chatbot RAG sobre un catálogo amplio de papers de computer vision,
combinando:

- búsqueda textual con SPIMI, BM25 y TF-IDF;
- búsqueda semántica con embeddings e índice HNSW;
- fusión híbrida mediante Reciprocal Rank Fusion (RRF);
- generación de respuestas con un LLM;
- citas verificables por paper, autor, año y páginas;
- API FastAPI e interfaz dentro de la aplicación web existente.

La ingesta debe poder recorrer todo el catálogo disponible, pero la primera
validación funcional y la evaluación controlada se realizarán sobre 20–50
papers. Esto permite probar el pipeline sin esperar a descargar y procesar
todo el universo antes de detectar errores.

El archivo `pwc.csv` de Papers With Code funciona como catálogo de metadata,
no como colección local de PDFs. En el snapshot consultado contiene
aproximadamente 1,847 líneas incluyendo el encabezado; solo una parte de los
registros apunta directamente a un archivo PDF y otros apuntan a páginas HTML.

## 2. Restricciones del proyecto

1. **No usar PostgreSQL como base principal del RAG.** El repositorio implementa
   su propio almacenamiento, parser SQL e índices.
2. **Reutilizar los componentes existentes:**
   - [`text_based/`](../../text_based/) para SPIMI, BM25 y TF-IDF.
   - [`indexes/hnsw.py`](../../indexes/hnsw.py) para la búsqueda vectorial.
   - [`web/backend/main.py`](../../web/backend/main.py) para la API.
   - [`web/app/page.tsx`](../../web/app/page.tsx) para la interfaz.
3. La demo actual de SQL, texto y multimedia debe continuar funcionando.
4. El catálogo puede contener todos los registros disponibles, pero el corpus
   de validación inicial debe limitarse a 20–50 papers.
5. Los embeddings y el índice HNSW se persistirán inicialmente en `.npz`, con un
   mapping separado `posición del vector → chunk_id`.
6. Se deben seleccionar PDFs con texto nativo. OCR queda fuera de la primera
   versión.
7. La descarga masiva debe ser gradual, con reintentos, límites de velocidad,
   registro de errores y respeto de las condiciones de cada fuente.

## 3. Arquitectura propuesta

```text
PDFs
  ↓
Extracción con PyMuPDF
  ↓
Limpieza y metadata validada
  ↓
Chunks de 200–500 palabras
  ↓
┌──────────────────────┬────────────────────────┐
│ Motor propio         │ Archivos de índices     │
│ papers/chunks        │ SPIMI + HNSW + mappings │
└──────────────────────┴────────────────────────┘
  ↓
Consulta del usuario
  ↓
BM25 + HNSW
  ↓
RRF
  ↓
Top 3–5 chunks
  ↓
Prompt restringido
  ↓
LLM
  ↓
Respuesta con citas y fuentes
```

La arquitectura debe aislar el motor mediante una capa de adaptación. El resto
del RAG no debe depender directamente de la implementación interna de
`StorageManager`, `ExecuteVisitor` o del parser.

## 4. Estructura de archivos propuesta

```text
rag/
├── config.py
├── models.py
├── ingestion/
│   ├── pdf_reader.py
│   ├── cleaner.py
│   ├── metadata.py
│   └── chunker.py
├── preprocessing/
│   └── text_pipeline.py
├── storage/
│   ├── corpus_store.py
│   └── chunk_mapping.py
├── embeddings/
│   └── encoder.py
├── indexing/
│   ├── build_text_index.py
│   └── build_vector_index.py
├── retrieval/
│   ├── text_retrieval.py
│   ├── vector_retrieval.py
│   └── hybrid.py
├── generation/
│   ├── prompt.py
│   └── llm_client.py
└── evaluation/
    ├── metrics.py
    └── runner.py

scripts/
├── ingest_papers.py
├── import_pwc_catalog.py
├── download_papers.py
├── build_rag_indexes.py
└── evaluate_rag.py

data/papers/
data/rag/
├── pwc_catalog.csv
├── metadata.csv
├── download_manifest.jsonl
├── failed_downloads.jsonl
├── chunks.jsonl
├── chunk_mapping.json
├── embeddings.npz
└── hnsw.npz
```

La lógica de RAG debe permanecer separada de `web/backend/`. El backend solo
debe validar solicitudes, llamar al servicio RAG y serializar respuestas.

## 5. Fase previa: importar el catálogo y adquirir papers

Esta fase debe ejecutarse antes de la preparación manual del corpus.

### 5.1 [LISTO] Importar `pwc.csv`

Crear `scripts/import_pwc_catalog.py` para:

1. descargar el CSV desde:
   `https://raw.githubusercontent.com/zziz/pwc/master/src/pwc.csv`;
2. leerlo con `csv.DictReader`, no separarlo manualmente por comas;
3. conservar:
   - `paper_title`;
   - `paper_link`;
   - `conference`;
   - `year`;
   - `repo_link`;
   - `repo_star`;
   - `last_update`;
4. normalizar espacios, URLs y años;
5. deduplicar por URL y, como respaldo, por título normalizado;
6. guardar `data/rag/pwc_catalog.csv`.

El catálogo debe conservarse aunque el PDF no pueda descargarse. Un registro de
catálogo no implica que exista un documento local utilizable por el RAG.

### 5.2 [LISTO] Descargar los documentos

Crear `scripts/download_papers.py` con estas reglas:

- procesar solo URLs `http` o `https`;
- identificar si el recurso es PDF por `Content-Type`, extensión o firma
  `%PDF`;
- guardar cada archivo con un nombre seguro basado en un hash de la URL;
- no usar `repo_link` para descargar el paper;
- aplicar timeout, reintentos limitados y espera entre solicitudes;
- no descargar el mismo recurso si ya tiene checksum registrado;
- limitar el número de descargas por ejecución mediante `--limit`;
- permitir reanudar con `--resume`;
- guardar errores sin detener todo el lote.

Ejemplo de uso:

```bash
python3 scripts/import_pwc_catalog.py
python3 scripts/download_papers.py --limit 50
python3 scripts/download_papers.py --resume
```

El manifiesto `data/rag/download_manifest.jsonl` debe registrar al menos:

```json
{
  "catalog_id": "sha256:url",
  "paper_title": "...",
  "source_url": "...",
  "local_path": "data/papers/...",
  "status": "downloaded",
  "http_status": 200,
  "content_type": "application/pdf",
  "sha256": "...",
  "bytes": 123456,
  "attempts": 1
}
```

Estados mínimos:

```text
cataloged
downloaded
not_pdf
http_error
invalid_pdf
extracted
failed
```

### 5.3 R[LISTO] esolver enlaces HTML

Los enlaces a páginas de CVPR, ICCV, NeurIPS u otros sitios no deben
interpretarse automáticamente como PDF sin validar el contenido. Para la
primera versión:

1. intentar descargar el enlace;
2. comprobar si la respuesta es un PDF;
3. si es HTML, registrar `not_pdf`;
4. dejar la resolución específica de cada portal como una mejora posterior.

No se debe hacer scraping agresivo ni asumir que todos los papers tienen la
misma licencia o estructura de URL.

### 5.4 Capas de disponibilidad

Separar explícitamente estos conteos:

| Capa | Significado |
|---|---|
| Catálogo | Registro existente en `pwc_catalog.csv` |
| Descargable | Se obtuvo una respuesta HTTP válida |
| PDF válido | El archivo contiene un PDF legible |
| Texto extraído | PyMuPDF extrajo texto suficiente |
| Indexable | Tiene chunks válidos y metadata utilizable |
| Evaluación | Fue seleccionado para preguntas de ground truth |

### 5.5 Criterios de aceptación

- El catálogo completo se importa sin perder filas con comas o comillas.
- El downloader puede detenerse y continuar sin duplicados.
- Cada intento queda registrado en el manifiesto.
- Un enlace HTML no se guarda incorrectamente como PDF.
- Los errores de una fuente no detienen el lote completo.
- Se puede obtener un subconjunto reproducible con `--limit` para la fase uno.
├── storage/
│   ├── corpus_store.py
│   └── chunk_mapping.py
├── embeddings/
│   └── encoder.py
├── indexing/
│   ├── build_text_index.py
│   └── build_vector_index.py
├── retrieval/
│   ├── text_retrieval.py
│   ├── vector_retrieval.py
│   └── hybrid.py
├── generation/
│   ├── prompt.py
│   └── llm_client.py
└── evaluation/
    ├── metrics.py
    └── runner.py

scripts/
├── ingest_papers.py
├── import_pwc_catalog.py
├── download_papers.py
├── build_rag_indexes.py
└── evaluate_rag.py

data/papers/
data/rag/
├── pwc_catalog.csv
├── metadata.csv
├── download_manifest.jsonl
├── failed_downloads.jsonl
├── chunks.jsonl
├── chunk_mapping.json
├── embeddings.npz
└── hnsw.npz
```

La lógica de RAG debe permanecer separada de `web/backend/`. El backend solo
debe validar solicitudes, llamar al servicio RAG y serializar respuestas.

## 5. Fase previa: importar el catálogo y adquirir papers

Esta fase debe ejecutarse antes de la preparación manual del corpus.

### 5.1 Importar `pwc.csv`

Crear `scripts/import_pwc_catalog.py` para:

1. descargar el CSV desde:
   `https://raw.githubusercontent.com/zziz/pwc/master/src/pwc.csv`;
2. leerlo con `csv.DictReader`, no separarlo manualmente por comas;
3. conservar:
   - `paper_title`;
   - `paper_link`;
   - `conference`;
   - `year`;
   - `repo_link`;
   - `repo_star`;
   - `last_update`;
4. normalizar espacios, URLs y años;
5. deduplicar por URL y, como respaldo, por título normalizado;
6. guardar `data/rag/pwc_catalog.csv`.

El catálogo debe conservarse aunque el PDF no pueda descargarse. Un registro de
catálogo no implica que exista un documento local utilizable por el RAG.

### 5.2 Descargar los documentos

Crear `scripts/download_papers.py` con estas reglas:

- procesar solo URLs `http` o `https`;
- identificar si el recurso es PDF por `Content-Type`, extensión o firma
  `%PDF`;
- guardar cada archivo con un nombre seguro basado en un hash de la URL;
- no usar `repo_link` para descargar el paper;
- aplicar timeout, reintentos limitados y espera entre solicitudes;
- no descargar el mismo recurso si ya tiene checksum registrado;
- limitar el número de descargas por ejecución mediante `--limit`;
- permitir reanudar con `--resume`;
- guardar errores sin detener todo el lote.

Ejemplo de uso:

```bash
python3 scripts/import_pwc_catalog.py
python3 scripts/download_papers.py --limit 50
python3 scripts/download_papers.py --resume
```

El manifiesto `data/rag/download_manifest.jsonl` debe registrar al menos:

```json
{
  "catalog_id": "sha256:url",
  "paper_title": "...",
  "source_url": "...",
  "local_path": "data/papers/...",
  "status": "downloaded",
  "http_status": 200,
  "content_type": "application/pdf",
  "sha256": "...",
  "bytes": 123456,
  "attempts": 1
}
```

Estados mínimos:

```text
cataloged
downloaded
not_pdf
http_error
invalid_pdf
extracted
failed
```

### 5.3 Resolver enlaces HTML

Los enlaces a páginas de CVPR, ICCV, NeurIPS u otros sitios no deben
interpretarse automáticamente como PDF sin validar el contenido. Para la
primera versión:

1. intentar descargar el enlace;
2. comprobar si la respuesta es un PDF;
3. si es HTML, registrar `not_pdf`;
4. dejar la resolución específica de cada portal como una mejora posterior.

No se debe hacer scraping agresivo ni asumir que todos los papers tienen la
misma licencia o estructura de URL.

### 5.4 Capas de disponibilidad

Separar explícitamente estos conteos:

| Capa | Significado |
|---|---|
| Catálogo | Registro existente en `pwc_catalog.csv` |
| Descargable | Se obtuvo una respuesta HTTP válida |
| PDF válido | El archivo contiene un PDF legible |
| Texto extraído | PyMuPDF extrajo texto suficiente |
| Indexable | Tiene chunks válidos y metadata utilizable |
| Evaluación | Fue seleccionado para preguntas de ground truth |

### 5.5 Criterios de aceptación

- El catálogo completo se importa sin perder filas con comas o comillas.
- El downloader puede detenerse y continuar sin duplicados.
- Cada intento queda registrado en el manifiesto.
- Un enlace HTML no se guarda incorrectamente como PDF.
- Los errores de una fuente no detienen el lote completo.
- Se puede obtener un subconjunto reproducible con `--limit` para la fase uno.

## 6. Fases de implementación

### Fase 0: Validar el motor existente

#### Tareas

1. Confirmar cómo crear tablas usando el dialecto propio:

   ```sql
   CREATE TABLE rag_papers (...) USING HEAP;
   CREATE TABLE rag_chunks (...) USING HEAP;
   ```

2. Confirmar los tipos admitidos para texto largo, enteros, identificadores y
   campos opcionales.
3. Confirmar la API Python para insertar registros y recuperar chunks por ID.
4. Implementar un adaptador en `rag/storage/corpus_store.py`.

#### Interfaz mínima

```python
class CorpusStore:
    def insert_paper(self, paper: Paper) -> int: ...
    def insert_chunks(self, chunks: list[Chunk]) -> None: ...
    def get_chunks(self, chunk_ids: list[int]) -> list[Chunk]: ...
```

#### Criterio de aceptación

Se puede insertar un paper, insertar sus chunks y recuperarlos por ID sin
modificar el motor base.

### Fase 1: Preparar el corpus de validación

#### Tareas

1. Seleccionar 20–50 papers del catálogo que tengan PDF válido y texto nativo.
2. Priorizar papers de visión por computador relacionados con:
   - detección;
   - segmentación;
   - embeddings;
   - búsqueda visual;
   - aprendizaje profundo.
3. Crear `data/rag/metadata.csv` a partir del manifiesto:

   ```csv
   source_file,title,authors,year,language
   paper_01.pdf,...,...,2023,en
   ```

4. Validar manualmente título, autores, año, idioma y cantidad de páginas.

#### Criterio de aceptación

Todos los PDFs tienen metadata válida y ningún archivo se procesa dos veces.

### Fase 2: Extraer, limpiar y dividir documentos

#### Extracción

Implementar `rag/ingestion/pdf_reader.py` con PyMuPDF:

- extraer texto por página;
- usar `sort=True`;
- conservar `page_start` y `page_end`;
- rechazar o registrar PDFs sin texto suficiente.

#### Limpieza

- unir palabras cortadas por guion;
- eliminar encabezados y pies repetidos;
- eliminar la sección `References`;
- normalizar espacios;
- conservar el texto original para las citas.

#### Chunking

- objetivo: 300 palabras;
- mínimo aproximado: 200;
- máximo: 500;
- solapamiento: 50 palabras;
- respetar oraciones y párrafos;
- detectar y guardar la sección cuando sea posible.

Modelo de chunk:

```python
{
    "chunk_id": 1,
    "paper_id": 4,
    "chunk_idx": 0,
    "page_start": 2,
    "page_end": 3,
    "section": "Introduction",
    "content": "...",
    "n_words": 302
}
```

#### Criterios de aceptación

- Ningún chunk supera 500 palabras salvo casos excepcionales documentados.
- Cada chunk conserva su paper y sus páginas.
- El pipeline se puede ejecutar nuevamente sin duplicar registros.
- Se revisa manualmente una muestra de al menos 10 papers.

### Fase 3: Persistir papers y chunks

Adaptar las tablas del documento base al dialecto de chutaDB:

```sql
CREATE TABLE rag_papers (
    paper_id INT PRIMARY KEY,
    title VARCHAR(500),
    authors VARCHAR(1000),
    year INT,
    language VARCHAR(10),
    source_file VARCHAR(255),
    n_pages INT
) USING HEAP;
```

```sql
CREATE TABLE rag_chunks (
    chunk_id INT PRIMARY KEY,
    paper_id INT,
    chunk_idx INT,
    page_start INT,
    page_end INT,
    section VARCHAR(255),
    content VARCHAR(3000),
    n_words INT
) USING HEAP;
```

No almacenar inicialmente `embedding FLOAT8[]`. El motor actual no demuestra
soporte equivalente para vectores en columnas, por lo que los vectores deben
mantenerse en `.npz`. Además, el motor limita el tamaño de un registro a
aproximadamente 4067 bytes; por eso los chunks grandes deben conservarse
también en `chunks.jsonl` y la tabla debe guardar únicamente contenido que
quepa en el límite seguro de 3000 bytes.

Para el índice textual, generar un JSONL compatible con el código existente:

```json
{"id": 120, "lyrics": "contenido del chunk", "title": "paper title"}
```

El campo `lyrics` se conserva por compatibilidad con `SPIMIIndex`, aunque
represente el contenido de un chunk.

### Fase 4: Preprocesamiento y embeddings

#### Texto

En la primera versión se debe reutilizar el tokenizador actual de SPIMI. No
introducir NLTK o spaCy salvo que la evaluación demuestre que son necesarios.

Como mejora posterior:

- stopwords por idioma;
- stemming Snowball;
- normalización de acentos solo para búsqueda.

#### Embeddings

Usar inicialmente:

```text
intfloat/multilingual-e5-small
```

Este modelo reduce el consumo respecto a `multilingual-e5-base`, funciona con
preguntas en español sobre papers en inglés y puede ejecutarse en CPU.

Implementar `rag/embeddings/encoder.py`:

- cargar el modelo una sola vez;
- procesar lotes de 16 o 32 textos;
- usar el prefijo `passage: ` para chunks;
- usar el prefijo `query: ` para preguntas;
- normalizar vectores con L2;
- guardar `embeddings.npz`.

Construir HNSW con:

```python
HNSW(
    M=16,
    ef_construction=200,
    metrica="coseno"
)
```

Usar `ef_search=50` inicialmente y guardar:

```text
data/rag/hnsw.npz
data/rag/chunk_mapping.json
```

Ejemplo de mapping:

```json
{
  "0": 1,
  "1": 2,
  "2": 3
}
```

### Fase 5: Implementar recuperación híbrida

Exponer tres modos:

1. `bm25`;
2. `vector`;
3. `hybrid`.

Para `hybrid`:

1. Recuperar 10–20 chunks con BM25.
2. Recuperar 10–20 chunks con HNSW.
3. Fusionar rankings mediante RRF.
4. Limitar a 5 chunks finales.
5. Limitar a 2 chunks por paper para evitar redundancia.

Resultado interno esperado:

```python
{
    "chunk_id": 120,
    "score": 0.031,
    "rank": 1,
    "paper": {...},
    "content": "...",
    "pages": "4-5"
}
```

Agregar filtros opcionales por año, autor e idioma después de validar la
búsqueda básica. El reranking con cross-encoder queda como experimento
posterior, no como requisito de la primera versión.

### Fase 6: Integrar el LLM

Crear una abstracción de proveedor:

```python
class LLMClient(Protocol):
    def generate(self, prompt: str) -> str: ...
```

Implementar inicialmente uno de estos proveedores:

- OpenAI;
- Anthropic;
- Ollama local.

La API key debe configurarse mediante variables de entorno y nunca incluirse en
el código o repositorio.

Prompt obligatorio:

```text
Responde usando únicamente el contexto proporcionado.
Si el contexto no contiene evidencia suficiente, indícalo explícitamente.
No inventes autores, años ni referencias.
Responde en el idioma de la pregunta.
Cita las afirmaciones con el formato (Autor, Año).
```

Configuración inicial:

```text
temperature = 0.0–0.2
top_k = 5
```

La respuesta debe incluir:

```json
{
  "answer": "...",
  "sources": [
    {
      "chunk_id": 120,
      "title": "...",
      "authors": "...",
      "year": 2023,
      "pages": "4-5"
    }
  ]
}
```

### Fase 7: Integrar FastAPI

Agregar endpoints a `web/backend/main.py` sin modificar el comportamiento de
los endpoints actuales.

#### `POST /api/rag/ingest`

Procesa un PDF nuevo:

```json
{
  "filename": "paper.pdf"
}
```

Debe validar extensión, guardar el archivo, extraer texto, generar chunks e
informar errores explícitamente.

#### `POST /api/rag/build`

Reconstruye índices:

```json
{
  "text": true,
  "vector": true
}
```

#### `POST /api/rag/ask`

```json
{
  "question": "¿Qué es la visión computacional?",
  "mode": "hybrid",
  "top_k": 5
}
```

Respuesta:

```json
{
  "answer": "...",
  "sources": [],
  "latency_ms": {
    "embedding": 120,
    "text_retrieval": 8,
    "vector_retrieval": 4,
    "generation": 1300,
    "total": 1432
  }
}
```

#### `GET /api/rag/papers`

Lista los papers disponibles y su estado de procesamiento.

#### `GET /api/rag/health`

Verifica:

- corpus disponible;
- índice textual;
- índice HNSW;
- proveedor LLM configurado.

### Fase 8: Integrar Next.js

Agregar un panel RAG en `web/app/page.tsx`, sin modificar los paneles actuales
de SQL, texto y multimedia.

La interfaz debe incluir:

- caja de pregunta;
- selector BM25/vectorial/híbrido;
- botón “Preguntar”;
- respuesta generada;
- tiempo total;
- panel desplegable de fuentes;
- contenido de cada chunk;
- paper, autor, año y páginas.

#### Criterio de aceptación

Desde `http://localhost:3000`, el usuario puede hacer una pregunta y comprobar
visualmente qué chunks originaron la respuesta.

## 7. Evaluación

Crear entre 20 y 30 preguntas en:

```text
data/rag/eval/questions.jsonl
```

Formato:

```json
{
  "question": "¿Qué diferencia existe entre BM25 y TF-IDF?",
  "relevant_chunks": [12, 45],
  "relevant_papers": [3]
}
```

Comparar:

- BM25;
- TF-IDF;
- HNSW;
- búsqueda híbrida.

Medir:

- Recall@5;
- Precision@5;
- MRR;
- nDCG@5;
- latencia p50 y p95;
- tiempo de construcción;
- tamaño de los índices;
- recall de HNSW frente a búsqueda lineal.

Para la generación, evaluar manualmente:

- fidelidad al contexto;
- relevancia;
- citas correctas;
- respuesta cuando no existe evidencia;
- idioma de la respuesta.

Guardar los resultados en `benchmark/rag/` y documentarlos en el informe.

## 8. Dependencias

Agregar únicamente las dependencias necesarias al archivo de dependencias del
backend:

```text
PyMuPDF
sentence-transformers
torch
python-dotenv
openai
anthropic
```

No instalar todos los proveedores si solo se usará uno. Tampoco agregar
PostgreSQL como requisito del RAG.

Variables recomendadas para `.env.example`:

```bash
RAG_EMBEDDING_MODEL=intfloat/multilingual-e5-small
RAG_LLM_PROVIDER=ollama
RAG_LLM_MODEL=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
RAG_TOP_K=5
RAG_CHUNK_TARGET_WORDS=300
RAG_CHUNK_OVERLAP_WORDS=50
RAG_HNSW_M=16
RAG_HNSW_EF_CONSTRUCTION=200
RAG_HNSW_EF_SEARCH=50
```

## 9. Orden recomendado de ejecución

1. Importar y versionar el catálogo `pwc.csv`.
2. Descargar un lote pequeño con `--limit`.
3. Validar persistencia de `rag_papers` y `rag_chunks`.
4. Procesar un solo PDF.
5. Probar recuperación BM25 sobre chunks.
6. Generar embeddings de esos chunks.
7. Probar HNSW y el mapping de IDs.
8. Implementar RRF.
9. Integrar un proveedor LLM.
10. Exponer `/api/rag/ask`.
11. Agregar el panel frontend.
12. Escalar el downloader al catálogo completo.
13. Seleccionar 20–50 papers para evaluación.
14. Ejecutar la evaluación.
15. Documentar resultados, cobertura y limitaciones.

No conviene comenzar por la interfaz ni por el LLM: la calidad de la demo
depende primero de que los chunks, IDs, páginas e índices sean trazables.

## 10. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| El motor no soporta textos largos | Guardar el contenido completo en JSONL y usar la tabla para metadata |
| El parser no soporta arrays o vectores | Usar campos escalares y archivos `.npz` |
| PDFs con columnas mal extraídas | Revisar una muestra y usar `sort=True` |
| Preguntas en español sobre papers en inglés | Usar embeddings multilingües |
| HNSW devuelve IDs internos | Mantener `chunk_mapping.json` |
| El LLM inventa información | Prompt restrictivo, temperatura baja y fuentes visibles |
| Dependencias pesadas | Usar `multilingual-e5-small` y CPU |
| API del LLM costosa | Ollama local o cache de preguntas |
| Reprocesamiento duplicado | Usar `source_file` como identificador único |
| Se rompe la demo actual | Mantener RAG en módulos, endpoints y componentes separados |
| El catálogo contiene páginas HTML en vez de PDFs | Registrar `not_pdf` y no enviarlas al extractor |
| Una fuente bloquea o limita descargas | Reintentos acotados, espera entre solicitudes y reanudación |
| El volumen hace lento el procesamiento | Separar catálogo, descarga, extracción e indexación por lotes |
| Se descargan documentos sin permiso claro | Revisar términos de la fuente y conservar solo enlaces/metadatos cuando corresponda |

## 11. Entregables finales

- Pipeline reproducible de ingestión.
- Catálogo completo importado desde `pwc.csv`.
- Manifiesto reproducible de descargas y errores.
- Corpus descargado de todos los PDFs accesibles y permitidos.
- Corpus de validación de 20–50 papers.
- Metadata validada.
- Tablas `rag_papers` y `rag_chunks`.
- Índice SPIMI/BM25/TF-IDF sobre chunks.
- Índice HNSW persistido.
- Recuperación híbrida con RRF.
- Cliente LLM configurable.
- Endpoint `/api/rag/ask`.
- Panel RAG en Next.js.
- Dataset de evaluación.
- Comparación BM25 vs HNSW vs híbrido.
- README actualizado con instrucciones locales y Docker.
- Informe con arquitectura, métricas, riesgos y limitaciones.

## 12. Criterio de finalización

La implementación se considera terminada cuando:

1. el catálogo puede importarse desde cero con un comando reproducible;
2. las descargas pueden pausarse y reanudarse sin duplicados;
3. cada paper tiene un estado de adquisición explícito;
4. el corpus disponible puede procesarse desde cero;
5. los índices pueden reconstruirse sin intervención manual;
6. una pregunta devuelve una respuesta con fuentes;
7. cada fuente permite localizar paper, páginas y chunk;
8. BM25, vectorial e híbrido pueden compararse desde la interfaz;
9. las pruebas existentes del proyecto continúan pasando;
10. existe evidencia cuantitativa de calidad, latencia y cobertura;
11. no hay claves secretas ni datos temporales obligatorios dentro del repositorio.
