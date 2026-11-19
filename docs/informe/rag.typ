// Sección del informe: Aplicación RAG sobre el motor de base de datos multimodal.
// Uso:  #include "rag.typ"  dentro del documento principal, o compilar solo:  typst compile rag.typ
// No requiere paquetes externos (los diagramas están hechos con primitivas nativas de Typst).

#let c-app = rgb("#dbeafe")
#let c-rag = rgb("#fef3c7")
#let c-db = rgb("#dcfce7")
#let c-ai = rgb("#fce7f3")
#let c-disk = rgb("#f3f4f6")

#let node(body, fill: white, w: auto) = rect(
  width: w,
  inset: 7pt,
  radius: 4pt,
  fill: fill,
  stroke: 0.7pt + luma(90),
  align(center, text(size: 8.5pt, body)),
)

#let layer(title, body, fill: white) = rect(
  width: 100%,
  inset: 8pt,
  radius: 6pt,
  fill: fill.lighten(55%),
  stroke: 0.8pt + luma(120),
  stack(
    dir: ttb,
    spacing: 6pt,
    align(left, text(size: 8pt, weight: "bold", fill: luma(60), upper(title))),
    body,
  ),
)

#let down = align(center, text(size: 11pt)[$arrow.b$])

= Aplicación RAG: asistente de consulta sobre papers científicos

== Visión general

La aplicación RAG (_Retrieval-Augmented Generation_) responde preguntas en lenguaje natural sobre un corpus de artículos científicos en PDF, citando los fragmentos que sustentan cada respuesta. Es, ante todo, una *aplicación de inteligencia artificial construida encima del motor de base de datos*: el modelo de lenguaje no responde "de memoria", sino que recibe únicamente evidencia recuperada por los índices que implementamos en el curso (SPIMI/BM25 para texto y HNSW para vectores) y guardada con el motor de almacenamiento propio.

La idea central es separar dos responsabilidades:

- *Recuperar* (determinista, auditable, hecho con nuestro motor): decidir qué fragmentos del corpus son relevantes para la pregunta.
- *Generar* (probabilístico, hecho con IA): redactar una respuesta en lenguaje natural usando _solo_ esos fragmentos.

Esto reduce las alucinaciones, permite mostrar las fuentes (título, autores, año y páginas) y hace que la calidad de la respuesta dependa directamente de la calidad de los índices, lo que conecta la parte de IA con los temas del curso.

El corpus actual contiene 619 papers divididos en 12 823 chunks.

== Componentes del motor reutilizados

El RAG no reimplementa ninguna estructura de búsqueda: reutiliza los módulos del motor y solo agrega la lógica de orquestación y los modelos de IA.

#figure(
  table(
    columns: (auto, 1fr, 1.4fr),
    align: (left, left, left),
    inset: 6pt,
    stroke: 0.5pt + luma(150),
    table.header([*Componente del motor*], [*Módulo*], [*Rol en el RAG*]),
    [Índice invertido SPIMI], [`text_based/spimi.py`], [Indexa el texto de los chunks en disco (`spimi.bin` + `spimi.json`) con construcción por bloques y fusión; es la base de la búsqueda léxica.],
    [Ranking BM25], [`text_based/bm25.py`], [Puntúa los chunks para una consulta ($k_1 = 1.2$, $b = 0.75$) leyendo postings desde el índice SPIMI.],
    [Índice vectorial HNSW], [`indexes/hnsw.py`], [Búsqueda aproximada de vecinos más cercanos sobre los embeddings (métrica coseno, $M = 16$, `ef_construction` = 200).],
    [Motor de almacenamiento], [`storage/` + `rag/storage/corpus_store.py`], [Persiste papers y chunks en las tablas `rag_papers` y `rag_chunks` del catálogo propio, con esquema tipado y validación, y los exporta a JSONL.],
    [Backend / catálogo], [`web/backend/main.py`], [Expone el servicio como endpoints HTTP junto a los de SQL, texto y multimedia, compartiendo el mismo proceso del motor.],
  ),
  caption: [Componentes del motor de base de datos que usa la aplicación RAG.],
)

Dos modificaciones al motor surgieron de esta integración y benefician al resto del sistema:

- *SPIMI/BM25*: se agregó `iter_postings_counts()` y `get_postings_count()`, que leen solo la cabecera de cada posting para obtener la frecuencia documental. Antes, BM25 decodificaba la lista completa de postings de cada término al calcular estadísticas, lo cual era muy costoso con 12 823 documentos.
- *HNSW*: `guardar` y `cargar` ahora serializan los identificadores como `int64` o texto en lugar de objetos Python, evitando `pickle` y permitiendo usar el `chunk_id` real como identificador del vector.

Además, el corpus del RAG reutiliza el contrato de documentos del índice de texto del proyecto (`id`, texto y título), de modo que un chunk se indexa con el mismo pipeline (tokenización, normalización con `casefold`) que las letras de canciones de la otra aplicación.

== Arquitectura del servicio

El servicio está organizado en capas. La clase `RAGService` (`rag/service.py`) es el orquestador: conoce las rutas de los artefactos, construye los índices, carga el recuperador bajo demanda y ejecuta el ciclo pregunta → contexto → respuesta.

#figure(
  stack(
    dir: ttb,
    spacing: 4pt,
    layer("Cliente", fill: c-app)[
      #node(w: 100%)[Interfaz web (Next.js) que consume la API: preguntas, selección de modo (`bm25` / `vector` / `hybrid`) y visualización de fuentes]
    ],
    down,
    layer("API HTTP (FastAPI)", fill: c-app)[
      #grid(
        columns: (1fr,) * 5,
        column-gutter: 4pt,
        node[`GET /health`],
        node[`POST /ingest`],
        node[`POST /build`],
        node[`POST /ask`],
        node[`GET /papers`],
      )
    ],
    down,
    layer("Servicio RAG: RAGService (orquestación)", fill: c-rag)[
      #grid(
        columns: (1fr,) * 3,
        column-gutter: 5pt,
        node(fill: white)[*Ingesta* \ lectura PDF, limpieza, chunking con solapamiento],
        node(fill: white)[*Indexación* \ normalización, SPIMI, embeddings + HNSW, caché por huella SHA-256],
        node(fill: white)[*Consulta* \ `HybridRetriever` (BM25 + HNSW + RRF), construcción del prompt, generación],
      )
    ],
    down,
    layer("Motor de base de datos (propio)", fill: c-db)[
      #grid(
        columns: (1fr,) * 3,
        column-gutter: 5pt,
        node(fill: white)[SPIMI + BM25 \ índice invertido en disco],
        node(fill: white)[HNSW \ grafo de vecinos, coseno],
        node(fill: white)[StorageManager \ `rag_papers`, `rag_chunks`],
      )
    ],
    v(2pt),
    layer("Modelos de IA (externos al motor)", fill: c-ai)[
      #grid(
        columns: (1fr,) * 2,
        column-gutter: 5pt,
        node(fill: white)[Encoder de embeddings \ `multilingual-e5-small` (384 dim.)],
        node(fill: white)[LLM generador \ Ollama (`qwen2.5:7b-instruct`) u OpenAI-compatible],
      )
    ],
    v(2pt),
    layer("Artefactos en disco (`data/rag/`)", fill: c-disk)[
      #node(w: 100%)[`chunks.jsonl` · `chunks_index.jsonl` · `spimi.bin/json` · `embeddings.npz` · `hnsw.npz` · `chunk_mapping.json` · `vector_manifest.json`]
    ],
  ),
  caption: [Arquitectura por capas del servicio RAG. El motor propio (verde) aporta toda la recuperación; la IA (rosa) interviene solo para codificar y redactar.],
)

=== Endpoints

#figure(
  table(
    columns: (auto, auto, 1fr),
    align: (left, left, left),
    inset: 6pt,
    stroke: 0.5pt + luma(150),
    table.header([*Método*], [*Ruta*], [*Función*]),
    [GET], [`/api/rag/health`], [Estado del proveedor LLM y disponibilidad del corpus y de los índices textual y vectorial.],
    [POST], [`/api/rag/ingest`], [Recibe un PDF, extrae, limpia y divide en chunks, y los agrega a `chunks.jsonl`.],
    [POST], [`/api/rag/build`], [Construye el índice textual y/o el vectorial a partir del corpus.],
    [POST], [`/api/rag/ask`], [Recupera, arma el contexto, genera la respuesta y devuelve fuentes y latencias.],
    [GET], [`/api/rag/papers`], [Lista los papers del corpus con su metadato y número de chunks.],
  ),
  caption: [API del servicio RAG (prefijo común `/api/rag`).],
)

== Ingesta del corpus

Cada PDF pasa por un pipeline de tres etapas (`rag/ingestion/`):

+ *Lectura*: se extrae el texto nativo página por página y se valida que el archivo sea un PDF legible con texto suficiente; los rechazados se registran en lugar de ingresar al corpus.
+ *Limpieza*: se detectan y eliminan encabezados y pies de página repetidos, se reconstruyen palabras cortadas por guiones de fin de línea y se normalizan los espacios. Se conserva además el texto original para mostrar citas exactas.
+ *Chunking*: el texto se divide en ventanas de aproximadamente 350 palabras (mínimo 200, máximo 500) con 50 palabras de solapamiento, de modo que una idea cortada en el límite aparezca completa en al menos un chunk. Cada chunk recuerda su `page_start` y `page_end`, lo que permite citar páginas.

Los chunks resultantes se guardan como JSONL y pueden persistirse también en las tablas `rag_papers` y `rag_chunks` del motor de almacenamiento, que valida tipos y longitudes máximas antes de insertar.

== Indexación

La construcción (`RAGService.build`) normaliza primero los registros: asigna identificadores enteros únicos y estables a papers y chunks, y propaga a todos los chunks de un paper su _abstract_, extraído con una expresión regular del inicio del documento. Luego construye dos índices independientes sobre el mismo identificador `chunk_id`, que es la llave que permite fusionar sus resultados.

=== Índice textual (SPIMI + BM25)

Los chunks se envían como flujo de documentos al constructor SPIMI, que particiona el vocabulario en bloques que caben en memoria, los vuelca a disco y los fusiona en un único archivo de postings con un diccionario de desplazamientos. En consulta, BM25 recorre solo los postings de los términos de la pregunta, por lo que el índice no necesita cargarse completo en memoria.

=== Índice vectorial (embeddings + HNSW)

+ Cada chunk se codifica con `intfloat/multilingual-e5-small` (384 dimensiones), un modelo multilingüe: el corpus mezcla español e inglés y las preguntas pueden estar en cualquiera de los dos.
+ El modelo exige prefijos distintos: los chunks se codifican con `passage: ` y las preguntas con `query: `. Los vectores se promedian sobre los tokens (_mean pooling_) y se normalizan con norma L2, así el coseno equivale al producto punto.
+ Los vectores se insertan en un HNSW propio con el `chunk_id` como identificador. Se guardan tres artefactos: `embeddings.npz`, `hnsw.npz` y `chunk_mapping.json`.

Codificar es la etapa más cara del sistema (corre un transformer sobre cada chunk), por lo que se implementó una *caché por huella*: se calcula un SHA-256 sobre el nombre del modelo, la longitud máxima de tokens y el par (`chunk_id`, contenido) de todos los chunks. Si el manifiesto `vector_manifest.json` coincide y los archivos existen, se reutiliza el índice sin recalcular nada. Cualquier cambio en el corpus o en el modelo invalida la caché automáticamente.

== Recuperación híbrida

Los dos índices son complementarios. BM25 acierta con términos exactos (nombres de métodos, siglas, números) pero no entiende sinónimos; HNSW captura similitud semántica y funciona entre idiomas, pero puede perder coincidencias literales. `HybridRetriever` (`rag/retrieval.py`) combina ambos con *Reciprocal Rank Fusion* (RRF):

$ "RRF"(d) = sum_(r in R) 1 / (k + "rank"_r (d)), quad k = 60 $

donde $R$ es el conjunto de rankings (BM25 y HNSW) y $"rank"_r (d)$ es la posición del chunk $d$ en el ranking $r$. RRF usa solo posiciones, no puntajes, así que no hace falta normalizar la escala de BM25 frente a la similitud coseno. Un chunk que aparece alto en ambos rankings queda primero.

El algoritmo de consulta es:

+ Validar la consulta y aplicar los filtros opcionales (año, autor, idioma) para obtener el conjunto de chunks elegibles.
+ Pedir `candidate_k = max(20, top_k)` candidatos a cada índice activo: BM25 sobre el índice SPIMI y HNSW con `ef_search` = 50 sobre el vector de la pregunta.
+ Descartar los candidatos no elegibles y fusionar los rankings con RRF.
+ Aplicar una *restricción de diversidad*: se admiten como máximo 2 chunks por paper, para que un solo artículo no monopolice el contexto.
+ Devolver los `top_k` mejores con su paper, páginas y puntaje.

El servicio admite tres modos: `bm25` (solo léxico), `vector` (solo semántico) e `hybrid` (ambos con RRF). Esto permite comparar los índices con la misma interfaz y demostrar el aporte de cada uno. El recuperador se carga de forma perezosa y se mantiene en memoria entre consultas; si falta algún artefacto, el servicio responde con un error explícito que indica qué construir.

== Generación con el modelo de lenguaje

Los chunks recuperados se ensamblan en un contexto etiquetado (`[Chunk id]`, título, autores, año, páginas y texto). `build_rag_prompt` lo inserta en un prompt con reglas explícitas:

- Responder usando únicamente el contexto proporcionado.
- Indicar de forma explícita cuando la evidencia sea insuficiente.
- No inventar autores, años ni referencias.
- Responder en el idioma de la pregunta y citar con el formato (Autor, Año).

La generación se abstrae detrás de un protocolo `LLMClient` con un único método `generate(prompt)`, lo que desacopla el servicio del proveedor. Hay dos implementaciones: `OllamaClient` (modelo local, por defecto `qwen2.5:7b-instruct`, sin enviar datos a terceros) y `OpenAICompatibleClient` (cubre servicios con la API de _chat completions_ como OpenAI, Groq u OpenRouter). El proveedor, modelo, temperatura (0.1 por defecto, para respuestas estables y apegadas al contexto), máximo de tokens y _timeout_ se configuran por variables de entorno `RAG_LLM_*`, sin tocar código.

== Flujo de una consulta

#figure(
  stack(
    dir: ttb,
    spacing: 4pt,
    node(w: 100%, fill: c-app)[*Pregunta* del usuario (`POST /api/rag/ask`: pregunta, modo, `top_k`)],
    down,
    grid(
      columns: (1fr, 1fr),
      column-gutter: 8pt,
      stack(
        dir: ttb,
        spacing: 4pt,
        node(w: 100%, fill: c-ai)[Encoder `query: ...` \ vector de 384 dim.],
        down,
        node(w: 100%, fill: c-db)[*HNSW* \ `candidate_k` vecinos más cercanos],
      ),
      stack(
        dir: ttb,
        spacing: 4pt,
        node(w: 100%, fill: c-db)[Tokenización y consulta de postings],
        down,
        node(w: 100%, fill: c-db)[*SPIMI + BM25* \ `candidate_k` mejores por puntaje],
      ),
    ),
    down,
    node(w: 100%, fill: c-rag)[*Fusión RRF* · filtros (año, autor, idioma) · máx. 2 chunks por paper · `top_k`],
    down,
    node(w: 100%, fill: c-rag)[Contexto con metadatos de cita $arrow$ prompt con reglas anti-alucinación],
    down,
    node(w: 100%, fill: c-ai)[*LLM* (Ollama / OpenAI-compatible) genera la respuesta],
    down,
    node(w: 100%, fill: c-app)[Respuesta + fuentes (título, autores, año, páginas, puntaje) + latencias por etapa],
  ),
  caption: [Flujo de una consulta en modo `hybrid`. En los modos `bm25` y `vector` solo se ejecuta la rama correspondiente.],
)

La respuesta incluye las fuentes con su puntaje y un desglose de latencias (recuperación, generación y total), lo que permite medir en qué etapa se invierte el tiempo. Como la recuperación se apoya en índices en disco y en memoria, se espera que la generación del LLM sea la etapa dominante de la latencia total.

== Decisiones de diseño y limitaciones

- *Híbrido sobre un solo índice*: se mantienen ambos índices y se fusionan por posición con RRF, evitando calibrar pesos entre puntajes de escalas distintas.
- *Identificador común*: usar el mismo `chunk_id` en SPIMI, HNSW y el almacenamiento hace trivial la fusión y la trazabilidad de cada fuente hasta su página en el PDF.
- *Caché por huella*: evita recalcular embeddings, que es lo más costoso, y se invalida solo cuando cambian los datos o el modelo.
- *Modelo intercambiable*: el LLM es configurable; el motor de recuperación no depende de él.
- *Limitaciones*: la extracción solo cubre PDFs con texto nativo (no hay OCR); HNSW es un índice aproximado, por lo que puede omitir algún vecino exacto; y la calidad final depende de que el LLM respete las instrucciones del prompt, por lo que se muestran siempre las fuentes para que el usuario pueda verificar.
