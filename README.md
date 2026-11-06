Como un spotify, un shazam y google lens pero todo a la vez shalom

chutaDB es un DBMS educativo en Python. Las Partes 1 y 2 (parser SQL, archivos, B+, hash, bitmap,
R-Tree, transacciones) están explicadas en [`README-chuta.md`](README-chuta.md). Este README cubre
el dataset, la instalación y la Parte 4 (búsqueda de imágenes y audio por similitud).

## Instalación

```bash
conda create -n vision python=3.12 && conda activate vision
pip install -r requirements.txt                  # motor, multimedia, benchmark (faiss-cpu, hnswlib)
pip install -r web/backend/requirements.txt      # API (FastAPI)
cd web && npm install && cd ..                   # frontend (Next.js, necesita Node)
./run_web.sh                                     # backend :8000 y frontend :3000
```

Pruebas: `python -m unittest discover -s tests -t .`. Las de la Parte 4 que necesitan `data/` se
saltan solas si no está.

## Organización

| Carpeta | Qué hay |
|---|---|
| `parser/` | scanner, parser (AST), executor y `EXPLAIN` |
| `storage/` | páginas, buffer pool, heap / sequential file, catálogo, WAL, `vector_index.py` |
| `indexes/` | B+, hash extensible, bitmap, R-Tree, **IVF, HNSW y métricas** ([`VECTOR_README.md`](indexes/VECTOR_README.md)) |
| `text_based/` | Parte 3: SPIMI, TF-IDF y BM25 ([`docs/SPIMI.md`](docs/SPIMI.md)) |
| `multimedia/` | Parte 4: SIFT, MFCC, codebook y `ExtractFeatures` ([`README`](multimedia/README.md)) |
| `benchmark/` | comparaciones de cada parte, la de la Parte 4 en [`benchmark/multimedia/`](benchmark/multimedia/README.md) |
| `scripts/` | descarga del dataset |
| `web/` | backend FastAPI (`web/backend/`) y frontend Next.js (`web/app/`) |
| `tests/` | pruebas con `unittest` |

## Dataset

Canciones con audio, letra, portada y fotos de artistas e integrantes de bandas.
Se descarga con los scripts de `scripts/` (detalles en [`data/README.md`](data/README.md)).
No se sube al repo.

| Dato | Fuente | Cantidad |
|---|---|---|
| Audio (preview 30 s), portada, foto del artista | Deezer API | 13 643 canciones |
| Letra (texto y `.lrc` sincronizado) | LRCLIB | 10 031 con letra |
| Portadas de otros álbumes de los mismos artistas | Deezer API | 20 005 portadas distintas |
| Fotos extra de artistas | TheAudioDB, Wikidata + Wikimedia Commons | |
| Persona o banda, integrantes y puesto | MusicBrainz (+ Wikidata) | |

## Parte 4: búsqueda multimedia

Dada una imagen o un audio, devolver los k más parecidos. Como no hay palabras, se fabrican
(*bag of visual / audio words*):

```
archivo ─► SIFT / MFCC ─► codebook K-Means (K = 1024) ─► histograma TF-IDF ─► IVF / HNSW ─► SIMILAR_TO
```

| Paso | Dónde | Detalle |
|---|---|---|
| Descriptores locales | `multimedia/sift.py`, `mfcc.py` | [`multimedia/README.md`](multimedia/README.md#1-descriptores-locales) |
| Codebook y TF-IDF | `multimedia/codebook.py`, `features.py` | [`multimedia/README.md`](multimedia/README.md#2-codebook-k-means) |
| Métricas, IVF, HNSW | `indexes/metricas.py`, `ivf.py`, `hnsw.py` | [`indexes/VECTOR_README.md`](indexes/VECTOR_README.md) |
| Índice en el motor | `storage/vector_index.py` | [`indexes/VECTOR_README.md`](indexes/VECTOR_README.md#en-el-motor-storagevector_indexpy) |
| SQL | `parser/` | abajo |
| Comparación | `benchmark/multimedia/` | [`benchmark/multimedia/README.md`](benchmark/multimedia/README.md) |

### Preparar los vectores

Con `data/` descargado (todo se puede cortar y volver a correr):

```bash
python -m multimedia.sift --source covers               # descriptores de portadas
python -m multimedia.mfcc                               # descriptores de audio
python -m multimedia.codebook --tipo ambos --ks 1024    # codebook
python -m multimedia.features --tipo ambos --k 1024     # vectores TF-IDF de la colección
```

### SQL

```
Select     ::= SELECT Proj {, Proj}* FROM id WHERE Cond [UsingVec] [ORDER BY ...] [LIMIT num]
SimilarTo  ::= SIMILAR_TO ( 'ruta' , [k =] num )
UsingVec   ::= USING (IVF | HNSW) [WITH METRIC = (cosine | euclidean | dot)]
Proj       ::= ... | SIMILARITY_SCORE ( ) [AS id]
CreateIdx  ::= CREATE INDEX ON id ( columna ) USING (IVF | HNSW)
```

```sql
CREATE TABLE canciones (id INT PRIMARY KEY, titulo VARCHAR(80), artista VARCHAR(60),
                        audio VARCHAR(40), portada VARCHAR(40)) USING HEAP;
INSERT INTO canciones VALUES (1, 'MATADORA', 'KAROL G', 'audio/4176013632.mp3', 'covers/1037643992.jpg');
CREATE INDEX ON canciones (portada) USING HNSW;
CREATE INDEX ON canciones (audio) USING IVF;

-- portadas parecidas
SELECT * FROM canciones
    WHERE SIMILAR_TO('covers/1037643992.jpg', k=10)
    USING HNSW WITH METRIC=cosine;

-- canciones que suenan parecido
SELECT titulo, artista, SIMILARITY_SCORE() AS score FROM canciones
    WHERE SIMILAR_TO('audio/4176013632.mp3', k=5)
    USING IVF WITH METRIC=euclidean
    ORDER BY score DESC;
```

- La ruta de la consulta puede ser absoluta o relativa a `data/`, y no tiene que estar en la tabla:
  se le saca el vector con `ExtractFeatures`. La extensión dice si es imagen o audio
- La columna indexada es un `VARCHAR` con rutas de imágenes o de audios (no mezcladas)
- Sin `USING` se usa el índice vectorial que tenga la tabla para ese tipo, y si no tiene, búsqueda
  lineal (`Seq Scan`). Sin `WITH METRIC` se usa coseno
- `SIMILARITY_SCORE()`: mayor = más parecido. Coseno y dot dan la similitud, euclidean da 1 / (1 + d)
- Se puede combinar con otras condiciones con `AND` (`... AND genero = 'pop'`) y con `LIMIT`.
  No se permite dentro de un `OR`, con `JOIN` ni con agregaciones

```
EXPLAIN ANALYZE SELECT titulo, SIMILARITY_SCORE() AS score FROM canciones
    WHERE SIMILAR_TO('covers/110599312.jpg', k=5) USING HNSW WITH METRIC=cosine;

Index Scan using hnsw_canciones_portada on canciones  (cost=0.29..20.94 rows=5 width=224) (actual time=50.835..50.966 rows=5 loops=1)
  Order By: k-NN por similitud (k=5, metrica=coseno)
  Vectores revisados: 259
  Index Cond: (similar_to('covers/110599312.jpg', k=5))
Planning Time: 0.008 ms
Execution Time: 50.985 ms
```

### Resultados

Con 19 800 portadas y 10 000 audios, coseno, k = 10 (detalle, gráficas y galería en
[`benchmark/multimedia/README.md`](benchmark/multimedia/README.md)):

| | construir (portadas) | consulta (portadas) | recall@10 portadas | recall@10 audio |
|---|---|---|---|---|
| Lineal | — | 14 ms | 1.000 | 1.000 |
| IVF (nlist = √N, nprobe = 10) | 10.4 s | 2.6 ms | 0.941 | 0.922 |
| HNSW (M = 16, ef = 50) | 175 s | 1.8 ms | 0.981 | 0.987 |

HNSW para consultar (casi no crece con N), IVF si se reconstruye seguido, y coseno o vectores
normalizados: con euclidiana sin normalizar el recall cae a 0.80–0.84.

## Ideas para la Parte 5

1. **Shazam:** reconocer la canción grabando unos segundos aunque haya ruido, con *audio fingerprinting* (picos del espectrograma como huella de la grabación, como hace Shazam). Se puede hacer con `librosa` o ver `audfprint` / `dejavu`
2. **Detector cantando:** que alguien cante o tararee y diga qué canción es o cuál se parece más, con *query by humming*. Se saca la melodía con `librosa.pyin` y se compara con DTW (`librosa.sequence.dtw`), porque MFCC ignora la melodía a propósito
