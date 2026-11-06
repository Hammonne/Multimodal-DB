# Consultas para probar el frontend

Este es el guion de pruebas manuales para la interfaz web. No se deben
confundir estas pruebas con los scripts de benchmark del final del documento:
en esta sección la persona que prueba debe abrir el navegador, elegir la vista,
escribir la consulta indicada y pulsar el botón correspondiente.

## 1. Arranque y navegación

1. Ejecutar `./run_web.sh`.
2. Abrir [http://localhost:3000](http://localhost:3000).
3. Confirmar que la consola SQL carga tablas en el panel izquierdo.
4. Cambiar entre las pestañas **SQL**, **Texto**, **Multimedia** y
   **Benchmarks**.
5. Si los estilos o chunks aparecen como 404, detener el servidor, ejecutar
   `rm -rf web/.next` y volver a iniciar `./run_web.sh`.

Cada caso debe registrarse como **PASS** o **FAIL**, junto con una captura que
muestre la consulta, la configuración elegida y el resultado.

## 2. Casos de prueba: consola SQL

Estos casos prueban el editor SQL, el botón **EJECUTAR**, la tabla de
resultados, el plan y el historial. Las sentencias se deben pegar una por una
en el editor.

### 2.1 Crear una tabla de prueba

El parser requiere indicar la organización física con `USING HEAP` o
`USING SEQUENTIAL`. Los tipos disponibles para esta demostración son `INT`,
`FLOAT`, `BOOL`, `DATE`, `POINT` y `VARCHAR(n)`.

```sql
CREATE TABLE demo_productos (
  id INT PRIMARY KEY,
  nombre VARCHAR(80),
  categoria VARCHAR(40),
  precio FLOAT,
  disponible BOOL
) USING HEAP;
```

Resultado esperado: el mensaje de éxito indica que `demo_productos` fue
creada y la tabla aparece en el catálogo. Si la tabla ya existe, usar otro
nombre, por ejemplo `demo_productos_2`, o eliminarla con:

```sql
DROP TABLE demo_productos;
```

### 2.2 Insertar datos

El parser actual usa `INSERT INTO ... VALUES (...)` sin lista de columnas.
Por eso el orden debe coincidir exactamente con el `CREATE TABLE`.

```sql
INSERT INTO demo_productos VALUES (1, 'Camara web', 'multimedia', 149.90, true);
```

```sql
INSERT INTO demo_productos VALUES (2, 'Microfono USB', 'audio', 89.50, true);
```

```sql
INSERT INTO demo_productos VALUES (3, 'Disco SSD', 'almacenamiento', 210.00, true);
```

```sql
INSERT INTO demo_productos VALUES (4, 'Cable HDMI', 'accesorios', 18.75, false);
```

```sql
INSERT INTO demo_productos VALUES (5, 'Audifonos', 'audio', 59.99, true);
```

Comprobar las filas:

```sql
SELECT * FROM demo_productos ORDER BY id ASC LIMIT 10;
```

### 2.3 Crear y usar índices tradicionales

Crear un índice B-tree para filtros y orden:

```sql
CREATE INDEX ON demo_productos (precio) USING BTREE;
```

Crear un índice Hash para búsquedas de igualdad:

```sql
CREATE INDEX ON demo_productos (categoria) USING HASH;
```

Crear un índice Bitmap para una columna con pocos valores repetidos:

```sql
CREATE INDEX ON demo_productos (disponible) USING BITMAP;
```

Probar las consultas que pueden beneficiarse de esos índices:

```sql
SELECT * FROM demo_productos
WHERE precio BETWEEN 50 AND 160
ORDER BY precio ASC
LIMIT 10;
```

```sql
SELECT * FROM demo_productos
WHERE categoria = 'audio'
LIMIT 10;
```

```sql
SELECT * FROM demo_productos
WHERE disponible = true
LIMIT 10;
```

Mostrar el plan:

```sql
EXPLAIN SELECT * FROM demo_productos
WHERE precio BETWEEN 50 AND 160
ORDER BY precio ASC
LIMIT 10;
```

```sql
EXPLAIN ANALYZE SELECT * FROM demo_productos
WHERE categoria = 'audio'
LIMIT 10;
```

El plan debe aparecer en el panel de plan. `EXPLAIN` muestra la estrategia
estimada y `EXPLAIN ANALYZE` ejecuta la consulta y agrega filas y tiempos
observados.

| Caso | Consulta para pegar en el editor | Resultado esperado |
|---|---|---|
| SQL-01 | `SELECT * FROM ventas WHERE id BETWEEN 1 AND 10 ORDER BY id ASC LIMIT 25;` | La consulta termina correctamente y muestra filas o un resultado vacío sin romper la interfaz. |
| SQL-02 | `EXPLAIN SELECT * FROM ventas WHERE id BETWEEN 1 AND 10;` | Se muestra un plan con nodos de acceso, tabla y cantidad estimada de filas. |
| SQL-03 | `EXPLAIN ANALYZE SELECT * FROM ventas WHERE id BETWEEN 1 AND 10;` | Se muestra el plan y el tiempo de ejecución. |
| SQL-04 | `SELECT * FROM ventas LIMIT 5;` | La tabla muestra como máximo cinco filas y permite revisar sus columnas. |
| SQL-05 | `SELECT * FROM tabla_que_no_existe;` | Aparece un mensaje de error visible; no debe mostrarse un resultado exitoso falso. |
| SQL-06 | *(dejar el editor vacío)* | La interfaz indica que se debe escribir una consulta. |

Si `ventas` no aparece en el catálogo de una instalación limpia, usar
`demo_productos` después de ejecutar las sentencias anteriores.

### 2.5 Importar los archivos generados por los scripts

Los scripts no crean tablas automáticamente: generan archivos CSV en `data/`.
Ahora el frontend puede cargarlos con `IMPORT INTO`, después de crear una
tabla cuyo encabezado coincida con las columnas del CSV. La importación
convierte `INT`, `FLOAT` y `BOOL` y reporta la línea exacta si encuentra un
dato inválido.

Ejemplo mínimo para `data/songs.csv`:

```sql
CREATE TABLE demo_songs (
  track_id INT PRIMARY KEY,
  title VARCHAR(255),
  artist_id INT,
  artist VARCHAR(255),
  album_id INT,
  album VARCHAR(255),
  genre VARCHAR(80),
  duration_s INT,
  deezer_rank INT,
  explicit BOOL,
  has_lyrics BOOL,
  has_synced_lyrics BOOL,
  instrumental BOOL,
  audio_path VARCHAR(255),
  lyrics_path VARCHAR(255),
  lrc_path VARCHAR(255),
  cover_path VARCHAR(255),
  deezer_url VARCHAR(500)
) USING HEAP;
```

```sql
IMPORT INTO demo_songs FROM 'data/songs.csv';
```

Verificar la carga:

```sql
SELECT * FROM demo_songs LIMIT 10;
```

Y crear un índice sobre la clave:

```sql
CREATE INDEX ON demo_songs (track_id) USING BTREE;
```

Para otros archivos, repetir el patrón:

```sql
CREATE TABLE demo_artists (
  artist_id INT PRIMARY KEY,
  name VARCHAR(255),
  genre VARCHAR(80),
  wikidata_id VARCHAR(100),
  n_images INT,
  main_image_path VARCHAR(255),
  deezer_url VARCHAR(500)
) USING HEAP;
```

```sql
IMPORT INTO demo_artists FROM 'data/artists.csv';
```

El encabezado del CSV debe contener todas las columnas declaradas. El orden
de las columnas puede ser diferente, porque la importación las busca por
nombre. Las columnas adicionales del CSV se ignoran; las columnas declaradas
que no existan producen un error explícito.

## 3. Casos de prueba: pestaña Texto

La pestaña **Texto** tiene un campo **CONSULTA**, los selectores **Algoritmo** y
**Dataset**, y los botones **CONSTRUIR ÍNDICE** y **BUSCAR**.

Antes de probarla, generar el corpus que el backend espera:

```bash
python3 scripts/generate_text_corpus.py
```

Esto crea `data/lyrics.jsonl` con 100 000 documentos. Si solo se desea una
prueba rápida:

```bash
python3 scripts/generate_text_corpus.py --documents 1000
```

También se puede arrastrar un archivo `.json` o `.jsonl` en la zona de carga
de la consola SQL. Debe contener objetos con `id` y `lyrics` (o `text`), por
ejemplo:

```json
[
  {"id": 1, "title": "Documento 1", "lyrics": "base de datos vectorial"},
  {"id": 2, "title": "Documento 2", "lyrics": "machine learning y búsqueda"}
]
```

La interfaz valida el archivo, lo normaliza como `data/lyrics.jsonl` y cambia
a la pestaña **Texto**. Después solo hay que elegir el tamaño y pulsar
**CONSTRUIR ÍNDICE**. Los CSV continúan usando el flujo de tablas SQL; los
JSON/JSONL se reservan para el corpus textual.

### 3.1 Construcción del índice

Para cada fila, seleccionar el dataset indicado, pulsar **CONSTRUIR ÍNDICE** y
esperar el mensaje `Índice SPIMI listo`.

| Caso | Dataset | Consulta visible | Resultado esperado |
|---|---:|---|---|
| TXT-01 | 1 000 | `base de datos vectorial` | Se informa cantidad de documentos, términos y tiempo de construcción. |
| TXT-02 | 10 000 | `base de datos vectorial` | Se construye el índice del tamaño seleccionado sin cambiar la consulta. |
| TXT-03 | 100 000 | `base de datos vectorial` | Se construye o se informa claramente si ese dataset no está disponible. |

### 3.2 Búsquedas válidas

Para cada caso: escribir la consulta, elegir el algoritmo, elegir el dataset,
pulsar **BUSCAR** y revisar la tabla **Resultados**.

| Caso | Consulta | Algoritmo | Dataset | Resultado esperado |
|---|---|---|---:|---|
| TXT-04 | `vectorial` | BM25 | 1 000 | Hasta 10 resultados ordenados por score. |
| TXT-05 | `vectorial` | TF-IDF + Coseno | 1 000 | Hasta 10 resultados y score visible. |
| TXT-06 | `base de datos` | BM25 | 10 000 | La cabecera indica BM25 y 10 000 documentos. |
| TXT-07 | `base de datos` | TF-IDF + Coseno | 10 000 | La cabecera indica TF-IDF y el tiempo de consulta. |
| TXT-08 | `búsqueda eficiente en documentos de texto y datos vectoriales` | BM25 | 1 000 | Se procesa una consulta de 5+ palabras sin desbordar el panel. |
| TXT-09 | `búsqueda eficiente en documentos de texto y datos vectoriales` | TF-IDF + Coseno | 1 000 | Se puede comparar el ranking con TXT-08. |
| TXT-10 | `zzzzterminoquenoexiste` | BM25 | 1 000 | Se muestra una tabla vacía y no se inventan documentos. |

En cada resultado comprobar:

- el número de filas no supera 10;
- los puestos aparecen numerados;
- cada fila muestra documento, score y fragmento;
- la insignia muestra el tiempo en milisegundos;
- la cabecera muestra algoritmo, dataset, documentos, términos y tamaño;
- cambiar BM25 por TF-IDF actualiza la respuesta;
- cambiar el dataset actualiza la metadata de la respuesta.

### 3.4 Cómo funciona la búsqueda textual

La búsqueda textual no se ejecuta con el editor SQL todavía. Se prueba desde
la pestaña **Texto**, que llama a los endpoints `/api/text/index` y
`/api/text/search`.

1. **Construcción:** SPIMI lee el corpus, tokeniza cada documento y crea un
   índice invertido `término -> documentos y frecuencias`.
2. **TF-IDF + Coseno:** calcula la importancia del término en el documento y
   en el corpus, normaliza los vectores y ordena por similitud angular.
3. **BM25:** combina frecuencia del término, rareza del término y longitud del
   documento; sus parámetros principales son `k1` y `b`.
4. **Respuesta web:** el frontend muestra los 10 primeros documentos, score,
   fragmento, tiempo, cantidad de documentos, términos y espacio del índice.

La comparación textual SQL equivalente todavía no está disponible en el
parser: `MATCH(...) USING TF_IDF/BM25` es una sintaxis objetivo, no una
consulta aprobada para esta consola. Para probar texto se deben usar los
casos TXT-01 a TXT-14.

### 3.5 Pruebas SQL con los documentos descargados

Los scripts generan `data/lyrics.csv` y los archivos de texto asociados en
`data/lyrics/`. Esta tabla importa la metadata de esos documentos y permite
probar desde SQL que el corpus fue cargado correctamente. Los identificadores
se declaran como `VARCHAR` porque provienen de servicios externos y pueden
superar el rango del `INT` de 32 bits.

Ejecutar primero:

```sql
CREATE TABLE demo_lyrics (
  document_id VARCHAR(64) PRIMARY KEY,
  track_id VARCHAR(64),
  title VARCHAR(255),
  artist_id VARCHAR(64),
  artist VARCHAR(255),
  album_id VARCHAR(64),
  album VARCHAR(255),
  genre VARCHAR(100),
  duration_s INT,
  lyrics_path VARCHAR(255),
  lrc_path VARCHAR(255),
  has_synced_lyrics BOOL,
  instrumental BOOL,
  n_chars INT,
  n_words INT
) USING HEAP;
```

Después importar el CSV generado por `fetch_lyrics.py`:

```sql
IMPORT INTO demo_lyrics FROM 'data/lyrics.csv';
```

Casos de prueba SQL:

```sql
SELECT * FROM demo_lyrics
LIMIT 10;
```

```sql
SELECT document_id, title, artist, n_words, lyrics_path
FROM demo_lyrics
WHERE n_words BETWEEN 50 AND 500
ORDER BY n_words DESC
LIMIT 10;
```

```sql
SELECT document_id, title, artist, genre
FROM demo_lyrics
WHERE has_synced_lyrics = true
LIMIT 10;
```

```sql
SELECT document_id, title, artist, n_chars
FROM demo_lyrics
WHERE artist = 'Justin Bieber'
ORDER BY n_chars DESC
LIMIT 10;
```

Crear índices para los filtros de metadata:

```sql
CREATE INDEX ON demo_lyrics (artist) USING BTREE;
```

```sql
CREATE INDEX ON demo_lyrics (n_words) USING BTREE;
```

Mostrar el plan:

```sql
EXPLAIN SELECT document_id, title, artist, n_words
FROM demo_lyrics
WHERE n_words BETWEEN 50 AND 500
ORDER BY n_words DESC
LIMIT 10;
```

```sql
EXPLAIN ANALYZE SELECT *
FROM demo_lyrics
WHERE has_synced_lyrics = true
LIMIT 10;
```

Estas consultas prueban la carga, filtros, ordenamiento, límites e índices de
la tabla documental. El contenido completo de cada letra está en
`lyrics_path`; el parser SQL actual no lee el contenido de archivos ni
implementa `MATCH`. Por ello, la búsqueda por términos y el ranking
TF-IDF/BM25 se prueban desde la pestaña **Texto**, usando el mismo corpus
`data/lyrics.jsonl`.

Para comprobar la correspondencia entre SQL y la búsqueda textual:

1. Ejecuta `SELECT COUNT(*) FROM demo_lyrics;` y registra la cantidad de
   documentos importados.
2. Ejecuta `SELECT * FROM demo_lyrics WHERE n_words > 0 LIMIT 10;`.
3. En **Texto**, construye el dataset `1k` y busca `base de datos vectorial`.
4. Compara que el ranking web devuelva documentos del corpus y que sus
   fragmentos correspondan a los archivos indicados por `lyrics_path`.

### 3.6 Cómo probar los índices de texto

En esta implementación hay dos niveles de indexación:

#### A. Índice invertido SPIMI: búsqueda por contenido

SPIMI es el índice que usa la pestaña **Texto**. No se crea con `CREATE INDEX`
porque no indexa una columna de la tabla SQL: indexa el contenido de
`data/lyrics.jsonl` y persiste postings en `data/search_indexes/`.

#### Prueba completa de SPIMI

**Paso 1: preparar el corpus**

Si se usan los documentos descargados:

```bash
python3 scripts/fetch_lyrics.py
```

Ese script genera o actualiza:

```text
data/lyrics.jsonl
data/lyrics.csv
data/lyrics/*.txt
data/lyrics/*.lrc
```

Si todavía no hay letras disponibles, se puede generar un corpus reproducible
para la demostración:

```bash
python3 scripts/generate_text_corpus.py --documents 100000
```

El backend espera que cada documento JSONL tenga al menos:

```json
{"id": 1, "lyrics": "base de datos vectorial ..."}
```

**Paso 2: construir SPIMI desde el frontend**

1. Abrir **Texto**.
2. Seleccionar `1 000`.
3. Pulsar **CONSTRUIR ÍNDICE**.
4. Esperar el mensaje `Índice SPIMI listo`.
5. Repetir con `10 000` y `100 000`.

La respuesta debe mostrar:

```text
dataset
algorithm = spimi
documents
terms
build_ms
index_size_bytes
```

**Paso 3: comprobar los artefactos**

Después de construir el índice:

```text
data/search_indexes/1k/spimi.bin
data/search_indexes/1k/spimi.json
data/search_indexes/1k/documents.json
```

Para los demás tamaños se crean las carpetas `10k` y `100k`. El archivo
`spimi.bin` contiene los postings y `spimi.json` contiene offsets y metadata
necesaria para leerlos; `documents.json` conserva el título y el texto usado
para mostrar los fragmentos.

**Paso 4: probar que el índice responde**

En **Texto**, ejecutar estas consultas con dataset `1 000`:

```text
vectorial
```

```text
base de datos
```

```text
base de datos vectorial
```

Para cada consulta repetir:

- algoritmo **BM25**;
- algoritmo **TF-IDF + Coseno**.

El resultado esperado es una tabla de hasta 10 filas con `rank`, documento,
score y fragmento. La cabecera debe indicar que se usó el dataset seleccionado
y mostrar los términos y el tamaño del índice.

**Paso 5: probar casos límite**

Consulta sin coincidencias:

```text
termino_que_no_existe_123
```

Debe devolver cero resultados sin error.

Consulta vacía: borrar el campo y pulsar **BUSCAR**. El botón debe quedar
deshabilitado.

**Paso 6: evidencia experimental**

Registrar una fila por combinación:

| Dataset | Consulta | Ranking | Construcción | Consulta | Términos | Tamaño |
|---|---|---|---:|---:|---:|---:|
| 1k | `vectorial` | BM25 | ... | ... | ... | ... |
| 1k | `vectorial` | TF-IDF | ... | ... | ... | ... |
| 10k | `base de datos` | BM25 | ... | ... | ... | ... |
| 10k | `base de datos` | TF-IDF | ... | ... | ... | ... |
| 100k | `base de datos vectorial` | BM25 | ... | ... | ... | ... |
| 100k | `base de datos vectorial` | TF-IDF | ... | ... | ... | ... |

SPIMI construye el índice una vez por dataset. BM25 y TF-IDF reutilizan los
postings de SPIMI y solo cambian la fórmula de ranking; por eso no se debe
construir un índice SPIMI diferente para cada algoritmo.

**Qué se demuestra**

1. SPIMI procesa documentos en memoria y crea un índice invertido.
2. Cada término apunta a los documentos donde aparece y su frecuencia.
3. La consulta recupera postings, en lugar de leer todos los documentos.
4. BM25 o TF-IDF calculan el score de los candidatos.
5. El frontend muestra el top 10 y los metadatos de rendimiento.

SPIMI no se prueba con una consulta SQL como `CREATE INDEX`. La tabla
`demo_lyrics` y sus índices B-tree/Hash/Bitmap sirven para consultar metadata;
la búsqueda de palabras se ejecuta mediante `/api/text/index` y
`/api/text/search`.

Para construirlo desde el frontend:

1. Abrir **Texto**.
2. Elegir `1 000`, `10 000` o `100 000`.
3. Pulsar **CONSTRUIR ÍNDICE**.
4. Confirmar el mensaje `Índice SPIMI listo`.

Luego ejecutar las mismas búsquedas con los dos rankers:

```text
base de datos vectorial
```

Seleccionar **BM25** y pulsar **BUSCAR**. Repetir seleccionando **TF-IDF +
Coseno**.

La evidencia que se debe registrar es:

- documentos indexados;
- cantidad de términos;
- tamaño del índice;
- tiempo de construcción;
- tiempo de consulta;
- top 10 y score.

La comparación de resultados se hace así:

| Índice lógico | Ranking | Interfaz |
|---|---|---|
| SPIMI | BM25 | Texto |
| SPIMI | TF-IDF + Coseno | Texto |
| PostgreSQL GIN | `plainto_tsquery` | Benchmark GIN |

Estas consultas SQL son la referencia conceptual del baseline GIN, pero no se
ejecutan en el motor SQL local:

```sql
SELECT document_id, title, artist
FROM demo_lyrics
WHERE to_tsvector('simple', contenido)
      @@ plainto_tsquery('simple', 'base de datos vectorial')
LIMIT 10;
```

El benchmark de GIN usa PostgreSQL y el script
`benchmark/text_gin.py`, no el parser local.

#### B. Índices SQL sobre metadata de documentos

Estos sí se crean desde la consola SQL y aceleran filtros sobre columnas
estructuradas. No reemplazan a SPIMI para buscar palabras dentro de una letra.

Crear un B-tree para ordenar y filtrar por cantidad de palabras:

```sql
CREATE INDEX ON demo_lyrics (n_words) USING BTREE;
```

Probarlo con rango:

```sql
EXPLAIN SELECT document_id, title, artist, n_words
FROM demo_lyrics
WHERE n_words BETWEEN 50 AND 500
ORDER BY n_words DESC
LIMIT 10;
```

Crear un B-tree para búsquedas por artista:

```sql
CREATE INDEX ON demo_lyrics (artist) USING BTREE;
```

```sql
EXPLAIN ANALYZE SELECT document_id, title, artist, n_words
FROM demo_lyrics
WHERE artist = 'Justin Bieber'
LIMIT 10;
```

Crear un Hash para igualdad exacta:

```sql
CREATE INDEX ON demo_lyrics (genre) USING HASH;
```

```sql
EXPLAIN SELECT document_id, title, artist, genre
FROM demo_lyrics
WHERE genre = 'Pop'
LIMIT 10;
```

Crear un Bitmap para una columna con pocos valores:

```sql
CREATE INDEX ON demo_lyrics (has_synced_lyrics) USING BITMAP;
```

```sql
EXPLAIN ANALYZE SELECT document_id, title, artist
FROM demo_lyrics
WHERE has_synced_lyrics = true
LIMIT 10;
```

El plan debe indicar el acceso al índice cuando el optimizador lo seleccione.
Para verificar la diferencia, ejecutar la consulta antes y después de crear el
índice y registrar el plan y el tiempo.

#### C. Orden recomendado de la demostración

Ejecutar en este orden:

```sql
CREATE INDEX ON demo_lyrics (n_words) USING BTREE;
```

```sql
CREATE INDEX ON demo_lyrics (artist) USING BTREE;
```

```sql
CREATE INDEX ON demo_lyrics (genre) USING HASH;
```

```sql
CREATE INDEX ON demo_lyrics (has_synced_lyrics) USING BITMAP;
```

Después probar:

```sql
EXPLAIN ANALYZE SELECT *
FROM demo_lyrics
WHERE n_words BETWEEN 50 AND 500
ORDER BY n_words DESC
LIMIT 10;
```

Finalmente, cambiar a **Texto** y ejecutar:

```text
machine learning
```

con **BM25** y **TF-IDF + Coseno**. Esto demuestra la diferencia entre un
índice SQL estructurado y un índice invertido especializado para contenido
textual.

### 3.7 Demostración rápida de búsqueda textual

1. Arrastrar `data/lyrics.jsonl` a la consola SQL.
2. Abrir la pestaña **Texto**.
3. Seleccionar `1 000` y pulsar **CONSTRUIR ÍNDICE**.
4. Buscar:

```text
base de datos vectorial
```

5. Ejecutar con **BM25** y luego con **TF-IDF + Coseno**.
6. Mostrar el top 10, los scores, el tiempo de consulta y el tamaño del
   índice.
7. Repetir con `10 000` y `100 000` para comparar rendimiento.

**Evidencia:** SPIMI construye el índice invertido; BM25 y TF-IDF generan el
ranking de documentos.

### 3.3 Validaciones de errores

| Caso | Acción en la interfaz | Resultado esperado |
|---|---|---|
| TXT-11 | Borrar el contenido de **CONSULTA** y pulsar **BUSCAR** | El botón queda deshabilitado. |
| TXT-12 | Escribir solo espacios | El botón queda deshabilitado. |
| TXT-13 | Elegir 100 000 sin haber generado ese índice y buscar | Se muestra un error explícito; la pantalla no queda simulando resultados. |
| TXT-14 | Pulsar dos veces **CONSTRUIR ÍNDICE** rápidamente | El estado de carga evita acciones inconsistentes y luego muestra el resultado final. |

## 4. Casos de prueba: pestaña Multimedia

La pestaña **Multimedia** necesita un ID que exista en las bases de features.
Tomar un ID real de `data/features/sift_covers.csv` para SIFT o de
`data/features/mfcc.csv` para MFCC; pegar únicamente ese valor en
**ID DEL OBJETO DE CONSULTA**.

### 4.0 Demostración rápida

1. Generar features:

```bash
python3 -m multimedia.sift
python3 -m multimedia.mfcc
python3 -m multimedia.features --tipo sift
python3 -m multimedia.features --tipo mfcc
```

2. Abrir **Multimedia**.
3. Elegir **SIFT / imagen** o **MFCC / audio**.
4. Copiar un ID real de `data/features/sift_covers.csv` o
   `data/features/mfcc.csv`.
5. Elegir `HNSW`, métrica `Coseno` y pulsar **BUSCAR SIMILARES**.
6. Repetir con `IVF` y comparar vecinos, distancia, candidatos y tiempo.

**Evidencia:** SIFT/MFCC convierten el objeto en un vector; IVF/HNSW recuperan
vecinos; la métrica define cómo se ordenan.

### 4.1 Búsquedas válidas

| Caso | ID | Descriptor | Índice | Métrica | Resultado esperado |
|---|---|---|---|---|---|
| MM-01 | ID real de `sift_covers.csv` | SIFT / imagen | HNSW | Coseno | Aparecen vecinos, distancia y marca de consulta. |
| MM-02 | El mismo ID de MM-01 | SIFT / imagen | IVF | Coseno | Se puede comparar con HNSW y se muestran candidatos. |
| MM-03 | El mismo ID de MM-01 | SIFT / imagen | HNSW | Euclidiana | La métrica cambia en la cabecera y puede cambiar el orden. |
| MM-04 | ID real de `mfcc.csv` | MFCC / audio | HNSW | Coseno | Se muestran vecinos de audio y dimensión del vector. |
| MM-05 | El mismo ID de MM-04 | MFCC / audio | IVF | Euclidiana | Se muestran candidatos y tiempo de consulta. |
| MM-06 | El mismo ID de MM-04 | MFCC / audio | IVF | Producto punto | La búsqueda termina sin error y muestra la métrica elegida. |

Para cada caso comprobar:

- como máximo aparecen 10 vecinos;
- los vecinos tienen `rank` y `distance`;
- la fila de la consulta se identifica como **CONSULTA** cuando corresponde;
- la cabecera muestra índice, métrica, objetos, dimensiones y candidatos;
- cambiar el índice o métrica no deja datos de la búsqueda anterior.

### 4.3 Pruebas SQL de índices vectoriales

Además de la pestaña **Multimedia**, el parser SQL soporta `IVF`, `HNSW`,
`SIMILAR_TO(...)` y `SIMILARITY_SCORE()`. Estas consultas requieren que las
rutas usadas en `ruta` existan físicamente y sean compatibles con el extractor
configurado.

Crear la tabla:

```sql
CREATE TABLE demo_imagenes (
  id INT PRIMARY KEY,
  nombre VARCHAR(100),
  ruta VARCHAR(255)
) USING HEAP;
```

Insertar rutas reales del proyecto. Reemplazar los valores de ejemplo por
archivos que existan en `data/`:

```sql
INSERT INTO demo_imagenes VALUES (1, 'Imagen de consulta', 'data/images/query.jpg');
```

```sql
INSERT INTO demo_imagenes VALUES (2, 'Imagen relacionada', 'data/images/related.jpg');
```

```sql
INSERT INTO demo_imagenes VALUES (3, 'Otra imagen', 'data/images/other.jpg');
```

Crear los índices vectoriales sobre la columna que contiene las rutas:

```sql
CREATE INDEX ON demo_imagenes (ruta) USING HNSW;
```

```sql
CREATE INDEX ON demo_imagenes (ruta) USING IVF;
```

Ejecutar una búsqueda HNSW con coseno:

```sql
SELECT *, SIMILARITY_SCORE() AS score
FROM demo_imagenes
WHERE SIMILAR_TO('data/images/query.jpg', k=3)
USING HNSW WITH METRIC=cosine
ORDER BY SIMILARITY_SCORE() DESC
LIMIT 3;
```

Ejecutar una búsqueda IVF con distancia euclidiana:

```sql
SELECT *, SIMILARITY_SCORE() AS score
FROM demo_imagenes
WHERE SIMILAR_TO('data/images/query.jpg', k=3)
USING IVF WITH METRIC=euclidean
ORDER BY SIMILARITY_SCORE() DESC
LIMIT 3;
```

Ejecutar producto punto:

```sql
SELECT id, nombre, SIMILARITY_SCORE() AS score
FROM demo_imagenes
WHERE SIMILAR_TO('data/images/query.jpg', 3)
USING HNSW WITH METRIC=dot
ORDER BY SIMILARITY_SCORE() DESC;
```

La consulta vectorial funciona así:

1. `SIMILAR_TO` convierte el archivo de consulta en un vector.
2. El índice IVF reduce la búsqueda a grupos candidatos, mientras HNSW
   recorre un grafo de vecinos aproximados.
3. La métrica calcula la distancia o similitud contra los candidatos.
4. `k` limita la cantidad de vecinos recuperados.
5. `SIMILARITY_SCORE()` expone el score para mostrarlo u ordenar.

Para validar el plan:

```sql
EXPLAIN SELECT *, SIMILARITY_SCORE() AS score
FROM demo_imagenes
WHERE SIMILAR_TO('data/images/query.jpg', k=3)
USING HNSW WITH METRIC=cosine
ORDER BY SIMILARITY_SCORE() DESC
LIMIT 3;
```

El plan debe indicar un `VECTOR INDEX SCAN` cuando el índice y la ruta son
válidos. Si no hay índice compatible, el plan puede indicar un `VECTOR SEQ
SCAN`, que compara los registros sin usar un índice especializado.

### 4.2 Validaciones de errores

| Caso | Acción | Resultado esperado |
|---|---|---|
| MM-07 | Dejar vacío el ID | **BUSCAR SIMILARES** queda deshabilitado. |
| MM-08 | Escribir `id-inexistente-frontend` | Aparece un error explícito indicando que no existe el objeto o la feature. |
| MM-09 | Elegir SIFT y pegar un ID exclusivo de MFCC | Aparece un error de datos incompatible; no se muestran resultados falsos. |

## 5. Casos de prueba: pestaña Benchmarks

1. Abrir **Benchmarks**.
2. Pulsar **CARGAR MULTIMEDIA**.
3. Confirmar que aparecen las columnas `Tipo`, `N`, `Índice`, `Métrica`,
   `Build (s)`, `Consulta (ms)`, `Recall@10` y `KB`.
4. Confirmar que se muestra el número de filas cargadas.
5. Pulsar **CARGAR GIN**.
6. Confirmar que PostgreSQL GIN aparece en una tabla separada.

| Caso | Acción | Resultado esperado |
|---|---|---|
| BEN-01 | Cargar multimedia con `resultados.json` disponible | Aparecen filas IVF/HNSW y sus métricas. |
| BEN-02 | Cargar multimedia sin `resultados.json` | Aparece un error claro indicando que se debe ejecutar el benchmark. |
| BEN-03 | Cargar GIN con `results_gin.json` disponible | Aparecen las filas del baseline PostgreSQL. |
| BEN-04 | Cargar GIN sin haber ejecutado el benchmark | La interfaz informa que GIN todavía no está disponible. |

Esta pestaña no ejecuta el benchmark: solo carga sus archivos JSON y los
presenta. Por eso BEN-02 y BEN-04 son comportamientos válidos en una
instalación que aún no tiene resultados.

## 6. Matriz mínima para la demostración

Si hay poco tiempo, ejecutar estos ocho casos y tomar una captura de cada
resultado: **SQL-01, SQL-02, TXT-01, TXT-04, TXT-07, MM-01, MM-04 y BEN-01**.
Con ellos se demuestra navegación, consulta SQL, plan de ejecución, SPIMI,
BM25, TF-IDF, SIFT, MFCC, HNSW y carga de benchmarks.

---

# Apéndice: ejecución técnica de benchmarks

Este plan sirve para probar la implementación actual, capturar evidencia para
la presentación y detectar qué resultados todavía dependen de datasets no
generados. Todas las corridas deben guardar el comando, tamaño del dataset,
parámetros, fecha y resultado.

## 0. Preparación común

Desde la raíz de `Multimodal-DB`:

```bash
cd /ruta/al/proyecto/Multimodal-DB
source .venv/bin/activate  # si existe el entorno local
export PYTHONPATH=.
```

Iniciar la web:

```bash
./run_web.sh
```

Comprobar servicios:

```bash
curl http://localhost:8000/api/health
curl -I http://localhost:3000/
```

Respuesta esperada del backend:

```json
{"status":"ok"}
```

Si Next.js devuelve 404 para archivos de `.next`, detener el guion y limpiar
solo el build temporal:

```bash
rm -rf web/.next
./run_web.sh
```

## 1. Pruebas unitarias de la parte textual

Estas pruebas verifican el algoritmo antes de medir rendimiento:

```bash
PYTHONPATH=. python3 tests/text_based/test_spimi.py
PYTHONPATH=. python3 tests/text_based/test_tfidf.py
PYTHONPATH=. python3 tests/text_based/test_bm25.py
```

Resultado mínimo esperado:

```text
9 tests pasaron.
13 tests pasaron.
15 tests pasaron.
```

Qué demuestra cada grupo:

| Suite | Demuestra |
|---|---|
| SPIMI | tokenización, bloques, merge, postings y persistencia |
| TF-IDF | pesos, normas, coseno, ranking y `top_k` |
| BM25 | `k1`, `b`, longitud de documentos, ranking y `top_k` |

## 2. Dataset textual

El corpus por defecto de la API es:

```text
data/lyrics.jsonl
```

Cada línea debe tener al menos:

```json
{"id": 1, "title": "Documento 1", "lyrics": "base de datos vectorial"}
```

Para comprobar que el corpus tiene los tamaños requeridos:

```bash
wc -l data/lyrics.jsonl
```

Si no tiene 100 000 documentos, no reportar la corrida de 10 000 como si fuera
la de 100 000. Usar un dataset sintético solo si se etiqueta explícitamente
como sintético.

## 3. Plan textual A: construcción SPIMI

Construir cada tamaño desde la pestaña **Texto** o con la API:

```bash
for size in 1k 10k 100k; do
  curl -sS -X POST http://localhost:8000/api/text/index \
    -H 'Content-Type: application/json' \
    -d "{\"dataset\":\"$size\"}"
  echo
done
```

Registrar por cada tamaño:

- documentos indexados;
- cantidad de términos;
- tiempo de construcción;
- tamaño de `.bin` + `.json`;
- memoria máxima del proceso, si se ejecuta desde consola.

Los artefactos quedan en:

```text
data/search_indexes/{1k,10k,100k}/
```

## 4. Plan textual B: ranking TF-IDF y BM25

Ejecutar exactamente estas consultas con ambos algoritmos:

| Longitud | Consulta |
|---|---|
| 1 palabra | `vectorial` |
| 3 palabras | `base de datos` |
| 5+ palabras | `búsqueda eficiente en documentos de texto y datos vectoriales` |

Ejemplo BM25:

```bash
curl -sS -X POST http://localhost:8000/api/text/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "base de datos",
    "algorithm": "bm25",
    "dataset": "1k",
    "top_k": 10
  }'
```

Ejemplo TF-IDF:

```bash
curl -sS -X POST http://localhost:8000/api/text/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "base de datos",
    "algorithm": "tfidf",
    "dataset": "1k",
    "top_k": 10
  }'
```

En cada respuesta guardar:

- algoritmo;
- dataset;
- ranking y score;
- tiempo de consulta;
- términos;
- tamaño del índice.

### Criterio de revisión manual

Para cada consulta revisar:

1. Los documentos del top 10 contienen al menos uno de los términos.
2. Los primeros resultados tienen mayor score que los siguientes.
3. La consulta desconocida devuelve `results: []`.
4. Una consulta vacía responde HTTP 400.
5. Cambiar BM25 por TF-IDF cambia los scores o el orden cuando la longitud
   de los documentos es diferente.

## 5. Plan textual C: PostgreSQL GIN

Levantar PostgreSQL:

```bash
docker compose up -d postgres
```

Ejecutar el baseline:

```bash
python -m benchmark.text_gin \
  --source data/lyrics.jsonl \
  --sizes 1000 10000 100000 \
  --output benchmark/text/results_gin.json
```

Si se usa PostgreSQL fuera de Docker:

```bash
POSTGRES_DSN='postgresql://usuario:clave@host:5432/base' \
python -m benchmark.text_gin
```

Validar el JSON:

```bash
python3 -m json.tool benchmark/text/results_gin.json >/dev/null
```

Luego entrar a la pestaña **Benchmarks** y pulsar **CARGAR GIN**.

La comparación correcta debe tener:

- mismas consultas;
- mismos tamaños;
- mismo corpus;
- `construir_s`;
- `consulta_ms`;
- espacio del índice.

GIN es un baseline externo de PostgreSQL, no un índice implementado dentro de
Chuta DB.

## 6. Plan multimedia A: preparación de features

Generar descriptores:

```bash
python -m multimedia.sift --source all
python -m multimedia.mfcc
```

Construir codebooks y bases TF-IDF:

```bash
python -m multimedia.features --tipo sift
python -m multimedia.features --tipo mfcc
```

Verificar que existan:

```text
data/features/codebook/sift_k1024.npz
data/features/codebook/mfcc_k1024.npz
data/features/tfidf_sift_k1024.npz
data/features/tfidf_mfcc_k1024.npz
```

## 7. Plan multimedia B: búsqueda interactiva

En la pestaña **Multimedia**, probar estas combinaciones:

| Descriptor | Índice | Métrica |
|---|---|---|
| SIFT | HNSW | coseno |
| SIFT | IVF | coseno |
| SIFT | HNSW | euclidiana |
| MFCC | HNSW | coseno |
| MFCC | IVF | euclidiana |

El `query_id` debe existir en el CSV correspondiente:

```text
data/features/sift_covers.csv
data/features/mfcc.csv
```

La misma prueba por API:

```bash
curl -sS -X POST http://localhost:8000/api/multimedia/search \
  -H 'Content-Type: application/json' \
  -d '{
    "kind": "sift",
    "index": "hnsw",
    "metric": "coseno",
    "query_id": "ID_EXISTENTE",
    "top_k": 10
  }'
```

Validar:

- el resultado tiene como máximo 10 vecinos;
- la distancia está ordenada de menor a mayor;
- el objeto consulta aparece identificado;
- HNSW e IVF devuelven el mismo tipo de ids;
- cambiar la métrica puede cambiar el orden.

## 8. Plan multimedia C: benchmark completo

Para un dataset disponible de 1 000:

```bash
python -m benchmark.multimedia.benchmark_vectorial \
  --data data/sim_1k \
  --salida benchmark/multimedia/graficas
```

Para la colección principal:

```bash
python -m benchmark.multimedia.benchmark_vectorial \
  --salida benchmark/multimedia/graficas
```

Regenerar gráficas desde el JSON:

```bash
python -m benchmark.multimedia.benchmark_vectorial --solo-graficas
```

Validar el resultado:

```bash
python3 -m json.tool benchmark/multimedia/graficas/resultados.json >/dev/null
```

Registrar por cada fila:

- `tipo`;
- `n`;
- `indice`;
- `metrica`;
- `construir_s`;
- `consulta_ms`;
- `recall10`;
- `espacio_kb`.

La búsqueda lineal es el ground truth y debe conservar `recall10 = 1.0`.
IVF y HNSW se evalúan contra ese resultado exacto.

## 9. Plan multimedia D: galería

Después del benchmark, entrar a **Benchmarks** y cargar los resultados.

Probar las configuraciones:

```text
sift_hnsw_coseno
sift_ivf_coseno
mfcc_hnsw_coseno
mfcc_ivf_coseno
```

Revisar los casos:

- recorte al 70 %;
- oscurecimiento;
- JPEG pequeño;
- audio recortado;
- audio con ruido.

La API también permite revisar los casos:

```bash
curl -sS \
  http://localhost:8000/api/multimedia/benchmarks/gallery/sift_hnsw_coseno
```

Un caso es acierto si `rank` está entre 1 y 10. Si `rank` es 0, el original
no fue recuperado.

## 10. Plan SQL y estado del parser

El motor sí soporta `EXPLAIN` para sentencias SQL generales:

```sql
EXPLAIN SELECT * FROM ventas WHERE id = 1;
EXPLAIN ANALYZE SELECT * FROM ventas WHERE id = 1;
```

Estas consultas muestran el plan del almacenamiento relacional y sirven para
la demo de índices B-tree, hash o bitmap. El parser también soporta la
extensión vectorial `SIMILAR_TO`, `USING IVF/HNSW`,
`WITH METRIC=...` y `SIMILARITY_SCORE()`. Las consultas de esa extensión están
documentadas en la sección 4.3.

La búsqueda textual `MATCH` todavía no está conectada al parser. Por eso esta
consulta es una especificación futura y no debe presentarse como una prueba
SQL que ya funciona:

```sql
SELECT * FROM documentos
WHERE MATCH(contenido, 'base de datos vectorial')
USING TF_IDF LIMIT 10;

```

Para texto, las pruebas funcionales actuales se ejecutan desde la pestaña
**Texto**, mediante los endpoints `/api/text/index` y `/api/text/search`.
Para multimedia se pueden usar tanto la pestaña **Multimedia** como las
consultas SQL vectoriales de la sección 4.3.

## 11. Checklist final de presentación

- [ ] Tests SPIMI, TF-IDF y BM25 pasan.
- [ ] Se muestran las tres longitudes de consulta.
- [ ] Se comparan BM25 y TF-IDF sobre el mismo dataset.
- [ ] GIN usa el mismo corpus y consultas.
- [ ] Se muestran tiempos y tamaño de índice.
- [ ] Se ejecuta al menos una búsqueda SIFT y una MFCC.
- [ ] Se comparan IVF y HNSW.
- [ ] Se comparan coseno y euclidiana.
- [ ] Se muestra Recall@10 contra búsqueda lineal.
- [ ] Se muestra al menos una galería de aciertos y errores.
- [ ] Todo resultado indica si fue ejecutado, pendiente o generado con datos
  sintéticos.
## Integración SQL de SPIMI

Para una demostración rápida se recomienda `demo_songs`, porque indexa las
letras existentes en `data/lyrics/*.txt` sin importar el corpus sintético de
100 000 documentos:

```sql
CREATE INDEX ON demo_songs (lyrics_path) USING SPIMI;

SELECT title, artist, SCORE() AS relevancia
FROM demo_songs
WHERE MATCH(lyrics_path, 'love tonight') USING BM25
LIMIT 10;

SELECT title, artist, SCORE() AS relevancia
FROM demo_songs
WHERE MATCH(lyrics_path, 'love tonight') USING TF_IDF
LIMIT 10;
```

`BM25` y `TF_IDF` reutilizan el mismo índice invertido SPIMI; solo cambia el
ranking. Las rutas relativas como `lyrics/track.txt` se resuelven desde
`data/`.

```sql
CREATE INDEX ON demo_lyrics (lyrics_path) USING SPIMI;
SELECT *, SCORE() AS relevancia
FROM demo_lyrics
WHERE MATCH(lyrics_path, 'machine learning') USING BM25
ORDER BY SCORE() DESC
LIMIT 10;
```

También se puede cambiar `BM25` por `TF_IDF`. El plan debe mostrar `TEXT INDEX
SCAN` y la salida debe incluir las filas originales y su score. Si no existe el
índice, el motor muestra la sentencia `CREATE INDEX` necesaria.
