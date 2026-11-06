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
resultados, el plan y el historial.

| Caso | Consulta para pegar en el editor | Resultado esperado |
|---|---|---|
| SQL-01 | `SELECT * FROM ventas WHERE id BETWEEN 1 AND 10 ORDER BY id ASC LIMIT 25;` | La consulta termina correctamente y muestra filas o un resultado vacío sin romper la interfaz. |
| SQL-02 | `EXPLAIN SELECT * FROM ventas WHERE id BETWEEN 1 AND 10;` | Se muestra un plan con nodos de acceso, tabla y cantidad estimada de filas. |
| SQL-03 | `EXPLAIN ANALYZE SELECT * FROM ventas WHERE id BETWEEN 1 AND 10;` | Se muestra el plan y el tiempo de ejecución. |
| SQL-04 | `SELECT * FROM ventas LIMIT 5;` | La tabla muestra como máximo cinco filas y permite revisar sus columnas. |
| SQL-05 | `SELECT * FROM tabla_que_no_existe;` | Aparece un mensaje de error visible; no debe mostrarse un resultado exitoso falso. |
| SQL-06 | *(dejar el editor vacío)* | La interfaz indica que se debe escribir una consulta. |

Si `ventas` no aparece en el catálogo de una instalación limpia, seleccionar
una tabla real del panel izquierdo y reemplazar `ventas` en SQL-01 a SQL-04.
No usar `MATCH` o `SIMILAR_TO` en esta consola todavía: esas extensiones aún
no están conectadas al parser SQL.

## 3. Casos de prueba: pestaña Texto

La pestaña **Texto** tiene un campo **CONSULTA**, los selectores **Algoritmo** y
**Dataset**, y los botones **CONSTRUIR ÍNDICE** y **BUSCAR**.

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
la demo de índices B-tree, hash o bitmap.

La sintaxis siguiente es el objetivo de la extensión, pero no debe usarse como
prueba aprobada hasta integrar los nodos al parser:

```sql
SELECT * FROM documentos
WHERE MATCH(contenido, 'base de datos vectorial')
USING TF_IDF LIMIT 10;

SELECT * FROM imagenes
WHERE SIMILAR_TO('foto.jpg', k=10)
USING HNSW WITH METRIC=cosine;
```

Mientras esos nodos no estén integrados, las pruebas textuales y multimedia
deben ejecutarse por los endpoints `/api/text/*` y `/api/multimedia/*`.

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
