# SPIMI: índice invertido para búsqueda de texto

Este documento explica qué se cambió para la Parte 3, cómo funciona SPIMI y
cómo está implementado.

Por ahora **solo se construye el índice**. No están implementados TF-IDF +
coseno, BM25, la comparación contra GIN ni las consultas `MATCH` del parser.

---

## 1. Qué se hizo

- Se agregó `text_based/spimi.py` con un índice invertido SPIMI.
- Se agregó `text_based/__init__.py` para exponer la API pública.
- El presupuesto de memoria se mide en **bytes estimados** (por defecto
  **64 MiB**).
- El diccionario en RAM se puede vaciar en medio de un documento si el
  presupuesto se supera.
- Los bloques temporales son binarios y se fusionan de forma progresiva con
  como máximo `MAX_OPEN_RUNS` archivos abiertos (por defecto 16).
- El índice final son dos archivos: un binario con los términos/postings y un
  JSON con metadata y offsets.
- Al cargar el índice solo se lee la metadata. `get_postings()` usa el offset
  del término y lee únicamente ese registro.
- Se agregaron pruebas offline en:
  - `tests/text_based/test_spimi.py`
  - `tests/text_based/test_fetch_lyrics.py`
- Se usó `scripts/fetch_lyrics.py` para generar un dataset chico de letras y
  probar el pipeline completo.
- No se modificó el parser ni el ejecutor SQL.

Archivos:

```text
text_based/
├── __init__.py
└── spimi.py

tests/text_based/
├── test_fetch_lyrics.py
└── test_spimi.py

docs/
└── SPIMI.md
```

Archivos generados para un índice con base `data/spimi_index`:

```text
data/spimi_index.bin
data/spimi_index.json
```

---

## 2. Definiciones

**Definición de documento:** cada texto que se quiere indexar. En este proyecto
un documento es una canción con su letra:

```json
{"id": 123, "lyrics": "letra de la canción..."}
```

**Definición de corpus:** el conjunto completo de documentos que se indexan.

**Definición de token:** un pedazo de texto producido por el tokenizador. Por
ejemplo, `"Hola, mundo!"` produce `["hola", "mundo"]`.

**Definición de término:** un token que se guarda en el índice. En esta
implementación todo token en minúsculas es un término; no se eliminan stopwords
ni se aplica stemming.

**Definición de posting:** el par `(id_documento, frecuencia)`. Por ejemplo,
para el término `"gato"`:

```text
(1, 2)  -> "gato" aparece dos veces en el documento 1
(5, 1)  -> "gato" aparece una vez en el documento 5
```

**Definición de diccionario temporal:** la estructura en RAM que relaciona cada
término con sus postings mientras se procesan documentos:

```python
{
    "gato": {1: 2, 5: 1},
    "perro": {1: 1}
}
```

**Definición de índice invertido:** una estructura que va desde las palabras
hacia los documentos.

**Definición de bloque o run:** un archivo temporal en disco con una parte
ordenada del índice. Se crea cuando el diccionario en RAM supera el presupuesto
de memoria.

**Definición de merge progresivo:** si hay más bloques que `max_open_runs`, se
fusionan grupos de bloques en varias pasadas hasta que quepan todos abiertos a
la vez.

**Definición de SPIMI:** *Single-Pass In-Memory Indexing*. Construye el índice
leyendo cada documento una sola vez y volcando partes a disco cuando se agota
la memoria estimada.

**Definición de longitud de documento (`dl`):** la cantidad total de tokens de
un documento. Se guarda en `doc_lengths` porque TF-IDF y BM25 la van a necesitar.

---

## 3. El algoritmo SPIMI

```text
Mientras haya documentos por procesar:
    Mantener un diccionario temporal en RAM.
    Para cada documento:
        Tokenizarlo y contar sus términos.
        Agregar o actualizar cada término en el diccionario temporal.
        Si la memoria estimada supera max_memory:
            Ordenar los términos del bloque.
            Guardar el bloque binario en disco.
            Vaciar el diccionario temporal.
Al terminar los documentos:
    Guardar el último bloque.
    Mientras haya más de max_open_runs bloques:
        Fusionar grupos de max_open_runs bloques en nuevos bloques.
    Fusionar los bloques restantes.
    Por cada término, juntar sus postings y escribir:
        .bin
        .json
```

Dos detalles importantes:

1. El volcado puede ocurrir **en medio de un documento**.
2. Los bloques no se abren todos juntos. Se fusionan por grupos de tamaño
   `max_open_runs`.

---

## 4. Implementación de bajo nivel

### 4.1 Tokenización

```python
TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
```

- `[^\W_]+` acepta letras Unicode (acentos, `ñ`, etc.) y evita guiones bajos.
- `(?:['’][^\W_]+)*` permite apóstrofes internos (`don't`, `rock’n’roll`).
- Después se aplica `casefold()`.

No se eliminan stopwords ni se aplica stemming.

### 4.2 Lectura de documentos

`iter_documents(ruta)` acepta:

- **JSONL:** un objeto JSON por línea.
- **Arreglo JSON:** un archivo que empieza con `[`.

### 4.3 Ids de documento

`normalize_doc_id(valor)` convierte ids numéricos guardados como texto a
enteros:

```python
normalize_doc_id("123")  -> 123
normalize_doc_id(" -4 ") -> -4
normalize_doc_id(7)      -> 7
```

Los ids de texto no están soportados: el formato binario guarda cada id como
`int64`. Eso alcanza para los `track_id` de Deezer / `fetch_lyrics.py`.

### 4.4 Diccionario temporal y presupuesto en bytes

```python
self.index = {
    "gato": {1: 2, 5: 1},
    "perro": {1: 1}
}
```

Costos estimados:

```text
costo_término = TERM_OVERHEAD + len(término en UTF-8)
costo_posting = POSTING_OVERHEAD
```

```python
TERM_OVERHEAD = 64
POSTING_OVERHEAD = 48
MAX_MEMORY_BYTES = 64 * 1024 * 1024
```

Después de insertar cada término:

```python
if self.memory_bytes >= self.max_memory:
    self.flush_block()
```

Si el mismo término aparece en dos bloques para el mismo documento, el merge
suma las frecuencias.

### 4.5 Volcado de un bloque binario

Cuando la memoria estimada llega a `max_memory`, `flush_block()`:

1. Crea `block_0.run`, `block_1.run`, etc. en un directorio temporal.
2. Recorre los términos en orden lexicográfico.
3. Escribe un registro por término.
4. Vacía el diccionario y reinicia `memory_bytes`.

Formato de cada registro:

```text
uint32  term_len
bytes   término en UTF-8
uint32  df
repetido df veces:
    int64   doc_id
    uint32  tf
```

### 4.6 Merge progresivo

`merge_progressive()` agrupa runs de a `max_open_runs` hasta que quepan todos
abiertos a la vez. Cada grupo se fusiona con un min-heap:

1. Pide el primer registro a cada run.
2. Saca el término más chico.
3. Lo entrega al consumidor.
4. Pide el siguiente registro al mismo run.
5. Repite hasta vaciar todos los runs.

En el merge final, si el mismo término llega desde varios runs, se juntan sus
postings y se suman las frecuencias.

### 4.7 Índice final

El índice se escribe primero como archivos `.part` y después se publica con
`os.replace()`:

```text
data/spimi_index.bin.part  -> data/spimi_index.bin
data/spimi_index.json.part -> data/spimi_index.json
```

#### `.bin`

Contiene un registro por término, en orden lexicográfico, con el mismo formato
de los bloques temporales.

#### `.json`

```json
{
  "doc_count": 3,
  "doc_lengths": {
    "1": 3,
    "2": 2,
    "3": 3
  },
  "term_offsets": {
    "gato": 0,
    "perro": 42,
    "raton": 90
  }
}
```

| Campo | Significado |
| --- | --- |
| `doc_count` | cantidad de documentos indexados |
| `doc_lengths` | tokens por documento |
| `term_offsets` | offset de cada término dentro de `.bin` |

`load()` lee solamente el JSON. `get_postings(term)` hace:

1. Buscar el término en `term_offsets`.
2. Abrir `.bin`, hacer `seek()` al offset y leer ese registro.
3. Devolver `[(doc_id, tf), ...]`.

Si el término no existe, devuelve una lista vacía.

### 4.8 Limpieza de temporales

Los runs viven en un `tempfile.TemporaryDirectory` y se borran al terminar
`build()`, tanto si salió bien como si hubo un error.

### 4.9 Errores controlados

El código lanza `ValueError` cuando:

- un documento no tiene campo `id`;
- el `id` no es numérico;
- `max_memory` es menor que 1;
- `max_open_runs` es menor que 2;
- un archivo binario está incompleto o truncado.

Un corpus vacío es válido: genera un `.bin` vacío y un JSON con
`doc_count = 0`, `doc_lengths = {}` y `term_offsets = {}`.

---

## 5. Uso desde Python

```python
from text_based.spimi import SPIMIIndex

index = SPIMIIndex(
    max_memory=64 * 1024 * 1024,
    output_file="data/spimi_index",
    max_open_runs=16,
)
index.build_from_json("data/lyrics.jsonl")

print(index.doc_count)
print(index.block_count)

loaded = SPIMIIndex(output_file="data/spimi_index").load()
print(loaded.get_postings("amor"))
print(loaded.doc_lengths)
print(loaded.get_doc_length(1))
```

`output_file` es la **base** del índice. A partir de ella se generan `.bin` y
`.json`.

También se puede construir desde una lista de documentos:

```python
documents = [
    {"id": 1, "lyrics": "gato gato perro"},
    {"id": 2, "lyrics": "gato raton"},
]
index = SPIMIIndex(max_memory=1, output_file="data/spimi_index")
index.build(documents)
```

---

## 6. Complejidad y uso de recursos

Sea:

- `D` = cantidad de documentos.
- `T` = cantidad total de tokens.
- `P` = cantidad de postings únicos `(término, documento)`.
- `V` = cantidad de términos distintos.
- `B` = cantidad de bloques iniciales.
- `F` = `max_open_runs`.
- `df_t` = cantidad de documentos del término `t`.

| Etapa | Tiempo | Memoria |
| --- | --- | --- |
| Tokenizar y contar | `O(T)` | `O(tamaño del documento)` |
| Ordenar y escribir un bloque | `O(P_b log P_b)` | `O(P_b)` |
| Merge progresivo | `O(P log F)` por pasada, `O(P log B)` en total | `O(F)` archivos abiertos |
| Escribir el índice final | `O(P)` | `O(max df_t + F)` |
| Cargar el índice | `O(D + V)` | `O(D + V)` |
| `get_postings(term)` | `O(df_t)` | `O(df_t)` |

Relación entre parámetros:

- `max_memory` chico: más bloques, menos RAM, más E/S.
- `max_memory` grande: menos bloques, más RAM, merge más barato.
- `max_open_runs` chico: menos archivos abiertos, más pasadas.
- `max_open_runs` grande: menos pasadas, más presión sobre el límite de
  archivos del sistema operativo.

Limitación: el merge final junta en RAM todos los postings de un mismo término
antes de escribirlo.

---

## 7. Datos de letras y `fetch_lyrics.py`

`scripts/fetch_lyrics.py` arma los archivos que consume SPIMI:

- `data/lyrics.jsonl`: un documento por línea, con `id` y `lyrics`.
- `data/lyrics.csv`: metadata de cada letra.

Si faltan letras, `--fetch` las descarga de LRCLIB. El script es reanudable.

```bash
python3 scripts/fetch_lyrics.py --fetch --songs 50
```

```python
from text_based.spimi import SPIMIIndex

index = SPIMIIndex(output_file="data/spimi_index")
index.build_from_json("data/lyrics.jsonl")
print(index.doc_count, len(index.term_offsets))
```

Los archivos de `data/` no se versionan: están en `.gitignore`.

---

## 8. Pruebas

```bash
PYTHONPATH=. python3 tests/text_based/test_spimi.py
PYTHONPATH=. python3 tests/text_based/test_fetch_lyrics.py
```

`test_spimi.py` cubre:

- tokenización de Unicode, apóstrofes y guiones bajos;
- normalización de ids numéricos;
- lectura de JSONL y de arreglos JSON;
- presupuesto en bytes y volcado en medio de un documento;
- merge progresivo con `max_open_runs` chico;
- formato binario y metadata con offsets;
- consulta de términos, incluidos Unicode e inexistentes;
- merge de un término repetido en varios bloques;
- corpus vacío;
- documentos sin `id`;
- parámetros inválidos.

`test_fetch_lyrics.py` cubre:

- conversión de `.lrc` a texto plano;
- construcción offline de `lyrics.jsonl` y `lyrics.csv`;
- integración directa con `SPIMIIndex`.

---

## 9. Lo que todavía no está implementado

- TF-IDF + similitud del coseno
- BM25
- Precision@10 / Recall@10
- Comparación contra GIN de PostgreSQL
- Extensión del parser SQL con `MATCH` / `SCORE()`
- Búsqueda por imagen o audio
- Integración con `ExternalSorter` / `IndexManager`

Cuando se implemente el parser, `MATCH` va a tener que:

1. tokenizar la consulta;
2. buscar cada término en el índice invertido;
3. obtener los documentos candidatos;
4. aplicar TF-IDF + coseno o BM25;
5. devolver los documentos ordenados por `SCORE()`.

---

## 10. Ventajas y límites de esta implementación

Ventajas:

- Cada documento se lee una sola vez.
- La memoria en RAM está acotada por `max_memory` en bytes estimados.
- El volcado puede ocurrir en medio de un documento.
- El merge progresivo abre como máximo `max_open_runs` runs a la vez.
- El índice final es binario y compacto.
- Al cargar solo se lee la metadata.
- `get_postings()` lee únicamente el registro del término buscado.
- El archivo final se escribe de forma atómica.
- Los tests son offline y deterministas.

Límites:

- No hay posiciones de palabras, así que no hay búsquedas de frases exactas.
- No se aplica stemming ni se eliminan stopwords.
- El límite de memoria es una estimación, no el uso exacto del proceso.
- El merge final junta en RAM todos los postings de un término.
- Los ids de documento deben ser enteros.
- El índice no se actualiza incrementalmente: si cambian los documentos, hay
  que reconstruirlo.
- No está integrado con transacciones, locks ni con el `IndexManager`.
- Todavía no hay ranking: `get_postings` no ordena por relevancia.
