# chutaDB como base de datos vectorial: qué hace, qué le falta y cómo lo resuelven los gestores reales

Este documento compara la Parte 4 con gestores vectoriales reales (pgvector, Qdrant, Weaviate, Milvus) y
bibliotecas como Faiss. Para cada limitación dice si se arregló, cómo se arregló y, si no, cómo se haría.
Las fuentes están al final.

| # | Tema | Estado |
|---|---|---|
| 1 | Velocidad: Faiss | se explica, no se cambia (el curso pide índices propios) |
| 2 | `SIMILAR_TO` con filtros | **arreglado**: pre-filtro exacto o búsqueda filtrada dentro del índice |
| 3 | Borrados | se explica cómo arreglarlo; no implementado |
| 4 | Codebook fijo | se explica cómo lo resuelven los gestores reales; no implementado |
| 5 | SQL limitado | **arreglado** `JOIN` y `GROUP BY`; quedan `OR` y varios `SIMILAR_TO` |
| 6 | Parámetros fijos | **arreglado**: `WITH NPROBE = n` y `WITH EF_SEARCH = n` |
| 7 | Vectores con bolsa de palabras | se explica por qué *embeddings* de redes neuronales serían mejores |

Todos los números de este documento salen de una tabla con **1 000 portadas** (columna `IMAGE`, un HNSW y un
IVF con su propio codebook, K = 671), 20 consultas, `k = 10` y recall@10 contra `USING SEQ`.

---

## 1. Velocidad: qué es Faiss

**Faiss** (*Facebook AI Similarity Search*) es la biblioteca de Meta para buscar vecinos cercanos entre
millones o miles de millones de vectores [1, 2]. Tiene los mismos índices que nosotros (IVF, HNSW) y otros
más (cuantización de productos, PQ), escritos en C++ con instrucciones SIMD y opcionalmente en GPU. Muchos
gestores vectoriales la usan por debajo (por ejemplo Milvus) [7].

En el benchmark (`benchmark/multimedia/README.md`, "Contra Faiss y hnswlib") nuestro HNSW llega al mismo
recall que Faiss y hnswlib, pero es entre 8 y 30 veces más lento, y construirlo con 20 000 portadas tarda
~3 min contra 6 s de Faiss. La diferencia no es el algoritmo, es el lenguaje: Python puro hace cada
comparación por separado, C++ compara muchos números a la vez.

**¿Se arregla?** No conviene: el curso pide implementar los índices. En producción se usaría Faiss o
hnswlib detrás de la misma interfaz (`VectorSecondaryIndex`), sin cambiar el SQL.

---

## 2. `SIMILAR_TO` con filtros (arreglado)

### El problema

```sql
SELECT titulo FROM canciones WHERE genero = 'Rock' AND SIMILAR_TO('foto.jpg', k=10) USING HNSW;
```

Antes, el motor pedía los 10 vecinos al índice y después tiraba los que no eran de Rock; si quedaban menos
de 10, pedía 20, 40, 80… (*post-filtro*). Con un filtro que deja pocas filas, había que pedir casi toda la
tabla al índice para completar 10, y el índice no sabía nada del filtro.

### Cómo lo hacen los gestores reales

- **Pre-filtro exacto cuando quedan pocas filas.** Qdrant estima cuántas filas cumplen el filtro y, si son
  menos que `full_scan_threshold`, compara contra todas ellas sin usar el grafo [4]. Weaviate hace lo mismo
  con `flatSearchCutoff` (40 000 por defecto) [5]
- **Filtro dentro del índice cuando quedan muchas.** Qdrant y hnswlib recorren el grafo HNSW completo pero
  solo dejan entrar al resultado los nodos que cumplen el filtro [4, 6]. Los nodos que no cumplen siguen
  sirviendo de "puente" para llegar a otras zonas del grafo. ACORN [11] y Filtered-DiskANN [12] son versiones
  más elaboradas de la misma idea
- **pgvector** (desde la versión 0.8.0) hace *iterative index scans*: si después de filtrar no llegó a k,
  sigue buscando en el índice en vez de devolver menos filas [3]

### Lo que hace ahora chutaDB

1. Primero se evalúa el resto del `WHERE` sobre la tabla y se arma el conjunto de filas que cumplen
2. Si cumplen **2 000 filas o menos** (`UMBRAL_PREFILTRO` en `parser/executor.py`), **pre-filtro exacto**:
   se comparan los vectores de esas filas con la consulta, sin usar el grafo. Es exacto (recall 1.0) y barato
3. Si cumplen más, **búsqueda filtrada dentro del índice**:
   - HNSW (`indexes/hnsw.py`): en el nivel 0 el grafo se recorre igual, pero a la lista de resultados solo
     entran los nodos permitidos, como hnswlib y Qdrant
   - IVF (`indexes/ivf.py`): solo se comparan los vectores permitidos de cada celda. Si el filtro deja una
     fracción *f* de las filas, se abren `nprobe / f` celdas (con un filtro del 5 %, 20 veces más), y si
     aun así no hay k se siguen abriendo celdas, como el *iterative scan* de pgvector
4. `EXPLAIN` muestra qué estrategia se usó: `Filtro: pre-filtro exacto (50 filas cumplen)` o
   `Filtro: filtro dentro del indice (...)`

### Resultados (1 000 portadas, recall@10 contra `USING SEQ` con el mismo filtro)

| Filtro deja | Estrategia | HNSW | IVF |
|---|---|---|---|
| 30 % (300 filas) | filtro dentro del índice | 1.000 · 7.5 ms | 0.970 · 6.3 ms |
| 30 % | pre-filtro exacto | 1.000 · 5.3 ms | 1.000 · 5.7 ms |
| 5 % (50 filas) | filtro dentro del índice | 1.000 · 9.7 ms | 1.000 · 7.4 ms |
| 5 % | pre-filtro exacto | 1.000 · 4.8 ms | 1.000 · 5.2 ms |
| sin filtro | índice normal | 1.000 · 2.5 ms | 0.930 · 1.9 ms |

- Siempre salen las 10 filas pedidas, y todas cumplen el filtro
- Con filtros que dejan pocas filas, el pre-filtro es el más rápido y además exacto, por eso es lo que se
  usa por debajo de 2 000 filas (con 1 000 portadas, siempre)
- Sin el escalado de `nprobe`, el IVF filtrado caía a 0.825 (30 %) y 0.63 (5 %): con pocas filas por celda,
  abrir solo 10 celdas no alcanza
- El costo que queda: para saber qué filas cumplen se recorre la tabla. Un gestor real usa un índice del
  filtro (Qdrant crea índices de *payload* [4]); aquí se podría usar el B+ o el bitmap de la columna filtrada

---

## 3. Borrados

### Qué pasa hoy

IVF y HNSW no borran: un `DELETE` marca el vector como borrado (`borrados` en `storage/vector_index.py`) y
la búsqueda lo salta. El vector sigue ocupando espacio y, en HNSW, sigue en el grafo. El índice solo crece
hasta que se vuelve a crear.

### ¿Pasa también en la base relacional?

Sí, el borrado lógico es normal en cualquier base, pero las tablas sí recuperan el espacio:

- **Heap** (`storage/files/heap_file.py`): marca el registro como borrado y la página guarda una lista de
  *slots* libres que el próximo `INSERT` reutiliza. Además tiene `compact()`/`reorganize()` para
  desfragmentar páginas
- **Archivo secuencial** (`storage/files/sequential_file.py`): marca el registro y, cuando el espacio
  desperdiciado pasa de un umbral (`WASTED_RATIO`), se reorganiza solo

El índice vectorial es el único que nunca recupera el espacio.

### Cómo lo resuelven los gestores reales y cómo se haría aquí

| Gestor | Qué hace |
|---|---|
| hnswlib | `mark_deleted` + `replace_deleted`: un `INSERT` nuevo ocupa el lugar de un nodo borrado y se reconectan sus vecinos [6] |
| pgvector / PostgreSQL | `VACUUM` limpia las filas muertas del HNSW y del IVF; `REINDEX` reconstruye [3] |
| Milvus | los datos van en segmentos; la *compaction* junta segmentos y descarta lo borrado [7] |
| Lucene / Elasticsearch | los documentos borrados se descartan al fusionar segmentos (*merge*) [8] |

Para chutaDB, lo más parecido a lo que ya hace el archivo secuencial:

1. **Compactar al pasar un umbral.** Si `borrados > 20 %` de los vectores, reconstruir solo con los vigentes.
   En el IVF es barato: los centroides se mantienen y basta con quitar los borrados de cada lista. En el HNSW
   hay que reconstruir el grafo, lo que con 20 000 vectores tarda minutos; por eso conviene hacerlo con un
   umbral alto o en un comando aparte (`REINDEX`)
2. **Reutilizar lugares en el HNSW** como hnswlib: el próximo vector ocupa el nodo borrado y se reconecta

---

## 4. Codebook fijo: cómo lo gestionan los gestores reales

En chutaDB, cada columna `IMAGE`/`AUDIO` entrena su codebook (K-Means) al crear el índice. Si después entran
muchos archivos distintos, las "palabras" ya no describen bien la colección y el idf queda viejo.

Los gestores reales tienen dos partes distintas:

- **El modelo que convierte el archivo en vector no lo entrena la base.** Es un modelo ya entrenado (CLIP,
  un modelo de audio, etc.) que se usa igual para todos los datos. Weaviate o Milvus llaman a ese modelo al
  insertar, pero no lo reentrenan [5, 7]. Si se cambia el modelo, hay que volver a calcular todos los vectores
- **Lo que sí se entrena es el índice IVF**: los centroides salen de K-Means sobre los datos, igual que
  nuestro IVF. pgvector recomienda crear el `ivfflat` **después** de cargar los datos y hacer `REINDEX` si
  los datos cambian mucho [3]. Faiss obliga a llamar `train()` antes de agregar vectores [2]. HNSW no tiene
  esta dependencia: no aprende nada de los datos, solo conecta vecinos

Para chutaDB habría dos opciones:

1. **Reentrenar cuando la tabla crece**, por ejemplo al duplicarse: rehacer el codebook, los vectores y el
   índice (como un `REINDEX` automático)
2. **Usar un modelo ya entrenado** (punto 7): el vector de cada archivo no depende de la colección, así que no
   hace falta reentrenar nada

---

## 5. SQL (arreglado en parte)

### Lo que se agregó

```sql
-- SIMILAR_TO + JOIN: las 5 canciones más parecidas a este audio, con el nombre de su artista,
-- solo de artistas de rock (el filtro de la otra tabla se aplica después de unir y siguen saliendo 5)
SELECT canciones.titulo, artistas.nombre, SIMILARITY_SCORE() AS score
    FROM canciones JOIN artistas ON canciones.id_artista = artistas.id
    WHERE artistas.genero = 'Rock' AND SIMILAR_TO('consulta.mp3', k=5);

-- SIMILAR_TO + GROUP BY: de las 20 portadas más parecidas, cuántas hay de cada género
SELECT genero, COUNT(*) AS n FROM canciones
    WHERE SIMILAR_TO('foto.jpg', k=20) GROUP BY genero ORDER BY n DESC;
```

- **`JOIN`:** la columna de imagen o audio puede estar en cualquiera de las dos tablas. Las condiciones que
  solo miran esa tabla se aplican antes de buscar (con el filtro del punto 2); las que miran la otra, después
  de unir. Si después de unir faltan filas, se piden el doble de vecinos hasta completar k
- **`GROUP BY` y agregaciones** (`COUNT`, `SUM`, `AVG`, `MIN`, `MAX`) se calculan sobre las k filas más
  parecidas. `SIMILARITY_SCORE()` no se puede agregar
- **`USING SEQ`** (`LINEAL`, `SEQUENTIAL`) fuerza la búsqueda exacta. Si la columna tiene índice, compara
  con todos sus vectores, con el mismo modelo, así es la respuesta exacta de lo que el índice aproxima

### Lo que sigue sin estar

| Falta | Cómo lo resuelven otros |
|---|---|
| `SIMILAR_TO` dentro de un `OR` | en pgvector se puede escribir `WHERE embedding <-> q < 0.5 OR genero = 'Rock'`, pero el índice solo se usa con `ORDER BY distancia LIMIT k`, así que eso recorre toda la tabla [3, 21]. Qdrant, Milvus y Weaviate permiten `OR` en los filtros de los datos, pero la búsqueda vectorial es una sola por consulta y no puede ir dentro del `OR` [4, 5, 7]. Es decir: un "k más parecidos" dentro de un `OR` no tiene un significado claro y ningún índice lo acelera |
| Dos `SIMILAR_TO` en la misma consulta (por ejemplo, portada y audio) | sí lo hacen varios, como *búsqueda híbrida o multivector*: cada búsqueda va por separado y después se combinan los rankings. Milvus: `hybrid_search` con un `AnnSearchRequest` por vector y `RRFRanker` o `WeightedRanker` (desde 2.4) [22]. Qdrant: `prefetch` + `fusion: rrf` o `dbsf` (desde 1.10) [23]. Weaviate: varios vectores objetivo combinados por mínimo, suma o promedio [24]. Elasticsearch: el *retriever* `rrf` con varios hijos [25]. pgvector no lo trae: hay que escribir las dos consultas y combinarlas en SQL |
| Buscar con un vector en vez de un archivo | pgvector acepta literales `'[0.1, 0.2, ...]'` [3]; aquí se podría aceptar `SIMILAR_TO(id = 5)` para buscar lo parecido a una fila de la tabla |
| Alias de tabla (`FROM canciones c`) | falta en el parser en general, no solo para `SIMILAR_TO` |

---

## 6. Parámetros desde SQL (arreglado)

```sql
SELECT id FROM fotos WHERE SIMILAR_TO('q.jpg', k=10) USING IVF  WITH NPROBE = 30;
SELECT id FROM fotos WHERE SIMILAR_TO('q.jpg', k=10) USING HNSW WITH EF_SEARCH = 200, METRIC = cosine;
SELECT id FROM fotos WHERE SIMILAR_TO('q.jpg', k=10) WITH EF_SEARCH = 100;   -- sin USING: el índice de la tabla
```

Es lo mismo que `SET ivfflat.probes` y `SET hnsw.ef_search` en pgvector [3] o el `params` de la búsqueda en
Qdrant y Milvus [4, 7]. `NPROBE` va solo con IVF y `EF_SEARCH` solo con HNSW (si no, error). `EXPLAIN` los
muestra (`Parametros: nprobe=30`).

| Consulta (1 000 portadas, k = 10) | Tiempo | Recall@10 |
|---|---|---|
| `USING IVF WITH NPROBE = 1` | 1.7 ms | 0.55 |
| `USING IVF WITH NPROBE = 10` (por defecto) | 2.0 ms | 0.93 |
| `USING IVF WITH NPROBE = 30` | 1.9 ms | 0.99 |
| `USING HNSW WITH EF_SEARCH = 10` | 1.7 ms | 1.00 |
| `USING HNSW WITH EF_SEARCH = 50` (por defecto) | 2.3 ms | 1.00 |
| `USING HNSW WITH EF_SEARCH = 200` | 4.3 ms | 1.00 |
| `USING SEQ` | 14 ms | 1.00 |

Con `NPROBE` se ve en vivo el cambio de velocidad por precisión del IVF. Con 1 000 portadas el HNSW ya acierta
todo con `EF_SEARCH = 10`; con 20 000 portadas el recall por defecto es 0.98 (ver el benchmark).

---

## 7. Vectores: por qué *embeddings* de redes neuronales serían mejores

### Qué usamos

SIFT [15] y MFCC [16] sacan descriptores locales a mano, K-Means los agrupa en "palabras" y cada archivo es un
histograma TF-IDF (*bag of visual words* [17], el enfoque que pide el curso). Funciona muy bien para encontrar
**el mismo objeto** modificado (recortado, rotado, en blanco y negro: recall 0.99–1.00 en el benchmark), pero:

- Con **fotos reales** de CDs y vinilos, solo el 66 % encuentra su portada en el top 10: el reflejo, la
  perspectiva y el fondo cambian los puntos SIFT
- Con **audio grabado con celular** no acierta nunca: el celular pierde graves y agudos, y MFCC describe
  justamente ese espectro
- **No entiende de qué trata** la imagen: dos portadas con un atardecer no se parecen para SIFT si los
  detalles son distintos. SIFT trabaja en gris, así que tampoco ve el color

### Qué son los *embeddings*

Un *embedding* es el vector que da una red neuronal entrenada con millones de ejemplos: la red aprende sola
qué rasgos importan. Archivos parecidos quedan cerca y el vector se usa igual en IVF y HNSW.

| Tipo | Modelo | Qué aporta |
|---|---|---|
| Imagen | **CLIP** [9] | imagen y texto en el mismo espacio: se puede buscar una portada escribiendo "atardecer en la playa" |
| Imagen | **DINOv2** [10] | aprendido sin etiquetas; muy bueno para encontrar la misma imagen o una parecida |
| Imagen | redes convolucionales (*neural codes*) [13] | ya en 2014 superaban a la bolsa de palabras visuales en recuperación de imágenes |
| Audio | **VGGish** [14], **PANNs** [18] | entrenados con millones de clips de YouTube; resisten mejor el ruido y la grabación |
| Audio | **CLAP** [19] | como CLIP pero para audio: buscar una canción escribiendo "guitarra acústica tranquila" |
| Audio (misma grabación) | huellas tipo **Shazam** [20] | no es un *embedding*, pero es lo que se usa para reconocer una canción grabada con celular (prototipo en la rama `avances-y-correcciones`) |

### Por qué sería mejor

1. **Más robusto con fotos y grabaciones reales**, que es lo que haría un usuario de una app: los modelos se
   entrenan con fotos con luz, ángulo y fondo distintos, y con audio con ruido
2. **Buscan por significado**, no solo por el mismo objeto, y CLIP o CLAP permiten buscar con texto
3. **No dependen de la colección**: el modelo ya está entrenado, así que no hay codebook que reentrenar
   cuando la tabla crece (resuelve el punto 4)
4. **Vectores más chicos**: 512–768 números contra 1 024 del codebook, con lo que IVF y HNSW son más rápidos

**Lo que cuesta:** hace falta PyTorch y el modelo (cientos de MB), y sacar el vector de cada archivo es más
lento sin GPU. Además el curso pide SIFT/MFCC y el codebook, así que sería un modo extra, no un reemplazo.

**Cómo se integraría:** `ModeloMultimedia` (`storage/modelo_multimedia.py`) ya es la pieza que convierte un
archivo en vector para cada columna. Bastaría otra clase con la misma interfaz (`vector(ruta)`) que llame a
CLIP o a PANNs, y una opción al crear la tabla o el índice, por ejemplo
`CREATE INDEX ON fotos (foto) USING HNSW WITH MODEL = clip`.

---

## Referencias

1. Johnson, J., Douze, M. y Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*, 7(3), 535–547. https://arxiv.org/abs/1702.08734
2. Douze, M. et al. (2024). The Faiss library. *arXiv:2401.08281*. https://github.com/facebookresearch/faiss
3. pgvector: open-source vector similarity search for Postgres (índices `ivfflat` y `hnsw`, `probes`, `ef_search`, *iterative index scans* desde la v0.8.0). https://github.com/pgvector/pgvector
4. Qdrant, documentación: *Indexing* (HNSW filtrable, `full_scan_threshold`, índices de *payload*) y *Filtering*. https://qdrant.tech/documentation/concepts/indexing/
5. Weaviate, documentación: *Filtering* (`flatSearchCutoff`) y *Hybrid search*. https://weaviate.io/developers/weaviate/concepts/filtering
6. hnswlib (`mark_deleted`, `replace_deleted`, filtros en la búsqueda). https://github.com/nmslib/hnswlib
7. Wang, J. et al. (2021). Milvus: a purpose-built vector data management system. *SIGMOD 2021*, 2614–2627. https://milvus.io/docs/
8. Elasticsearch, documentación: *k-nearest neighbor search* y *Reciprocal rank fusion*. https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html
9. Radford, A. et al. (2021). Learning transferable visual models from natural language supervision (CLIP). *ICML 2021*. https://arxiv.org/abs/2103.00020
10. Oquab, M. et al. (2023). DINOv2: Learning robust visual features without supervision. *arXiv:2304.07193*.
11. Patel, L., Kraft, P., Guestrin, C. y Zaharia, M. (2024). ACORN: Performant and predicate-agnostic search over vector embeddings and structured data. *SIGMOD 2024*. https://arxiv.org/abs/2403.04871
12. Gollapudi, S. et al. (2023). Filtered-DiskANN: Graph algorithms for approximate nearest neighbor search with filters. *WWW 2023*, 3406–3416.
13. Babenko, A., Slesarev, A., Chigorin, A. y Lempitsky, V. (2014). Neural codes for image retrieval. *ECCV 2014*. https://arxiv.org/abs/1404.1777
14. Hershey, S. et al. (2017). CNN architectures for large-scale audio classification (VGGish). *ICASSP 2017*. https://arxiv.org/abs/1609.09430
15. Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. *International Journal of Computer Vision*, 60(2), 91–110.
16. Davis, S. y Mermelstein, P. (1980). Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 28(4), 357–366.
17. Sivic, J. y Zisserman, A. (2003). Video Google: A text retrieval approach to object matching in videos. *ICCV 2003*, 1470–1477.
18. Kong, Q. et al. (2020). PANNs: Large-scale pretrained audio neural networks for audio pattern recognition. *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 28, 2880–2894. https://arxiv.org/abs/1912.10211
19. Elizalde, B., Deshmukh, S., Al Ismail, M. y Wang, H. (2023). CLAP: Learning audio concepts from natural language supervision. *ICASSP 2023*. https://arxiv.org/abs/2206.04769
20. Wang, A. (2003). An industrial-strength audio search algorithm. *ISMIR 2003*.
21. pgEdge, documentación de pgvector 0.8: *Iterative index scans* (el índice solo se usa con `ORDER BY` de la distancia y `LIMIT`). https://docs.pgedge.com/pgvector/v0-8-5/iterative-index-scans/
22. Zilliz / Milvus, referencia de `hybrid_search` (`AnnSearchRequest`, `RRFRanker`, `WeightedRanker`). https://docs.zilliz.com/reference/python/python/Vector-hybrid_search
23. Qdrant, documentación: *Hybrid and multi-stage queries* (`prefetch`, `fusion`). https://qdrant.tech/documentation/concepts/hybrid-search/
24. Weaviate, documentación: *Multiple target vectors* (estrategias mínimo, suma, promedio). https://weaviate.io/developers/weaviate/search/multi-vector
25. Elasticsearch, documentación: *RRF retriever*. https://www.elastic.co/docs/reference/elasticsearch/rest-apis/retrievers/rrf-retriever
