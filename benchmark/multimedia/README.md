# Benchmark de la Parte 4: IVF vs HNSW, euclidiana vs coseno

Compara los dos índices vectoriales (`indexes/ivf.py`, `indexes/hnsw.py`) y las dos métricas
sobre los vectores TF-IDF de las portadas (SIFT) y del audio (MFCC), con consultas k-NN de k = 10.

```bash
python -m benchmark.multimedia.benchmark_vectorial                      # con data/ completa (10k)
python -m benchmark.multimedia.benchmark_vectorial --data data/sim_1k   # con 1k, ~2 min
python -m benchmark.multimedia.benchmark_vectorial --solo-graficas      # rehace las imágenes
```

Sin `--tamanos` cada tipo elige según la data que tenga: portadas 1 000, 2 500, 5 000, 10 000 y
19 800; audio hasta 10 000; con `data/sim_1k` corre 250, 500 y 750. Las imágenes y `resultados.json` van a `graficas/`
(o a `--salida`), junto con `resultados.json`.

## Cómo se mide

- **Base:** los primeros N objetos de `sift_covers.csv` y `mfcc.csv`, con el codebook K = 1024.
  El idf se calcula con esos N, como lo haría la base de datos real.
- **Consultas:** 200 objetos que no están en ninguna base (100 con 1k).
- **Recall@10:** cuántos de los 10 vecinos exactos (búsqueda lineal con la misma métrica)
  devuelve el índice.
- **Vectores sin normalizar:** si se normalizan, euclidiana y coseno ordenan igual
  (‖a − b‖² = 2 − 2·cos), así que para compararlas se usa el TF-IDF tal cual.
- **Parámetros:** IVF con nlist = √N y nprobe = 10; HNSW con M = 16, ef_construction = 100 y
  ef_search = 50. Faiss (`IndexIVFFlat`, `IndexHNSWFlat`) y hnswlib usan los mismos, con 1 hilo.
  Necesitan `pip install faiss-cpu hnswlib`.
- **Espacio:** lo que ocupa el índice guardado en disco (vectores + celdas o grafo).
- **Original de una consulta modificada:** 100 portadas recortadas al 70 %, oscurecidas al 60 % o
  bajadas a 200 px en jpeg, y 100 canciones cortadas a 5 s (con y sin ruido). Se mira si el
  original sale entre los 10 primeros. Sale de ahí la galería.

## Resultados

Portadas hasta **19 800** (de las 20 005 se separan 200 como consultas) y audio hasta **10 000**.
La corrida completa, con las librerías, tarda **27 min 26 s** en un MacBook; lo que más pesa es
construir nuestro HNSW en Python (~3–4 min con 20k).

| tipo | n | índice | métrica | construir (s) | consulta (ms) | recall@10 |
|---|---|---|---|---|---|---|
| portadas | 19 800 | lineal | coseno | — | 14.1 | 1.000 |
| portadas | 19 800 | IVF | coseno | 10.4 | 2.6 | 0.941 |
| portadas | 19 800 | HNSW | coseno | 175 | 1.8 | 0.981 |
| portadas | 19 800 | lineal | euclidiana | — | 52.1 | 1.000 |
| portadas | 19 800 | IVF | euclidiana | 7.9 | 16.0 | 0.839 |
| portadas | 19 800 | HNSW | euclidiana | 253 | 2.3 | 0.801 |
| audio | 10 000 | lineal | coseno | — | 7.2 | 1.000 |
| audio | 10 000 | IVF | coseno | 3.2 | 1.5 | 0.922 |
| audio | 10 000 | HNSW | coseno | 81 | 1.7 | 0.987 |
| audio | 10 000 | lineal | euclidiana | — | 23.3 | 1.000 |
| audio | 10 000 | IVF | euclidiana | 3.6 | 19.3 | 0.998 |
| audio | 10 000 | HNSW | euclidiana | 156 | 2.3 | 0.983 |

Con 10 000 portadas: IVF coseno 3.3 s / 1.5 ms / 0.924 y HNSW coseno 100 s / 2.3 ms / 0.985.

![tiempo de construcción](graficas/construccion_vs_n.png)
![tiempo por consulta](graficas/consulta_vs_n.png)
![recall@10](graficas/recall_vs_n.png)

El espacio crece lineal con N y es casi igual en todos: lo que pesa son los vectores
(19 800 × 1 024 floats ≈ 77 MB). Las celdas del IVF suman ~0.7 MB y el grafo del HNSW ~4 MB.

![espacio](graficas/espacio_vs_n.png)

¿Sale el original en el top 10? (portadas con 19 800, audio con 10 000)

| consulta | IVF coseno | IVF euclidiana | HNSW coseno | HNSW euclidiana |
|---|---|---|---|---|
| portada recortada 70 % | .63 | .64 | .65 | .49 |
| portada oscurecida 60 % | .88 | .90 | .90 | .82 |
| portada jpeg + 200 px | .39 | .23 | .38 | .18 |
| audio fragmento 5 s | .79 | .86 | .85 | .81 |
| audio 5 s + ruido | .34 | .36 | .38 | .31 |

Con el doble de portadas hay más candidatas parecidas y acierta un poco menos (recortada .78 con
10k, .65 con 20k usando HNSW coseno).

![original en el top 10](graficas/calidad_original.png)

### Contra Faiss y hnswlib (coseno)

Las mismas estructuras en C++, con los mismos parámetros y 1 hilo:

| | construir 20k portadas (s) | consulta (ms) | recall@10 portadas 20k | recall@10 audio 10k |
|---|---|---|---|---|
| IVF nuestro | 10.4 | 2.6 | 0.941 | 0.922 |
| Faiss IVFFlat | 0.2 | 0.19 | 0.910 | 0.912 |
| HNSW nuestro | 175 | 1.8 | 0.981 | 0.987 |
| Faiss HNSWFlat | 4.6 | 0.15 | 0.970 | 0.984 |
| hnswlib | 36.6 | 0.92 | 0.966 | 0.978 |

![nuestro vs librerías](graficas/nuestro_vs_librerias.png)

El recall es el mismo o mejor: el algoritmo está bien implementado. La diferencia de tiempo es
de lenguaje (Python vs C++ con SIMD): ~40 veces al construir el HNSW y ~12 veces al consultar.

## Qué se ve

- **HNSW es el más rápido al consultar** y casi no crece con N: ~1.5 ms con 1k y ~1.8 ms con 20k,
  mientras la búsqueda lineal pasa de 0.7 a 14 ms. Lo caro es construirlo: ~3 min para 20k en
  Python puro, contra ~10 s del IVF.
- **IVF construye ~17 veces más rápido** pero con nprobe = 10 pierde más recall (0.92–0.94 con
  coseno). Subiendo nprobe llega a 0.96–0.98 y sigue siendo varias veces más rápido que lo lineal.

  ![recall vs tiempo](graficas/recall_vs_tiempo.png)

- **La euclidiana sobre vectores sin normalizar empeora con N.** Las normas van de 0 a 2.2
  (mediana 0.1), así que el vecino "más cercano" depende del tamaño del vector y no de lo que
  muestra. En el IVF, K-Means agrupa por tamaño: con 10k las 10 celdas más grandes tienen el 66 %
  de la base, y con nprobe = 10 termina revisando casi todo (16 ms, casi como lo lineal). Con 20k
  portadas el recall con euclidiana cae a 0.80–0.84 en nuestros índices y a 0.73–0.85 en las
  librerías, mientras con coseno se mantiene en 0.91–0.98.
- **Coseno encuentra mejor el original** (jpeg chico .38 vs .18 con HNSW): coseno ignora el
  tamaño del vector.
- Las consultas difíciles (jpeg chico, audio con ruido) fallan igual en los cuatro: el límite ahí
  es el descriptor y el codebook, no el índice.

## Galería: aciertos y errores

Consulta modificada a la izquierda y los 5 primeros resultados; borde verde = el original.
Arriba los aciertos, abajo los errores. Están las 8 combinaciones en `graficas/galeria_*.jpg`.

![galería portadas](graficas/galeria_sift_hnsw_coseno.jpg)
![galería audio](graficas/galeria_mfcc_hnsw_coseno.jpg)

## Recomendación

| Si… | Usar |
|---|---|
| se consulta mucho y la base cambia poco | **HNSW + coseno** (o vectores normalizados) |
| se reconstruye seguido o se carga mucha data nueva | **IVF + coseno**, con nprobe ≈ 20 % de nlist |
| la base es chica (< 1 000) | búsqueda lineal: ya tarda menos de 1 ms |
| se quiere euclidiana | normalizar antes; si no, el IVF degenera |
| hace falta velocidad real en producción | Faiss o hnswlib: mismo recall, 10–45 veces más rápido |

Por eso el motor (`storage/vector_index.py`) guarda los vectores normalizados: así un solo índice
sirve para las tres métricas del `WITH METRIC`.

## Imágenes en `graficas/`

| archivo | qué muestra |
|---|---|
| `construccion_vs_n.png` | tiempo de construcción vs tamaño |
| `consulta_vs_n.png` | tiempo por consulta vs tamaño, con la búsqueda lineal |
| `recall_vs_n.png` | recall@10 contra la búsqueda exacta |
| `espacio_vs_n.png` | espacio en disco |
| `recall_vs_tiempo.png` | curvas nprobe 1–50 y ef_search 10–160 con 10k |
| `nuestro_vs_librerias.png` | nuestro IVF y HNSW contra Faiss y hnswlib |
| `calidad_original.png` | si sale el original de una consulta modificada |
| `galeria_<tipo>_<índice>_<métrica>.jpg` | aciertos y errores con las portadas (verde = el original) |
| `tabla.md` | la tabla de arriba, con las librerías |
| `resultados.json` | todos los números |

El enunciado pide también 100 000 objetos. El dataset tiene 10 000 canciones y 20 005 portadas
(`covers.csv`), así que llega a 10k en audio y ~20k en imágenes. Si se bajan más portadas con
`scripts/fetch_covers.py --total 100000`, el script sigue solo a 50k y 100k.
