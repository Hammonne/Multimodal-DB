| tipo | índice | métrica | construir (s) | consulta (ms) | recall@10 | espacio (KB) |
|---|---|---|---|---|---|---|
| sift | lineal | coseno | 0.0 | 9.75 | 1.000 | 40000 |
| sift | ivf | coseno | 3.7 | 1.57 | 0.903 | 40481 |
| sift | hnsw | coseno | 91.9 | 1.47 | 0.980 | 42298 |
| sift | faiss_ivf | coseno | 0.1 | 0.15 | 0.915 | 40479 |
| sift | faiss_hnsw | coseno | 2.0 | 0.13 | 0.981 | 41407 |
| sift | hnswlib | coseno | 16.6 | 0.88 | 0.978 | 41454 |
| sift | lineal | euclidiana | 0.0 | 24.24 | 1.000 | 40000 |
| sift | ivf | euclidiana | 4.2 | 10.72 | 0.966 | 40481 |
| sift | hnsw | euclidiana | 116.0 | 2.04 | 0.961 | 42062 |
| sift | faiss_ivf | euclidiana | 0.1 | 0.42 | 0.962 | 40479 |
| sift | faiss_hnsw | euclidiana | 0.9 | 0.06 | 0.844 | 41407 |
| sift | hnswlib | euclidiana | 8.7 | 0.40 | 0.858 | 41454 |
| mfcc | lineal | coseno | 0.0 | 10.12 | 1.000 | 40000 |
| mfcc | ivf | coseno | 3.5 | 1.69 | 0.922 | 40481 |
| mfcc | hnsw | coseno | 82.2 | 1.66 | 0.987 | 42320 |
| mfcc | faiss_ivf | coseno | 0.1 | 0.13 | 0.912 | 40479 |
| mfcc | faiss_hnsw | coseno | 1.7 | 0.12 | 0.984 | 41407 |
| mfcc | hnswlib | coseno | 15.2 | 0.86 | 0.978 | 41454 |
| mfcc | lineal | euclidiana | 0.0 | 23.24 | 1.000 | 40000 |
| mfcc | ivf | euclidiana | 3.3 | 20.00 | 0.998 | 40481 |
| mfcc | hnsw | euclidiana | 143.5 | 1.99 | 0.983 | 42128 |
| mfcc | faiss_ivf | euclidiana | 0.1 | 0.38 | 0.980 | 40479 |
| mfcc | faiss_hnsw | euclidiana | 1.3 | 0.09 | 0.913 | 41407 |
| mfcc | hnswlib | euclidiana | 8.3 | 0.51 | 0.921 | 41454 |
