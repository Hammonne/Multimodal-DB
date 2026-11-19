| tipo | índice | métrica | construir (s) | consulta (ms) | recall@10 | espacio (KB) |
|---|---|---|---|---|---|---|
| sift | lineal | coseno | 0.0 | 14.42 | 1.000 | 79200 |
| sift | ivf | coseno | 11.0 | 2.60 | 0.941 | 79922 |
| sift | hnsw | coseno | 174.0 | 1.51 | 0.981 | 83713 |
| sift | faiss_ivf | coseno | 0.3 | 0.19 | 0.910 | 79920 |
| sift | faiss_hnsw | coseno | 6.0 | 0.19 | 0.970 | 81989 |
| sift | hnswlib | coseno | 38.5 | 0.96 | 0.966 | 82075 |
| sift | lineal | euclidiana | 0.0 | 61.07 | 1.000 | 79200 |
| sift | ivf | euclidiana | 13.3 | 19.03 | 0.839 | 79922 |
| sift | hnsw | euclidiana | 239.2 | 1.91 | 0.801 | 83202 |
| sift | faiss_ivf | euclidiana | 0.3 | 0.68 | 0.848 | 79920 |
| sift | faiss_hnsw | euclidiana | 3.0 | 0.09 | 0.748 | 81989 |
| sift | hnswlib | euclidiana | 19.1 | 0.47 | 0.733 | 82075 |
| mfcc | lineal | coseno | 0.0 | 7.70 | 1.000 | 40000 |
| mfcc | ivf | coseno | 3.9 | 1.80 | 0.922 | 40481 |
| mfcc | hnsw | coseno | 83.3 | 1.49 | 0.987 | 42320 |
| mfcc | faiss_ivf | coseno | 0.1 | 0.14 | 0.912 | 40479 |
| mfcc | faiss_hnsw | coseno | 2.0 | 0.12 | 0.984 | 41407 |
| mfcc | hnswlib | coseno | 16.1 | 0.87 | 0.978 | 41454 |
| mfcc | lineal | euclidiana | 0.0 | 23.07 | 1.000 | 40000 |
| mfcc | ivf | euclidiana | 4.6 | 19.58 | 0.998 | 40481 |
| mfcc | hnsw | euclidiana | 134.1 | 2.06 | 0.983 | 42128 |
| mfcc | faiss_ivf | euclidiana | 0.1 | 0.39 | 0.980 | 40479 |
| mfcc | faiss_hnsw | euclidiana | 1.3 | 0.10 | 0.913 | 41407 |
| mfcc | hnswlib | euclidiana | 8.4 | 0.56 | 0.921 | 41454 |
