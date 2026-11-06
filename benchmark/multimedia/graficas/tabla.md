| tipo | índice | métrica | construir (s) | consulta (ms) | recall@10 | espacio (KB) |
|---|---|---|---|---|---|---|
| sift | lineal | coseno | 0.0 | 14.14 | 1.000 | 79200 |
| sift | ivf | coseno | 10.4 | 2.61 | 0.941 | 79922 |
| sift | hnsw | coseno | 175.2 | 1.77 | 0.981 | 83713 |
| sift | faiss_ivf | coseno | 0.2 | 0.19 | 0.910 | 79920 |
| sift | faiss_hnsw | coseno | 4.6 | 0.15 | 0.970 | 81989 |
| sift | hnswlib | coseno | 36.6 | 0.92 | 0.966 | 82075 |
| sift | lineal | euclidiana | 0.0 | 52.14 | 1.000 | 79200 |
| sift | ivf | euclidiana | 7.9 | 16.00 | 0.839 | 79922 |
| sift | hnsw | euclidiana | 253.2 | 2.28 | 0.801 | 83202 |
| sift | faiss_ivf | euclidiana | 0.2 | 0.58 | 0.848 | 79920 |
| sift | faiss_hnsw | euclidiana | 2.9 | 0.09 | 0.748 | 81989 |
| sift | hnswlib | euclidiana | 18.4 | 0.45 | 0.733 | 82075 |
| mfcc | lineal | coseno | 0.0 | 7.18 | 1.000 | 40000 |
| mfcc | ivf | coseno | 3.2 | 1.48 | 0.922 | 40481 |
| mfcc | hnsw | coseno | 80.9 | 1.66 | 0.987 | 42320 |
| mfcc | faiss_ivf | coseno | 0.1 | 0.15 | 0.912 | 40479 |
| mfcc | faiss_hnsw | coseno | 1.7 | 0.12 | 0.984 | 41407 |
| mfcc | hnswlib | coseno | 15.1 | 0.86 | 0.978 | 41454 |
| mfcc | lineal | euclidiana | 0.0 | 23.27 | 1.000 | 40000 |
| mfcc | ivf | euclidiana | 3.6 | 19.26 | 0.998 | 40481 |
| mfcc | hnsw | euclidiana | 156.3 | 2.31 | 0.983 | 42128 |
| mfcc | faiss_ivf | euclidiana | 0.1 | 0.54 | 0.980 | 40479 |
| mfcc | faiss_hnsw | euclidiana | 1.2 | 0.10 | 0.913 | 41407 |
| mfcc | hnswlib | euclidiana | 8.2 | 0.51 | 0.921 | 41454 |
