# Resultados del benchmark textual

Ground truth: un documento es relevante si contiene al menos 1 termino de una consulta de 1 palabra, 2 terminos de una de 3 palabras, o 3 terminos de una de 5+ palabras. El criterio se calcula una sola vez y se aplica igual a las tres tecnicas.

## Construccion, espacio y memoria

| N | TF-IDF + coseno | BM25 | GIN | Espacio TF-IDF/BM25 | Indice GIN | Tabla GIN | Total GIN | Memoria SPIMI | Memoria TF-IDF | Memoria BM25 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 1303.5 ms | 1372.6 ms | 805.8 ms | 1836.5 KB | 1328.0 KB | 4632.0 KB | 5960.0 KB | 7909.2 KB | 2290.8 KB | 2290.8 KB |
| 2500 | 2408.6 ms | 2332.5 ms | 1657.6 ms | 4043.1 KB | 2616.0 KB | 10256.0 KB | 12872.0 KB | 16632.7 KB | 4623.9 KB | 4623.9 KB |
| 5000 | 4505.6 ms | 4375.5 ms | 2792.4 ms | 7628.0 KB | 4608.0 KB | 19464.0 KB | 24072.0 KB | 29882.7 KB | 7939.6 KB | 7939.6 KB |
| 7500 | 6984.1 ms | 6525.2 ms | 3595.6 ms | 11323.0 KB | 6616.0 KB | 28968.0 KB | 35584.0 KB | 44206.2 KB | 10036.1 KB | 10036.1 KB |
| 10000 | 7723.3 ms | 7187.5 ms | 5911.8 ms | 14876.1 KB | 8424.0 KB | 38144.0 KB | 46568.0 KB | 56835.0 KB | 11486.3 KB | 11486.3 KB |

## Tiempo de consulta, Precision@10 y Recall@10

| N | Consulta | Tecnica | ms | p95 ms | P@10 | R@10 |
|---:|---|---|---:|---:|---:|---:|
| 1000 | love | TF-IDF + coseno | 1.225 | 1.313 | 1.000 | 0.022 |
| 1000 | love | BM25 | 1.093 | 1.571 | 1.000 | 0.022 |
| 1000 | love | GIN (PostgreSQL) | 5.136 | 5.601 | 1.000 | 0.022 |
| 1000 | fire | TF-IDF + coseno | 0.158 | 0.191 | 1.000 | 0.175 |
| 1000 | fire | BM25 | 0.106 | 0.126 | 1.000 | 0.175 |
| 1000 | fire | GIN (PostgreSQL) | 0.743 | 0.999 | 1.000 | 0.175 |
| 1000 | thunder | TF-IDF + coseno | 0.041 | 0.045 | 0.500 | 1.000 |
| 1000 | thunder | BM25 | 0.024 | 0.027 | 0.500 | 1.000 |
| 1000 | thunder | GIN (PostgreSQL) | 0.226 | 0.241 | 0.500 | 1.000 |
| 1000 | midnight | TF-IDF + coseno | 0.072 | 0.085 | 1.000 | 0.625 |
| 1000 | midnight | BM25 | 0.046 | 0.048 | 1.000 | 0.625 |
| 1000 | midnight | GIN (PostgreSQL) | 0.344 | 0.431 | 1.000 | 0.625 |
| 1000 | love heart baby | TF-IDF + coseno | 2.032 | 2.260 | 0.900 | 0.032 |
| 1000 | love heart baby | BM25 | 1.771 | 1.900 | 1.000 | 0.036 |
| 1000 | love heart baby | GIN (PostgreSQL) | 7.954 | 9.309 | 1.000 | 0.036 |
| 1000 | fire water soul | TF-IDF + coseno | 0.460 | 0.549 | 0.100 | 0.077 |
| 1000 | fire water soul | BM25 | 0.457 | 0.878 | 0.800 | 0.615 |
| 1000 | fire water soul | GIN (PostgreSQL) | 1.740 | 1.868 | 0.100 | 0.077 |
| 1000 | rain dream winter | TF-IDF + coseno | 0.402 | 0.430 | 0.000 | 0.000 |
| 1000 | rain dream winter | BM25 | 0.296 | 0.311 | 0.400 | 1.000 |
| 1000 | rain dream winter | GIN (PostgreSQL) | 1.315 | 1.552 | 0.100 | 0.250 |
| 1000 | love heart baby night fire | TF-IDF + coseno | 2.766 | 3.321 | 0.500 | 0.035 |
| 1000 | love heart baby night fire | BM25 | 2.219 | 2.782 | 1.000 | 0.071 |
| 1000 | love heart baby night fire | GIN (PostgreSQL) | 12.270 | 15.079 | 0.800 | 0.057 |
| 1000 | love time never without your | TF-IDF + coseno | 3.391 | 4.330 | 0.700 | 0.024 |
| 1000 | love time never without your | BM25 | 2.798 | 3.465 | 1.000 | 0.034 |
| 1000 | love time never without your | GIN (PostgreSQL) | 17.670 | 20.886 | 0.700 | 0.024 |
| 1000 | heart baby night rain dream | TF-IDF + coseno | 2.614 | 5.559 | 0.100 | 0.023 |
| 1000 | heart baby night rain dream | BM25 | 1.457 | 1.947 | 0.600 | 0.136 |
| 1000 | heart baby night rain dream | GIN (PostgreSQL) | 11.460 | 12.456 | 0.300 | 0.068 |
| 2500 | love | TF-IDF + coseno | 1.845 | 2.333 | 1.000 | 0.010 |
| 2500 | love | BM25 | 1.917 | 2.756 | 1.000 | 0.010 |
| 2500 | love | GIN (PostgreSQL) | 6.354 | 8.149 | 1.000 | 0.010 |
| 2500 | fire | TF-IDF + coseno | 0.337 | 0.444 | 1.000 | 0.066 |
| 2500 | fire | BM25 | 0.209 | 0.221 | 1.000 | 0.066 |
| 2500 | fire | GIN (PostgreSQL) | 1.316 | 1.444 | 1.000 | 0.066 |
| 2500 | thunder | TF-IDF + coseno | 0.066 | 0.070 | 1.000 | 0.556 |
| 2500 | thunder | BM25 | 0.054 | 0.065 | 1.000 | 0.556 |
| 2500 | thunder | GIN (PostgreSQL) | 0.792 | 1.356 | 1.000 | 0.556 |
| 2500 | midnight | TF-IDF + coseno | 0.087 | 0.089 | 1.000 | 0.333 |
| 2500 | midnight | BM25 | 0.069 | 0.090 | 1.000 | 0.333 |
| 2500 | midnight | GIN (PostgreSQL) | 0.609 | 0.683 | 1.000 | 0.333 |
| 2500 | love heart baby | TF-IDF + coseno | 5.378 | 7.430 | 0.900 | 0.015 |
| 2500 | love heart baby | BM25 | 3.537 | 5.990 | 1.000 | 0.016 |
| 2500 | love heart baby | GIN (PostgreSQL) | 19.734 | 22.010 | 0.800 | 0.013 |
| 2500 | fire water soul | TF-IDF + coseno | 0.977 | 1.110 | 0.100 | 0.037 |
| 2500 | fire water soul | BM25 | 0.693 | 1.017 | 1.000 | 0.370 |
| 2500 | fire water soul | GIN (PostgreSQL) | 5.243 | 6.951 | 0.000 | 0.000 |
| 2500 | rain dream winter | TF-IDF + coseno | 0.598 | 0.806 | 0.100 | 0.125 |
| 2500 | rain dream winter | BM25 | 0.381 | 0.443 | 0.700 | 0.875 |
| 2500 | rain dream winter | GIN (PostgreSQL) | 4.242 | 4.690 | 0.000 | 0.000 |
| 2500 | love heart baby night fire | TF-IDF + coseno | 5.777 | 8.682 | 0.800 | 0.027 |
| 2500 | love heart baby night fire | BM25 | 3.811 | 4.500 | 1.000 | 0.034 |
| 2500 | love heart baby night fire | GIN (PostgreSQL) | 30.375 | 33.271 | 0.400 | 0.014 |
| 2500 | love time never without your | TF-IDF + coseno | 7.292 | 10.535 | 0.500 | 0.008 |
| 2500 | love time never without your | BM25 | 6.637 | 8.056 | 1.000 | 0.016 |
| 2500 | love time never without your | GIN (PostgreSQL) | 36.986 | 38.095 | 0.500 | 0.008 |
| 2500 | heart baby night rain dream | TF-IDF + coseno | 3.738 | 4.368 | 0.200 | 0.023 |
| 2500 | heart baby night rain dream | BM25 | 2.948 | 3.694 | 1.000 | 0.114 |
| 2500 | heart baby night rain dream | GIN (PostgreSQL) | 25.016 | 26.226 | 0.100 | 0.011 |
| 5000 | love | TF-IDF + coseno | 3.621 | 4.531 | 1.000 | 0.006 |
| 5000 | love | BM25 | 2.695 | 2.992 | 1.000 | 0.006 |
| 5000 | love | GIN (PostgreSQL) | 16.098 | 17.807 | 1.000 | 0.006 |
| 5000 | fire | TF-IDF + coseno | 0.905 | 1.327 | 1.000 | 0.033 |
| 5000 | fire | BM25 | 1.010 | 1.569 | 1.000 | 0.033 |
| 5000 | fire | GIN (PostgreSQL) | 3.076 | 4.302 | 1.000 | 0.033 |
| 5000 | thunder | TF-IDF + coseno | 0.155 | 0.177 | 1.000 | 0.294 |
| 5000 | thunder | BM25 | 0.119 | 0.129 | 1.000 | 0.294 |
| 5000 | thunder | GIN (PostgreSQL) | 0.687 | 0.879 | 1.000 | 0.294 |
| 5000 | midnight | TF-IDF + coseno | 0.206 | 0.234 | 1.000 | 0.196 |
| 5000 | midnight | BM25 | 0.164 | 0.192 | 1.000 | 0.196 |
| 5000 | midnight | GIN (PostgreSQL) | 0.935 | 1.288 | 1.000 | 0.196 |
| 5000 | love heart baby | TF-IDF + coseno | 9.185 | 11.219 | 0.900 | 0.009 |
| 5000 | love heart baby | BM25 | 7.825 | 10.864 | 1.000 | 0.009 |
| 5000 | love heart baby | GIN (PostgreSQL) | 42.688 | 44.178 | 0.700 | 0.007 |
| 5000 | fire water soul | TF-IDF + coseno | 1.348 | 1.386 | 0.100 | 0.015 |
| 5000 | fire water soul | BM25 | 2.295 | 4.638 | 1.000 | 0.147 |
| 5000 | fire water soul | GIN (PostgreSQL) | 10.243 | 11.819 | 0.000 | 0.000 |
| 5000 | rain dream winter | TF-IDF + coseno | 1.327 | 1.590 | 0.000 | 0.000 |
| 5000 | rain dream winter | BM25 | 1.133 | 1.240 | 0.800 | 0.400 |
| 5000 | rain dream winter | GIN (PostgreSQL) | 6.141 | 7.395 | 0.000 | 0.000 |
| 5000 | love heart baby night fire | TF-IDF + coseno | 7.690 | 9.190 | 0.600 | 0.012 |
| 5000 | love heart baby night fire | BM25 | 6.292 | 7.002 | 1.000 | 0.020 |
| 5000 | love heart baby night fire | GIN (PostgreSQL) | 61.353 | 65.255 | 0.500 | 0.010 |
| 5000 | love time never without your | TF-IDF + coseno | 14.550 | 17.025 | 0.500 | 0.004 |
| 5000 | love time never without your | BM25 | 11.512 | 13.563 | 1.000 | 0.009 |
| 5000 | love time never without your | GIN (PostgreSQL) | 99.941 | 124.726 | 0.500 | 0.004 |
| 5000 | heart baby night rain dream | TF-IDF + coseno | 6.804 | 8.610 | 0.200 | 0.013 |
| 5000 | heart baby night rain dream | BM25 | 5.922 | 7.997 | 0.900 | 0.056 |
| 5000 | heart baby night rain dream | GIN (PostgreSQL) | 55.261 | 58.800 | 0.100 | 0.006 |
| 7500 | love | TF-IDF + coseno | 6.691 | 7.366 | 1.000 | 0.004 |
| 7500 | love | BM25 | 4.736 | 5.329 | 1.000 | 0.004 |
| 7500 | love | GIN (PostgreSQL) | 13.983 | 15.065 | 1.000 | 0.004 |
| 7500 | fire | TF-IDF + coseno | 0.832 | 0.890 | 1.000 | 0.022 |
| 7500 | fire | BM25 | 0.740 | 0.864 | 1.000 | 0.022 |
| 7500 | fire | GIN (PostgreSQL) | 3.079 | 3.446 | 1.000 | 0.022 |
| 7500 | thunder | TF-IDF + coseno | 0.218 | 0.362 | 1.000 | 0.213 |
| 7500 | thunder | BM25 | 0.083 | 0.102 | 1.000 | 0.213 |
| 7500 | thunder | GIN (PostgreSQL) | 0.656 | 0.739 | 1.000 | 0.213 |
| 7500 | midnight | TF-IDF + coseno | 0.171 | 0.243 | 1.000 | 0.137 |
| 7500 | midnight | BM25 | 0.104 | 0.118 | 1.000 | 0.137 |
| 7500 | midnight | GIN (PostgreSQL) | 0.742 | 0.820 | 1.000 | 0.137 |
| 7500 | love heart baby | TF-IDF + coseno | 11.180 | 14.305 | 0.900 | 0.006 |
| 7500 | love heart baby | BM25 | 9.482 | 11.553 | 1.000 | 0.007 |
| 7500 | love heart baby | GIN (PostgreSQL) | 41.398 | 43.746 | 0.700 | 0.005 |
| 7500 | fire water soul | TF-IDF + coseno | 2.311 | 4.489 | 0.400 | 0.038 |
| 7500 | fire water soul | BM25 | 2.153 | 2.950 | 1.000 | 0.094 |
| 7500 | fire water soul | GIN (PostgreSQL) | 12.661 | 14.416 | 0.200 | 0.019 |
| 7500 | rain dream winter | TF-IDF + coseno | 2.049 | 2.954 | 0.100 | 0.034 |
| 7500 | rain dream winter | BM25 | 1.420 | 1.535 | 0.800 | 0.276 |
| 7500 | rain dream winter | GIN (PostgreSQL) | 8.352 | 8.772 | 0.000 | 0.000 |
| 7500 | love heart baby night fire | TF-IDF + coseno | 14.136 | 16.089 | 0.500 | 0.007 |
| 7500 | love heart baby night fire | BM25 | 9.362 | 13.937 | 1.000 | 0.015 |
| 7500 | love heart baby night fire | GIN (PostgreSQL) | 72.045 | 80.322 | 0.500 | 0.007 |
| 7500 | love time never without your | TF-IDF + coseno | 16.322 | 21.442 | 0.600 | 0.004 |
| 7500 | love time never without your | BM25 | 13.823 | 14.851 | 1.000 | 0.006 |
| 7500 | love time never without your | GIN (PostgreSQL) | 98.276 | 127.129 | 0.500 | 0.003 |
| 7500 | heart baby night rain dream | TF-IDF + coseno | 8.785 | 10.563 | 0.200 | 0.010 |
| 7500 | heart baby night rain dream | BM25 | 4.827 | 6.413 | 0.900 | 0.043 |
| 7500 | heart baby night rain dream | GIN (PostgreSQL) | 55.257 | 59.832 | 0.100 | 0.005 |
| 10000 | love | TF-IDF + coseno | 6.755 | 8.408 | 1.000 | 0.003 |
| 10000 | love | BM25 | 5.616 | 6.944 | 1.000 | 0.003 |
| 10000 | love | GIN (PostgreSQL) | 23.923 | 27.930 | 1.000 | 0.003 |
| 10000 | fire | TF-IDF + coseno | 1.038 | 1.148 | 1.000 | 0.017 |
| 10000 | fire | BM25 | 1.228 | 2.255 | 1.000 | 0.017 |
| 10000 | fire | GIN (PostgreSQL) | 5.328 | 8.770 | 1.000 | 0.017 |
| 10000 | thunder | TF-IDF + coseno | 0.136 | 0.144 | 1.000 | 0.159 |
| 10000 | thunder | BM25 | 0.094 | 0.132 | 1.000 | 0.159 |
| 10000 | thunder | GIN (PostgreSQL) | 1.080 | 1.478 | 1.000 | 0.159 |
| 10000 | midnight | TF-IDF + coseno | 0.131 | 0.174 | 1.000 | 0.119 |
| 10000 | midnight | BM25 | 0.093 | 0.097 | 1.000 | 0.119 |
| 10000 | midnight | GIN (PostgreSQL) | 0.748 | 1.012 | 1.000 | 0.119 |
| 10000 | love heart baby | TF-IDF + coseno | 13.578 | 15.697 | 0.900 | 0.005 |
| 10000 | love heart baby | BM25 | 11.554 | 13.736 | 1.000 | 0.005 |
| 10000 | love heart baby | GIN (PostgreSQL) | 86.002 | 115.752 | 0.700 | 0.004 |
| 10000 | fire water soul | TF-IDF + coseno | 3.148 | 3.835 | 0.500 | 0.035 |
| 10000 | fire water soul | BM25 | 2.903 | 3.309 | 1.000 | 0.071 |
| 10000 | fire water soul | GIN (PostgreSQL) | 17.302 | 20.724 | 0.200 | 0.014 |
| 10000 | rain dream winter | TF-IDF + coseno | 2.238 | 2.856 | 0.000 | 0.000 |
| 10000 | rain dream winter | BM25 | 1.581 | 1.945 | 0.800 | 0.235 |
| 10000 | rain dream winter | GIN (PostgreSQL) | 11.373 | 14.030 | 0.000 | 0.000 |
| 10000 | love heart baby night fire | TF-IDF + coseno | 16.094 | 18.212 | 0.500 | 0.006 |
| 10000 | love heart baby night fire | BM25 | 12.525 | 13.365 | 1.000 | 0.012 |
| 10000 | love heart baby night fire | GIN (PostgreSQL) | 150.935 | 185.606 | 0.500 | 0.006 |
| 10000 | love time never without your | TF-IDF + coseno | 26.796 | 30.739 | 0.600 | 0.003 |
| 10000 | love time never without your | BM25 | 19.760 | 21.232 | 1.000 | 0.005 |
| 10000 | love time never without your | GIN (PostgreSQL) | 153.841 | 182.461 | 0.400 | 0.002 |
| 10000 | heart baby night rain dream | TF-IDF + coseno | 12.323 | 14.890 | 0.300 | 0.011 |
| 10000 | heart baby night rain dream | BM25 | 8.038 | 10.969 | 0.900 | 0.034 |
| 10000 | heart baby night rain dream | GIN (PostgreSQL) | 134.316 | 163.683 | 0.100 | 0.004 |

## Ventajas y desventajas

| Tecnica | Ventajas | Desventajas | Cuando conviene |
|---|---|---|---|
| TF-IDF + coseno | Simple e interpretable; el idf premia terminos raros; coseno compensa diferencias de longitud; no necesita parametros | No satura la frecuencia como BM25; un termino presente en todos los documentos tiene idf 0; sensible a la tokenizacion | Documentos cortos, baseline explicable y comparaciones didacticas |
| BM25 | Satura la frecuencia y normaliza por longitud; suele rankear mejor texto heterogeneo; idf siempre positivo | Requiere calibrar `k1` y `b`; necesita la longitud media del corpus | Ranking textual principal con letras de distinta extension |
| GIN (PostgreSQL) | Indice maduro y muy optimizado; sintaxis SQL; integrado con transacciones, concurrencia y mantenimiento | Depende de PostgreSQL; el ranking con `ts_rank_cd` no equivale a BM25; mayor costo de infraestructura | Produccion y baseline externo, no como sustituto del indice propio |

