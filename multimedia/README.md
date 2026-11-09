# Multimedia

Parte 4 (§2.4.1): `ExtractFeatures(objeto)` convierte una imagen o un audio en un vector de dimensión K
para buscar contenido parecido. Es la misma idea que en texto, pero aquí no hay palabras: hay píxeles y
muestras de audio. Entonces las palabras se fabrican. De cada imagen o audio se sacan muchos vectores
chiquitos (descriptores locales), se agrupan todos los de la colección con K-Means y cada grupo pasa a ser
una "palabra". Así cada objeto queda como un histograma de cuántas veces aparece cada palabra, igual que
un documento. A esto se le llama *bag of visual words* (o *audio words*)

![k-means: cada grupo de descriptores es una palabra](../docs/img/parte4/kmeans_palabras.png)

```
archivo ─► descriptores locales ─► palabra más cercana ─► histograma tf ─► × idf ─► vector TF-IDF (K = 1024)
           SIFT (128) / MFCC (39)    codebook K-Means                                  │
                                                                                      ▼
                                                              IVF / HNSW (indexes/) ─► SIMILAR_TO en SQL
```

| Texto (Parte 3) | Multimedia (Parte 4) |
|---|---|
| documento | imagen o audio |
| token (palabra después de stemming) | descriptor local: punto SIFT o ventana MFCC |
| vocabulario | codebook: K centroides de K-Means |
| tf: veces que aparece la palabra | tf: fracción de descriptores que cae en cada centroide |
| idf | igual: `log(N / df)` |
| índice invertido SPIMI | índice vectorial IVF o HNSW |
| `MATCH(...)` | `SIMILAR_TO(...)` |

Todo corre en el entorno `vision` y guarda en `data/features/`, que no se sube al repo.
Para MFCC instala primero las dependencias del proyecto:

```bash
python -m pip install -r requirements.txt
```

Los pasos, en orden (todos se pueden cortar y volver a correr, siguen donde se quedaron):

```bash
conda activate vision
python -m multimedia.sift --source covers                 # 1. descriptores de las portadas
python -m multimedia.mfcc                                 # 1. descriptores de los audios
python -m multimedia.codebook --tipo ambos --ks 1024      # 2. codebook con K-Means
python -m multimedia.features --tipo ambos --k 1024       # 3. vectores TF-IDF de toda la colección
```

Para una demostración rápida con pocas portadas:

```bash
python scripts/prepare_multimedia_demo.py --tipo sift --limite 50 --k 64
```

La web detecta automáticamente la base `tfidf_sift_k64.npz` si todavía no
existe la base grande de K=1024.

Para preparar audio rápidamente:

```bash
python scripts/prepare_multimedia_demo.py --tipo mfcc --limite 50 --k 64
```

Si no está instalado `librosa`, el comando se detiene indicando la instalación
pendiente; no genera un índice vacío.

Los archivos de `data/` cumplen roles distintos:

| Carpeta/archivo | Uso |
|---|---|
| `songs.csv` | catálogo de canciones y rutas |
| `audio/` | archivos usados por MFCC |
| `covers/` | portadas usadas por SIFT |
| `artist_images.csv` y `artists/` | catálogo/fotos de artistas; no son la colección SIFT por defecto |
| `features/` | descriptores, codebooks y vectores que consumen IVF/HNSW |

No se insertan los binarios en una tabla SQL. Una tabla SQL puede guardar
metadatos y rutas, pero la búsqueda multimedia usa los artefactos generados en
`data/features/`.

| Paso | Salida en `data/features/` | Tamaño con la data completa |
|---|---|---|
| SIFT | `sift/<ruta>.npy` + `sift_covers.csv` | 20 005 portadas, ~457 descriptores c/u (9.1 M) |
| MFCC | `mfcc/<track_id>.npy` + `mfcc.csv` | 13 643 audios, ~1 290 ventanas c/u (12.9 M con 10k) |
| Codebook | `codebook/<tipo>_k1024.npz` (`centroides, media, desv`) | 1024 × 128 y 1024 × 39 |
| TF-IDF | `tfidf_<tipo>_k1024.npz` (`ids, tf, idf, vectores`) | 20 005 × 1024 (sift), 10 000 × 1024 (mfcc) |

## 1. Descriptores locales

| | SIFT (imágenes) | MFCC (audio) |
|---|---|---|
| Archivo | `sift.py` | `mfcc.py` |
| Pedazo | punto clave (esquina, borde, textura) | ventana de 93 ms que avanza 23 ms |
| Vectores por objeto | ~457 (máx. 500) | ~1 290 por preview de 30 s |
| Dimensión | 128 | 39 |
| Aguanta | escala, rotación, brillo | volumen |
| No ve | color | melodía |

### SIFT

*Scale-Invariant Feature Transform*. Busca puntos de la imagen que se pueden volver a encontrar aunque
la imagen esté más chica, rotada o más oscura (esquinas, manchas, bordes) y describe la zona alrededor
de cada uno con 128 números. Cada círculo es un punto clave: su tamaño es la escala y la raya su orientación

![puntos SIFT en una portada](../docs/img/parte4/sift_puntos.jpg)

**Cómo encuentra puntos a distintas escalas.** La imagen se desenfoca con gaussianas cada vez más anchas
(σ, kσ, k²σ...), mientras más se desenfoca es como verla más de lejos. Cada vez que σ se duplica la imagen
se reduce a la mitad (una octava) y así se arma una pirámide. Luego se restan imágenes vecinas:
`D = L(kσ) − L(σ)`. Esa resta de gaussianas (DoG) da valores muy altos o muy bajos justo donde hay una
mancha del tamaño de σ

![restar dos gaussianas da un detector de manchas](../docs/img/parte4/dog_curvas.png)
![DoG en una portada](../docs/img/parte4/dog_portada.png)

**El descriptor.** Alrededor del punto (ya girado a su orientación) se toma una ventana de 16 × 16, se parte
en 4 × 4 celdas y en cada una se hace un histograma de 8 direcciones del gradiente (hacia dónde cambia el
brillo): 4 × 4 × 8 = 128

![descriptor 4 x 4 x 8](../docs/img/parte4/sift_descriptor.png)

Detalles:
- Detector: busca extremos en la pirámide de diferencias de gaussianas (DoG), cada punto es más
  alto o más bajo que sus 26 vecinos (8 en su escala, 9 arriba, 9 abajo). Eso da la posición y el tamaño
- Se descartan los de poco contraste y se calcula la orientación dominante (histograma de 36 ángulos
  del gradiente), por eso aguanta rotaciones
- Descriptor: ventana de 16 × 16 alrededor del punto, 4 × 4 celdas con un histograma de 8 orientaciones
  cada una = 128 números
- La imagen se pasa a gris y se achica a 512 px de lado (`MAX_SIDE`), más grande es más lento y no da puntos mejores
- Se quedan los 500 puntos de mayor contraste (`N_FEATURES`), el 64 % de las portadas llega a ese tope
- RootSIFT: se normaliza L1 y se saca raíz, así la distancia euclidiana entre descriptores equivale a
  Hellinger, que compara mejor histogramas
- Portadas: las de `data/covers.csv` sin `duplicada_de`. Con `--source all` también artistas e
  integrantes (solo las fotos que aceptó el filtro de caras, para la Parte 5)
- 15 de las 20 005 portadas no tienen ningún punto clave: son de un color plano o texto y fondo de
  colores distintos pero mismo brillo (en gris quedan lisas). Su vector queda en cero. Para esas sirve
  más el histograma de color (abajo, en Extras)

![portadas sin puntos clave](../docs/img/parte4/sin_puntos.png)
![a color y en gris](../docs/img/parte4/sin_puntos_gris.png)

### MFCC

*Mel-Frequency Cepstral Coefficients*, es como SIFT pero para audio. En vez de puntos clave, la canción
se parte en ventanas muy cortas (en 93 ms el sonido casi no cambia) y cada ventana se describe con pocos
números que resumen el timbre, o sea cómo suena: voz, bajo, platillos. Se propuso en 1980 para
reconocimiento de voz y es el descriptor clásico de audio

![ventanas que se solapan](../docs/img/parte4/mfcc_ventanas.png)

Paso a paso: la onda → espectrograma (FFT por ventana) → espectrograma mel (40 bandas, como escucha el
oído: `mel(f) = 2595 · log10(1 + f / 700)`) → log → DCT, que deja los primeros coeficientes con la forma
general del espectro

![de la onda a los MFCC](../docs/img/parte4/mfcc_pasos.jpg)

Al final cada canción es una matriz de 39 filas (13 MFCC + 13 Δ + 13 ΔΔ) por ~1 290 ventanas

![matriz MFCC de una canción](../docs/img/parte4/mfcc_matriz.png)

Detalles:
- Audio mono a 22 050 Hz
- Ventana de 2048 muestras (93 ms) con avance de 512 (23 ms, 75 % de solapamiento) y ventana de Hann
- Por ventana: FFT, 40 bandas mel, log y DCT, se guardan los coeficientes 1 a 13
- El coeficiente 0 es el volumen, se descarta para que la misma canción más fuerte o más bajita dé lo mismo
- Se agregan delta y delta-delta (cuánto cambia el timbre entre ventanas): 13 + 13 + 13 = 39
- La fila `i` empieza en el segundo `i * 512 / 22050`, sirve para buscar por fragmentos
  (`extraer_mfcc(path, offset=10, duration=5)`)
- Antes del K-Means se estandariza cada dimensión (z-score), las deltas son mucho más chicas que los coeficientes

## 2. Codebook (K-Means)

Los millones de descriptores se agrupan en K centroides, cada centroide es una "palabra"

- `MiniBatchKMeans` de sklearn, k-means++, `n_init=3`, semilla 0
- Se entrena con una muestra de máximo 50 descriptores por objeto (tope 300 000), con las 10 000 canciones
- MFCC se estandariza antes y la media y la desviación se guardan con los centroides, para estandarizar
  igual las consultas. SIFT no (ya viene con RootSIFT)

### Por qué K = 1024

Se probó K de 64 a 4096 (`codebook-k-finding.ipynb`) con consultas de respuesta conocida: portadas
rotadas, recortadas, oscurecidas o comprimidas y fragmentos de 5 s de audio, con y sin ruido.
recall@10 = la original sale entre las 10 primeras (búsqueda exacta por coseno)

| K (10k) | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|
| portada recortada al 70 % | .74 | .81 | .87 | .92 |
| portada jpeg + 200 px | .30 | .45 | .59 | .71 |
| audio fragmento 5 s | .80 | .88 | .94 | .97 |
| audio 5 s + ruido | .54 | .62 | .69 | .73 |
| ms por consulta (sift / mfcc) | 1.1 / 0.8 | 2.0 / 1.6 | 3.8 / 3.4 | 8.1 / 5.9 |

- La inercia baja casi en recta con K (en escala log), no hay un codo claro, así que el método del codo no
  sirve para elegir. Davies-Bouldin también mejora siempre con K

![inercia vs K](../docs/img/parte4/codebook_inercia.png)
![Davies-Bouldin vs K](../docs/img/parte4/codebook_davies_bouldin.png)

- Lo que decide es el recall de las consultas modificadas contra el tiempo:

![recall@10 vs K](../docs/img/parte4/codebook_recall.png)

- Más K = palabras más específicas y mejor recall, pero vectores más grandes, consultas más lentas y
  palabras casi vacías (con 1k, en 4096 un tercio de las palabras tiene menos de 5 descriptores)
- 1024 es el punto medio: buen recall sin duplicar el tiempo por consulta. K también es la dimensión
  de los vectores que indexan IVF y HNSW

## 3. TF-IDF (`features.py`)

```
tf(w, d) = n(w, d) / Σ n(w', d)      idf(w) = log(N / df(w))      v_d = tf · idf / ‖tf · idf‖₂
```

- `palabras()`: cada descriptor va a su centroide más cercano (euclidiana, en lotes de 50 000)
- `histograma()`: tf = fracción de descriptores en cada palabra, así no importa cuántos puntos tenga el objeto
- `construir_base()`: tf de toda la colección, idf y vectores normalizados en `tfidf_<tipo>_k1024.npz`
- `ExtractFeatures(path)`: decide SIFT o MFCC por la extensión y usa el mismo codebook e idf de la base,
  así una consulta nueva queda en el mismo espacio. `normalizar=False` deja el vector sin normalizar
  (para comparar euclidiana y coseno en el benchmark)

```python
from multimedia.features import ExtractFeatures
v = ExtractFeatures("data/covers/110599312.jpg")   # np.ndarray (1024,), norma 1
```

Los índices y las métricas están en [`indexes/VECTOR_README.md`](../indexes/VECTOR_README.md) y la
comparación en [`benchmark/multimedia/README.md`](../benchmark/multimedia/README.md)

## Extras

### Histograma de color (`color.py`)
Descriptor global de las portadas, para lo que SIFT no ve. 12 tonos x 4 saturaciones + 2 grises (oscuro y claro) = 50 dimensiones, con raíz como RootSIFT

![histograma de color de una portada](../docs/img/parte4/histograma_color.png)

Casi no se mira el brillo: con 3 niveles de brillo por color una portada oscurecida al 60 % encontraba la original en el top 10 el 17 % de las veces, así el 91 %

Probado con 300 portadas modificadas buscando entre las 8 400:

| Cambio | Top 1 | Top 10 |
|---|---|---|
| Rotar 90° / 15° | 100 % / 99 % | 100 % |
| Recortar al 70 % | 86 % | 95 % |
| Oscurecer al 60 % | 84 % | 91 % |
| Aclarar x1.3 | 87 % | 94 % |
| JPEG comprimido + 200 px | 69 % | 87 % |

Solo encuentra colores parecidos, no la misma portada, por eso la idea es sumarlo a SIFT: `a * sim_sift + (1 - a) * sim_color`

### Embeddings de cara (`face_embeddings.py`)
Igual que en el Lab 11: ResNet de dlib vía `face_recognition`, 128 dimensiones, misma persona si la distancia es < 0.6. Usa la caja de la cara que guardó el filtro (`face_box`) así que no vuelve a detectar, y achica a 1000 px porque a resolución completa es ~10 veces más lento

Sale en `faces.npz` con `embedding, path, kind, owner_id, name`. Es un vector por cara, no necesita codebook. Sirve para buscar por artista (Parte 5)

### Búsqueda por cara con HNSW (`caras.py`)
A diferencia de las portadas y el audio, acá se usa un **descriptor global**: toda la cara es un solo vector de 128 dimensiones, como en el Lab 11. No hay descriptores locales, codebook ni TF-IDF: los embeddings de `faces.npz` entran directo a nuestro HNSW (`indexes/hnsw.py`) con distancia **euclidiana**, que es la que usa `face_recognition` para su umbral de 0.6

```bash
python -m multimedia.caras foto.jpg --k 5      # quién se parece a cada cara de la foto
python -m multimedia.caras --evaluar           # mide el índice con caras que ya están en la base
```

- El índice se guarda en `data/features/faces_hnsw.npz` la primera vez y después solo se carga
- Una persona puede tener varias fotos: el resultado deja una fila por persona, con su cara más cercana, y marca `misma_persona` si la distancia es < 0.45 (ver abajo por qué no 0.6)
- Si la foto tiene varias caras (una banda), busca cada una por separado
- La primera vez construye el índice (~50 s); después cada búsqueda tarda ~1 s, casi todo en detectar la cara y sacar su embedding

```
$ python -m multimedia.caras data/artists/lady-gaga_75491/main.jpg --k 3
cara 1 (290, 675, 675, 290):
  0.000  Lady Gaga (artist)  artists/lady-gaga_75491/main.jpg  misma persona
  0.537  Cat Power (artist)  artists/cat-power_1039/extra_05.jpg
  0.541  Mariah Carey (artist)  artists/mariah-carey_65/main.jpg
```

**Cómo le va** (`--evaluar`): 8 937 caras de 2 693 personas (artistas e integrantes de bandas). Se sacan de la base 300 caras de personas que tienen 2 fotos o más y se buscan; acierta si la cara más cercana es de la misma persona, en otra foto.

| | |
|---|---|
| Construir el HNSW (8 637 caras) | 50 s |
| Consulta HNSW vs lineal | 1.05 ms vs 2.30 ms (2.2× más rápido) |
| Recall@10 contra la lineal | 0.995 |
| La persona correcta sale primera | 79 % |
| La persona correcta está entre las 5 caras más cercanas | 92 % |

**El umbral de 0.6 no sirve con tantas personas.** `face_recognition` propone 0.6 y en el Lab 11 funcionaba, pero acá la cara más cercana queda bajo 0.6 en el 100 % de las consultas, también cuando es de otra persona. Con la búsqueda exacta:

| umbral | responde "misma persona" en | y acierta |
|---|---|---|
| 0.6 | 100 % | 79 % |
| 0.5 | 94 % | 83 % |
| **0.45** | **70 %** | **90 %** |
| 0.4 | 48 % | 92 % |

La mediana de la distancia es 0.39 cuando acierta y 0.46 cuando se equivoca. Se usa 0.45: responde en 7 de cada 10 consultas y acierta 9 de cada 10.

Con ~9 000 caras de 128 dimensiones la lineal es rápida (2 ms), así que el HNSW gana poco; la diferencia crecería con más artistas.

En la web se usa en la app (`/musica` → Reconocer → Cara, también con la cámara en vivo) y en la página
`/como-funciona`, que muestra la detección con los 68 puntos, el embedding y el histograma de distancias.

## Referencias

- Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. *International Journal of Computer Vision*, 60(2), 91–110. (SIFT)
- Arandjelović, R. y Zisserman, A. (2012). Three things everyone should know to improve object retrieval. *CVPR 2012*, 2911–2918. (RootSIFT)
- Sivic, J. y Zisserman, A. (2003). Video Google: a text retrieval approach to object matching in videos. *ICCV 2003*, 1470–1477. (bag of visual words)
- Davis, S. y Mermelstein, P. (1980). Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 28(4), 357–366. (MFCC)
- Lloyd, S. (1982). Least squares quantization in PCM. *IEEE Transactions on Information Theory*, 28(2), 129–137. (K-Means)
- Arthur, D. y Vassilvitskii, S. (2007). k-means++: the advantages of careful seeding. *SODA 2007*, 1027–1035.
- Sculley, D. (2010). Web-scale k-means clustering. *WWW 2010*, 1177–1178. (Mini-batch K-Means)
- Salton, G. y Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management*, 24(5), 513–523. (TF-IDF)
- Bradski, G. (2000). The OpenCV Library. *Dr. Dobb's Journal of Software Tools*.
- McFee, B. et al. (2015). librosa: audio and music signal analysis in Python. *Proceedings of the 14th Python in Science Conference*, 18–25.
- Pedregosa, F. et al. (2011). Scikit-learn: machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
