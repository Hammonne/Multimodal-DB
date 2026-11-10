# Pendientes de la Parte 4 y para la Parte 5

Lo que quedó abierto al 9 de noviembre, para que el equipo pueda seguir. Cada punto dice dónde está
el código y qué falta.

> **Dos ramas.** Esta rama (`parte-4-correcciones`) tiene solo la Parte 4. Todo lo de la Parte 5 ya
> empezado (app `/musica`, caras en vivo, Shazam, tarareo y la página "Cómo funciona") está en la rama
> **`avances-y-correcciones`**, que se deja como base de la Parte 5 y **no se une a `main` todavía**.
> Los archivos marcados con *(rama avances)* solo existen allá.

## Parte 5: la aplicación (todo pendiente en esta rama)

Ya hecho en `avances-y-correcciones`, falta revisarlo y traerlo:

- **App de música** (`web/app/musica/` *(rama avances)*): colección, páginas de artista, grupos y
  "Reconocer" por portada, audio o cara, con el top k editable
- **Reconocimiento de caras en vivo** (`multimedia/caras.py` *(rama avances)*): desde una foto, la cámara
  o en vivo, con el top k del HNSW. El Docker del backend instala `face_recognition` con `dlib-bin`.
  Con ~2 700 personas, el umbral de 0.45 acierta el 90 % de las veces que responde
  (`python -m multimedia.caras --evaluar`)
- **Shazam** (`multimedia/huellas.py` *(rama avances)*): acierta el 99 % con audio grabado con celular.
  Falta bajar el espacio (guardar menos picos por segundo), guardar el índice en disco y llevarlo al
  SQL o a la web
- **Tarareo** (`multimedia/tarareo.py` *(rama avances)*): el híbrido (vector fijo con coseno → DTW sobre
  50 candidatos) funciona con tarareos simulados. Falta separar la voz de la mezcla (`demucs`), probar
  con gente cantando de verdad (dataset MIR-QBSH) y meter los vectores fijos en el HNSW. DTW no es una
  métrica: solo puede reordenar candidatos, nunca podar dentro del índice
- **"Cómo funciona"** (`web/app/como-funciona/` y `web/backend/explica.py` *(rama avances)*): explica
  SIFT, MFCC, IVF, HNSW y caras con nuestra data. Falta una sección de texto (ver abajo)
- Las mediciones de Shazam y tarareo están en `docs/INVESTIGACION_PARTE5.md` *(rama avances)*

## Columnas `IMAGE` y `AUDIO` (hecho en esta rama, queda por mejorar)

- **SQL de caras:** una columna de fotos de caras con `SIMILAR_TO`, con embeddings de cara en vez de SIFT
  (por ejemplo un tipo `FACE`, o `IMAGE` con `WITH MODEL = caras`)
- **Reentrenar el modelo:** hoy el codebook se entrena una vez al crear el índice. Si después se insertan
  muchos archivos distintos, el codebook queda viejo. Se podría reentrenar cuando la tabla crece, por
  ejemplo, al doble
- **Alias de tabla:** el parser no acepta `FROM fotos f`; hay que usar el nombre completo
- **Base vectorial frente a gestores reales:** ya se agregaron la búsqueda con filtro, `WITH NPROBE` /
  `EF_SEARCH` y `SIMILAR_TO` con `JOIN` y `GROUP BY`. Lo que falta, con cómo lo resuelven pgvector, Qdrant,
  Weaviate y Milvus, está en [`BASE_VECTORIAL.md`](BASE_VECTORIAL.md): compactar los borrados del índice,
  reentrenar el codebook, `SIMILAR_TO` en un `OR` o dos a la vez (búsqueda híbrida), buscar con un vector o
  una fila, y *embeddings* de redes neuronales (CLIP, PANNs) como modo extra
- **Filtro con índice:** para saber qué filas cumplen el filtro se recorre la tabla; se podría usar el B+ o el
  bitmap de la columna filtrada
- **Probar la consola web en Docker** subiendo un `.zip` (ver la sección SQL del README)

## Revisar en la Parte 3 (texto)

Los tests pasan, pero las fórmulas no coinciden con las de clase:

- **`text_based/tfidf.py`:** el peso es `tf / df`. El TF-IDF usual es `tf · log(N / df)`: el idf es
  un logaritmo de N sobre df, no una división por df
- **`text_based/bm25.py`:** el TF se calcula como `f / (f + k1 + norm)`. BM25 usa
  `f · (k1 + 1) / (f + k1 · norm)`, con `norm = 1 − b + b · |d| / avgdl`. El idf
  `log((N − df + 0.5) / (df + 0.5))` sale negativo para términos que están en más de la mitad de los
  documentos; muchas implementaciones usan `log(1 + (N − df + 0.5) / (df + 0.5))` para evitarlo

### Texto en "Cómo funciona"

Con el mismo estilo que las otras secciones de la página:

1. **Tokens:** una letra de canción real partida en palabras (minúsculas, sin signos), como hace `tokenize`
2. **SPIMI:** los documentos entrando en bloques que se escriben a disco cuando se llena la memoria, y la
   mezcla de los bloques en un índice invertido (término → lista de documentos con su frecuencia)
3. **Una consulta:** escribir unas palabras y ver la lista de postings de cada término, su df y su idf
4. **El puntaje:** para los primeros resultados, cuánto aporta cada término con TF-IDF y con BM25
   (tf, idf, largo del documento contra el promedio, k1 y b), y cómo cambia el ranking

Antes conviene corregir las fórmulas, para no explicar un cálculo que no es el de clase. El índice SPIMI
de las ~10 000 letras se arma en unos segundos.

## ¿Codebook de 2048 palabras? (experimento)

Hoy K = 1024 (`codebook-k-finding.ipynb`). Con 2048 el recall de las consultas modificadas sube (portada en
jpeg chico .45 → .59, audio de 5 s .88 → .94), pero cada consulta tarda el doble (2.0 → 3.8 ms en portadas).
Cambiarlo obliga a rehacer el codebook, los vectores, los índices, el benchmark y las colecciones de la web.

## Fotos reales de portadas

Hay fotos reales de 600 CDs, vinilos y casetes de Discogs (3 409 fotos, `scripts/fetch_fotos_reales.py`) y su
evaluación (`benchmark/multimedia/fotos_reales.py`): el 66 % encuentra la portada digital en el top 10
(~70 % en CDs y vinilos, 26 % en casetes). Se podría:

- Quitar a mano las fotos principales que en realidad son el disco o la contraportada, para que el
  número mida solo al buscador
- Sumar el histograma de color (`multimedia/color.py`) a SIFT, que no ve el color
- Bajar más: el script sigue donde se quedó (`--total 500`). Discogs sin token deja 25 pedidos por
  minuto y solo ~1 de cada 3 álbumes tiene edición física (más entre los populares: `--orden popular`)

## Más datos

- **Agregar artistas puntuales:** el dataset tomó ~200 artistas por género de las radios de Deezer, así que
  faltan algunos famosos (por ejemplo Kanye West, que solo aparece como colaborador). Habría que sumar a
  `scripts/fetch_music.py` una opción `--artista "Nombre"` y después calcular sus MFCC, SIFT y vectores
  con el mismo codebook
- `scripts/fetch_covers.py --total 100000` sigue bajando portadas de los mismos artistas: ~94 000
  candidatas, ~3 h y ~32 GB. El benchmark agrega 50k y 100k solo si hay data
- Ojo: nuestro HNSW está en Python puro. Construirlo con 100k tardaría ~15–20 min por métrica, y la
  corrida completa del benchmark, varias horas
- El dataset está en Kaggle (`jimenagurbillon/chutadb-canciones-10k`)
