# Pendientes de la Parte 4 y para la Parte 5

Lo que quedó abierto al 9 de noviembre, para que el equipo pueda seguir. Cada punto dice dónde está
el código y qué falta.

## Revisar en la Parte 3 (texto)

Los tests pasan, pero las fórmulas no coinciden con las de clase:

- **`text_based/tfidf.py`:** el peso es `tf / df`. El TF-IDF usual es `tf · log(N / df)`: el idf es
  un logaritmo de N sobre df, no una división por df
- **`text_based/bm25.py`:** el TF se calcula como `f / (f + k1 + norm)`. BM25 usa
  `f · (k1 + 1) / (f + k1 · norm)`, con `norm = 1 − b + b · |d| / avgdl`. El idf
  `log((N − df + 0.5) / (df + 0.5))` sale negativo para términos que están en más de la mitad de los
  documentos; muchas implementaciones usan `log(1 + (N − df + 0.5) / (df + 0.5))` para evitarlo

## Fotos reales de portadas

Ya hay 250 fotos reales de CDs, vinilos y casetes de Discogs (`scripts/fetch_fotos_reales.py`) y su
evaluación (`benchmark/multimedia/fotos_reales.py`; resultados en `benchmark/multimedia/README.md`):
el 70 % encuentra la portada digital en el top 10. Se podría:

- Quitar a mano las fotos principales que en realidad son el disco o la contraportada, para que el
  número mida solo al buscador
- Sumar el histograma de color (`multimedia/color.py`) a SIFT, que no ve el color
- Bajar más: el script sigue donde se quedó (`--total 500`). Discogs sin token deja 25 pedidos por
  minuto y solo ~1 de cada 4 álbumes tiene edición física, así que 250 tardan ~2 h

## Búsqueda por cara de artista (extra)

Ya funciona por consola: `python -m multimedia.caras foto.jpg` mete los embeddings de 128 dimensiones
de `data/features/faces.npz` en nuestro HNSW con euclidiana y devuelve los artistas más parecidos
(resultados y por qué el umbral es 0.45 en `multimedia/README.md`). Falta:

- Mostrarla en la web: un endpoint que reciba la foto, llame a `multimedia.caras.buscar_foto` y
  devuelva nombre, foto y distancia
- Opcional: llevarla al SQL, por ejemplo con `SIMILAR_TO` sobre una columna de fotos de caras
- `face_recognition` (dlib) no está en `web/backend/requirements.txt`; habría que agregarlo a la
  imagen de Docker

## Shazam y "cantas y reconoce" (Parte 5)

Hay un prototipo medido de cada uno en `docs/INVESTIGACION_PARTE5.md`. Falta:

- **Shazam** (`multimedia/huellas.py`): ya acierta 99 % con audio grabado con celular. Falta bajar el
  espacio (guardar menos picos por segundo), guardar el índice en disco y llevarlo al SQL o la web
- **Tarareo** (`multimedia/tarareo.py`): el híbrido (vector fijo con coseno → DTW sobre 50 candidatos)
  funciona con tarareos simulados. Falta separar la voz de la mezcla (`demucs`), probar con gente
  cantando de verdad (dataset MIR-QBSH) y meter los vectores fijos en el HNSW. Recordar que DTW no es
  una métrica: solo puede reordenar candidatos, nunca podar dentro del índice

## Más datos

- **Agregar artistas puntuales:** el dataset tomó ~200 artistas por género de las radios de Deezer, así que
  faltan algunos famosos (por ejemplo Kanye West, que solo aparece como colaborador). Habría que sumar a
  `scripts/fetch_music.py` una opción `--artista "Nombre"` y después calcular sus MFCC, SIFT y vectores
  con el mismo codebook (`multimedia/mfcc.py`, `multimedia/sift.py`, `python -m web.backend.multimedia_search`)
- `scripts/fetch_covers.py --total 100000` sigue bajando portadas de los mismos artistas: ~94 000
  candidatas, ~3 h y ~32 GB. El benchmark agrega 50k y 100k solo si hay data
- Ojo: nuestro HNSW está en Python puro. Construirlo con 100k tardaría ~15–20 min por métrica, y la
  corrida completa del benchmark, varias horas
- El dataset en Kaggle (`jimenagurbillon/chutadb-canciones-10k`, versión 3) ya tiene las 20k portadas
