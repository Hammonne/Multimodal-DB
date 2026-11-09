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

El benchmark ya prueba con **fotos simuladas**: la portada en perspectiva sobre un fondo, con tinte de
color, reflejo, desenfoque, ruido y jpeg (`foto_simulada` en `benchmark/multimedia/benchmark_vectorial.py`).
El paso siguiente es probar con fotos de verdad de CDs, vinilos y pósters:

- **Discogs** es la mejor fuente: los usuarios suben fotos de sus discos físicos. La API pide un
  token personal para bajar imágenes (https://www.discogs.com/developers) y permite 60 pedidos por
  minuto
- Solo sirven álbumes que ya estén entre las 20 005 portadas; si no, no hay respuesta correcta
- Con 100–300 fotos alcanza para medir; para la demo bastan ~20

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

Están las ideas y los cuidados en el README principal, en *Ideas para la Parte 5*. Lo más importante:
DTW no es una métrica, así que no se puede usar tal cual con los índices.

## Más datos

- `scripts/fetch_covers.py --total 100000` sigue bajando portadas de los mismos artistas: ~94 000
  candidatas, ~3 h y ~32 GB. El benchmark agrega 50k y 100k solo si hay data
- Ojo: nuestro HNSW está en Python puro. Construirlo con 100k tardaría ~15–20 min por métrica, y la
  corrida completa del benchmark, varias horas
- El dataset en Kaggle (`jimenagurbillon/chutadb-canciones-10k`, versión 3) ya tiene las 20k portadas
