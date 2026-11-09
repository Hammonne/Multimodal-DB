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

## Texto en "Cómo funciona" (Parte 3)

La página `/como-funciona` (`web/app/como-funciona/page.tsx` y `web/backend/explica.py`) explica imágenes,
audio, índices y caras con nuestra data. Falta una sección de texto, con el mismo estilo:

1. **Tokens:** una letra de canción real partida en palabras (minúsculas, sin signos), como hace `tokenize`
2. **SPIMI:** los documentos entrando en bloques que se escriben a disco cuando se llena la memoria, y la mezcla
   de los bloques en un índice invertido (término → lista de documentos con su frecuencia)
3. **Una consulta:** escribir unas palabras y ver la lista de postings de cada término, su df y su idf
4. **El puntaje:** para los primeros resultados, cuánto aporta cada término con TF-IDF y con BM25
   (tf, idf, largo del documento contra el promedio, k1 y b), y cómo cambia el ranking

Antes conviene corregir las fórmulas de `text_based/tfidf.py` y `text_based/bm25.py` (sección siguiente), para
no explicar un cálculo que no es el de clase. No falta calcular nada pesado: el índice SPIMI de las ~10 000
letras se arma en unos segundos.

## Fotos reales de portadas

Ya hay fotos reales de 600 CDs, vinilos y casetes de Discogs (3 409 fotos, `scripts/fetch_fotos_reales.py`) y su
evaluación (`benchmark/multimedia/fotos_reales.py`; resultados en `benchmark/multimedia/README.md`):
el 66 % encuentra la portada digital en el top 10 (~70 % en CDs y vinilos, 26 % en casetes). Se podría:

- Quitar a mano las fotos principales que en realidad son el disco o la contraportada, para que el
  número mida solo al buscador
- Sumar el histograma de color (`multimedia/color.py`) a SIFT, que no ve el color
- Bajar más: el script sigue donde se quedó (`--total 500`). Discogs sin token deja 25 pedidos por
  minuto y solo ~1 de cada 3 álbumes tiene edición física (más entre los populares: `--orden popular`)

## Búsqueda por cara de artista (extra)

Ya está en la web: en la app (`/musica` → Reconocer → Cara) se reconoce desde una foto, la cámara o en vivo,
con el top k del HNSW, y en la página de cada artista sale "los artistas que más se parecen". El Docker del
backend instala `face_recognition` con `dlib-bin`. Queda opcional:

- Llevarla al SQL, por ejemplo con `SIMILAR_TO` sobre una columna de fotos de caras
- Medir el umbral con más personas: con ~2 700, 0.45 acierta 90 % de las veces que responde
  (`python -m multimedia.caras --evaluar`)

## ¿Codebook de 2048 palabras? (experimento para la Parte 5)

Hoy K = 1024 (`codebook-k-finding.ipynb`). Con 2048 el recall de las consultas modificadas sube (portada en
jpeg chico .45 → .59, audio de 5 s .88 → .94) pero cada consulta tarda el doble (2.0 → 3.8 ms en portadas).
Cambiarlo obliga a rehacer el codebook, los vectores, los índices, el benchmark y las colecciones de la web.

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
