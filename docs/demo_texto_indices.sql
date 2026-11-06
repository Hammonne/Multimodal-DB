-- Demo de búsqueda textual e índices.
-- Ejecutar desde el workspace web, una sentencia a la vez.

-- 1. Comprobar que la tabla tiene canciones y rutas de letras.
SELECT track_id, title, artist, lyrics_path
FROM demo_songs
LIMIT 10;

SELECT COUNT(*) AS canciones
FROM demo_songs;

-- 2. Construir un único índice invertido SPIMI.
-- BM25 y TF_IDF reutilizan este mismo índice.
CREATE INDEX ON demo_songs (lyrics_path) USING SPIMI;

-- 3. Búsqueda BM25: consulta corta.
SELECT track_id, title, artist, SCORE() AS relevancia
FROM demo_songs
WHERE MATCH(lyrics_path, 'love tonight') USING BM25
ORDER BY SCORE() DESC
LIMIT 10;

-- 4. Búsqueda TF-IDF + coseno con la misma consulta.
SELECT track_id, title, artist, SCORE() AS relevancia
FROM demo_songs
WHERE MATCH(lyrics_path, 'love tonight') USING TF_IDF
ORDER BY SCORE() DESC
LIMIT 10;

-- 5. Consultas de distinta longitud.
SELECT track_id, title, artist, SCORE() AS relevancia
FROM demo_songs
WHERE MATCH(lyrics_path, 'love') USING BM25
ORDER BY SCORE() DESC
LIMIT 10;

SELECT track_id, title, artist, SCORE() AS relevancia
FROM demo_songs
WHERE MATCH(lyrics_path, 'love tonight baby') USING BM25
ORDER BY SCORE() DESC
LIMIT 10;

SELECT track_id, title, artist, SCORE() AS relevancia
FROM demo_songs
WHERE MATCH(lyrics_path, 'love tonight baby heart forever') USING TF_IDF
ORDER BY SCORE() DESC
LIMIT 10;

-- 6. Verificar el plan: debe mostrar TEXT INDEX SCAN y SPIMI.
EXPLAIN SELECT track_id, title, SCORE() AS relevancia
FROM demo_songs
WHERE MATCH(lyrics_path, 'love tonight') USING BM25
ORDER BY SCORE() DESC
LIMIT 10;

-- 7. Índices estructurados para metadata.
CREATE INDEX ON demo_songs (artist) USING BTREE;
CREATE INDEX ON demo_songs (genre) USING HASH;
CREATE INDEX ON demo_songs (has_lyrics) USING BITMAP;

EXPLAIN SELECT track_id, title, artist
FROM demo_songs
WHERE artist = 'HUGEL'
LIMIT 10;

EXPLAIN SELECT track_id, title, artist, genre
FROM demo_songs
WHERE genre = 'Dance'
LIMIT 10;

EXPLAIN SELECT track_id, title, artist
FROM demo_songs
WHERE has_lyrics = true
LIMIT 10;

-- 8. Combinar filtro estructurado con búsqueda textual.
SELECT track_id, title, artist, SCORE() AS relevancia
FROM demo_songs
WHERE artist = 'HUGEL'
  AND MATCH(lyrics_path, 'love tonight') USING BM25
ORDER BY SCORE() DESC
LIMIT 10;
