-- ================== Texto ==================



-- ================== Multimedia ==================
-- 1. Tablas: portada es IMAGE (SIFT) y audio es AUDIO (MFCC).
CREATE TABLE mm_artistas (id INT PRIMARY KEY, nombre VARCHAR(80), genero VARCHAR(30)) USING HEAP;
CREATE TABLE mm_canciones (id INT PRIMARY KEY, titulo VARCHAR(120), artista_id INT, genero VARCHAR(30),
                           audio AUDIO, portada IMAGE) USING HEAP;

-- 2. Datos: 30 canciones del dataset (10 Rock, 10 Pop, 10 Latino) y sus artistas.
--    Las rutas (covers/..., audio/...) se resuelven contra data/.
INSERT INTO mm_artistas VALUES (1, 'The White Stripes', 'Rock');
INSERT INTO mm_artistas VALUES (2, 'Avril Lavigne', 'Rock');
INSERT INTO mm_artistas VALUES (3, 'Keane', 'Rock');
INSERT INTO mm_artistas VALUES (4, 'Meryl', 'Rock');
INSERT INTO mm_artistas VALUES (5, 'Trinix', 'Rock');
INSERT INTO mm_artistas VALUES (6, 'Noga Erez', 'Rock');
INSERT INTO mm_artistas VALUES (7, 'Bilal Hassani', 'Rock');
INSERT INTO mm_artistas VALUES (8, 'Little Mix', 'Rock');
INSERT INTO mm_artistas VALUES (9, 'Snow Patrol', 'Rock');
INSERT INTO mm_artistas VALUES (10, 'Calvin Harris', 'Rock');
INSERT INTO mm_artistas VALUES (11, 'Lewis Capaldi', 'Pop');
INSERT INTO mm_artistas VALUES (12, 'Billie Eilish', 'Pop');
INSERT INTO mm_artistas VALUES (13, 'Katy Perry', 'Pop');
INSERT INTO mm_artistas VALUES (14, 'The Outfield', 'Pop');
INSERT INTO mm_artistas VALUES (15, 'HUNTR/X', 'Pop');
INSERT INTO mm_artistas VALUES (16, 'Dua Lipa', 'Pop');
INSERT INTO mm_artistas VALUES (17, 'Dove Cameron', 'Pop');
INSERT INTO mm_artistas VALUES (18, 'Bruce Hornsby & The Range', 'Pop');
INSERT INTO mm_artistas VALUES (19, 'Kiko Chicabana', 'Latino');
INSERT INTO mm_artistas VALUES (20, 'FM O Dia', 'Latino');
INSERT INTO mm_artistas VALUES (21, 'Salsa Power Band', 'Latino');
INSERT INTO mm_artistas VALUES (22, 'Luisito Ayala Y La Puerto Rican Power', 'Latino');
INSERT INTO mm_artistas VALUES (23, 'Maria Bethânia', 'Latino');
INSERT INTO mm_artistas VALUES (24, 'Rubén Blades', 'Latino');
INSERT INTO mm_artistas VALUES (25, 'alleh', 'Latino');
INSERT INTO mm_artistas VALUES (26, 'Victor Manuelle', 'Latino');
INSERT INTO mm_artistas VALUES (27, 'Joanna', 'Latino');
INSERT INTO mm_artistas VALUES (28, 'Jerry Rivera', 'Latino');

SELECT id, titulo, genero, audio, portada FROM mm_canciones LIMIT 10;
SELECT genero, COUNT(*) AS n FROM mm_canciones GROUP BY genero;

-- 3. Búsqueda lineal (sin índice): entrena el codebook de la columna en el momento
--    y compara contra todas las filas. La portada de la canción 1 debe salir primera.
SELECT track_id, album, SIMILARITY_SCORE() AS score FROM demo_songs
    WHERE SIMILAR_TO('covers/982091821.jpg', k=5) USING SEQ
    ORDER BY score DESC;

-- 4. Índices vectoriales: cada columna entrena su propio codebook con sus archivos.
CREATE INDEX ON demo_songs (cover_path) USING HNSW;
CREATE INDEX ON demo_songs (cover_path) USING IVF;
CREATE INDEX ON demo_songs (audio_path) USING HNSW;
CREATE INDEX ON demo_songs (audio_path) USING IVF;

-- 5. Imágenes: portadas parecidas con HNSW y con IVF (la canción 1 debe salir primera).
--    IVF es aproximado: con NPROBE bajo puede devolver menos de k filas.
SELECT track_id, album, genre, SIMILARITY_SCORE() AS score FROM demo_songs
    WHERE SIMILAR_TO('covers/982091821.jpg', k=5) USING HNSW
    ORDER BY score DESC;

SELECT track_id, album, genre, SIMILARITY_SCORE() AS score FROM demo_songs
    WHERE SIMILAR_TO('covers/982091821.jpg', k=5)  USING IVF WITH NPROBE = 10
    ORDER BY score DESC;

-- 6. Audio: canciones que suenan parecido (la canción 2 debe salir primera).

SELECT track_id, album, genre, SIMILARITY_SCORE() AS score FROM demo_songs
    WHERE SIMILAR_TO('covers/982091821.jpg', k=5) USING HNSW
    ORDER BY score DESC;


SELECT track_id, album, genre, SIMILARITY_SCORE() AS score FROM demo_songs
    WHERE SIMILAR_TO('covers/982091821.jpg', k=5) USING IVF
    ORDER BY score DESC;

-- 7. Consulta con un archivo que NO está en la tabla (Californication, Rock).
SELECT track_id, album, genre, SIMILARITY_SCORE() AS score FROM demo_songs
    WHERE SIMILAR_TO('covers/86511.jpg', k=5) USING HNSW
    ORDER BY score DESC;

SELECT track_id, album, genre, SIMILARITY_SCORE() AS score FROM demo_songs
    WHERE SIMILAR_TO('audio/725274.mp3', k=5) USING HNSW
    ORDER BY score DESC;

-- 8. Por defecto la métrica es coseno; aquí se cambia a euclidiana y se sube ef_search.
SELECT track_id, album, genre, SIMILARITY_SCORE() AS score FROM demo_songs
    WHERE SIMILAR_TO('covers/982091821.jpg', k=5) USING HNSW WITH EF_SEARCH = 100, METRIC = euclidean
    ORDER BY score DESC;

-- 9. SIMILAR_TO con filtro: solo canciones de Pop.
SELECT id, titulo, genero, SIMILARITY_SCORE() AS score FROM mm_canciones
    WHERE genero = 'Pop' AND SIMILAR_TO('audio/637113922.mp3', k=5) USING HNSW
    ORDER BY score DESC;

-- 10. SIMILAR_TO con JOIN: nombre del artista de cada resultado.
SELECT mm_canciones.titulo, mm_artistas.nombre, SIMILARITY_SCORE() AS score
    FROM mm_canciones JOIN mm_artistas ON mm_canciones.artista_id = mm_artistas.id
    WHERE SIMILAR_TO('covers/86511.jpg', k=5);

-- 11. SIMILAR_TO con GROUP BY: de los 20 audios más parecidos, cuántos hay de cada género.
SELECT genero, COUNT(*) AS n FROM mm_canciones
    WHERE SIMILAR_TO('audio/725274.mp3', k=20) GROUP BY genero ORDER BY n DESC;

-- 12. Plan de ejecución: qué índice usa.
EXPLAIN SELECT id FROM mm_canciones WHERE SIMILAR_TO('covers/86511.jpg', k=5) USING HNSW;
EXPLAIN SELECT id FROM mm_canciones WHERE SIMILAR_TO('covers/86511.jpg', k=5) USING IVF;
EXPLAIN SELECT id FROM mm_canciones WHERE SIMILAR_TO('audio/725274.mp3', k=5) USING HNSW;
EXPLAIN SELECT id FROM mm_canciones WHERE SIMILAR_TO('audio/725274.mp3', k=5) USING IVF;

-- 13. Limpieza (opcional).
-- DROP TABLE mm_canciones;
-- DROP TABLE mm_artistas;


-- ================== PREGUNTAS RAG ==================
/*
- ¿Qué es el policy gradient y qué variantes aparecen en los papers? 
- ¿Cómo se entrena una GAN de forma estable?
- ¿Qué es Glow y qué aporta la convolución 1x1 invertible?
- ¿Qué técnica usa PointNet++ para procesar nubes de puntos?
- ¿En qué se diferencia ShuffleNet de otras redes eficientes para móviles?
- ¿Cómo funciona PWC-Net para estimar el flujo óptico?
- ¿Qué propone LightGBM para hacer más eficiente el gradient boosting?
- ¿Qué métodos de transferencia de estilo aparecen en el corpus?
- ¿Cómo se hace inpainting de imágenes con atención contextual?
*/