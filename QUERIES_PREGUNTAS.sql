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

INSERT INTO mm_canciones VALUES (1, 'Jolene (Live Under Blackpool Lights)', 1, 'Rock', 'audio/1164738362.mp3', 'covers/190752902.jpg');
INSERT INTO mm_canciones VALUES (2, 'What the Hell', 2, 'Rock', 'audio/637113922.mp3', 'covers/88170452.jpg');
INSERT INTO mm_canciones VALUES (3, 'Bedshaped', 3, 'Rock', 'audio/71967079.mp3', 'covers/7069140.jpg');
INSERT INTO mm_canciones VALUES (4, 'Shatta Confessions', 4, 'Rock', 'audio/3399817861.mp3', 'covers/767551171.jpg');
INSERT INTO mm_canciones VALUES (5, 'The Magic Key', 5, 'Rock', 'audio/2179672997.mp3', 'covers/414776687.jpg');
INSERT INTO mm_canciones VALUES (6, 'VIEWS', 6, 'Rock', 'audio/1638677992.mp3', 'covers/291487202.jpg');
INSERT INTO mm_canciones VALUES (7, 'Beaucoup', 7, 'Rock', 'audio/3915853641.mp3', 'covers/945356831.jpg');
INSERT INTO mm_canciones VALUES (8, 'Secret Love Song (feat. Jason Derulo)', 8, 'Rock', 'audio/1057395242.mp3', 'covers/168591722.jpg');
INSERT INTO mm_canciones VALUES (9, 'Open Your Eyes', 9, 'Rock', 'audio/2277698.mp3', 'covers/227412.jpg');
INSERT INTO mm_canciones VALUES (10, 'Promises', 10, 'Rock', 'audio/542335432.mp3', 'covers/70900922.jpg');
INSERT INTO mm_canciones VALUES (11, 'Forget Me', 11, 'Pop', 'audio/1896602817.mp3', 'covers/352670597.jpg');
INSERT INTO mm_canciones VALUES (12, 'What Was I Made For? [From The Motion Picture "Barbie"]', 12, 'Pop', 'audio/2366043655.mp3', 'covers/463935645.jpg');
INSERT INTO mm_canciones VALUES (13, 'Last Friday Night (T.G.I.F.)', 13, 'Pop', 'audio/17135109.mp3', 'covers/1603030.jpg');
INSERT INTO mm_canciones VALUES (14, 'All The Love In The World', 14, 'Pop', 'audio/854525.mp3', 'covers/97722.jpg');
INSERT INTO mm_canciones VALUES (15, 'How It’s Done', 15, 'Pop', 'audio/3412534551.mp3', 'covers/771853201.jpg');
INSERT INTO mm_canciones VALUES (16, 'Training Season', 16, 'Pop', 'audio/2661514912.mp3', 'covers/547520122.jpg');
INSERT INTO mm_canciones VALUES (17, 'Talk To Me', 14, 'Pop', 'audio/2141902657.mp3', 'covers/405437377.jpg');
INSERT INTO mm_canciones VALUES (18, 'Night Falls', 17, 'Pop', 'audio/720599002.mp3', 'covers/105319892.jpg');
INSERT INTO mm_canciones VALUES (19, 'The Way It Is (Remastered)', 18, 'Pop', 'audio/628518.mp3', 'covers/78639.jpg');
INSERT INTO mm_canciones VALUES (20, 'Bruises', 11, 'Pop', 'audio/678686642.mp3', 'covers/96636702.jpg');
INSERT INTO mm_canciones VALUES (21, 'Pega Pega', 19, 'Latino', 'audio/3173782691.mp3', 'covers/693993651.jpg');
INSERT INTO mm_canciones VALUES (22, 'Doce Amizade (Ao Vivo)', 20, 'Latino', 'audio/3941004621.mp3', 'covers/954004191.jpg');
INSERT INTO mm_canciones VALUES (23, 'La Noche Más Linda', 21, 'Latino', 'audio/2976476331.mp3', 'covers/637694271.jpg');
INSERT INTO mm_canciones VALUES (24, 'Tu Cariñito', 22, 'Latino', 'audio/10963411.mp3', 'covers/1003425.jpg');
INSERT INTO mm_canciones VALUES (25, 'Fera Ferida', 23, 'Latino', 'audio/4699807.mp3', 'covers/432633.jpg');
INSERT INTO mm_canciones VALUES (26, 'Amor y Control', 24, 'Latino', 'audio/70331590.mp3', 'covers/6900774.jpg');
INSERT INTO mm_canciones VALUES (27, 'la culpa (track 11)', 25, 'Latino', 'audio/3406148371.mp3', 'covers/769819031.jpg');
INSERT INTO mm_canciones VALUES (28, 'Apiádate de Mi Edited Version', 26, 'Latino', 'audio/88877217.mp3', 'covers/8983731.jpg');
INSERT INTO mm_canciones VALUES (29, 'Amor Bandido', 27, 'Latino', 'audio/14566716.mp3', 'covers/1339805.jpg');
INSERT INTO mm_canciones VALUES (30, 'Qué Hay de Malo', 28, 'Latino', 'audio/1031403.mp3', 'covers/113077.jpg');

SELECT id, titulo, genero, audio, portada FROM mm_canciones LIMIT 10;
SELECT genero, COUNT(*) AS n FROM mm_canciones GROUP BY genero;

-- 3. Búsqueda lineal (sin índice): entrena el codebook de la columna en el momento
--    y compara contra todas las filas. La portada de la canción 1 debe salir primera.
SELECT id, titulo, SIMILARITY_SCORE() AS score FROM mm_canciones
    WHERE SIMILAR_TO('covers/190752902.jpg', k=5) USING SEQ
    ORDER BY score DESC;

-- 4. Índices vectoriales: cada columna entrena su propio codebook con sus archivos.
CREATE INDEX ON mm_canciones (portada) USING HNSW;
CREATE INDEX ON mm_canciones (portada) USING IVF;
CREATE INDEX ON mm_canciones (audio) USING HNSW;
CREATE INDEX ON mm_canciones (audio) USING IVF;

-- 5. Imágenes: portadas parecidas con HNSW y con IVF (la canción 1 debe salir primera).
--    IVF es aproximado: con NPROBE bajo puede devolver menos de k filas.
SELECT id, titulo, genero, SIMILARITY_SCORE() AS score FROM mm_canciones
    WHERE SIMILAR_TO('covers/190752902.jpg', k=5) USING HNSW
    ORDER BY score DESC;

SELECT id, titulo, genero, SIMILARITY_SCORE() AS score FROM mm_canciones
    WHERE SIMILAR_TO('covers/190752902.jpg', k=5) USING IVF WITH NPROBE = 10
    ORDER BY score DESC;

-- 6. Audio: canciones que suenan parecido (la canción 2 debe salir primera).
SELECT id, titulo, genero, SIMILARITY_SCORE() AS score FROM mm_canciones
    WHERE SIMILAR_TO('audio/637113922.mp3', k=5) USING HNSW
    ORDER BY score DESC;

SELECT id, titulo, genero, SIMILARITY_SCORE() AS score FROM mm_canciones
    WHERE SIMILAR_TO('audio/637113922.mp3', k=5) USING IVF
    ORDER BY score DESC;

-- 7. Consulta con un archivo que NO está en la tabla (Californication, Rock).
SELECT id, titulo, genero, SIMILARITY_SCORE() AS score FROM mm_canciones
    WHERE SIMILAR_TO('covers/86511.jpg', k=5) USING HNSW
    ORDER BY score DESC;

SELECT id, titulo, genero, SIMILARITY_SCORE() AS score FROM mm_canciones
    WHERE SIMILAR_TO('audio/725274.mp3', k=5) USING HNSW
    ORDER BY score DESC;

-- 8. Por defecto la métrica es coseno; aquí se cambia a euclidiana y se sube ef_search.
SELECT id, titulo, SIMILARITY_SCORE() AS score FROM mm_canciones
    WHERE SIMILAR_TO('covers/190752902.jpg', k=5) USING HNSW WITH EF_SEARCH = 100, METRIC = euclidean
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