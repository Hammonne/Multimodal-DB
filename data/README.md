# Dataset de canciones

Dataset multimedia para las Partes 3 (full-text sobre letras) y 4 (similitud de
audio e imágenes). Se genera con `scripts/fetch_music.py`, que solo usa la
librería estándar de Python 3.9+, no necesita API keys y se puede interrumpir y
volver a correr: retoma desde los CSV.

```bash
python3 scripts/fetch_music.py --songs 1000      # canciones, portadas, fotos
python3 scripts/fetch_members.py                 # persona o banda, integrantes
conda activate vision                            # entorno con face_recognition
python scripts/filter_artist_faces.py --delete   # deja solo fotos de caras
python3 scripts/fetch_profiles.py                # una foto de perfil por artista
```

`fetch_music.py` y `fetch_members.py` pueden correr a la vez; el filtro con
`--delete` espera a que ambos terminen.

| Opción | Por defecto | Descripción |
|---|---|---|
| `--songs` | 1000 | canciones objetivo (puede pasarse por unas pocas) |
| `--per-artist` | 5 | canciones top por artista |
| `--extra-images` | 8 | máximo de fotos de Wikimedia Commons por artista |
| `--workers` | 6 | hilos; cada host tiene su propio límite de velocidad |
| `--out` | `data/` | carpeta de salida |
| `--con-letra` | 0 | en vez de `--songs`: llegar a N canciones con letra |
| `--mas` | 3 | con `--con-letra`: canciones extra que se miran por artista |

Para pasar de 1 000 a 10 000 basta con volver a correr con `--songs 10000` sobre
la misma carpeta: no se vuelve a descargar nada.

Solo 6 388 de las 10 000 tenían letra en LRCLIB, así que después se corrió
`--con-letra 10000`: a los artistas que ya estaban se les miran más canciones
de su top y solo entran las que tienen letra, sin artistas ni fotos nuevas.
Las nuevas van al final de `songs.csv`, las primeras 10 000 no cambian.

| | Total | Para los experimentos |
|---|---|---|
| Canciones (audio) | 13 643 | las primeras 10 000 filas |
| Con letra (texto) | 10 031 | las que tienen `has_lyrics` |
| Portadas distintas (imagen) | 10 732 | 10 000 de ellas |

## Contenido

Los datos no están en el repositorio (`.gitignore`): cada integrante los genera
localmente con los scripts. Solo se versiona este README.

```
data/
├── songs.csv
├── artists.csv
├── artist_images.csv
├── artist_types.csv
├── members.csv
├── band_members.csv
├── member_images.csv
├── artist_faces.csv
├── member_faces.csv
├── audio/<track_id>.mp3          preview de 30 s
├── lyrics/<track_id>.txt         letra en texto plano
├── lyrics/<track_id>.lrc         letra sincronizada (si existe)
├── covers/<album_id>.jpg         portada del álbum (1000×1000)
└── artists/<nombre>_<artist_id>/   p. ej. artists/bad-bunny_8355/
    ├── profile.jpg               foto de perfil para la interfaz
    ├── main.jpg                  foto principal (Deezer)
    └── extra_NN.jpg|png          fotos adicionales (TheAudioDB, Commons)
└── members/<nombre>_<id>/        integrante de una banda
    └── photo_NN.jpg|png          fotos (Wikimedia Commons)
```

El nombre de la carpeta del artista va sin tildes ni espacios; el id de Deezer
al final evita choques entre artistas con el mismo nombre.

Las rutas de los CSV son relativas a `data/`.

### songs.csv

`track_id, title, artist_id, artist, album_id, album, genre, duration_s,
deezer_rank, explicit, has_lyrics, has_synced_lyrics, instrumental, audio_path,
lyrics_path, lrc_path, cover_path, deezer_url`

- `genre` es el género **semilla** desde el que se llegó al artista (Deezer no
  da género por canción en este endpoint); los artistas descubiertos como
  "relacionados" heredan el género del artista de origen.
- `duration_s` es la duración de la canción completa; el audio es un preview
  de 30 s.

### artists.csv

`artist_id, name, genre, wikidata_id, n_images, main_image_path, deezer_url`

### artist_images.csv

`artist_id, path, source, license, author, page, url`: una fila por foto, con
su fuente (`deezer`, `theaudiodb`, `wikimedia_commons`), licencia, autor y
página de origen cuando la fuente los publica (Commons sí; Deezer y TheAudioDB
no).

### artist_profiles.csv

`artist_id, name, profile_path, source`: la imagen que muestra la interfaz
para cada artista, independiente del filtro de caras (que puede haber borrado
todas sus fotos). `source` dice de dónde sale: `deezer` (la foto oficial,
aunque sea un logo; se descarta la silueta gris genérica), `foto` (una que
pasó el filtro) o `portada` (la de su canción más popular). Todos los
artistas con canciones tienen perfil; `profile.jpg` no está en
`artist_images.csv`, así que no se usa para buscar por cara.

### artist_types.csv

`artist_id, name, musicbrainz_id, type, match`: `type` es `persona`, `banda`,
`otro` o `desconocido` (no está en MusicBrainz); `match` indica si se encontró
por el enlace a Deezer o por nombre.

### Integrantes de bandas

- `members.csv`: `member_id` (MusicBrainz), `name, wikidata_id, occupations,
  n_images, folder`. Un integrante de varias bandas aparece una sola vez.
- `band_members.csv`: `artist_id, band, member_id, member, roles,
  roles_source, founder, current, begin, end`. `roles` es el puesto en esa
  banda (guitarrista, bajista, baterista, vocalista, corista, tecladista, DJ…);
  si MusicBrainz no registra instrumentos se usan las ocupaciones generales de
  Wikidata (`roles_source = wikidata`) y si tampoco hay, `integrante`.
  `current` distingue integrantes actuales de anteriores.
- `member_images.csv`: igual que `artist_images.csv`, por integrante.

## Fuentes

| Dato | Fuente |
|---|---|
| Audio, portada, foto principal | [Deezer API](https://developers.deezer.com/api) (`preview`, `cover_xl`, `picture_xl`) |
| Letras | [LRCLIB](https://lrclib.net) (`/api/search`: `plainLyrics`, `syncedLyrics`) |
| Fotos extra | [TheAudioDB](https://www.theaudiodb.com) (thumbs y fanart) |
| Fotos extra con licencia | [Wikidata](https://www.wikidata.org) (P18, P373) + [Wikimedia Commons](https://commons.wikimedia.org) |
| Persona o banda, integrantes y puesto | [MusicBrainz](https://musicbrainz.org) (relación "member of band" con instrumentos); Wikidata P106 como respaldo |

### Cómo se eligen las canciones

1. **Semillas:** por cada uno de los 16 géneros, los artistas de las radios del
   género y de las playlists que lo mencionan, ordenados por frecuencia de
   aparición (`/genre/{id}/artists` devuelve el mismo chart global para todos
   los géneros, por eso no se usa). Los géneros se intercalan para que el
   dataset quede balanceado.
2. **Expansión:** cuando se acaban las semillas se siguen los artistas
   relacionados (`/artist/{id}/related`) de los ya descargados.
3. Por artista se toman sus `--per-artist` canciones top con preview.
4. **Letra:** se busca en LRCLIB y se acepta solo un candidato con el mismo
   artista y título (sin tildes, paréntesis ni sufijos tipo "- Remastered") y
   duración a ±15 s, prefiriendo los que tienen letra sincronizada.

## Filtro de caras: artist_faces.csv y member_faces.csv

Las fotos extra incluyen conciertos, público, logos y carteles.
`scripts/filter_artist_faces.py` deja solo fotos útiles para buscar por cara:

- **Solista o integrante:** fotos donde aparece esa persona sola (una única
  cara), con la cara de al menos el 8 % del alto de la imagen. Las fotos con
  compañeros de banda se rechazan aunque también se repitan.
- **Banda:** una sola foto de grupo (2+ caras de la banda y ninguna ajena),
  preferentemente la principal de Deezer. Las fotos de cara de cada
  integrante están en `members/`.

Cómo sabe quién es quién: detecta las caras de todas las fotos de un artista o
integrante con `face_recognition` y toma como suyas las que se repiten en
varias fotos (distancia < 0.55); los fans no se repiten. Una foto con cualquier
cara ajena (de más del 3 % del alto) o con más de 12 caras se rechaza. Si
ninguna cara se repite, solo acepta la foto principal si tiene una sola cara.

Columnas: `artist_id | member_id, path, source, n_faces, keep, kind,
face_box`, con `kind` = `cara` o `banda` y `face_box` = `top right bottom
left` en píxeles de la imagen original, para recortar la cara.

Sin `--delete` solo marca; con `--delete` borra del disco las fotos rechazadas
y las quita de `artist_images.csv` / `member_images.csv` (los `*_faces.csv`
quedan como registro). Necesita `artist_types.csv`, así que va después de
`fetch_members.py`. Es reanudable: solo procesa artistas e integrantes nuevos.

### Consideraciones

- Los previews de Deezer y las imágenes de Deezer y TheAudioDB tienen
  copyright: el dataset es solo para uso académico y no se redistribuye.
