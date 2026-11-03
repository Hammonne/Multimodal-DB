# Dataset de canciones

Dataset multimedia para las Partes 3 (full-text sobre letras) y 4 (similitud de
audio e imágenes). Se genera con `scripts/fetch_music.py`, que solo usa la
librería estándar de Python 3.9+, no necesita API keys y se puede interrumpir y
volver a correr: retoma desde los CSV.

```bash
python3 scripts/fetch_music.py --songs 1000      # luego 10000 y 100000
```

| Opción | Por defecto | Descripción |
|---|---|---|
| `--songs` | 1000 | canciones objetivo (puede pasarse por unas pocas) |
| `--per-artist` | 5 | canciones top por artista |
| `--extra-images` | 8 | máximo de fotos de Wikimedia Commons por artista |
| `--workers` | 6 | hilos; cada host tiene su propio límite de velocidad |
| `--out` | `data/` | carpeta de salida |

Para pasar de 1 000 a 10 000 basta con volver a correr con `--songs 10000` sobre
la misma carpeta: no se vuelve a descargar nada.

## Contenido

Solo los CSV y este README están en el repositorio; los archivos multimedia
quedan fuera por `.gitignore` y cada integrante los genera localmente.

```
data/
├── songs.csv
├── artists.csv
├── artist_images.csv
├── audio/<track_id>.mp3          preview de 30 s
├── lyrics/<track_id>.txt         letra en texto plano
├── lyrics/<track_id>.lrc         letra sincronizada (si existe)
├── covers/<album_id>.jpg         portada del álbum (1000×1000)
└── artists/<artist_id>/
    ├── main.jpg                  foto principal (Deezer)
    └── extra_NN.jpg|png          fotos adicionales (TheAudioDB, Commons)
```

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

## Fuentes

| Dato | Fuente |
|---|---|
| Audio, portada, foto principal | [Deezer API](https://developers.deezer.com/api) (`preview`, `cover_xl`, `picture_xl`) |
| Letras | [LRCLIB](https://lrclib.net) (`/api/search`: `plainLyrics`, `syncedLyrics`) |
| Fotos extra | [TheAudioDB](https://www.theaudiodb.com) (thumbs y fanart) |
| Fotos extra con licencia | [Wikidata](https://www.wikidata.org) (P18, P373) + [Wikimedia Commons](https://commons.wikimedia.org) |

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

### Consideraciones

- Las fotos extra de Commons salen de la categoría del artista y pueden incluir
  logos, carteles o fotos donde aparece otra persona; conviene filtrarlas con
  detección de rostros antes de usarlas para similitud.
- Los previews de Deezer y las imágenes de Deezer y TheAudioDB tienen
  copyright: el dataset es solo para uso académico y no se redistribuye.
