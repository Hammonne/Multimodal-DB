Como un spotify, un shazam y google lens pero todo a la vez shalom

## Dataset

Canciones con audio, letra, portada y fotos de artistas e integrantes de bandas.
Se descarga con los scripts de `scripts/` (detalles en [`data/README.md`](data/README.md)).

| Dato | Fuente |
|---|---|
| Audio (preview 30 s), portada, foto del artista | Deezer API |
| Letra (texto y `.lrc` sincronizado) | LRCLIB |
| Fotos extra de artistas | TheAudioDB, Wikidata + Wikimedia Commons |
| Persona o banda, integrantes y puesto | MusicBrainz (+ Wikidata) |

## Multimedia

Descriptores de imágenes (SIFT) y audio (MFCC) para la Parte 4, más color de portadas y caras como extra. Detalles en [`multimedia/README.md`](multimedia/README.md)

## Ideas para la Parte 5

1. **Shazam:** reconocer la canción grabando unos segundos aunque haya ruido, con *audio fingerprinting* (picos del espectrograma como huella de la grabación, como hace Shazam). Se puede hacer con `librosa` o ver `audfprint` / `dejavu`
2. **Detector cantando:** que alguien cante o tararee y diga qué canción es o cuál se parece más, con *query by humming*. Se saca la melodía con `librosa.pyin` y se compara con DTW (`librosa.sequence.dtw`), porque MFCC ignora la melodía a propósito
