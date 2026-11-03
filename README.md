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
