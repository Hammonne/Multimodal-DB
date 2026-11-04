# Multimedia

Extracción de características para la Parte 4 (§2.4.1). Cada imagen o audio se parte en pedazos y cada pedazo se vuelve un vector (descriptor local). Luego se cuantizan con K-Means (#4) y cada objeto queda como un histograma TF-IDF (#5) que es lo que se indexa con IVF y HNSW

Todo corre en el entorno `vision` y guarda en `data/features/`, que no se sube al repo

```bash
conda activate vision
python -m multimedia.sift --source all    # portadas, artistas e integrantes
python -m multimedia.mfcc                 # audios
python -m multimedia.color                # extra: color de portadas
python -m multimedia.face_embeddings      # extra: caras
```

Todos son reanudables, si se cortan se vuelven a correr y siguen

## Descriptores locales

| | SIFT (imágenes) | MFCC (audio) |
|---|---|---|
| Archivo | `sift.py` | `mfcc.py` |
| Pedazo | punto clave (esquina, borde, textura) | ventana de 93 ms que avanza 23 ms |
| Vectores por objeto | ~450 (máx. 500) | ~1290 por preview de 30 s |
| Dimensión | 128 | 39 |
| Aguanta | escala, rotación, brillo | volumen |
| No ve | color | melodía |
| Salida | `sift/<ruta de la imagen>.npy` + `sift_<source>.csv` | `mfcc/<track_id>.npy` + `mfcc.csv` |

### SIFT
- La imagen se pasa a gris y se achica a 512 px de lado, más grande es lento y no mejora
- Se quedan los 500 puntos de mayor contraste
- RootSIFT: se normaliza L1 y se saca raíz, así la distancia euclidiana compara mejor histogramas
- Portadas: todas las de `songs.csv`. Artistas e integrantes: solo las fotos que aceptó el filtro de caras
- 8 portadas no tienen ningún punto clave: son de un color plano o texto y fondo de colores distintos pero mismo brillo (en gris quedan lisas)

### MFCC
- Audio mono a 22 050 Hz
- Ventana de 2048 muestras con avance de 512 (75 % de solapamiento)
- Por ventana: FFT, 40 bandas mel, log y DCT, se guardan los coeficientes 1 a 13
- El coeficiente 0 es el volumen, se descarta para que la misma canción más fuerte o más bajita dé lo mismo
- Se agregan delta y delta-delta (cuánto cambia el timbre entre ventanas): 13 + 13 + 13 = 39
- La fila `i` empieza en el segundo `i * 512 / 22050`, sirve para buscar por fragmentos (`extraer_mfcc(path, offset=10, duration=5)`)
- Antes del K-Means hay que estandarizar cada dimensión, las deltas son mucho más chicas

## Extras

### Histograma de color (`color.py`)
Descriptor global de las portadas, para lo que SIFT no ve. 12 tonos x 4 saturaciones + 2 grises (oscuro y claro) = 50 dimensiones, con raíz como RootSIFT

Casi no se mira el brillo: con 3 niveles de brillo por color una portada oscurecida al 60 % encontraba la original en el top 10 el 17 % de las veces, así el 91 %

Probado con 300 portadas modificadas buscando entre las 8 400:

| Cambio | Top 1 | Top 10 |
|---|---|---|
| Rotar 90° / 15° | 100 % / 99 % | 100 % |
| Recortar al 70 % | 86 % | 95 % |
| Oscurecer al 60 % | 84 % | 91 % |
| Aclarar x1.3 | 87 % | 94 % |
| JPEG comprimido + 200 px | 69 % | 87 % |

Solo encuentra colores parecidos, no la misma portada, por eso la idea es sumarlo a SIFT: `a * sim_sift + (1 - a) * sim_color`

### Embeddings de cara (`face_embeddings.py`)
Igual que en el Lab 11: ResNet de dlib vía `face_recognition`, 128 dimensiones, misma persona si la distancia es < 0.6. Usa la caja de la cara que guardó el filtro (`face_box`) así que no vuelve a detectar, y achica a 1000 px porque a resolución completa es ~10 veces más lento

Sale en `faces.npz` con `embedding, path, kind, owner_id, name`. Es un vector por cara, no necesita codebook. Sirve para buscar por artista (Parte 5)
