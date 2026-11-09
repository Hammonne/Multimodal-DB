# Investigación para la Parte 5: Shazam y "cantas y reconoce"

Dos funciones que no se pueden hacer con lo de la Parte 4 (SIFT/MFCC + bolsa de palabras), con un
prototipo de cada una medido sobre nuestro dataset. Los prototipos están en `multimedia/huellas.py` y
`multimedia/tarareo.py`; todavía no están conectados al SQL ni a la web.

## Por qué no alcanza con MFCC

El benchmark de la Parte 4 (`benchmark/multimedia/README.md`) mostró que un fragmento de 5 s
**grabado con celular** no encuentra nunca su canción con MFCC (0 de 100). Probando cada efecto por
separado: el eco solo deja 0.76, pero quitar frecuencias la arruina (0.00 con el filtro de 300–3 400 Hz
de un celular y 0.04 incluso con 100–7 000 Hz). Los MFCC describen la *forma* del espectro y el filtro
la cambia entera.

Y MFCC no puede servir para tararear: describe el timbre (qué instrumentos suenan) e ignora a propósito
la melodía, que es justo lo único que tiene un tarareo.

## 1. Shazam: huellas de audio

### Cómo funciona (Wang, 2003)

1. Espectrograma de la canción (11 025 Hz, ventanas de 93 ms cada 23 ms)
2. **Picos**: puntos que son el máximo de energía en su vecindad (±7 bandas, ±5 ventanas)
3. **Pares de picos**: cada pico "ancla" se empareja con los 8 siguientes que estén a menos de ~1.5 s.
   Cada par da un hash de 24 bits: frecuencia del ancla, frecuencia del otro pico y distancia en tiempo
4. **Índice invertido** ordenado por hash: `hash → [(canción, tiempo del ancla)]`. Buscar un hash es una
   búsqueda binaria (`np.searchsorted`)
5. **Consulta**: se sacan los hashes del fragmento y cada coincidencia vota por
   `(canción, tiempo en la canción − tiempo en el fragmento)`. La canción correcta junta muchos votos
   **con el mismo desfase**; las coincidencias por azar se reparten en desfases distintos

Funciona aunque el celular filtre frecuencias porque no mira la forma del espectro, solo *dónde* están
los picos más fuertes, y esos siguen ahí.

### Resultados (`python -m multimedia.huellas --evaluar --n 2000`)

2 000 canciones (previews de 30 s), 100 consultas sacadas del segundo 10 de cada canción:

| consulta | top 1 | top 5 | con MFCC (top 10) |
|---|---|---|---|
| fragmento de 5 s | 0.99 | 1.00 | 0.85 |
| 5 s + volumen y ruido | 0.99 | 1.00 | 0.37 |
| **5 s grabado con celular** | **0.99** | **1.00** | **0.00** |
| 3 s grabado con celular | 0.99 | 1.00 | — |
| 5 s + ruido 3 veces más fuerte que la música | 0.30 | 0.46 | — |

- Armar el índice: 90 s (más 65 s de leer los mp3); consulta: ~58 ms
- **El problema es el espacio:** 26.5 M de hashes, ~500 MB para 2 000 canciones. Con las 13 643 serían
  >3 GB en memoria. Para escalar hay que guardar menos picos por segundo (subir el umbral de energía o
  bajar `FAN`) y guardar el índice en disco por bloques, como el SPIMI de la Parte 3

### Cómo llevarlo al motor

Encaja con lo que ya tenemos: es un índice invertido de enteros. Se puede guardar con el hash extensible
o con un archivo ordenado por hash (como SPIMI), y la consulta sería algo como
`WHERE RECOGNIZE('grabacion.mp3')`. Librerías de referencia: `audfprint` (Dan Ellis) y `dejavu`.

## 2. "Cantas y reconoce": búsqueda por tarareo

### El cuidado con DTW

Lo primero que aparece al buscar es comparar la melodía con **DTW** (*Dynamic Time Warping*), que
alinea dos secuencias aunque vayan a distinta velocidad. El profesor advirtió tener cuidado:

- **DTW no es una métrica**: no cumple la desigualdad triangular. IVF, HNSW y cualquier índice que pode
  la búsqueda suponen una métrica, así que con DTW pueden descartar candidatos correctos. No se puede
  meter tal cual en un índice
- Cuesta O(n·m) por comparación: comparar contra toda la base una por una es lento

### Qué probamos (`python -m multimedia.tarareo --n 200`)

1. **Melodía de cada canción** con `librosa.pyin` (una nota por ventana de 23 ms). Cada canción se
   parte en tramos de 5 s cada 1 s
2. **Sin el tono**: a cada tramo se le resta su mediana, así cantar más agudo o más grave da la misma curva
3. Tres formas de comparar:
   - **vector fijo + coseno**: el tramo se estira a 64 puntos. Es un vector normal, así que se puede
     indexar con IVF/HNSW con coseno
   - **DTW contra todos los tramos**: lo que advirtió el profesor, como referencia de precisión
   - **híbrido**: el vector fijo elige 50 candidatos (esa parte sí puede ir en el índice) y DTW solo
     reordena esos 50. DTW nunca se usa para podar, así que no importa que no sea métrica
4. **Tarareo simulado**: no tenemos grabaciones de gente cantando, así que se toma la melodía de 5 s
   de la canción (desde el segundo 10) y se le cambia el tono (±5 semitonos), la velocidad (±15 %), se
   desafina (±0.3 semitonos) y se le quita el 10 % de las notas

200 canciones, 5 101 tramos, 100 tarareos:

| método | top 1 | top 5 | por consulta | ¿se puede indexar? |
|---|---|---|---|---|
| vector fijo + coseno | 0.74 | 0.81 | 0.4 ms | sí (IVF / HNSW) |
| DTW contra todos los tramos | 0.86 | 0.95 | 895 ms | no |
| **híbrido: coseno 50 + DTW** | **0.86** | 0.90 | **8.6 ms** | sí, la primera etapa |

- **DTW es el más preciso, pero 2 000 veces más lento** que el vector fijo, y crece con toda la base:
  con las 13 643 canciones (~350 000 tramos) serían ~1 min por consulta
- **El vector fijo pierde precisión** porque estirar a 64 puntos solo corrige una velocidad pareja; si el
  tarareo se apura en una parte y se frena en otra, las curvas ya no calzan
- **El híbrido da lo mejor de los dos:** el mismo top 1 que DTW en ~1 % del tiempo. Es la forma correcta
  de usar DTW con un índice: el índice (con una métrica de verdad) recorta la base y DTW solo reordena
  unos pocos. Si el vector fijo deja afuera a la canción correcta, DTW ya no puede recuperarla; por eso el
  top 5 baja un poco (0.90 vs 0.95). Pedir más candidatos lo sube a cambio de tiempo
- Calcular las melodías con pyin es lo más caro: ~2.4 s por canción (8 min para 200). Para la base
  completa serían ~9 h, una sola vez; se guardan en `data/features/melodias_<n>.npz`

### Lo que falta para que funcione de verdad

- **La voz está mezclada con los instrumentos.** pyin sobre la mezcla completa saca a veces la guitarra
  o el bajo en vez de la voz. Lo usual es separar la voz primero (por ejemplo con `demucs`) o usar un
  extractor de melodía para música polifónica (Melodia, Salamon y Gómez, 2012)
- **Probar con gente cantando de verdad**, que desafina, cambia el ritmo y olvida partes. Hay datasets
  públicos de tarareos (MIR-QBSH)
- **Los previews son de 30 s**: si no incluyen el coro, nadie va a cantar esa parte
- La alternativa moderna es la de Google *Hum to Search* (2020): una red aprende un embedding donde el
  tarareo y la canción quedan cerca. El resultado es un vector, así que entraría directo a nuestro
  HNSW, pero hay que entrenar el modelo

## Referencias

- Wang, A. (2003). An industrial-strength audio search algorithm. *ISMIR 2003*.
- Ellis, D. (2014). audfprint: landmark-based audio fingerprinting. https://github.com/dpwe/audfprint
- Mauch, M. y Dixon, S. (2014). pYIN: a fundamental frequency estimator using probabilistic threshold distributions. *ICASSP 2014*, 659–663.
- Keogh, E. y Ratanamahatana, C. A. (2005). Exact indexing of dynamic time warping. *Knowledge and Information Systems*, 7(3), 358–386.
- Salamon, J. y Gómez, E. (2012). Melody extraction from polyphonic music signals using pitch contour characteristics. *IEEE TASLP*, 20(6), 1759–1770.
- Défossez, A. et al. (2021). Hybrid spectrogram and waveform source separation (Demucs). *ISMIR 2021 Workshop on Music Source Separation*.
