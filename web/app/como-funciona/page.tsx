"use client";

// cómo funciona: el pipeline de la parte 4 paso a paso, calculado en vivo sobre una portada o una canción real
import { useEffect, useMemo, useRef, useState } from "react";
import styles from "./explica.module.css";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const media = (p: string) => `${API}/api/multimedia/assets/${p}`;

type Resumen = { portadas: number; descriptores_sift: number; audios: number; ventanas_mfcc: number; k: number; vectores_sift: number; vectores_mfcc: number; palabras_por_portada: number; palabras_por_audio: number; caras: number; personas: number };
type Ejemplo = { id: string; imagen: string; titulo?: string; artista?: string };
type Punto = { x: number; y: number; tam: number; angulo: number; palabra: number };
type DatosSift = { path: string; ancho: number; alto: number; puntos: Punto[]; descriptores: number[][]; tf: number[]; vector: number[]; idf: number[]; palabras_distintas: number };
type DatosMfcc = { track_id: string; titulo: string; artista: string; audio: string; portada: string; duracion: number; onda: number[]; espectrograma: number[][]; mel: number[][]; mfcc: number[][]; n_ventanas: number; paso_ventanas: number; palabras: number[]; tf: number[]; vector: number[]; palabras_distintas: number };

type DatosDog = { ancho: number; alto: number; sigmas: number[]; gaussianas: string[]; dogs: string[]; extremos: number[][]; n_extremos: number };
type DatosPalabras = { fondo: number[][]; centroides: number[][]; portada: number[][]; varianza: number[] };
type DatosIvf = { puntos: number[][]; centroides: number[][]; nlist: number; consulta: number; orden_celdas: number[]; abiertas: number[]; revisados: number; resultado: number[]; exactos: number[]; aciertos: number; imagenes: Record<string, string> };
type DatosHnsw = { nodos: number[][]; aristas: Record<string, number[][]>; entrada: number; max_nivel: number; consulta: number; consulta_xy: number[]; niveles: { nivel: number; camino: number[]; visitados: number[] }[]; nivel0_visitados: number[]; resultado: number[]; por_nivel: number[]; imagenes: Record<string, string> };

type DatosCaras = { path: string; nombre: string; ancho: number; alto: number; caja: number[]; puntos: Record<string, number[][]>; embedding: number[]; bordes: number[]; hist_misma: number[]; hist_otras: number[]; n_misma: number; n_otras: number; umbral: number; otras_fotos: string[]; mapa: number[][]; mapa_persona: number[][]; mapa_yo: number[]; vecinos: { nombre: string; foto: string; distancia: number; misma_persona: boolean; xy: number[] }[]; revisados: number; total: number };

async function pedir<T>(path: string): Promise<T> {
  const r = await fetch(`${API}${path}`);
  const d = await r.json();
  if (!r.ok) throw new Error(typeof d.detail === "string" ? d.detail : "Error en el servidor");
  return d as T;
}

const n = (v: number) => v.toLocaleString("es");
// un color por palabra del codebook: el ángulo dorado reparte bien los tonos
const colorPalabra = (p: number, a = 1) => `hsla(${(p * 137.508) % 360}, 70%, 48%, ${a})`;

function colorCalor(t: number) {
  // de azul oscuro a amarillo (parecido a viridis), t entre 0 y 1
  const c = [[33, 24, 74], [56, 88, 140], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
  const x = Math.max(0, Math.min(1, t)) * (c.length - 1);
  const i = Math.min(c.length - 2, Math.floor(x));
  const f = x - i;
  return c[i].map((v, j) => Math.round(v + (c[i + 1][j] - v) * f));
}

function MapaCalor({ M, alto = 140, marca, onElegir, invertir = true, etiqueta, porFila = false }: { M: number[][]; alto?: number; marca?: number; onElegir?: (f: number) => void; invertir?: boolean; etiqueta?: string; porFila?: boolean }) {
  // matriz (filas x columnas) como imagen; marca = columna elegida (fracción 0-1)
  const ref = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c || !M.length) return;
    const filas = M.length, cols = M[0].length;
    c.width = cols; c.height = filas;
    const ctx = c.getContext("2d")!;
    // porFila: cada fila con su propia escala (en los mfcc el primer coeficiente es mucho mayor que el resto)
    const rango = (f: number[]) => [Math.min(...f), Math.max(...f)];
    let [min, max] = [Infinity, -Infinity];
    M.forEach((f) => { const [a, b] = rango(f); min = Math.min(min, a); max = Math.max(max, b); });
    const img = ctx.createImageData(cols, filas);
    for (let i = 0; i < filas; i++) {
      const [mi, ma] = porFila ? rango(M[i]) : [min, max];
      for (let j = 0; j < cols; j++) {
      const [r, g, b] = colorCalor((M[i][j] - mi) / (ma - mi || 1));
      const k = ((invertir ? filas - 1 - i : i) * cols + j) * 4;
      img.data[k] = r; img.data[k + 1] = g; img.data[k + 2] = b; img.data[k + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [M, invertir, porFila]);
  return (
    <div className={styles.mapa} style={{ height: alto }} onClick={(e) => { const r = e.currentTarget.getBoundingClientRect(); onElegir?.((e.clientX - r.left) / r.width); }}>
      <canvas ref={ref} />
      {marca !== undefined && <i style={{ left: `${marca * 100}%` }} />}
      {etiqueta && <span>{etiqueta}</span>}
    </div>
  );
}

function Onda({ valores, marca, onElegir }: { valores: number[]; marca: number; onElegir: (f: number) => void }) {
  // forma de onda espejada: cada barra es el máximo de amplitud de un pedacito del audio
  const max = Math.max(...valores, 1e-9);
  const ancho = 100 / valores.length;
  return (
    <div className={styles.mapa} style={{ height: 80 }} onClick={(e) => { const r = e.currentTarget.getBoundingClientRect(); onElegir((e.clientX - r.left) / r.width); }}>
      <svg viewBox="0 0 100 80" preserveAspectRatio="none" style={{ width: "100%", height: "100%", display: "block", background: "rgba(255,255,255,.45)" }}>
        {valores.map((v, i) => { const h = (v / max) * 38; return <rect key={i} x={i * ancho} y={40 - h} width={ancho} height={2 * h} fill={i / valores.length <= marca ? "var(--amber)" : "#9aa29d"} />; })}
      </svg>
      <i style={{ left: `${marca * 100}%` }} />
      <span>amplitud en el tiempo</span>
    </div>
  );
}

function Barras({ valores, resaltar, color = "var(--amber)", alto = 90, etiqueta }: { valores: number[]; resaltar?: number; color?: string; alto?: number; etiqueta?: string }) {
  const max = Math.max(...valores.map(Math.abs), 1e-9);
  const ancho = 100 / valores.length;
  const hayNeg = valores.some((v) => v < 0);
  return (
    <figure className={styles.figura}>
    {etiqueta && <figcaption>{etiqueta}</figcaption>}
    <svg className={styles.barras} viewBox={`0 0 100 ${alto}`} preserveAspectRatio="none" style={{ height: alto }}>
      {hayNeg && <line x1="0" x2="100" y1={alto / 2} y2={alto / 2} stroke="var(--line)" strokeWidth="0.3" />}
      {valores.map((v, i) => {
        const h = (Math.abs(v) / max) * (hayNeg ? alto / 2 : alto);
        const y = hayNeg ? (v >= 0 ? alto / 2 - h : alto / 2) : alto - h;
        return <rect key={i} x={i * ancho} y={y} width={Math.max(ancho * 0.85, 0.08)} height={Math.max(h, v ? 0.4 : 0)} fill={i === resaltar ? "#c0392b" : color} />;
      })}
      {resaltar !== undefined && <rect x={resaltar * ancho - 0.4} y={0} width={ancho + 0.8} height={alto} fill="none" stroke="#c0392b" strokeWidth="0.4" vectorEffect="non-scaling-stroke" />}
    </svg>
    </figure>
  );
}

function Descriptor({ valores }: { valores: number[] }) {
  // 128 = 4 x 4 celdas, cada una con 8 direcciones: se dibuja como flechas, más larga = más gradiente
  const max = Math.max(...valores, 1);
  return (
    <svg className={styles.descriptor} viewBox="0 0 160 160">
      {Array.from({ length: 16 }, (_, c) => {
        const cx = (c % 4) * 40 + 20, cy = Math.floor(c / 4) * 40 + 20;
        return <g key={c}>
          <rect x={cx - 20} y={cy - 20} width="40" height="40" fill="none" stroke="var(--line)" />
          {Array.from({ length: 8 }, (_, o) => {
            const v = valores[c * 8 + o] / max, a = (o * Math.PI) / 4, l = 3 + v * 15;
            return <line key={o} x1={cx} y1={cy} x2={cx + Math.cos(a) * l} y2={cy - Math.sin(a) * l} stroke="var(--amber)" strokeWidth={1 + v * 2} strokeLinecap="round" />;
          })}
        </g>;
      })}
    </svg>
  );
}

function Selector({ ejemplos, elegido, onElegir, onMas }: { ejemplos: Ejemplo[]; elegido: string; onElegir: (id: string) => void; onMas: () => void }) {
  return (
    <div className={styles.selector}>
      {ejemplos.map((e) => <button key={e.id} className={e.id === elegido ? styles.elegido : ""} onClick={() => onElegir(e.id)} title={e.titulo ? `${e.titulo} · ${e.artista}` : e.id}>
        <img src={media(e.imagen)} alt="" loading="lazy" />
        {e.titulo && <span>{e.titulo}</span>}
      </button>)}
      <button className={styles.mas} onClick={onMas}>otros ↻</button>
    </div>
  );
}

function SeccionSift({ r }: { r: Resumen | null }) {
  const [ejemplos, setEjemplos] = useState<Ejemplo[]>([]);
  const [path, setPath] = useState("");
  const [d, setD] = useState<DatosSift | null>(null);
  const [error, setError] = useState("");
  const [punto, setPunto] = useState(0);
  const [dog, setDog] = useState<DatosDog | null>(null);
  const [pal, setPal] = useState<DatosPalabras | null>(null);
  const cargarEjemplos = () => pedir<{ ids: Ejemplo[] }>("/api/explica/ejemplos?tipo=sift&n=11").then((x) => { setEjemplos(x.ids); if (!path) setPath(x.ids[0]?.id ?? ""); });
  useEffect(() => { void cargarEjemplos(); }, []); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!path) return;
    setD(null); setError(""); setPunto(0);
    pedir<DatosSift>(`/api/explica/sift?path=${encodeURIComponent(path)}`).then(setD).catch((e: Error) => setError(e.message));
    setDog(null); setPal(null);
    pedir<DatosDog>(`/api/explica/dog?path=${encodeURIComponent(path)}`).then(setDog).catch(() => setDog(null));
    pedir<DatosPalabras>(`/api/explica/palabras2d?path=${encodeURIComponent(path)}`).then(setPal).catch(() => setPal(null));
  }, [path]);
  const p = d?.puntos[punto];
  const misma = useMemo(() => (d && p ? d.puntos.filter((q) => q.palabra === p.palabra).length : 0), [d, p]);
  return (
    <section className={styles.seccion} id="sift">
      <h2><b>1</b> Imágenes: SIFT + bolsa de palabras visuales</h2>
      <p className={styles.bajada}>Una portada no se compara píxel por píxel: se buscan cientos de <i>puntos clave</i>, cada uno se describe con 128 números y se traduce a una de {r ? n(r.k) : "1 024"} "palabras visuales". La portada queda como un histograma de palabras. Elige una portada y toca un punto.</p>
      <Selector ejemplos={ejemplos} elegido={path} onElegir={setPath} onMas={cargarEjemplos} />
      {error && <p className={styles.error}>{error}</p>}
      {d && p && <div className={styles.dosColumnas}>
        <div>
          <div className={styles.imagenPuntos}>
            <img src={media(d.path)} alt="portada" />
            <svg viewBox={`0 0 ${d.ancho} ${d.alto}`}>
              {d.puntos.map((q, i) => <g key={i} onClick={() => setPunto(i)} className={styles.punto} opacity={q.palabra === p.palabra || i === punto ? 1 : 0.35}>
                <circle cx={q.x} cy={q.y} r={Math.max(4, q.tam / 2)} fill={i === punto ? "rgba(255,255,255,.25)" : "none"} stroke={colorPalabra(q.palabra)} strokeWidth={i === punto ? 3.5 : 1.8} vectorEffect="non-scaling-stroke" />
                <line x1={q.x} y1={q.y} x2={q.x + Math.cos((q.angulo * Math.PI) / 180) * Math.max(4, q.tam / 2)} y2={q.y - Math.sin((q.angulo * Math.PI) / 180) * Math.max(4, q.tam / 2)} stroke={colorPalabra(q.palabra)} strokeWidth={1.8} vectorEffect="non-scaling-stroke" />
              </g>)}
            </svg>
          </div>
          <p className={styles.nota}>Cada círculo es un punto clave: su tamaño es la escala donde se encontró y la rayita, su orientación. El color es su palabra visual.</p>
        </div>
        <div className={styles.pasos}>
          <div className={styles.paso}><span>Paso 1 · puntos clave</span>
            <p>Se pasa a <b>gris</b> (SIFT no usa el color: 1 canal) y se achica a {d.ancho}×{d.alto} px. SIFT busca esquinas y manchas que se mantienen al cambiar el tamaño (máximos de una diferencia de gaussianas) y se queda con los <b>{d.puntos.length}</b> de más contraste.</p></div>
          <div className={styles.paso} id="descriptor"><span>Paso 2 · descriptor del punto #{punto + 1}</span>
            <div className={styles.fila}><Descriptor valores={d.descriptores[punto]} />
              <p>Alrededor del punto se arma una grilla de <b>4 × 4</b> celdas, girada según su orientación (por eso aguanta rotaciones). En cada celda se cuenta cuánto cambia el brillo en <b>8 direcciones</b>: 4 × 4 × 8 = <b>128 números</b>. Después se normaliza (RootSIFT).</p></div></div>
          <div className={styles.paso}><span>Paso 3 · palabra visual</span>
            <p>El descriptor se compara con los {r ? n(r.k) : "1 024"} centroides del codebook (K-Means sobre los {r ? n(r.descriptores_sift) : "9 M"} descriptores de la colección) y cae en la <b style={{ color: colorPalabra(p.palabra) }}>palabra #{p.palabra}</b>. En esta portada hay <b>{misma}</b> {misma === 1 ? "punto" : "puntos"} con esa palabra (los que quedan resaltados).</p></div>
          <div className={styles.paso} id="histograma"><span>Paso 4 · histograma y TF-IDF</span>
            <p>Se cuenta cuántos puntos cayeron en cada palabra (TF). Esta portada usa <b>{d.palabras_distintas}</b> de las {n(d.tf.length)} palabras. Después se multiplica por el IDF: las palabras que aparecen en casi todas las portadas pesan poco.</p>
            <Barras valores={d.tf} resaltar={p.palabra} etiqueta="TF: fracción de puntos en cada palabra" />
            <Barras valores={d.vector} resaltar={p.palabra} color="var(--green)" etiqueta="TF-IDF normalizado: el vector que se indexa" />
            <p className={styles.nota}>Ese vector de {n(d.vector.length)} números es lo que guardan el IVF y el HNSW. Dos portadas se parecen si sus vectores apuntan hacia el mismo lado (coseno).</p></div>
        </div>
      </div>}
      {dog && <div className={styles.bloque} id="dog">
        <h3>¿De dónde salen los puntos? La diferencia de gaussianas (DoG)</h3>
        <p className={styles.bajada}>La portada se borronea cada vez más (gaussianas con σ creciente). Restando cada imagen borrosa de la siguiente quedan solo los detalles de un tamaño: eso es la <b>DoG</b>. Un punto clave es un píxel más claro (o más oscuro) que sus <b>26 vecinos</b>: 8 en su misma DoG y 9 arriba y 9 abajo. Así SIFT encuentra manchas y esquinas en cualquier escala.</p>
        <div className={styles.piramide}>
          <span className={styles.rotulo}>gaussianas</span>
          {dog.gaussianas.map((g, i) => <figure key={i}><img src={g} alt="" /><figcaption>σ = {dog.sigmas[i]}</figcaption></figure>)}
        </div>
        <div className={styles.piramide}>
          <span className={styles.rotulo}>DoG = resta</span>
          {dog.dogs.map((g, i) => <figure key={i} className={i === 1 ? styles.dogElegida : ""}><div className={styles.dogImg}><img src={g} alt="" />
            {i === 1 && <svg viewBox={`0 0 ${dog.ancho} ${dog.alto}`}>{dog.extremos.map(([x, y], j) => <circle key={j} cx={x} cy={y} r={4} fill="none" stroke="#c0392b" strokeWidth={1.5} vectorEffect="non-scaling-stroke" />)}</svg>}</div>
            <figcaption>σ{dog.sigmas[i + 1]} − σ{dog.sigmas[i]}{i === 1 ? ` · ${dog.n_extremos} extremos` : ""}</figcaption></figure>)}
        </div>
        <p className={styles.nota}>En rojo, los extremos de la segunda DoG comparada con la de arriba y la de abajo. SIFT hace esto en varias octavas (la imagen a la mitad, a un cuarto…), descarta los de poco contraste y los que están sobre un borde, y se queda con los 500 más fuertes.</p>
      </div>}
      {pal && p && <div className={styles.bloque} id="palabras">
        <h3>La bolsa de palabras visuales en 2D</h3>
        <p className={styles.bajada}>Cada puntito es un descriptor de 128 números (3 600 tomados de 60 portadas al azar), aplastado a 2 dimensiones con PCA. K-Means pone centroides (las <b>×</b>) y cada descriptor pertenece al centroide más cercano: <b>cada zona de color es una palabra</b>. La versión simple usa 12 palabras para que se vea; el codebook real tiene {n(pal.centroides.length)}. Los círculos grandes son los descriptores de esta portada; el elegido tiene una línea hacia el centroide de su palabra.</p>
        <EspacioPalabras pal={pal} punto={punto} setPunto={setPunto} />
        <p className={styles.nota}>Ojo: el plano explica solo {((pal.varianza[0] + pal.varianza[1]) * 100).toFixed(0)} % de la variación de los 128 números, así que dos puntos cercanos aquí no siempre lo están en 128D. Sirve para ver la idea: el espacio de descriptores se parte en {n(pal.centroides.length)} zonas y cada zona es una palabra.</p>
      </div>}
    </section>
  );
}

// 12 colores bien distintos, uno por palabra de la versión simple
const PALETA12 = ["#e6194b", "#3cb44b", "#ffb000", "#4363d8", "#f58231", "#911eb4", "#00a5a8", "#f032e6", "#7a8f00", "#9a6324", "#000075", "#808080"];
const colorZona = (z: number, a = 1) => {
  const h = PALETA12[z % 12];
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
  return `rgba(${r}, ${g}, ${b}, ${a})`;
};

function kmeans2d(pts: number[][], k: number) {
  // k-means de juguete en el plano (solo para la vista simple): k centroides y la zona de cada punto
  let C = pts.filter((_, i) => i % Math.floor(pts.length / k) === 0).slice(0, k).map((p) => [p[0], p[1]]);
  const zona = (x: number, y: number) => C.reduce((m, c, j) => ((x - c[0]) ** 2 + (y - c[1]) ** 2 < (x - C[m][0]) ** 2 + (y - C[m][1]) ** 2 ? j : m), 0);
  for (let it = 0; it < 20; it++) {
    const suma = C.map(() => [0, 0, 0]);
    pts.forEach(([x, y]) => { const z = zona(x, y); suma[z][0] += x; suma[z][1] += y; suma[z][2] += 1; });
    C = suma.map((s2, j) => (s2[2] ? [s2[0] / s2[2], s2[1] / s2[2]] : C[j]));
  }
  return { C, zona };
}

function EspacioPalabras(props: { pal: DatosPalabras; punto: number; setPunto: (i: number) => void }) {
  const [simple, setSimple] = useState(true);
  return <>
    <div className={styles.controles}>
      <button className={simple ? styles.activo : ""} onClick={() => setSimple(true)}>versión simple: 12 palabras</button>
      <button className={!simple ? styles.activo : ""} onClick={() => setSimple(false)}>codebook real: {n(props.pal.centroides.length)} palabras</button>
    </div>
    {simple ? <EspacioSimple {...props} /> : <EspacioReal {...props} />}
  </>;
}

function EspacioSimple({ pal, punto, setPunto }: { pal: DatosPalabras; punto: number; setPunto: (i: number) => void }) {
  const W = 1000, H = 560;
  const X = (x: number) => 20 + x * (W - 40), Y = (y: number) => 20 + (1 - y) * (H - 40);
  const { C, zona } = useMemo(() => kmeans2d(pal.fondo, 12), [pal]);
  const fondo = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    // cada píxel del plano se pinta del color de su centroide más cercano: las zonas son las palabras
    const c = fondo.current;
    if (!c) return;
    c.width = 250; c.height = 140;
    const ctx = c.getContext("2d")!;
    for (let i = 0; i < c.width; i++) for (let j = 0; j < c.height; j++) {
      const x = ((i / c.width) * W - 20) / (W - 40), y = 1 - ((j / c.height) * H - 20) / (H - 40);
      ctx.fillStyle = colorZona(zona(x, y), 0.16);
      ctx.fillRect(i, j, 1, 1);
    }
  }, [zona]);
  const elegido = pal.portada[punto];
  const zElegido = elegido ? zona(elegido[0], elegido[1]) : 0;
  return (
    <div className={styles.planoCapas}>
      <canvas ref={fondo} />
      <svg className={styles.plano} viewBox={`0 0 ${W} ${H}`} style={{ background: "transparent" }}>
        {pal.fondo.map(([x, y], i) => <circle key={i} cx={X(x)} cy={Y(y)} r={2.2} fill={colorZona(zona(x, y), 0.45)} />)}
        {pal.portada.map(([x, y], i) => <circle key={`p${i}`} cx={X(x)} cy={Y(y)} r={i === punto ? 9 : 4.5} fill={colorZona(zona(x, y), 0.95)} stroke={i === punto ? "#1d2728" : "#fff"} strokeWidth={i === punto ? 2.5 : 1} className={styles.punto} onClick={() => setPunto(i)} />)}
        {C.map(([x, y], j) => <g key={j}><path d={`M${X(x) - 9},${Y(y) - 9}L${X(x) + 9},${Y(y) + 9}M${X(x) - 9},${Y(y) + 9}L${X(x) + 9},${Y(y) - 9}`} stroke="#1d2728" strokeWidth={3} /><text x={X(x) + 12} y={Y(y) + 5} fontSize="15" fontWeight="700" fill="#1d2728">palabra {j + 1}</text></g>)}
        {elegido && <line x1={X(elegido[0])} y1={Y(elegido[1])} x2={X(C[zElegido][0])} y2={Y(C[zElegido][1])} stroke="#c0392b" strokeWidth={2.5} strokeDasharray="5 4" />}
      </svg>
    </div>
  );
}

function EspacioReal({ pal, punto, setPunto }: { pal: DatosPalabras; punto: number; setPunto: (i: number) => void }) {
  const W = 1000, H = 560;
  const X = (x: number) => 20 + x * (W - 40), Y = (y: number) => 20 + (1 - y) * (H - 40);
  const elegido = pal.portada[punto];
  const centro = elegido ? pal.centroides[elegido[2]] : null;
  return (
    <svg className={styles.plano} viewBox={`0 0 ${W} ${H}`}>
      {pal.fondo.map(([x, y, w], i) => <circle key={i} cx={X(x)} cy={Y(y)} r={2.2} fill={colorPalabra(w, 0.35)} />)}
      {pal.centroides.map(([x, y], i) => <path key={i} d={`M${X(x) - 3},${Y(y) - 3}L${X(x) + 3},${Y(y) + 3}M${X(x) - 3},${Y(y) + 3}L${X(x) + 3},${Y(y) - 3}`} stroke="rgba(29,39,40,.55)" strokeWidth={1} />)}
      {pal.portada.map(([x, y, w], i) => <circle key={`p${i}`} cx={X(x)} cy={Y(y)} r={i === punto ? 9 : 5} fill={colorPalabra(w, i === punto ? 1 : 0.8)} stroke={i === punto ? "#1d2728" : "#fff"} strokeWidth={i === punto ? 2.5 : 1} className={styles.punto} onClick={() => setPunto(i)} />)}
      {elegido && centro && <>
        <line x1={X(elegido[0])} y1={Y(elegido[1])} x2={X(centro[0])} y2={Y(centro[1])} stroke="#c0392b" strokeWidth={2} strokeDasharray="5 4" />
        <path d={`M${X(centro[0]) - 8},${Y(centro[1]) - 8}L${X(centro[0]) + 8},${Y(centro[1]) + 8}M${X(centro[0]) - 8},${Y(centro[1]) + 8}L${X(centro[0]) + 8},${Y(centro[1]) - 8}`} stroke="#c0392b" strokeWidth={3} />
        <text x={X(centro[0]) + 12} y={Y(centro[1]) - 10} fontSize="16" fill="#c0392b" fontWeight="700">palabra #{elegido[2]}</text>
      </>}
    </svg>
  );
}

function SeccionIndices({ r }: { r: Resumen | null }) {
  const [consulta, setConsulta] = useState(7);
  const [nprobe, setNprobe] = useState(3);
  const [iv, setIv] = useState<DatosIvf | null>(null);
  const [hn, setHn] = useState<DatosHnsw | null>(null);
  useEffect(() => { pedir<DatosIvf>(`/api/explica/ivf?consulta=${consulta}&nprobe=${nprobe}`).then(setIv).catch(() => setIv(null)); }, [consulta, nprobe]);
  useEffect(() => { pedir<DatosHnsw>(`/api/explica/hnsw?consulta=${consulta % 300}`).then(setHn).catch(() => setHn(null)); }, [consulta]);
  const W = 1000, H = 520;
  const X = (x: number) => 20 + x * (W - 40), Y = (y: number) => 20 + (1 - y) * (H - 40);
  const colorCelda = (c: number, a = 1) => `hsla(${(c * 137.508) % 360}, 60%, 50%, ${a})`;
  const abiertas = new Set(iv?.abiertas ?? []);
  const resultado = new Set(iv?.resultado ?? []);
  const exactos = new Set(iv?.exactos ?? []);
  return (
    <section className={styles.seccion} id="indices">
      <h2><b>3</b> Los índices: encontrar los parecidos sin revisar todo</h2>
      <p className={styles.bajada}>Ya tenemos un vector por portada. Buscar los parecidos comparando con los {r ? n(r.vectores_sift) : "20 005"} vectores (búsqueda lineal) funciona, pero es lento cuando la base crece. Los índices revisan solo una parte. Aquí hay <b>2 000 portadas reales</b>, aplastadas a 2D con PCA para poder dibujarlas (el índice trabaja en las 1 024 dimensiones).</p>
      <div className={styles.controles}>
        <button onClick={() => setConsulta(Math.floor(Math.random() * 2000))}>otra portada de consulta ↻</button>
        {iv && <span className={styles.consultaMini}><img src={media(iv.imagenes[String(iv.consulta)])} alt="" /> consulta</span>}
      </div>

      {iv && <div className={styles.bloque} id="ivf">
        <h3>IVF: dividir en celdas y abrir solo las más cercanas</h3>
        <p className={styles.bajada}>Antes de buscar, K-Means parte las 2 000 portadas en <b>{iv.nlist} celdas</b> (cada color es una celda, cada número su centroide). Para una consulta se miden las distancias a los {iv.nlist} centroides y solo se revisan las <b>nprobe</b> celdas más cercanas.</p>
        <div className={styles.controles}>
          <label>nprobe = <b>{nprobe}</b> <input type="range" min={1} max={iv.nlist} value={nprobe} onChange={(e) => setNprobe(Number(e.target.value))} /></label>
          <span>revisó <b>{n(iv.revisados)}</b> de 2 000 vectores ({(iv.revisados / 20).toFixed(0)} %) · encontró <b>{iv.aciertos} de 10</b> vecinos exactos</span>
        </div>
        <svg className={styles.plano} viewBox={`0 0 ${W} ${H}`}>
          {iv.puntos.map(([x, y, c], i) => <circle key={i} cx={X(x)} cy={Y(y)} r={exactos.has(i) || resultado.has(i) ? 6 : 3} fill={colorCelda(c, abiertas.has(c) ? 0.9 : 0.12)} stroke={resultado.has(i) ? "#1d2728" : exactos.has(i) ? "#c0392b" : "none"} strokeWidth={2} />)}
          {iv.centroides.map(([x, y], c) => <text key={c} x={X(x)} y={Y(y)} fontSize={abiertas.has(c) ? 15 : 10} fontWeight={abiertas.has(c) ? 800 : 400} fill={abiertas.has(c) ? "#1d2728" : "rgba(29,39,40,.35)"} textAnchor="middle">{c}</text>)}
          {iv.puntos[iv.consulta] && <path d={estrella(X(iv.puntos[iv.consulta][0]), Y(iv.puntos[iv.consulta][1]), 13)} fill="#f1c40f" stroke="#1d2728" strokeWidth={2} />}
        </svg>
        <p className={styles.nota}>★ la consulta · borde negro: lo que devolvió el IVF · borde rojo: los 10 vecinos exactos que no encontró. Con pocas celdas abiertas es rapidísimo pero se le escapan vecinos que quedaron en celdas cerradas; subiendo nprobe se acerca a la búsqueda lineal.</p>
        <Miniaturas titulo="Lo que devolvió el IVF" ids={iv.resultado} imagenes={iv.imagenes} ok={exactos} />
      </div>}

      {hn && <div className={styles.bloque} id="hnsw">
        <h3>HNSW: un grafo por niveles, como autopistas y calles</h3>
        <p className={styles.bajada}>Cada portada es un nodo conectado con sus vecinas. Al insertar una portada se sortea hasta qué nivel llega: casi todas se quedan en el nivel 0 y muy pocas suben (cada nivel tiene ~1/M de los nodos del de abajo). Arriba hay pocos nodos con conexiones largas (autopistas); abajo están todos, con conexiones cortas (calles). La búsqueda <b>entra por el nodo del nivel más alto</b>, salta al vecino más cercano a la consulta hasta no poder mejorar, <b>baja al mismo nodo un nivel más abajo</b> y repite hasta el nivel 0. Aquí, 300 portadas con M = 4 para que se pueda ver.</p>
        <NivelesHnsw hn={hn} X={X} />
        <CaminoHnsw hn={hn} />
      </div>}
    </section>
  );
}

type PasoHnsw = { tipo: "entrada" | "salto" | "baja" | "explora" | "resultado"; nivel: number; de?: number; a?: number; visto?: number; texto: string };

function pasosHnsw(hn: DatosHnsw): PasoHnsw[] {
  // la búsqueda como una lista de pasos para animarla
  const pasos: PasoHnsw[] = [{ tipo: "entrada", nivel: hn.max_nivel, a: hn.entrada, texto: `Entra por el único nodo del nivel ${hn.max_nivel} (el punto de entrada del grafo).` }];
  for (let l = hn.max_nivel; l >= 1; l--) {
    const cam = hn.niveles.find((x) => x.nivel === l)?.camino ?? [];
    if (cam.length <= 1) pasos.push({ tipo: "salto", nivel: l, texto: `Nivel ${l}: mira a sus vecinos y ninguno está más cerca de la ★, así que se queda donde está.` });
    cam.slice(1).forEach((x, i) => pasos.push({ tipo: "salto", nivel: l, de: cam[i], a: x, texto: `Nivel ${l}: salta a un vecino que está más cerca de la ★.` }));
    pasos.push({ tipo: "baja", nivel: l, texto: `No puede acercarse más en el nivel ${l}: baja al mismo nodo en el nivel ${l - 1}.` });
  }
  const total = hn.nivel0_visitados.length;
  for (let v = 8; v < total; v += 8) pasos.push({ tipo: "explora", nivel: 0, visto: v, texto: `Nivel 0: explora los vecinos de los mejores candidatos (${v} de 300 nodos mirados).` });
  pasos.push({ tipo: "explora", nivel: 0, visto: total, texto: `Nivel 0: ya no hay candidatos que mejoren; miró ${total} de 300 nodos.` });
  pasos.push({ tipo: "resultado", nivel: 0, visto: total, texto: `Devuelve los 10 nodos más cercanos a la ★ (en verde), sin haber revisado los 300.` });
  return pasos;
}

function NivelesHnsw({ hn, X }: { hn: DatosHnsw; X: (x: number) => number }) {
  // todos los niveles apilados con las mismas posiciones: arriba el más alto, abajo el 0; se puede animar
  const W = 1000, alto = 150, gap = 34;
  const pasos = useMemo(() => pasosHnsw(hn), [hn]);
  const [paso, setPaso] = useState(pasos.length - 1);
  const [reproduciendo, setReproduciendo] = useState(false);
  useEffect(() => { setPaso(pasos.length - 1); setReproduciendo(false); }, [pasos]);
  useEffect(() => {
    if (!reproduciendo) return;
    if (paso >= pasos.length - 1) { setReproduciendo(false); return; }
    const t = setTimeout(() => setPaso((p) => p + 1), pasos[paso + 1]?.tipo === "explora" ? 450 : 1100);
    return () => clearTimeout(t);
  }, [reproduciendo, paso, pasos]);
  const actual = pasos[paso];
  const hecho = pasos.slice(0, paso + 1);
  const niveles = Array.from({ length: hn.max_nivel + 1 }, (_, i) => hn.max_nivel - i);
  const top = (l: number) => 10 + (hn.max_nivel - l) * (alto + gap);
  const Yl = (y: number, l: number) => top(l) + 12 + (1 - y) * (alto - 24);
  const H = top(0) + alto + 10;
  const llego = (l: number) => hecho.some((p) => p.nivel === l || (p.tipo === "baja" && p.nivel === l + 1));
  const saltos = (l: number) => hecho.filter((p) => p.tipo === "salto" && p.nivel === l && p.de !== undefined) as Required<PasoHnsw>[];
  const bajo = (l: number) => hecho.some((p) => p.tipo === "baja" && p.nivel === l);
  const vistos0 = new Set(hn.nivel0_visitados.slice(0, Math.max(0, ...hecho.map((p) => p.visto ?? 0))));
  const finales = new Set(actual.tipo === "resultado" ? hn.resultado : []);
  const posicion = (l: number) => { const s2 = saltos(l); return s2.length ? s2[s2.length - 1].a : (l === hn.max_nivel ? hn.entrada : undefined); };
  let actualNodo = hn.entrada;
  hecho.forEach((p) => { if (p.tipo === "salto" && p.a !== undefined) actualNodo = p.a; });
  function reproducir() { if (paso >= pasos.length - 1) setPaso(0); setReproduciendo(true); }
  return <>
    <div className={styles.controles}>
      {reproduciendo ? <button onClick={() => setReproduciendo(false)}>⏸ pausa</button> : <button className={styles.activo} onClick={reproducir}>▶ animar la búsqueda</button>}
      <button onClick={() => { setReproduciendo(false); setPaso(Math.max(0, paso - 1)); }}>‹ paso</button>
      <button onClick={() => { setReproduciendo(false); setPaso(Math.min(pasos.length - 1, paso + 1)); }}>paso ›</button>
      <button onClick={() => { setReproduciendo(false); setPaso(0); }}>⟲ desde el inicio</button>
      <span className={styles.pasoTexto}><b>Paso {paso + 1} de {pasos.length}.</b> {actual.texto}</span>
    </div>
    <svg className={styles.plano} viewBox={`0 0 ${W} ${H}`}>
      <defs><marker id="flecha" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0L10,5L0,10z" fill="#c0392b" /></marker></defs>
      {niveles.map((l) => {
        const activo = actual.nivel === l;
        const visto = llego(l);
        const vis = l >= 1 ? new Set(visto ? hn.niveles.find((x) => x.nivel === l)?.visitados : []) : vistos0;
        const s2 = saltos(l), fin = posicion(l);
        return <g key={l} opacity={visto ? 1 : 0.45}>
          <rect x={4} y={top(l)} width={W - 8} height={alto} fill={activo ? "rgba(231,214,183,.45)" : "rgba(255,255,255,.4)"} stroke={activo ? "var(--amber)" : "var(--line)"} strokeWidth={activo ? 2.5 : 1} />
          <text x={12} y={top(l) + 18} fontSize="14" fontWeight="700" fill="#1d2728">nivel {l}</text>
          <text x={12} y={top(l) + 34} fontSize="12" fill="var(--muted)">{hn.por_nivel[l]} {hn.por_nivel[l] === 1 ? "nodo" : "nodos"}</text>
          {(hn.aristas[String(l)] ?? []).map(([a2, b2], i) => <line key={i} x1={X(hn.nodos[a2][0])} y1={Yl(hn.nodos[a2][1], l)} x2={X(hn.nodos[b2][0])} y2={Yl(hn.nodos[b2][1], l)} stroke="rgba(29,39,40,.15)" strokeWidth={l > 0 ? 1.5 : 0.6} />)}
          {s2.map((p, i) => <line key={`c${i}`} x1={X(hn.nodos[p.de][0])} y1={Yl(hn.nodos[p.de][1], l)} x2={X(hn.nodos[p.a][0])} y2={Yl(hn.nodos[p.a][1], l)} stroke="#c0392b" strokeWidth={3} markerEnd="url(#flecha)" />)}
          {hn.nodos.map(([x, y, lv], i) => lv >= l ? <circle key={i} cx={X(x)} cy={Yl(y, l)} r={finales.has(i) && l === 0 ? 7 : i === fin && visto ? 7 : vis.has(i) ? 4 : 2.6}
            fill={finales.has(i) && l === 0 ? "var(--green)" : i === fin && visto ? "#c0392b" : i === hn.entrada ? "#1d2728" : vis.has(i) ? "var(--amber)" : "#5b6b8c"} stroke="#fff" strokeWidth={0.8} /> : null)}
          {activo && l > 0 && <circle cx={X(hn.nodos[actualNodo][0])} cy={Yl(hn.nodos[actualNodo][1], l)} r={13} fill="none" stroke="#c0392b" strokeWidth={2} className={styles.pulso} />}
          <path d={estrella(X(hn.consulta_xy[0]), Yl(hn.consulta_xy[1], l), 9)} fill="#f1c40f" stroke="#1d2728" strokeWidth={1.5} />
          {l > 0 && bajo(l) && fin !== undefined && <line x1={X(hn.nodos[fin][0])} y1={Yl(hn.nodos[fin][1], l)} x2={X(hn.nodos[fin][0])} y2={Yl(hn.nodos[fin][1], l - 1)} stroke="#c0392b" strokeWidth={1.8} strokeDasharray="4 4" />}
        </g>;
      })}
    </svg>
  </>;
}

function CaminoHnsw({ hn }: { hn: DatosHnsw }) {
  // el recorrido como una tira: por dónde entra, a qué portada salta en cada nivel y qué encuentra
  const pasos: { texto: string; nodo: number }[] = [{ texto: `entra en el nivel ${hn.max_nivel}`, nodo: hn.entrada }];
  hn.niveles.forEach((nv) => nv.camino.slice(1).forEach((x) => pasos.push({ texto: `salta en el nivel ${nv.nivel}`, nodo: x })));
  return (
    <div className={styles.camino}>
      <span>El camino de la búsqueda</span>
      <div>
        <figure className={styles.pasoConsulta}><img src={media(hn.imagenes[String(hn.consulta)])} alt="" /><figcaption>★ consulta</figcaption></figure>
        {pasos.map((p, i) => <figure key={i}><img src={media(hn.imagenes[String(p.nodo)])} alt="" /><figcaption>{i + 1}. {p.texto}</figcaption></figure>)}
        <figure className={styles.pasoFinal}><div className={styles.miniGrid}>{hn.resultado.slice(0, 4).map((x) => <img key={x} src={media(hn.imagenes[String(x)])} alt="" />)}</div><figcaption>nivel 0: explora {hn.nivel0_visitados.length} de 300 nodos y devuelve los 10 más parecidos</figcaption></figure>
      </div>
      <p className={styles.nota}>En cada panel: ● negro la entrada · rojo el camino (flechas) · ámbar los nodos que miró · línea punteada: baja al mismo nodo en el nivel de abajo · verde (nivel 0) los resultados · ★ la consulta, dibujada en todos los niveles para ver cómo se acerca.</p>
    </div>
  );
}

const recorte = (p: string) => `${API}/api/app/cara-recorte?path=${encodeURIComponent(p)}`;

function SeccionCaras({ r }: { r: Resumen | null }) {
  const [ejemplos, setEjemplos] = useState<Ejemplo[]>([]);
  const [path, setPath] = useState("");
  const [d, setD] = useState<DatosCaras | null>(null);
  const [error, setError] = useState("");
  const cargarEjemplos = () => pedir<{ ids: Ejemplo[] }>("/api/explica/caras-ejemplos?n=11").then((x) => { setEjemplos(x.ids); if (!path) setPath(x.ids[0]?.id ?? ""); });
  useEffect(() => { void cargarEjemplos(); }, []); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!path) return;
    setD(null); setError("");
    pedir<DatosCaras>(`/api/explica/caras?path=${encodeURIComponent(path)}`).then(setD).catch((e: Error) => setError(e.message));
  }, [path]);
  const W = 1000, H = 420;
  const X = (x: number) => 20 + x * (W - 40), Y = (y: number) => 20 + (1 - y) * (H - 40);
  return (
    <section className={styles.seccion} id="caras">
      <h2><b>4</b> Caras: un descriptor global</h2>
      <p className={styles.bajada}>Para las caras no hay puntos clave ni palabras: una red neuronal (la ResNet de dlib, vía <code>face_recognition</code>, como en el Lab 11) mira la cara entera y la resume en <b>un solo vector de 128 números</b>. Es un <i>descriptor global</i>: una cara = un vector. Hay {r ? n(r.caras) : "8 937"} caras de {r ? n(r.personas) : "2 693"} artistas e integrantes, y esos vectores van directo al HNSW con distancia euclidiana. Elige una cara.</p>
      <Selector ejemplos={ejemplos} elegido={path} onElegir={setPath} onMas={cargarEjemplos} />
      {error && <p className={styles.error}>{error}</p>}
      {d && <>
        <div className={styles.dosColumnas}>
          <div>
            <div className={styles.imagenPuntos}>
              <img src={media(d.path)} alt={d.nombre} />
              <svg viewBox={`0 0 ${d.ancho} ${d.alto}`}>
                <rect x={d.caja[3]} y={d.caja[0]} width={d.caja[1] - d.caja[3]} height={d.caja[2] - d.caja[0]} fill="none" stroke="var(--amber)" strokeWidth={3} vectorEffect="non-scaling-stroke" />
                {Object.entries(d.puntos).map(([parte, pts]) => <g key={parte}>
                  <polyline points={pts.map((q) => q.join(",")).join(" ")} fill="none" stroke="#3cb44b" strokeWidth={1.6} vectorEffect="non-scaling-stroke" />
                  {pts.map((q, i) => <circle key={i} cx={q[0]} cy={q[1]} r={Math.max(1.5, d.ancho / 300)} fill="#e6194b" />)}
                </g>)}
              </svg>
            </div>
            <p className={styles.nota}>{d.nombre}. Recuadro: la cara detectada (HOG). Puntos rojos y líneas verdes: los 68 puntos de referencia (cejas, ojos, nariz, boca y mentón).</p>
          </div>
          <div className={styles.pasos}>
            <div className={styles.paso}><span>Paso 1 · encontrar la cara</span>
              <p>Un detector (HOG: histogramas de la dirección de los bordes, parecido a la idea de SIFT) encuentra el recuadro de la cara. Después ubica <b>68 puntos de referencia</b> y gira y escala la cara para que los ojos y la boca queden siempre en el mismo lugar.</p></div>
            <div className={styles.paso}><span>Paso 2 · 128 números</span>
              <p>La cara alineada pasa por una red neuronal entrenada para que las fotos de una misma persona den vectores cercanos y las de personas distintas, lejanos. Sale un vector de <b>128 números</b>. No hay histograma ni codebook: el vector ya es la "huella" de la cara.</p>
              <Barras valores={d.embedding} color="#4363d8" alto={80} etiqueta="los 128 números de esta cara" /></div>
            <div className={styles.paso}><span>Paso 3 · ¿misma persona?</span>
              <p>Dos caras son de la misma persona si la <b>distancia euclidiana</b> entre sus vectores es chica. face_recognition propone 0.6; con tantos artistas usamos <b>{d.umbral}</b>, que se equivoca menos (ver el histograma de abajo).</p></div>
          </div>
        </div>

        <div className={styles.bloque}>
          <h3>La distancia de esta cara a las {n(d.total)} caras</h3>
          <p className={styles.bajada}>En verde, sus otras {d.n_misma} fotos (la misma persona); en gris, las {n(d.n_otras)} caras de otras personas. Las líneas son los umbrales. Lo que queda a la izquierda del umbral se considera "la misma persona".</p>
          <HistogramaDistancias d={d} />
        </div>

        <div className={styles.bloque}>
          <h3>El mapa de las caras en 2D</h3>
          <p className={styles.bajada}>Los vectores de 128 números aplastados a 2D con PCA (una de cada tres caras). En azul las fotos de {d.nombre}, la ★ es la elegida y los círculos verdes, los vecinos que encontró el HNSW. Igual que antes, el plano pierde información: cerca en 2D no siempre es cerca en 128D.</p>
          <svg className={styles.plano} viewBox={`0 0 ${W} ${H}`}>
            {d.mapa.map(([x, y], i) => <circle key={i} cx={X(x)} cy={Y(y)} r={1.8} fill="rgba(29,39,40,.18)" />)}
            {d.vecinos.map((v, i) => <circle key={`v${i}`} cx={X(v.xy[0])} cy={Y(v.xy[1])} r={7} fill="none" stroke="var(--green)" strokeWidth={2.5} />)}
            {d.mapa_persona.map(([x, y], i) => <circle key={`p${i}`} cx={X(x)} cy={Y(y)} r={5.5} fill="#4363d8" stroke="#fff" strokeWidth={1} />)}
            <path d={estrella(X(d.mapa_yo[0]), Y(d.mapa_yo[1]), 12)} fill="#f1c40f" stroke="#1d2728" strokeWidth={2} />
          </svg>
        </div>

        <div className={styles.bloque}>
          <h3>Los 10 más parecidos según el HNSW</h3>
          <p className={styles.bajada}>El HNSW revisó <b>{n(d.revisados)}</b> de {n(d.total)} caras (sin contar esta misma foto). Las tarjetas son el recorte de la cara con la que se calculó cada vector.</p>
          <div className={styles.vecinosCaras}>
            <figure className={styles.caraConsulta}><img src={recorte(d.path)} alt="" /><figcaption>★ {d.nombre}</figcaption></figure>
            {d.vecinos.map((v, i) => <figure key={i} className={v.nombre === d.nombre ? styles.caraMisma : ""}><img src={recorte(v.foto)} alt="" /><figcaption>{i + 1}. {v.nombre}<br /><b>{v.distancia.toFixed(3)}</b> {v.misma_persona ? "· misma persona" : ""}</figcaption></figure>)}
          </div>
          {d.otras_fotos.length > 0 && <p className={styles.nota}>Borde azul: es la misma persona en otra foto. Si una foto suya queda por encima de {d.umbral}, la app diría "se parece" en vez de "misma persona": las fotos con otra luz, ángulo o edad cambian el vector.</p>}
        </div>
      </>}
    </section>
  );
}

function HistogramaDistancias({ d }: { d: DatosCaras }) {
  // dos histogramas en el mismo eje: la misma persona (escala propia, son pocas) y las demás
  const W = 1000, H = 220, n0 = d.hist_otras.length;
  const maxO = Math.max(...d.hist_otras, 1), maxM = Math.max(...d.hist_misma, 1);
  const X = (v: number) => 40 + (v / 1.2) * (W - 60);
  const ancho = (W - 60) / n0;
  return (
    <svg className={styles.plano} viewBox={`0 0 ${W} ${H + 30}`}>
      {d.hist_otras.map((c, i) => <rect key={i} x={X(d.bordes[i])} y={H - (c / maxO) * (H - 20)} width={ancho * 0.9} height={(c / maxO) * (H - 20)} fill="rgba(91,107,140,.35)" />)}
      {d.hist_misma.map((c, i) => c ? <rect key={`m${i}`} x={X(d.bordes[i])} y={H - (c / maxM) * (H - 20)} width={ancho * 0.9} height={(c / maxM) * (H - 20)} fill="var(--green)" opacity={0.85} /> : null)}
      {[[d.umbral, `nuestro umbral ${d.umbral}`], [0.6, "face_recognition 0.6"]].map(([u, t]) => <g key={String(t)}>
        <line x1={X(Number(u))} x2={X(Number(u))} y1={8} y2={H} stroke={Number(u) === d.umbral ? "#c0392b" : "#1d2728"} strokeWidth={2} strokeDasharray={Number(u) === d.umbral ? "0" : "5 4"} />
        <text x={Number(u) === d.umbral ? X(Number(u)) - 6 : X(Number(u)) + 6} y={20} fontSize="13" textAnchor={Number(u) === d.umbral ? "end" : "start"} fill={Number(u) === d.umbral ? "#c0392b" : "#1d2728"}>{t}</text>
      </g>)}
      <line x1={40} x2={W - 20} y1={H} y2={H} stroke="var(--line)" />
      {[0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2].map((v) => <text key={v} x={X(v)} y={H + 18} fontSize="12" fill="var(--muted)" textAnchor="middle">{v.toFixed(1)}</text>)}
    </svg>
  );
}

function estrella(cx: number, cy: number, r: number) {
  const p = Array.from({ length: 10 }, (_, i) => { const a = (Math.PI / 5) * i - Math.PI / 2, rr = i % 2 ? r * 0.45 : r; return `${cx + Math.cos(a) * rr},${cy + Math.sin(a) * rr}`; });
  return `M${p.join("L")}Z`;
}

function Miniaturas({ titulo, ids, imagenes, ok }: { titulo: string; ids: number[]; imagenes: Record<string, string>; ok?: Set<number> }) {
  return (
    <div className={styles.miniaturas}>
      <span>{titulo}</span>
      <div>{ids.map((i) => <figure key={i} className={ok && !ok.has(i) ? styles.noExacto : ""}><img src={media(imagenes[String(i)])} alt="" /></figure>)}</div>
    </div>
  );
}

function SeccionMfcc({ r }: { r: Resumen | null }) {
  const [ejemplos, setEjemplos] = useState<Ejemplo[]>([]);
  const [id, setId] = useState("");
  const [d, setD] = useState<DatosMfcc | null>(null);
  const [error, setError] = useState("");
  const [t, setT] = useState(0.3);
  const audio = useRef<HTMLAudioElement | null>(null);
  const cargarEjemplos = () => pedir<{ ids: Ejemplo[] }>("/api/explica/ejemplos?tipo=mfcc&n=8").then((x) => { setEjemplos(x.ids); if (!id) setId(x.ids[0]?.id ?? ""); });
  useEffect(() => { void cargarEjemplos(); }, []); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!id) return;
    setD(null); setError("");
    pedir<DatosMfcc>(`/api/explica/mfcc?track=${id}`).then(setD).catch((e: Error) => setError(e.message));
  }, [id]);
  const col = d ? Math.min(d.mfcc[0].length - 1, Math.floor(t * d.mfcc[0].length)) : 0;
  const coef = d ? d.mfcc.map((f) => f[col]) : [];
  const palabra = d ? d.palabras[col] : 0;
  function elegir(f: number) {
    setT(f);
    if (audio.current && d) audio.current.currentTime = f * d.duracion;
  }
  return (
    <section className={styles.seccion} id="mfcc">
      <h2><b>2</b> Audio: MFCC + bolsa de palabras de audio</h2>
      <p className={styles.bajada}>El mismo truco con sonido: el audio se corta en ventanas cortísimas, cada una se describe con 39 números (los MFCC, que capturan el <i>timbre</i>) y se traduce a una "palabra de audio". Elige una canción y toca cualquier gráfico para moverte en el tiempo.</p>
      <Selector ejemplos={ejemplos} elegido={id} onElegir={setId} onMas={cargarEjemplos} />
      {error && <p className={styles.error}>{error}</p>}
      {!d && !error && id && <p className={styles.nota}>Calculando la onda, el espectrograma y los MFCC…</p>}
      {d && <>
        <div className={styles.cancion}>
          <img src={media(d.portada)} alt="" />
          <div><b>{d.titulo}</b><span>{d.artista} · preview de {d.duracion.toFixed(0)} s</span></div>
          <audio ref={audio} controls src={media(d.audio)} onTimeUpdate={(e) => setT(e.currentTarget.currentTime / d.duracion)} />
        </div>
        <div className={styles.pasosAudio}>
          <div className={styles.paso}><span>Paso 1 · la onda y las ventanas</span>
            <p>El audio se pasa a mono a 22 050 muestras por segundo. Se corta en ventanas de <b>2 048 muestras (93 ms)</b> que avanzan de a 512 (23 ms): en 93 ms el sonido casi no cambia. Este preview tiene <b>{n(d.n_ventanas)}</b> ventanas.</p>
            <Onda valores={d.onda} marca={t} onElegir={elegir} /></div>
          <div className={styles.paso}><span>Paso 2 · espectrograma</span>
            <p>A cada ventana se le aplica la <b>transformada de Fourier</b>: qué tan fuerte suena cada frecuencia. Abajo los graves, arriba los agudos; más claro = más fuerte.</p>
            <MapaCalor M={d.espectrograma} alto={130} marca={t} onElegir={elegir} etiqueta="frecuencia (0 a ~5.5 kHz)" /></div>
          <div className={styles.paso}><span>Paso 3 · escala mel</span>
            <p>El oído distingue mucho mejor los graves que los agudos, así que las frecuencias se agrupan en <b>40 bandas mel</b>: angostas abajo y anchas arriba.</p>
            <MapaCalor M={d.mel} alto={110} marca={t} onElegir={elegir} etiqueta="40 bandas mel" /></div>
          <div className={styles.paso}><span>Paso 4 · MFCC: 39 números por ventana</span>
            <p>Del logaritmo de las bandas se sacan <b>13 coeficientes</b> (una transformada coseno): describen la forma del espectro, o sea el timbre. Se suman cuánto cambian (<b>13 deltas</b>) y cuánto cambia el cambio (<b>13 delta-deltas</b>): 39 en total.</p>
            <MapaCalor M={d.mfcc} alto={130} marca={t} onElegir={elegir} etiqueta="39 coeficientes (cada fila con su propia escala)" porFila invertir={false} />
            <Barras valores={coef} color="var(--amber)" etiqueta={`los 39 números de la ventana en ${(t * d.duracion).toFixed(1)} s (13 MFCC · 13 delta · 13 delta-delta)`} /></div>
          <div className={styles.paso}><span>Paso 5 · palabra de audio e histograma</span>
            <p>Cada ventana cae en una de las {r ? n(r.k) : "1 024"} palabras del codebook de audio (K-Means sobre {r ? n(r.ventanas_mfcc) : "17 M"} ventanas): la de {(t * d.duracion).toFixed(1)} s es la <b style={{ color: colorPalabra(palabra) }}>palabra #{palabra}</b>. La canción usa <b>{d.palabras_distintas}</b> palabras distintas y queda como un histograma TF-IDF, igual que una portada.</p>
            <Barras valores={d.tf} resaltar={palabra} etiqueta="TF: fracción de ventanas en cada palabra" />
            <Barras valores={d.vector} resaltar={palabra} color="var(--green)" etiqueta="TF-IDF normalizado: el vector que se indexa" />
            <p className={styles.nota}>Por eso dos canciones "suenan parecido" cuando usan las mismas palabras de timbre, aunque la melodía sea otra: MFCC ignora la melodía a propósito.</p></div>
        </div>
      </>}
    </section>
  );
}

function ir(id: string) {
  // baja suave hasta la parte del recorrido y la hace destellar
  const el = document.getElementById(id);
  if (!el) return;
  el.scrollIntoView({ behavior: "smooth", block: "start" });
  el.classList.remove(styles.destello);
  void el.offsetWidth;
  el.classList.add(styles.destello);
}

function Caja({ id, titulo, detalle, tipo }: { id: string; titulo: string; detalle: string; tipo?: "imagen" | "audio" | "indice" | "cara" }) {
  return <button className={`${styles.caja} ${tipo ? styles[`caja_${tipo}`] : ""}`} onClick={() => ir(id)} title="ir a esta parte">
    <b>{titulo}</b><span>{detalle}</span>
  </button>;
}

function Flujo() {
  // el pipeline como diagrama: cada caja lleva a su explicación
  return (
    <nav className={styles.diagrama} aria-label="recorrido">
      <Caja id="sift" titulo="archivo" detalle="una portada o una canción" />
      <span className={styles.flecha}>→</span>
      <div className={styles.rama}>
        <Caja id="dog" tipo="imagen" titulo="imagen → SIFT" detalle="puntos clave (DoG) · 128 números c/u" />
        <Caja id="mfcc" tipo="audio" titulo="audio → MFCC" detalle="ventanas de 93 ms · 39 números c/u" />
        <Caja id="caras" tipo="cara" titulo="cara → 128 números" detalle="descriptor global: va directo al índice" />
      </div>
      <span className={styles.flecha}>→</span>
      <Caja id="palabras" titulo="palabra más cercana" detalle="codebook K-Means de 1 024" />
      <span className={styles.flecha}>→</span>
      <Caja id="histograma" titulo="histograma TF" detalle="cuántas veces sale cada palabra" />
      <span className={styles.flecha}>→</span>
      <Caja id="histograma" titulo="× IDF" detalle="las palabras comunes pesan menos" />
      <span className={styles.flecha}>→</span>
      <Caja id="histograma" titulo="vector de 1 024" detalle="TF-IDF normalizado" />
      <span className={styles.flecha}>→</span>
      <div className={styles.rama}>
        <Caja id="ivf" tipo="indice" titulo="IVF" detalle="celdas de K-Means" />
        <Caja id="hnsw" tipo="indice" titulo="HNSW" detalle="grafo por niveles" />
      </div>
    </nav>
  );
}

export default function ComoFunciona() {
  const [r, setR] = useState<Resumen | null>(null);
  useEffect(() => { pedir<Resumen>("/api/explica/resumen").then(setR).catch(() => setR(null)); }, []);
  return (
    <div className={styles.pagina}>
      <header className={styles.cabecera}>
        <div><span className={styles.ceja}>CHUTA DB · PARTE 4</span><h1>Cómo funciona la búsqueda por imagen y audio</h1></div>
        <nav><a href="/">Consola técnica</a><a href="/musica">chuta app ▸</a></nav>
      </header>
      {r && <div className={styles.numeros}>
        <div><b>{n(r.portadas)}</b><span>portadas</span></div>
        <div><b>{n(r.descriptores_sift)}</b><span>descriptores SIFT (128 c/u)</span></div>
        <div><b>{n(r.audios)}</b><span>audios</span></div>
        <div><b>{n(r.ventanas_mfcc)}</b><span>ventanas MFCC (39 c/u)</span></div>
        <div><b>{n(r.k)}</b><span>palabras por codebook</span></div>
        <div><b>{n(r.vectores_sift + r.vectores_mfcc)}</b><span>histogramas TF-IDF de {n(r.k)} dims</span></div>
        <div><b>{n(r.caras)}</b><span>caras de {n(r.personas)} personas (128 c/u)</span></div>
      </div>}
      <Flujo />
      <SeccionSift r={r} />
      <SeccionMfcc r={r} />
      <SeccionIndices r={r} />
      <SeccionCaras r={r} />
    </div>
  );
}
