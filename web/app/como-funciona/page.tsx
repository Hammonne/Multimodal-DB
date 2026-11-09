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
  const cargarEjemplos = () => pedir<{ ids: Ejemplo[] }>("/api/explica/ejemplos?tipo=sift&n=11").then((x) => { setEjemplos(x.ids); if (!path) setPath(x.ids[0]?.id ?? ""); });
  useEffect(() => { void cargarEjemplos(); }, []); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!path) return;
    setD(null); setError(""); setPunto(0);
    pedir<DatosSift>(`/api/explica/sift?path=${encodeURIComponent(path)}`).then(setD).catch((e: Error) => setError(e.message));
  }, [path]);
  const p = d?.puntos[punto];
  const misma = useMemo(() => (d && p ? d.puntos.filter((q) => q.palabra === p.palabra).length : 0), [d, p]);
  return (
    <section className={styles.seccion}>
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
          <div className={styles.paso}><span>Paso 2 · descriptor del punto #{punto + 1}</span>
            <div className={styles.fila}><Descriptor valores={d.descriptores[punto]} />
              <p>Alrededor del punto se arma una grilla de <b>4 × 4</b> celdas, girada según su orientación (por eso aguanta rotaciones). En cada celda se cuenta cuánto cambia el brillo en <b>8 direcciones</b>: 4 × 4 × 8 = <b>128 números</b>. Después se normaliza (RootSIFT).</p></div></div>
          <div className={styles.paso}><span>Paso 3 · palabra visual</span>
            <p>El descriptor se compara con los {r ? n(r.k) : "1 024"} centroides del codebook (K-Means sobre los {r ? n(r.descriptores_sift) : "9 M"} descriptores de la colección) y cae en la <b style={{ color: colorPalabra(p.palabra) }}>palabra #{p.palabra}</b>. En esta portada hay <b>{misma}</b> {misma === 1 ? "punto" : "puntos"} con esa palabra (los que quedan resaltados).</p></div>
          <div className={styles.paso}><span>Paso 4 · histograma y TF-IDF</span>
            <p>Se cuenta cuántos puntos cayeron en cada palabra (TF). Esta portada usa <b>{d.palabras_distintas}</b> de las {n(d.tf.length)} palabras. Después se multiplica por el IDF: las palabras que aparecen en casi todas las portadas pesan poco.</p>
            <Barras valores={d.tf} resaltar={p.palabra} etiqueta="TF: fracción de puntos en cada palabra" />
            <Barras valores={d.vector} resaltar={p.palabra} color="var(--green)" etiqueta="TF-IDF normalizado: el vector que se indexa" />
            <p className={styles.nota}>Ese vector de {n(d.vector.length)} números es lo que guardan el IVF y el HNSW. Dos portadas se parecen si sus vectores apuntan hacia el mismo lado (coseno).</p></div>
        </div>
      </div>}
    </section>
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
    <section className={styles.seccion}>
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

export default function ComoFunciona() {
  const [r, setR] = useState<Resumen | null>(null);
  useEffect(() => { pedir<Resumen>("/api/explica/resumen").then(setR).catch(() => setR(null)); }, []);
  return (
    <div className={styles.pagina}>
      <header className={styles.cabecera}>
        <div><span className={styles.ceja}>CHUTA DB · PARTE 4</span><h1>Cómo funciona la búsqueda por imagen y audio</h1></div>
        <nav><a href="/">Consola técnica</a><a href="/musica">Abrir la app ▸</a></nav>
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
      <div className={styles.flujo}>
        {["archivo", "descriptores locales", "palabra más cercana", "histograma TF", "× IDF", "vector de 1 024", "IVF / HNSW"].map((x, i, a) => <span key={x}>{x}{i < a.length - 1 && <i>→</i>}</span>)}
      </div>
      <SeccionSift r={r} />
      <SeccionMfcc r={r} />
      <p className={styles.proximamente}>Próximamente en esta página: cómo IVF y HNSW encuentran los vecinos sin revisar toda la base, y el reconocimiento de caras.</p>
    </div>
  );
}
