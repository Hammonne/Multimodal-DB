"use client";

// la app: catálogo para navegar como en un reproductor de música, con búsquedas que usan nuestros índices
import { useEffect, useRef, useState } from "react";
import { ArrowLeft, Camera, Disc3, Home, Info, Mic2, Pause, Play, Search, Sparkles, Terminal, Users } from "lucide-react";
import styles from "./app.module.css";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const media = (path: string) => `${API}/api/multimedia/assets/${path}`;

type Artista = { id: string; nombre: string; genero: string; foto: string; tipo: string };
type Album = { id: string; titulo: string; artista_id: string; artista: string; portada: string; fecha: string; n_canciones: number; distancia?: number; ediciones?: number };
type Cancion = { id: string; titulo: string; artista: string; artista_id: string; con?: { id: string; nombre: string } | null; album: string; album_id: string; portada: string; audio: string; duracion: number; genero: string; explicita: boolean; letra: boolean; rank: number; distancia?: number; score?: number; fragmento?: string };
type Integrante = { nombre: string; rol: string; actual: boolean; fundador: boolean; foto: string };
type Tecnico = { indice: string; revisados?: number; total?: number; ms: number; extraccion_ms?: number };
type Vista = { tipo: "inicio" } | { tipo: "artista"; id: string } | { tipo: "album"; id: string } | { tipo: "cancion"; id: string } | { tipo: "buscar"; q?: string } | { tipo: "reconocer" };

async function pedir<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, init);
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Error en el servidor");
  return data as T;
}

function duracion(s: number) {
  return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
}

function Imagen({ src, redonda, alt }: { src: string; redonda?: boolean; alt: string }) {
  const [error, setError] = useState(false);
  if (!src || error) return <div className={`${styles.imagenVacia} ${redonda ? styles.redonda : ""}`}><Mic2 size={28} /></div>;
  return <img className={`${styles.imagen} ${redonda ? styles.redonda : ""}`} src={media(src)} alt={alt} loading="lazy" onError={() => setError(true)} />;
}

function ComoFunciona({ tecnico, descripcion }: { tecnico: Tecnico | null; descripcion: string }) {
  const [abierto, setAbierto] = useState(false);
  if (!tecnico) return null;
  return (
    <div className={styles.comoFunciona}>
      <button className={styles.botonTexto} onClick={() => setAbierto(!abierto)}><Info size={14} /> {abierto ? "Ocultar" : "Ver cómo funciona"}</button>
      {abierto && <div className={styles.cajaTecnica}>
        <p>{descripcion}</p>
        <p><b>Índice:</b> {tecnico.indice}{tecnico.revisados !== undefined && tecnico.total ? <> · revisó <b>{tecnico.revisados.toLocaleString()}</b> de {tecnico.total.toLocaleString()} vectores ({(100 * tecnico.revisados / tecnico.total).toFixed(1)} %)</> : null} · {tecnico.ms.toFixed(1)} ms{tecnico.extraccion_ms ? ` + ${tecnico.extraccion_ms.toFixed(0)} ms sacando descriptores` : ""}</p>
        <p>El detalle de cada búsqueda queda en la consola técnica, pestaña <b>Actividad de la app</b>.</p>
      </div>}
    </div>
  );
}

export default function AppMusica() {
  const [historial, setHistorial] = useState<Vista[]>([{ tipo: "inicio" }]);
  const vista = historial[historial.length - 1];

  // la vista va en la dirección (/musica?album=123), así se puede recargar o compartir
  useEffect(() => {
    const p = new URLSearchParams(window.location.search);
    const inicial: Vista | null = p.get("artista") ? { tipo: "artista", id: p.get("artista")! }
      : p.get("album") ? { tipo: "album", id: p.get("album")! }
      : p.get("cancion") ? { tipo: "cancion", id: p.get("cancion")! }
      : p.has("buscar") ? { tipo: "buscar", q: p.get("buscar") ?? "" }
      : p.has("reconocer") ? { tipo: "reconocer" } : null;
    if (inicial) setHistorial([{ tipo: "inicio" }, inicial]);
  }, []);
  useEffect(() => {
    const q = vista.tipo === "inicio" ? "" : vista.tipo === "buscar" ? `?buscar=${encodeURIComponent(vista.q ?? "")}`
      : vista.tipo === "reconocer" ? "?reconocer" : `?${vista.tipo}=${vista.id}`;
    // se conserva el estado que guarda next.js en el historial; con null se rompe su navegación
    window.history.replaceState(window.history.state, "", `/musica${q}`);
  }, [vista]);
  const [sonando, setSonando] = useState<Cancion | null>(null);
  const [pausa, setPausa] = useState(true);
  const [tiempo, setTiempo] = useState(0);
  const audio = useRef<HTMLAudioElement | null>(null);
  // cuántos parecidos mostrar en todas las búsquedas (10 por defecto); el navegador lo recuerda
  const [k, setKEstado] = useState(10);
  useEffect(() => { try { const v = Number(localStorage.getItem("chuta-k")); if (v >= 1 && v <= 50) setKEstado(v); } catch { /* sin almacenamiento */ } }, []);
  function setK(v: number) {
    setKEstado(v);
    try { localStorage.setItem("chuta-k", String(v)); } catch { /* sin almacenamiento */ }
  }

  function ir(v: Vista) {
    setHistorial((h) => [...h, v]);
    document.getElementById("contenido")?.scrollTo({ top: 0 });
  }
  function volver() {
    setHistorial((h) => (h.length > 1 ? h.slice(0, -1) : h));
  }

  function reproducir(c: Cancion) {
    if (sonando?.id === c.id && audio.current) {
      if (audio.current.paused) void audio.current.play(); else audio.current.pause();
      return;
    }
    setSonando(c);
  }

  useEffect(() => {
    if (sonando && audio.current) {
      audio.current.src = media(sonando.audio);
      void audio.current.play();
    }
  }, [sonando]);

  const ctx = { ir, reproducir, sonando, pausa, tiempo, k, setK };

  return (
    <div className={styles.app}>
      <aside className={styles.lateral}>
        <div className={styles.logo}><Disc3 size={26} /> <b>chutaDB</b> <span>music</span></div>
        <nav className={styles.menu}>
          <button className={vista.tipo === "inicio" ? styles.menuActivo : ""} onClick={() => ir({ tipo: "inicio" })}><Home size={18} /> <span>Inicio</span></button>
          <button className={vista.tipo === "buscar" ? styles.menuActivo : ""} onClick={() => ir({ tipo: "buscar" })}><Search size={18} /> <span>Buscar</span></button>
          <button className={vista.tipo === "reconocer" ? styles.menuActivo : ""} onClick={() => ir({ tipo: "reconocer" })}><Camera size={18} /> <span>Reconocer</span></button>
        </nav>
        <a className={styles.consola} href="/como-funciona" onClick={(e) => { e.preventDefault(); window.location.assign("/como-funciona"); }}><Info size={16} /> <span>Cómo funciona</span></a>
        <a className={`${styles.consola} ${styles.consolaSegunda}`} href="/" onClick={(e) => { e.preventDefault(); window.location.assign("/"); }}><Terminal size={16} /> <span>Consola técnica</span></a>
        <p className={styles.notaLateral}>Las búsquedas de imágenes, audio y caras usan nuestros índices (HNSW, IVF). Cada una queda registrada en la consola.</p>
      </aside>

      <main id="contenido" className={styles.contenido}>
        {historial.length > 1 && <button className={styles.volver} onClick={volver}><ArrowLeft size={18} /></button>}
        {vista.tipo === "inicio" && <Inicio {...ctx} />}
        {vista.tipo === "artista" && <PaginaArtista key={vista.id} id={vista.id} {...ctx} />}
        {vista.tipo === "album" && <PaginaAlbum key={vista.id} id={vista.id} {...ctx} />}
        {vista.tipo === "cancion" && <PaginaCancion key={vista.id} id={vista.id} {...ctx} />}
        {vista.tipo === "buscar" && <PaginaBuscar inicial={vista.q ?? ""} {...ctx} />}
        {vista.tipo === "reconocer" && <PaginaReconocer {...ctx} />}
      </main>

      <footer className={styles.reproductor}>
        {sonando ? <>
          <div className={styles.sonandoInfo} onClick={() => ir({ tipo: "cancion", id: sonando.id })}>
            <Imagen src={sonando.portada} alt={sonando.album} />
            <div><b>{sonando.titulo}</b><span>{sonando.artista}</span></div>
          </div>
          <div className={styles.controles}>
            <button className={styles.botonPlay} onClick={() => reproducir(sonando)}>{pausa ? <Play size={18} fill="currentColor" /> : <Pause size={18} fill="currentColor" />}</button>
            <div className={styles.barra}><span>{duracion(tiempo)}</span><div className={styles.progreso}><div style={{ width: `${Math.min(100, (tiempo / 30) * 100)}%` }} /></div><span>0:30</span></div>
          </div>
          <div className={styles.notaPreview}>preview de 30 s (Deezer)</div>
        </> : <div className={styles.notaPreview}>Elige una canción para escucharla</div>}
        <audio ref={audio} onPlay={() => setPausa(false)} onPause={() => setPausa(true)} onEnded={() => setPausa(true)} onTimeUpdate={(e) => setTiempo(e.currentTarget.currentTime)} />
      </footer>
    </div>
  );
}

type Ctx = { ir: (v: Vista) => void; reproducir: (c: Cancion) => void; sonando: Cancion | null; pausa: boolean; tiempo: number; k: number; setK: (k: number) => void };

function SelectorK({ k, setK }: { k: number; setK: (k: number) => void }) {
  // número editable dentro de los títulos ("los [10] más parecidos"); se aplica al salir o con Enter
  const [texto, setTexto] = useState(String(k));
  useEffect(() => { setTexto(String(k)); }, [k]);
  function aplicar() {
    const n = Math.round(Number(texto));
    if (Number.isFinite(n) && n >= 1) setK(Math.min(50, n)); else setTexto(String(k));
  }
  return <input className={styles.selectorK} type="number" min={1} max={50} value={texto} title="cuántos mostrar (1 a 50)"
    onChange={(e) => setTexto(e.target.value)} onBlur={aplicar} onKeyDown={(e) => { if (e.key === "Enter") { aplicar(); (e.target as HTMLInputElement).blur(); } }} />;
}

function useDatos<T>(path: string | null) {
  const [datos, setDatos] = useState<T | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    if (!path) return;
    let cancelado = false;
    setDatos(null);
    setError("");
    pedir<T>(path).then((d) => { if (!cancelado) setDatos(d); }).catch((e: Error) => { if (!cancelado) setError(e.message); });
    return () => { cancelado = true; };
  }, [path]);
  return { datos, error };
}

function Cargando({ error }: { error: string }) {
  return <div className={styles.cargando}>{error || "Cargando..."}</div>;
}

function TarjetaAlbum({ a, ir, extra }: { a: Album; ir: (v: Vista) => void; extra?: string }) {
  return (
    <button className={styles.tarjeta} onClick={() => ir({ tipo: "album", id: a.id })}>
      <Imagen src={a.portada} alt={a.titulo} />
      <b>{a.titulo}</b>
      <span>{extra ?? a.artista}{a.ediciones ? ` · +${a.ediciones} ${a.ediciones === 1 ? "edición" : "ediciones"}` : ""}</span>
    </button>
  );
}

function TarjetaArtista({ a, ir }: { a: Artista; ir: (v: Vista) => void }) {
  return (
    <button className={styles.tarjeta} onClick={() => ir({ tipo: "artista", id: a.id })}>
      <Imagen src={a.foto} redonda alt={a.nombre} />
      <b>{a.nombre}</b>
      <span>{a.tipo === "grupo" || a.tipo === "banda" ? "Banda" : "Artista"}</span>
    </button>
  );
}

function NombreArtista({ c, ir, fuerte }: { c: Cancion; ir: (v: Vista) => void; fuerte?: boolean }) {
  // colaboraciones: el nombre de la canción puede no tener perfil; se enlaza al artista con quien aparece
  const clase = fuerte ? styles.enlace : styles.enlaceSuave;
  return <>
    {c.artista_id ? <button className={clase} onClick={() => ir({ tipo: "artista", id: c.artista_id })}>{fuerte ? <b>{c.artista}</b> : c.artista}</button> : fuerte ? <b>{c.artista}</b> : c.artista}
    {c.con && c.con.id !== c.artista_id && <> · con <button className={clase} onClick={() => ir({ tipo: "artista", id: c.con!.id })}>{c.con.nombre}</button></>}
  </>;
}

function ListaCanciones({ canciones, ctx, conPortada = true, detalle }: { canciones: Cancion[]; ctx: Ctx; conPortada?: boolean; detalle?: (c: Cancion) => string }) {
  return (
    <div className={styles.lista}>
      {canciones.map((c, i) => {
        const activa = ctx.sonando?.id === c.id;
        return (
          <div key={`${c.id}-${i}`} className={`${styles.fila} ${activa ? styles.filaActiva : ""}`}>
            <button className={styles.filaPlay} onClick={() => ctx.reproducir(c)}>{activa && !ctx.pausa ? <Pause size={15} fill="currentColor" /> : <><span className={styles.numero}>{i + 1}</span><Play size={15} fill="currentColor" className={styles.iconoPlay} /></>}</button>
            {conPortada && <div className={styles.miniPortada}><Imagen src={c.portada} alt={c.album} /></div>}
            <div className={styles.filaTexto}>
              <button className={styles.enlace} onClick={() => ctx.ir({ tipo: "cancion", id: c.id })}><b>{c.titulo}</b>{c.explicita && <em className={styles.e}>E</em>}</button>
              <span><NombreArtista c={c} ir={ctx.ir} />{detalle ? ` · ${detalle(c)}` : ""}</span>
            </div>
            <button className={`${styles.enlaceSuave} ${styles.filaAlbum}`} onClick={() => ctx.ir({ tipo: "album", id: c.album_id })}>{c.album}</button>
            <span className={styles.filaDuracion}>{duracion(c.duracion)}</span>
          </div>
        );
      })}
    </div>
  );
}

function Inicio(ctx: Ctx) {
  const { datos, error } = useDatos<{ secciones: { genero: string; artistas: Artista[] }[]; albumes: Album[]; canciones: Cancion[]; totales: { canciones: number; artistas: number; albumes: number } }>("/api/app/inicio");
  if (!datos) return <Cargando error={error} />;
  return (
    <div>
      <section className={styles.portadaInicio}>
        <span className={styles.etiqueta}>chutaDB music</span>
        <h1>Tu música, buscada por cómo se ve, cómo suena y lo que dice</h1>
        <p>{datos.totales.canciones.toLocaleString()} canciones · {datos.totales.artistas.toLocaleString()} artistas · {datos.totales.albumes.toLocaleString()} álbumes, todo en una base de datos hecha desde cero.</p>
        <div className={styles.accionesInicio}>
                    <button onClick={() => ctx.ir({ tipo: "reconocer" })}><Camera size={16} /> Reconocer la foto de un CD</button>
        </div>
      </section>
      <h2 className={styles.titulo}>Álbumes populares</h2>
      <div className={styles.grilla}>{datos.albumes.map((a) => <TarjetaAlbum key={a.id} a={a} ir={ctx.ir} />)}</div>
      <h2 className={styles.titulo}>Lo más escuchado</h2>
      <ListaCanciones canciones={datos.canciones.slice(0, 6)} ctx={ctx} />
      {datos.secciones.map((s) => <div key={s.genero}>
        <h2 className={styles.titulo}>{s.genero}</h2>
        <div className={styles.grilla}>{s.artistas.map((a) => <TarjetaArtista key={a.id} a={a} ir={ctx.ir} />)}</div>
      </div>)}
    </div>
  );
}

function PaginaArtista({ id, ...ctx }: Ctx & { id: string }) {
  const { datos, error } = useDatos<Artista & { canciones: Cancion[]; albumes: Album[]; integrantes: Integrante[]; fotos: string[] }>(`/api/app/artista/${id}`);
  const { datos: parecido } = useDatos<{ parecidos: (Artista & { foto_cara: string; distancia: number; misma_persona: boolean })[]; tecnico: Tecnico | null }>(`/api/app/artista/${id}/se-parece?k=${ctx.k}`);
  const [todasFotos, setTodasFotos] = useState(false);
  const [grande, setGrande] = useState("");
  if (!datos) return <Cargando error={error} />;
  const actuales = datos.integrantes.filter((m) => m.actual);
  const integrantes = actuales.length ? actuales : datos.integrantes;
  return (
    <div>
      <section className={styles.cabecera}>
        <div className={styles.cabeceraImagen}><Imagen src={datos.foto} redonda alt={datos.nombre} /></div>
        <div>
          <span className={styles.etiqueta}>{integrantes.length ? "Banda" : "Artista"} · {datos.genero}</span>
          <h1 className={styles.nombreGrande}>{datos.nombre}</h1>
          <p className={styles.subtitulo}>{datos.albumes.length} álbumes · {datos.canciones.length} canciones en la base</p>
          {datos.canciones[0] && <button className={styles.botonVerde} onClick={() => ctx.reproducir(datos.canciones[0])}><Play size={18} fill="currentColor" /> Escuchar</button>}
        </div>
      </section>
      {datos.canciones.length > 0 && <><h2 className={styles.titulo}>Populares</h2><ListaCanciones canciones={datos.canciones.slice(0, 10)} ctx={ctx} /></>}
      {integrantes.length > 0 && <>
        <h2 className={styles.titulo}><Users size={18} /> Integrantes</h2>
        <div className={styles.grilla}>{integrantes.slice(0, 12).map((m, i) => <div key={`${m.nombre}-${i}`} className={styles.tarjetaFija}>
          <Imagen src={m.foto} redonda alt={m.nombre} /><b>{m.nombre}</b><span>{m.rol}{m.fundador ? " · fundador" : ""}</span>
        </div>)}</div>
      </>}
      <h2 className={styles.titulo}>Discografía</h2>
      <div className={styles.grilla}>{datos.albumes.map((a) => <TarjetaAlbum key={a.id} a={a} ir={ctx.ir} extra={a.n_canciones ? `${a.n_canciones} ${a.n_canciones === 1 ? "canción" : "canciones"}` : "álbum"} />)}</div>
      {datos.fotos.length > 1 && <>
        <h2 className={styles.titulo}>Fotos</h2>
        <div className={styles.galeria}>{(todasFotos ? datos.fotos : datos.fotos.slice(0, 6)).map((f) => <button key={f} className={styles.foto} onClick={() => setGrande(f)}><Imagen src={f} alt={datos.nombre} /></button>)}</div>
        {datos.fotos.length > 6 && <button className={styles.botonBorde} onClick={() => setTodasFotos(!todasFotos)}>{todasFotos ? "Ver menos" : `Ver las ${datos.fotos.length} fotos`}</button>}
      </>}
      {parecido && parecido.parecidos.length > 0 && <>
        <h2 className={styles.titulo}><Sparkles size={18} /> Los <SelectorK k={ctx.k} setK={ctx.setK} /> artistas que más se parecen</h2>
        <p className={styles.ayuda}>Artistas cuya cara se parece, según los embeddings de cara (como en el Lab 11) buscados con nuestro HNSW.</p>
        <div className={styles.grilla}>{parecido.parecidos.map((p) => <button key={p.id} className={styles.tarjeta} onClick={() => ctx.ir({ tipo: "artista", id: p.id })}>
          <Imagen src={p.foto_cara} redonda alt={p.nombre} /><b>{p.nombre}</b><span>{p.misma_persona ? "¡muy parecido!" : `distancia ${p.distancia.toFixed(2)}`}</span>
        </button>)}</div>
        <ComoFunciona tecnico={parecido.tecnico} descripcion="Cada foto de cara se convierte en un vector de 128 números con la red de dlib (face_recognition). Se promedian las caras del artista y el HNSW busca las caras más cercanas con distancia euclidiana entre ~9 000 caras de artistas e integrantes." />
      </>}
      {grande && <div className={styles.visor} onClick={() => setGrande("")}><img src={media(grande)} alt={datos.nombre} /></div>}
    </div>
  );
}

function PaginaAlbum({ id, ...ctx }: Ctx & { id: string }) {
  const { datos, error } = useDatos<Album & { canciones: Cancion[] }>(`/api/app/album/${id}`);
  const [parecidos, setParecidos] = useState<{ parecidos: Album[]; fotos: { imagen: string; que_es: string; puesto: number }[]; tecnico: Tecnico } | null>(null);
  const [buscando, setBuscando] = useState(false);
  async function buscarParecidos() {
    setBuscando(true);
    try { setParecidos(await pedir(`/api/app/album/${id}/parecidos?k=${ctx.k}`)); } catch { setParecidos(null); } finally { setBuscando(false); }
  }
  useEffect(() => { void buscarParecidos(); }, [id, ctx.k]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!datos) return <Cargando error={error} />;
  return (
    <div>
      <section className={styles.cabecera}>
        <div className={styles.cabeceraImagen}><Imagen src={datos.portada} alt={datos.titulo} /></div>
        <div>
          <span className={styles.etiqueta}>Álbum</span>
          <h1 className={styles.nombreGrande}>{datos.titulo}</h1>
          <p className={styles.subtitulo}><button className={styles.enlace} onClick={() => ctx.ir({ tipo: "artista", id: datos.artista_id })}><b>{datos.artista}</b></button>{datos.fecha ? ` · ${datos.fecha.slice(0, 4)}` : ""} · {datos.canciones.length ? `${datos.canciones.length} canciones en la base` : "solo la portada está en la base"}</p>
          <div className={styles.botones}>
            {datos.canciones[0] && <button className={styles.botonVerde} onClick={() => ctx.reproducir(datos.canciones[0])}><Play size={18} fill="currentColor" /> Escuchar</button>}
            <button className={styles.botonBorde} onClick={buscarParecidos} disabled={buscando}><Sparkles size={16} /> {buscando ? "Buscando..." : "Buscar parecidas otra vez"}</button>
          </div>
        </div>
      </section>
      {datos.canciones.length > 0 && <ListaCanciones canciones={datos.canciones} ctx={ctx} conPortada={false} />}
      {parecidos && <>
        {parecidos.fotos.length > 0 && <>
          <h2 className={styles.titulo}>Fotos de este álbum en la colección</h2>
          <p className={styles.ayuda}>Fotos reales del CD o vinilo (Discogs) o versiones modificadas de este álbum, y en qué puesto las encontró el buscador al buscar con la portada.</p>
          <div className={styles.grilla}>{parecidos.fotos.map((f) => <div key={f.imagen} className={styles.tarjetaFija}><Imagen src={f.imagen} alt={f.que_es} /><b>{f.que_es}</b><span>{f.puesto ? `encontrada en el puesto #${f.puesto}` : "no salió entre las primeras 50"}</span></div>)}</div>
        </>}
        <h2 className={styles.titulo}>Los <SelectorK k={ctx.k} setK={ctx.setK} /> álbumes más parecidos</h2>
        <div className={styles.grilla}>{parecidos.parecidos.map((a) => <TarjetaAlbum key={a.id} a={a} ir={ctx.ir} />)}</div>
        <ComoFunciona tecnico={parecidos.tecnico} descripcion="Cada portada se parte en cientos de puntos SIFT; cada punto se vuelve una 'palabra visual' del codebook (K-Means, 1 024 palabras) y la portada queda como un histograma TF-IDF. El HNSW encuentra los histogramas más parecidos con coseno sin revisar toda la base." />
      </>}
    </div>
  );
}

function PaginaCancion({ id, ...ctx }: Ctx & { id: string }) {
  const { datos, error } = useDatos<Cancion & { texto: string; lrc: { t: number; texto: string }[] }>(`/api/app/cancion/${id}`);
  const [parecidas, setParecidas] = useState<{ parecidas: Cancion[]; tecnico: Tecnico } | null>(null);
  const [sinAudio, setSinAudio] = useState(false);
  const [buscando, setBuscando] = useState(false);
  const esta = ctx.sonando?.id === id;
  // no sabemos dónde empieza el preview de 30 s dentro de la canción: el usuario toca la línea que está
  // escuchando y con eso se calcula el desfase; se guarda por canción en el navegador
  const [desfase, setDesfase] = useState<number | null>(null);
  const lineaActiva = useRef<HTMLParagraphElement | null>(null);
  useEffect(() => {
    try { const v = localStorage.getItem(`desfase-${id}`); setDesfase(v === null ? null : Number(v)); } catch { setDesfase(null); }
  }, [id]);
  function sincronizar(t: number) {
    if (!esta) return;
    const nuevo = t - ctx.tiempo;
    setDesfase(nuevo);
    try { localStorage.setItem(`desfase-${id}`, String(nuevo)); } catch { /* sin almacenamiento, solo dura esta visita */ }
  }
  let actual = -1;
  if (datos && esta && desfase !== null) {
    datos.lrc.forEach((l, i) => { if (l.t <= ctx.tiempo + desfase + 0.15) actual = i; });
  }
  useEffect(() => { lineaActiva.current?.scrollIntoView({ block: "center", behavior: "smooth" }); }, [actual]);
  async function buscarParecidas() {
    setBuscando(true);
    try { setParecidas(await pedir(`/api/app/cancion/${id}/parecidas?k=${ctx.k}`)); setSinAudio(false); } catch { setParecidas(null); setSinAudio(true); } finally { setBuscando(false); }
  }
  useEffect(() => { void buscarParecidas(); }, [id, ctx.k]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!datos) return <Cargando error={error} />;
  return (
    <div>
      <section className={styles.cabecera}>
        <div className={styles.cabeceraImagen}><Imagen src={datos.portada} alt={datos.album} /></div>
        <div>
          <span className={styles.etiqueta}>Canción · {datos.genero}</span>
          <h1 className={styles.nombreGrande}>{datos.titulo}</h1>
          <p className={styles.subtitulo}><NombreArtista c={datos} ir={ctx.ir} fuerte /> · <button className={styles.enlace} onClick={() => ctx.ir({ tipo: "album", id: datos.album_id })}>{datos.album}</button> · {duracion(datos.duracion)}</p>
          <div className={styles.botones}>
            <button className={styles.botonVerde} onClick={() => ctx.reproducir(datos)}>{esta && !ctx.pausa ? <Pause size={18} fill="currentColor" /> : <Play size={18} fill="currentColor" />} {esta && !ctx.pausa ? "Pausa" : "Escuchar"}</button>
            <button className={styles.botonBorde} onClick={buscarParecidas} disabled={buscando}><Sparkles size={16} /> {buscando ? "Buscando..." : "Buscar parecidas otra vez"}</button>
          </div>
        </div>
      </section>
      {parecidas && <>
        <h2 className={styles.titulo}>Las <SelectorK k={ctx.k} setK={ctx.setK} /> que más suenan parecido</h2>
        <ListaCanciones canciones={parecidas.parecidas} ctx={ctx} />
        <ComoFunciona tecnico={parecidas.tecnico} descripcion="El audio se corta en ventanas de 93 ms y cada una se describe con 39 coeficientes MFCC (el timbre). Las ventanas se agrupan en 1 024 'palabras de audio' y la canción queda como un histograma TF-IDF que se busca con HNSW." />
      </>}
      {sinAudio && <p className={styles.ayuda}>Esta canción todavía no tiene su vector de audio en la colección, así que no se puede buscar lo que suena parecido.</p>}
      <h2 className={styles.titulo}>Letra</h2>
      {datos.lrc.length > 0 ? <div className={styles.letra}>
        <p className={styles.ayuda}>{!esta ? "Dale a Escuchar y toca la línea que estás oyendo: desde ahí la letra se pinta sola." : desfase === null ? "Toca la línea que estás escuchando para sincronizar la letra." : "Sincronizada. Si se adelanta o se atrasa, toca la línea correcta otra vez."}</p>
        {datos.lrc.map((l, i) => <p key={i} ref={i === actual ? lineaActiva : null} onClick={() => sincronizar(l.t)} className={`${esta ? styles.lineaTocable : ""} ${i === actual ? styles.lineaActiva : i < actual ? styles.lineaPasada : ""}`}><span className={styles.marcaTiempo}>{duracion(l.t)}</span>{l.texto || "♪"}</p>)}
      </div> : datos.texto ? <pre className={styles.letraPlana}>{datos.texto}</pre> : <p className={styles.ayuda}>Esta canción no tiene letra en la base (puede ser instrumental).</p>}
    </div>
  );
}

function PaginaBuscar(_: Ctx & { inicial: string }) {
  // la búsqueda por texto (nombres y letras) es la parte 3, de otros compañeros: todavía no está conectada
  return (
    <div>
      <h1 className={styles.nombreGrande}>Buscar</h1>
      <div className={styles.cajaTecnica}>
        <p><b>Aún no implementado.</b></p>
        <p>La búsqueda por texto (nombres y letras con el índice SPIMI) es la Parte 3 del proyecto. Mientras tanto puedes explorar desde el inicio o reconocer una portada o una cara.</p>
      </div>
    </div>
  );
}

type Candidato = { nombre: string; artista_id: string; banda: string; foto: string; distancia: number; misma_persona: boolean; otras: string[] };
type RespuestaCaras = { caras: { caja: number[]; candidatos: Candidato[] }[]; ancho: number; alto: number; tecnico: Tecnico };
type RespuestaPortada = { tipo: string; resultados: (Album & Cancion)[]; tecnico: Tecnico };

function leer64(blob: Blob) {
  return new Promise<string>((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result).split(",", 2)[1] ?? "");
    r.onerror = () => reject(new Error("No se pudo leer la imagen"));
    r.readAsDataURL(blob);
  });
}

function PaginaReconocer(ctx: Ctx) {
  const [que, setQue] = useState<"portada" | "cara">("portada");
  const [fuente, setFuente] = useState<"archivo" | "camara" | "vivo">("archivo");
  const [archivo, setArchivo] = useState<File | null>(null);
  const [vista, setVista] = useState("");
  const [caras, setCaras] = useState<RespuestaCaras | null>(null);
  const [portada, setPortada] = useState<RespuestaPortada | null>(null);
  const [cargando, setCargando] = useState(false);
  const [error, setError] = useState("");
  const [enVivo, setEnVivo] = useState(false);
  const [elegida, setElegida] = useState(0);
  const video = useRef<HTMLVideoElement | null>(null);
  const flujo = useRef<MediaStream | null>(null);
  const vivo = useRef(false);

  function limpiar() { setCaras(null); setPortada(null); setError(""); setElegida(0); }

  // la cámara se prende solo en los modos cámara y en vivo, y se apaga al salir
  useEffect(() => {
    if (fuente === "archivo") return;
    let cancelado = false;
    navigator.mediaDevices?.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false })
      .then((s) => {
        if (cancelado) { s.getTracks().forEach((t) => t.stop()); return; }
        flujo.current = s;
        if (video.current) { video.current.srcObject = s; void video.current.play(); }
      })
      .catch(() => setError("No se pudo abrir la cámara. Revisa los permisos del navegador."));
    return () => {
      cancelado = true;
      vivo.current = false;
      setEnVivo(false);
      flujo.current?.getTracks().forEach((t) => t.stop());
      flujo.current = null;
    };
  }, [fuente]);

  useEffect(() => {
    if (!archivo) return;
    const url = URL.createObjectURL(archivo);
    setVista(url);
    return () => URL.revokeObjectURL(url);
  }, [archivo]);

  function cuadro(): Promise<Blob | null> {
    // el cuadro actual del video, achicado a 640 px de ancho
    const v = video.current;
    if (!v || !v.videoWidth) return Promise.resolve(null);
    const c = document.createElement("canvas");
    const escala = Math.min(1, 640 / v.videoWidth);
    c.width = Math.round(v.videoWidth * escala);
    c.height = Math.round(v.videoHeight * escala);
    c.getContext("2d")?.drawImage(v, 0, 0, c.width, c.height);
    return new Promise((resolve) => c.toBlob((b) => resolve(b), "image/jpeg", 0.85));
  }

  async function enviar(blob: Blob, nombre: string, deVivo: boolean) {
    const b64 = await leer64(blob);
    if (que === "cara") {
      setCaras(await pedir<RespuestaCaras>("/api/app/reconocer-cara", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ content_base64: b64, en_vivo: deVivo, k: ctx.k }) }));
    } else if (fuente === "archivo") {
      setPortada(await pedir<RespuestaPortada>("/api/app/reconocer", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ filename: nombre, content_base64: b64, k: ctx.k }) }));
    } else {
      setPortada(await pedir<RespuestaPortada>("/api/app/reconocer-portada", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ content_base64: b64, en_vivo: deVivo, k: ctx.k }) }));
    }
  }

  async function reconocer() {
    setCargando(true);
    setError("");
    try {
      if (fuente === "archivo" && archivo) await enviar(archivo, archivo.name, false);
      if (fuente === "camara") {
        const b = await cuadro();
        if (b) { setVista(URL.createObjectURL(b)); await enviar(b, "camara.jpg", false); }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Error");
    } finally {
      setCargando(false);
    }
  }

  useEffect(() => {
    if (!enVivo && fuente === "archivo" && archivo && (caras || portada)) void reconocer();
  }, [ctx.k]); // eslint-disable-line react-hooks/exhaustive-deps

  async function alternarVivo() {
    // detener congela el video en el último cuadro (con sus recuadros); empezar lo vuelve a mover
    if (vivo.current) { vivo.current = false; setEnVivo(false); video.current?.pause(); return; }
    vivo.current = true;
    setEnVivo(true);
    limpiar();
    void video.current?.play();
    // un cuadro a la vez: se manda el siguiente cuando vuelve la respuesta (~1 por segundo)
    while (vivo.current) {
      const inicio = Date.now();
      try {
        const b = await cuadro();
        if (b) await enviar(b, "vivo.jpg", true);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Error");
      }
      await new Promise((r) => setTimeout(r, Math.max(150, 700 - (Date.now() - inicio))));
    }
  }

  const esAudio = archivo?.type.startsWith("audio");
  const marco = fuente === "archivo" ? (archivo && vista && !esAudio ? "img" : null) : "video";
  const recuadros = que === "cara" && caras ? caras.caras : [];
  // la cara del top 10: la elegida con un clic o, si no, la más grande del cuadro
  const masGrande = recuadros.reduce((m, c, i) => ((c.caja[2] - c.caja[0]) * (c.caja[1] - c.caja[3]) > (recuadros[m].caja[2] - recuadros[m].caja[0]) * (recuadros[m].caja[1] - recuadros[m].caja[3]) ? i : m), 0);
  const cara = recuadros[elegida] ?? recuadros[masGrande];
  const indiceCara = recuadros[elegida] ? elegida : masGrande;

  return (
    <div>
      <h1 className={styles.nombreGrande}>Reconocer</h1>
      <p className={styles.subtitulo}>Muéstrale la portada de un CD o la cara de un artista. Puedes subir una foto, tomarla con la cámara o dejarla reconociendo en vivo.</p>
      <div className={styles.selector}>
        <span>¿Qué reconocer?</span>
        <button className={que === "portada" ? styles.selectorActivo : ""} onClick={() => { setQue("portada"); limpiar(); }}><Disc3 size={15} /> Portada</button>
        <button className={que === "cara" ? styles.selectorActivo : ""} onClick={() => { setQue("cara"); limpiar(); }}><Users size={15} /> Cara</button>
      </div>
      <div className={styles.selector}>
        <span>¿Desde dónde?</span>
        <button className={fuente === "archivo" ? styles.selectorActivo : ""} onClick={() => { setFuente("archivo"); limpiar(); }}>Subir foto</button>
        <button className={fuente === "camara" ? styles.selectorActivo : ""} onClick={() => { setFuente("camara"); limpiar(); }}><Camera size={15} /> Cámara</button>
        <button className={fuente === "vivo" ? styles.selectorActivo : ""} onClick={() => { setFuente("vivo"); limpiar(); }}><Sparkles size={15} /> En vivo</button>
      </div>

      <div className={styles.reconocerFila}><div>
      {fuente === "archivo" && <label className={styles.zonaArchivo}>
        <input type="file" accept={que === "cara" ? "image/*" : "image/*,audio/*"} onChange={(e) => { setArchivo(e.target.files?.[0] ?? null); limpiar(); }} />
        {archivo && esAudio && vista ? <audio controls src={vista} /> : !archivo && <><Camera size={34} /><span>Toca para elegir {que === "cara" ? "una foto con caras" : "una foto o un audio"}</span></>}
        {marco === "img" && <div className={styles.marco}><img src={vista} alt="tu foto" />{recuadros.map((c, i) => <Recuadro key={i} c={c} ancho={caras!.ancho} alto={caras!.alto} activa={recuadros.length > 1 && i === indiceCara} onClick={() => setElegida(i)} />)}</div>}
      </label>}
      {fuente !== "archivo" && <div className={styles.marcoVideo}>
        <video ref={video} playsInline muted />
        {recuadros.map((c, i) => <Recuadro key={i} c={c} ancho={caras!.ancho} alto={caras!.alto} espejo activa={recuadros.length > 1 && i === indiceCara} onClick={() => setElegida(i)} />)}
        {fuente === "vivo" && enVivo && <span className={styles.enVivo}>● EN VIVO</span>}
        {fuente === "vivo" && !enVivo && caras && <span className={`${styles.enVivo} ${styles.pausado}`}>❚❚ PAUSADO</span>}
      </div>}
      <div className={styles.botones}>
        {fuente === "vivo"
          ? <button className={styles.botonVerde} onClick={alternarVivo}>{enVivo ? <><Pause size={16} fill="currentColor" /> Detener</> : <><Play size={16} fill="currentColor" /> Empezar a reconocer</>}</button>
          : <button className={styles.botonVerde} onClick={reconocer} disabled={cargando || (fuente === "archivo" && !archivo)}><Sparkles size={16} /> {cargando ? "Reconociendo..." : fuente === "camara" ? "Tomar foto y reconocer" : "Reconocer"}</button>}
      </div>
      {error && <p className={styles.ayuda}>{error}</p>}
      {que === "cara" && cara?.candidatos[0] && cara.candidatos[0].otras.length > 0 && <div className={styles.otrasFotos}>
        <span className={styles.etiqueta}>Otras fotos de {cara.candidatos[0].nombre}</span>
        <p className={styles.ayuda}>Las caras con las que face_recognition calculó sus embeddings.</p>
        <div className={styles.galeriaCaras}>{cara.candidatos[0].otras.map((f) => <CaraImg key={f} path={f} alt={cara.candidatos[0].nombre} />)}</div>
      </div>}
      </div>
      {que === "cara" && cara && <aside className={styles.top10}>
        {cara.candidatos[0] && <button className={styles.top1} onClick={() => cara.candidatos[0].artista_id && ctx.ir({ tipo: "artista", id: cara.candidatos[0].artista_id })}>
          <CaraImg path={cara.candidatos[0].foto} alt={cara.candidatos[0].nombre} />
          <span className={styles.top1Texto}>
            <span className={styles.etiqueta}>Te pareces más a</span>
            <b>{cara.candidatos[0].nombre}</b>
            <small>{cara.candidatos[0].banda ? `integrante de ${cara.candidatos[0].banda} · ` : ""}distancia {cara.candidatos[0].distancia.toFixed(3)}</small>
            <em className={cara.candidatos[0].misma_persona ? styles.insigniaOk : styles.insignia}>{cara.candidatos[0].misma_persona ? "misma persona (< 0.45)" : "se parece (≥ 0.45)"}</em>
          </span>
        </button>}
        <span className={styles.etiqueta}>Los <SelectorK k={ctx.k} setK={ctx.setK} /> más parecidos · HNSW, euclidiana</span>
        {recuadros.length > 1 && <p className={styles.ayuda}>Hay {recuadros.length} caras: toca un recuadro para ver su top.</p>}
        {cara.candidatos.map((m, i) => <button key={`${m.nombre}-${i}`} className={`${styles.top10Fila} ${i === 0 ? styles.top10Primero : ""}`} onClick={() => m.artista_id && ctx.ir({ tipo: "artista", id: m.artista_id })}>
          <span className={styles.top10Puesto}>{i + 1}</span>
          <CaraImg path={m.foto} alt={m.nombre} redonda />
          <span className={styles.top10Texto}><b>{m.nombre}</b><small>{m.banda ? `integrante de ${m.banda}` : m.misma_persona ? "misma persona" : "se parece"}</small></span>
          <span className={styles.top10Distancia}><b>{m.distancia.toFixed(3)}</b><i style={{ width: `${Math.max(4, 100 - m.distancia * 140)}%` }} /></span>
        </button>)}
        <p className={styles.ayuda}>Distancia euclidiana entre embeddings de 128 dims; menor = más parecido. Bajo 0.45 se considera la misma persona.</p>
      </aside>}
      </div>

      {que === "cara" && caras && <>
        {!caras.caras.length && <p className={styles.ayuda}>No encontré caras. Acércate un poco y mira a la cámara con buena luz.</p>}
        <ComoFunciona tecnico={caras.tecnico} descripcion="Como en el Lab 11: face_recognition detecta cada cara y la convierte en un vector de 128 números (ResNet de dlib). El HNSW busca la cara más cercana con distancia euclidiana entre ~9 000 caras de artistas e integrantes; si la distancia es menor a 0.45 se considera la misma persona." />
      </>}
      {que === "portada" && portada && <>
        <h2 className={styles.titulo}>{portada.tipo === "álbum" ? <>Los <SelectorK k={ctx.k} setK={ctx.setK} /> álbumes más parecidos</> : <>Las <SelectorK k={ctx.k} setK={ctx.setK} /> canciones más parecidas</>}</h2>
        {portada.tipo === "álbum"
          ? <div className={styles.grilla}>{portada.resultados.map((a, i) => <TarjetaAlbum key={a.id} a={a} ir={ctx.ir} extra={i === 0 ? `${a.artista} · el más parecido` : a.artista} />)}</div>
          : <ListaCanciones canciones={portada.resultados} ctx={ctx} />}
        <ComoFunciona tecnico={portada.tecnico} descripcion={portada.tipo === "álbum" ? "A la imagen se le sacan los puntos SIFT, se convierten en palabras visuales con el mismo codebook de la colección y el histograma TF-IDF se busca en el HNSW. Como SIFT es local y en gris, aguanta rotación, cambios de color y fotos con otra luz." : "Al audio se le sacan los MFCC, se convierten en palabras de audio y el histograma se busca en el HNSW. Funciona con el audio original; grabado con un celular no (para eso sirven las huellas tipo Shazam de la Parte 5)."} />
      </>}
    </div>
  );
}

function CaraImg({ path, alt, redonda }: { path: string; alt: string; redonda?: boolean }) {
  // el recorte de la cara que usó face_recognition (no la foto de perfil completa)
  const [error, setError] = useState(false);
  if (!path || error) return <div className={`${styles.imagenVacia} ${redonda ? styles.redonda : ""}`}><Users size={22} /></div>;
  return <img className={`${styles.imagen} ${redonda ? styles.redonda : ""}`} src={`${API}/api/app/cara-recorte?path=${encodeURIComponent(path)}`} alt={alt} loading="lazy" onError={() => setError(true)} />;
}

function Recuadro({ c, ancho, alto, espejo, activa, onClick }: { c: { caja: number[]; candidatos: Candidato[] }; ancho: number; alto: number; espejo?: boolean; activa?: boolean; onClick?: () => void }) {
  // caja = (arriba, derecha, abajo, izquierda) en píxeles del cuadro enviado; se pasa a porcentajes.
  // la cámara se ve en espejo (como un selfie), así que ahí la x se refleja: izquierda = ancho - derecha
  const [top, right, bottom, left] = c.caja;
  const x = espejo ? ancho - right : left;
  const m = c.candidatos[0];
  return (
    <div onClick={onClick} className={`${styles.recuadro} ${m?.misma_persona ? styles.recuadroOk : ""} ${activa ? styles.recuadroActivo : ""}`} style={{ left: `${(100 * x) / ancho}%`, top: `${(100 * top) / alto}%`, width: `${(100 * (right - left)) / ancho}%`, height: `${(100 * (bottom - top)) / alto}%` }}>
      <span>{m ? (m.misma_persona ? m.nombre : `¿${m.nombre}?`) : "?"}</span>
    </div>
  );
}
