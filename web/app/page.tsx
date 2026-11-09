"use client";

import { useEffect, useRef, useState } from "react";
import type { DragEvent } from "react";
import dynamic from "next/dynamic";
import { AlertCircle, CheckCircle2, ChevronDown, ChevronRight, CircleStop, Copy, Database, FileBox, History, LockKeyhole, Play, RefreshCw, RotateCcw, Search, Table2, Terminal, X, Zap } from "lucide-react";
import styles from "./page.module.css";
import type { SpatialPoint } from "./MapPanel";

const MapPanel = dynamic(() => import("./MapPanel"), { ssr: false });

type Column = { name: string; type: string; position: number; primary_key: boolean };
type TextResult = { rank: number; id: number; title: string; snippet: string; score: number; length: number | null };
type TextSearchResponse = { results: TextResult[]; duration_ms: number; documents: number; terms: number; index_size_bytes: number; algorithm: string; dataset: string };
type MultimediaResult = { rank: number; id: string; distance: number; is_query: boolean; label?: string; original?: string };
type MultimediaResponse = { results: MultimediaResult[]; duration_ms: number; build_ms: number; documents: number; dimension: number; candidates: number; kind: string; index: string; metric: string; query_id: string; extract_ms?: number };
type BenchmarkRow = { tipo: string; n: number; metrica: string; indice: string; construir_s: number; consulta_ms: number; recall10: number; espacio_kb: number };
type BenchmarkQuality = { tipo: string; indice: string; metrica: string; cambio: string; recall10_original: number };
type BenchmarkCase = { cambio: string; objetivo: number; rank: number; resultados: number[] };
type BenchmarkResponse = { filas: BenchmarkRow[]; calidad: BenchmarkQuality[]; tamanos: number[] | Record<string, number[]>; k: number; nq: number; parametros: Record<string, string | number>; galeria: Record<string, BenchmarkCase[]> };
type GinRow = { n: number; consulta_tipo: string; consulta: string; construir_s: number; consulta_ms: number; resultados: number; espacio_bytes: number };
type GinResponse = { algoritmo: string; tamanos: number[]; filas: GinRow[] };
type TableInfo = { name: string; file_type: string; key_index: number; columns: Column[]; indexes: { index_name: string; index_type: string; column_name: string }[] };
type PlanNode = { node: string; table?: string; index?: string; operation?: string; access?: string; column?: string; rows?: number };
type QueryResult = { message: string; columns: string[]; rows: unknown[][]; plan: PlanNode[]; row_count: number; affected_rows?: number; duration_ms?: number };
type HistoryItem = { sql: string; status: "ok" | "error"; duration: number; rows: number; at: string };
type DemoEvent = { type: string; label: string; detail: string; tone: "neutral" | "active" | "success" | "warning" };
type LiveTransactionStatus = { transactions: { transaction_id: number; status: string; last_lsn: number }[]; locks: { resource: string; transaction_id: number | null; mode: string; count: number }[]; wal: { lsn: number; prev_lsn: number; transaction_id: number; type: string; operation: string | null; resource?: string }[]; recovery: { recovered_transactions: number[]; last_checkpoint: number | null; redo_records: number; clr_records: number } };
type TransactionDetail = { transaction_id: number; status: string; last_lsn: number; locks: { resource: string; mode: string; count: number }[]; events: { type: string; lsn: number; operation: string | null; resource: string }[] };

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const POR_PAGINA = 16;
const starterSql = "SELECT *\nFROM ventas\nWHERE id BETWEEN 1 AND 10\nORDER BY id ASC\nLIMIT 25;";

function apiErrorMessage(detail: unknown, fallback: string): string {
  if (typeof detail === "string" && detail.trim()) return detail;
  if (Array.isArray(detail)) {
    return detail.map((item) => {
      if (typeof item === "string") return item;
      if (item && typeof item === "object") {
        const entry = item as { msg?: unknown; loc?: unknown };
        const location = Array.isArray(entry.loc) ? ` (${entry.loc.join(".")})` : "";
        return `${String(entry.msg ?? "Error de validación")}${location}`;
      }
      return String(item);
    }).join("; ");
  }
  if (detail && typeof detail === "object") {
    const entry = detail as { message?: unknown; msg?: unknown; detail?: unknown };
    return apiErrorMessage(entry.message ?? entry.msg ?? entry.detail, fallback);
  }
  return fallback;
}

function limitarTopK(valor: string) {
  const n = Math.round(Number(valor));
  return Number.isFinite(n) ? Math.min(100, Math.max(1, n)) : 10;
}

function leerBase64(file: File) {
  // el backend recibe el archivo en base64 dentro del json, igual que la carga de csv
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",", 2)[1] ?? "");
    reader.onerror = () => reject(new Error("No se pudo leer el archivo"));
    reader.readAsDataURL(file);
  });
}

function TextSearchPanel() {
  const [query, setQuery] = useState("base de datos vectorial");
  const [algorithm, setAlgorithm] = useState("bm25");
  const [dataset, setDataset] = useState("1k");
  const [result, setResult] = useState<TextSearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");

  async function search() {
    setLoading(true);
    setMessage("");
    try {
      const response = await fetch(`${API}/api/text/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, algorithm, dataset, top_k: 10 }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? "No se pudo ejecutar la búsqueda");
      setResult(data);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Error de búsqueda");
      setResult(null);
    } finally {
      setLoading(false);
    }

  }

  async function build() {
    setLoading(true);
    setMessage("");
    try {
      const response = await fetch(`${API}/api/text/index`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? "No se pudo construir el índice");
      setMessage(`Índice SPIMI listo: ${data.documents} documentos, ${data.terms} términos, ${data.build_ms.toFixed(0)} ms`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Error de construcción");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className={`${styles.workspace} ${styles.searchWorkspace}`}>
      <section className={`${styles.panel} ${styles.queryPanel}`}>
        <div className={styles.panelHeader}><div><span className={styles.eyebrow}>05 / TEXT SEARCH</span><h2 className={styles.panelTitle}>Búsqueda textual</h2></div><div className={styles.searchBadge}><Search size={14} /> SPIMI</div></div>
        <div className={styles.panelBody}>
          <label className={styles.eyebrow} htmlFor="text-query">CONSULTA</label>
          <textarea id="text-query" className={styles.editorInput} value={query} onChange={(event) => setQuery(event.target.value)} />
          <div className={styles.queryMeta}>
            <label>Algoritmo <select value={algorithm} onChange={(event) => setAlgorithm(event.target.value)}><option value="bm25">BM25</option><option value="tfidf">TF-IDF + Coseno</option></select></label>
            <label>Dataset <select value={dataset} onChange={(event) => setDataset(event.target.value)}><option value="1k">1 000</option><option value="10k">10 000</option><option value="100k">100 000</option></select></label>
          </div>
          <div className={styles.queryActions}><button className={styles.secondaryButton} onClick={build} disabled={loading}><Zap size={14} /> CONSTRUIR ÍNDICE</button><button className={styles.runButton} onClick={search} disabled={loading || !query.trim()}><Play size={14} fill="currentColor" /> {loading ? "EJECUTANDO" : "BUSCAR"}</button></div>
          {message && <div className={styles.catalogHint}><AlertCircle size={15} /><span>{message}</span></div>}
        </div>
      </section>
      <section className={`${styles.panel} ${styles.resultsPanel}`}>
        <div className={styles.panelHeader}><div><span className={styles.eyebrow}>06 / RANKING</span><h2 className={styles.panelTitle}>Resultados <em className={styles.resultCount}>{result ? `${result.results.length} filas` : ""}</em></h2></div><div className={styles.searchBadge}><Search size={14} /> {result ? `${result.duration_ms.toFixed(2)} ms` : "WAITING"}</div></div>
        {result && <div className={styles.queryMeta}><span>{result.algorithm.toUpperCase()} · {result.dataset} · {result.documents} documentos · {result.terms} términos</span><span>{(result.index_size_bytes / 1024).toFixed(1)} KB</span></div>}
        <div className={styles.resultTableWrap}>{result?.results.length ? <table className={styles.resultTable}><thead><tr className={styles.resultRow}><th className={styles.resultHeader}>#</th><th className={styles.resultHeader}>Documento</th><th className={styles.resultHeader}>Score</th><th className={styles.resultHeader}>Fragmento</th></tr></thead><tbody>{result.results.map((item) => <tr className={styles.resultRow} key={`${item.id}-${item.rank}`}><td className={styles.resultCell}>{item.rank}</td><td className={styles.resultCell}><b>{item.title}</b><br /><small>ID {item.id} · {item.length ?? 0} tokens</small></td><td className={styles.resultCell}>{item.score.toFixed(5)}</td><td className={styles.resultCell}>{item.snippet}</td></tr>)}</tbody></table> : <div className={styles.empty}><Search size={25} /><p>Construye un índice y ejecuta una consulta para ver el ranking.</p></div>}</div>
      </section>
    </div>
  );
}

function MultimediaSearchPanel() {
  const [kind, setKind] = useState("sift");
  const [index, setIndex] = useState("hnsw");
  const [metric, setMetric] = useState("coseno");
  const [queryId, setQueryId] = useState("");
  const [mode, setMode] = useState<"base" | "archivo">("base");
  const [file, setFile] = useState<File | null>(null);
  const [filePreview, setFilePreview] = useState("");
  const [topK, setTopK] = useState(10);
  const [grupo, setGrupo] = useState("fotos");
  const [pagina, setPagina] = useState(0);
  const [totalEjemplos, setTotalEjemplos] = useState(0);
  const [result, setResult] = useState<MultimediaResponse | null>(null);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [examples, setExamples] = useState<string[]>([]);
  const [consultasEjemplo, setConsultasEjemplo] = useState<{ path: string; etiqueta: string; original: string }[]>([]);

  useEffect(() => {
    let cancelled = false;
    fetch(`${API}/api/multimedia/examples?kind=${kind}&offset=${pagina * POR_PAGINA}&limit=${POR_PAGINA}${kind === "sift" ? `&grupo=${grupo}` : ""}`)
      .then((response) => response.ok ? response.json() : null)
      .then((data: { ids?: string[]; total?: number } | null) => {
        if (!cancelled) { setExamples(data?.ids ?? []); setTotalEjemplos(data?.total ?? 0); }
      })
      .catch(() => { if (!cancelled) setExamples([]); });
    return () => { cancelled = true; };
  }, [kind, grupo, pagina]);
  useEffect(() => { setPagina(0); }, [kind, grupo]);
  const paginas = Math.max(1, Math.ceil(totalEjemplos / POR_PAGINA));

  useEffect(() => {
    fetch(`${API}/api/multimedia/consultas-ejemplo`)
      .then((response) => response.ok ? response.json() : null)
      .then((data: { consultas?: { path: string; etiqueta: string; original: string }[] } | null) => setConsultasEjemplo(data?.consultas ?? []))
      .catch(() => setConsultasEjemplo([]));
  }, []);

  async function usarEjemplo(path: string) {
    // se baja la imagen y se usa como si el usuario la hubiera subido
    const response = await fetch(`${API}/api/multimedia/assets/${path}`);
    if (!response.ok) { setMessage("No se pudo cargar el ejemplo"); return; }
    const blob = await response.blob();
    elegirArchivo(new File([blob], path.split("/").pop() ?? "consulta.jpg", { type: blob.type || "image/jpeg" }));
  }

  useEffect(() => {
    if (!file) { setFilePreview(""); return; }
    const url = URL.createObjectURL(file);
    setFilePreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  function elegirArchivo(nuevo: File | null) {
    setFile(nuevo);
    setResult(null);
    if (nuevo) setKind(nuevo.type.startsWith("audio") ? "mfcc" : "sift");
  }

  async function search() {
    setLoading(true);
    setMessage("");
    try {
      const response = mode === "archivo" && file
        ? await fetch(`${API}/api/multimedia/search-file`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ index, metric, filename: file.name, content_base64: await leerBase64(file), top_k: topK }) })
        : await fetch(`${API}/api/multimedia/search`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ kind, index, metric, query_id: queryId, top_k: topK }) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? "No se pudo ejecutar la búsqueda multimedia");
      setResult(data);
    } catch (error) {
      setResult(null);
      setMessage(error instanceof Error ? error.message : "Error de búsqueda");
    } finally {
      setLoading(false);
    }

  }

  return (
    <div className={`${styles.workspace} ${styles.searchWorkspace}`}>
      <section className={`${styles.panel} ${styles.queryPanel}`}>
        <div className={styles.panelHeader}><div><span className={styles.eyebrow}>07 / MULTIMEDIA SEARCH</span><h2 className={styles.panelTitle}>Similitud multimedia</h2></div><div className={styles.searchBadge}><Search size={14} /> k-NN</div></div>
        <div className={styles.panelBody}>
          <div className={styles.modeSwitch}>
            <button className={mode === "base" ? styles.viewTabActive : styles.viewTab} onClick={() => setMode("base")}>Objeto de la base</button>
            <button className={mode === "archivo" ? styles.viewTabActive : styles.viewTab} onClick={() => setMode("archivo")}>Subir mi archivo</button>
          </div>
          {mode === "archivo" ? <>
          <label className={styles.eyebrow} htmlFor="multimedia-file">TU CONSULTA</label>
          <p className={styles.helpText}>Sube una foto de una portada (por ejemplo una foto del CD con el celular) o un fragmento de audio. Se le sacan los descriptores con el mismo codebook de la colección.</p>
          <input id="multimedia-file" className={styles.fileInput} type="file" accept="image/*,audio/*" onChange={(event) => elegirArchivo(event.target.files?.[0] ?? null)} />
          {consultasEjemplo.length > 0 && <>
          <p className={styles.helpText}>O prueba con un ejemplo: fotos reales de CDs y vinilos (Discogs) y una portada con distintos cambios.</p>
          <div className={styles.exampleChoices}>
            {consultasEjemplo.map((c) => <div key={c.path} role="button" tabIndex={0} className={file?.name === c.path.split("/").pop() ? styles.exampleChoiceActive : styles.exampleChoice} onClick={() => usarEjemplo(c.path)} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") usarEjemplo(c.path); }}>
              <img className={styles.exampleThumbnail} src={`${API}/api/multimedia/assets/${c.path}`} alt={c.etiqueta} loading="lazy" />
              <span>{c.etiqueta}</span>
            </div>)}
          </div>
          </>}
          {file && filePreview && <div className={styles.filePreview}>{file.type.startsWith("audio") ? <audio controls src={filePreview} /> : <img src={filePreview} alt="consulta subida" />}<span>{file.name}</span></div>}
          </> : <>
          <label className={styles.eyebrow} htmlFor="multimedia-query">ELIGE UN OBJETO</label>
          {kind === "sift" && <div className={styles.modeSwitch}>
            {[["fotos", "Álbumes con foto CD / vinilo"], ["aumentadas", "Álbumes con imágenes aumentadas"], ["ambos", "Ambos"], ["ninguno", "Ninguno"]].map(([valor, texto]) =>
              <button key={valor} className={grupo === valor ? styles.viewTabActive : styles.viewTab} onClick={() => { setGrupo(valor); setQueryId(""); setResult(null); }}>{texto}</button>)}
          </div>}
          <p className={styles.helpText}>{kind !== "sift" ? "Selecciona un ID de ejemplo y busca objetos parecidos." : ({
            fotos: "Portadas con fotos reales de sus CDs o vinilos (Discogs) en la colección: al buscarlas salen al menos 5 de esas fotos entre los 10 primeros.",
            aumentadas: "Portadas con versiones modificadas (rotada, espejo, blanco y negro, otros colores...): al buscarlas salen al menos 5 entre los 10 primeros.",
            ambos: "Portadas con fotos reales y versiones modificadas: al buscarlas salen al menos 5 de ellas entre los 10 primeros.",
            ninguno: "Portadas que solo están una vez en la colección: los vecinos son otros álbumes parecidos.",
          } as Record<string, string>)[grupo]}</p>
          <div className={styles.exampleChoices}>
            {examples.map((id) => {
              const isImage = kind === "sift";
              const mediaPath = isImage ? id : `audio/${id}.mp3`;
              const mediaUrl = `${API}/api/multimedia/assets/${mediaPath}`;
              return <div key={id} role="button" tabIndex={0} className={queryId === id ? styles.exampleChoiceActive : styles.exampleChoice} onClick={() => setQueryId(id)} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") setQueryId(id); }}>
                {isImage ? <img className={styles.exampleThumbnail} src={mediaUrl} alt={`Portada ${id}`} loading="lazy" /> : <audio className={styles.exampleAudio} controls preload="none" src={mediaUrl} onClick={(event) => event.stopPropagation()} />}
                <span>{id}</span>
              </div>;
            })}
          </div>
          {totalEjemplos > POR_PAGINA && <div className={styles.paginador}>
            <button className={styles.viewTab} disabled={pagina === 0} onClick={() => setPagina(pagina - 1)} aria-label="página anterior">‹</button>
            <span>página {pagina + 1} de {paginas.toLocaleString()} · {totalEjemplos.toLocaleString()} {kind === "sift" ? "portadas" : "canciones"}</span>
            <button className={styles.viewTab} disabled={pagina + 1 >= paginas} onClick={() => setPagina(pagina + 1)} aria-label="página siguiente">›</button>
          </div>}
          <input id="multimedia-query" className={styles.editorInput} list="multimedia-examples" placeholder={examples.length ? "También puedes escribir o pegar un ID" : "Genera primero la base multimedia"} value={queryId} onChange={(event) => setQueryId(event.target.value)} />
          <datalist id="multimedia-examples">{examples.map((id) => <option key={id} value={id} />)}</datalist>
          </>}
          <div className={styles.queryMeta}>
            <label>Descriptor <select value={kind} disabled={mode === "archivo"} onChange={(event) => setKind(event.target.value)}><option value="sift">SIFT / imagen</option><option value="mfcc">MFCC / audio</option></select></label>
            <label>Índice <select value={index} onChange={(event) => setIndex(event.target.value)}><option value="hnsw">HNSW</option><option value="ivf">IVF</option></select></label>
            <label>Métrica <select value={metric} onChange={(event) => setMetric(event.target.value)}><option value="coseno">Coseno</option><option value="euclidiana">Euclidiana</option><option value="producto_punto">Producto punto</option></select></label>
            <label>Top-k <input className={styles.topKInput} type="number" min={1} max={100} value={topK} onChange={(event) => setTopK(limitarTopK(event.target.value))} /></label>
          </div>
          <div className={styles.queryActions}><button className={styles.runButton} onClick={search} disabled={loading || (mode === "archivo" ? !file : !queryId.trim())}><Play size={14} fill="currentColor" /> {loading ? "BUSCANDO..." : "BUSCAR SIMILARES"}</button></div>
          {message && <div className={styles.catalogHint}><AlertCircle size={15} /><span>{message}</span></div>}
        </div>
      </section>
      <section className={`${styles.panel} ${styles.resultsPanel}`}>
        <div className={styles.panelHeader}><div><span className={styles.eyebrow}>08 / NEIGHBORS</span><h2 className={styles.panelTitle}>Vecinos encontrados</h2></div><div className={styles.searchBadge}><Search size={14} /> {result ? `${result.duration_ms.toFixed(2)} ms` : "WAITING"}</div></div>
        {result && <div className={styles.queryMeta}><span>{result.index.toUpperCase()} · {result.metric} · top {result.results.length} · {result.documents} objetos · {result.dimension} dimensiones</span><span>{result.candidates} candidatos{result.extract_ms ? ` · descriptores ${result.extract_ms.toFixed(0)} ms` : ""}</span></div>}
        <div className={styles.resultTableWrap}>{result?.results.length ? <table className={styles.resultTable}><thead><tr className={styles.resultRow}><th className={styles.resultHeader}>#</th><th className={styles.resultHeader}>Vista previa</th><th className={styles.resultHeader}>Objeto parecido</th><th className={styles.resultHeader}>Distancia</th><th className={styles.resultHeader}>Qué es</th></tr></thead><tbody>{result.results.map((item) => { const isImage = result.kind === "sift"; const mediaPath = isImage ? item.id : `audio/${item.id}.mp3`; const preview = `${API}/api/multimedia/assets/${mediaPath}`; return <tr className={styles.resultRow} key={`${item.id}-${item.rank}`}><td className={styles.resultCell}>{item.rank}</td><td className={styles.resultCell}>{isImage ? <img className={styles.multimediaThumbnail} src={preview} alt={`Portada ${item.id}`} loading="lazy" /> : <audio className={styles.resultAudio} controls preload="none" src={preview} />}</td><td className={styles.resultCell}>{item.id}</td><td className={styles.resultCell}>{item.distance.toFixed(6)}</td><td className={styles.resultCell}>{item.is_query ? "CONSULTA" : item.label && item.label !== "portada" && item.label !== "canción" ? <>{item.label}<br /><small>{item.original === result.query_id ? "del mismo álbum" : `de ${item.original}`}</small></> : "VECINO"}</td></tr>; })}</tbody></table> : <div className={styles.empty}><Search size={25} /><p>{mode === "archivo" ? "Sube una imagen o un audio y pulsa BUSCAR SIMILARES." : examples.length ? "Elige un ID arriba y pulsa BUSCAR SIMILARES." : "No hay una base multimedia disponible. Genera las features antes de buscar."}</p></div>}</div>
      </section>
    </div>
  );
}

type Actividad = { hora: string; ts?: number; accion: string; titulo?: string; indice?: string; metrica?: string; descriptor?: string; sql?: string; revisados?: number; total?: number; ms?: number; resultados?: number; terminos?: string[] };

function ActividadPanel() {
  // lo que se hace en la app (/app) queda registrado en data/app_log.jsonl; se refresca cada 3 s
  const [items, setItems] = useState<Actividad[]>([]);
  const [filtro, setFiltro] = useState("todos");
  useEffect(() => {
    let vivo = true;
    const cargar = () => fetch(`${API}/api/app/actividad?limite=150`).then((r) => r.json()).then((d) => { if (vivo) setItems(d.actividad ?? []); }).catch(() => undefined);
    cargar();
    const t = setInterval(cargar, 3000);
    return () => { vivo = false; clearInterval(t); };
  }, []);
  const indices = ["todos", "HNSW", "ninguno"];
  const visibles = items.filter((i) => filtro === "todos" || (i.indice ?? "").startsWith(filtro));
  return (
    <div className={`${styles.workspace} ${styles.searchWorkspace}`}>
      <section className={`${styles.panel} ${styles.resultsPanel}`} style={{ gridColumn: "1 / -1" }}>
        <div className={styles.panelHeader}><div><span className={styles.eyebrow}>09 / APP ACTIVITY</span><h2 className={styles.panelTitle}>Actividad de la app <em className={styles.resultCount}>{visibles.length} registros</em></h2></div><a className={styles.appButton} href="/musica" target="_blank" rel="noreferrer">Abrir la app ▸</a></div>
        <div className={styles.queryMeta}><span>Cada acción en la app muestra qué índice usó, cuánto revisó y cuánto tardó. Se actualiza solo.</span><span>{indices.map((i) => <button key={i} className={filtro === i ? styles.viewTabActive : styles.viewTab} onClick={() => setFiltro(i)}>{i}</button>)}</span></div>
        <div className={styles.resultTableWrap}>{visibles.length ? <table className={styles.resultTable}><thead><tr className={styles.resultRow}><th className={styles.resultHeader}>Hora</th><th className={styles.resultHeader}>Acción</th><th className={styles.resultHeader}>Índice / métrica</th><th className={styles.resultHeader}>Revisado</th><th className={styles.resultHeader}>Tiempo</th><th className={styles.resultHeader}>SQL equivalente</th></tr></thead><tbody>{visibles.map((i, n) => <tr className={styles.resultRow} key={`${i.hora}-${n}`}><td className={styles.resultCell}>{i.ts ? new Date(i.ts * 1000).toLocaleTimeString() : i.hora}</td><td className={styles.resultCell}><b>{i.accion}</b><br /><small>{i.titulo}</small></td><td className={styles.resultCell}>{i.indice}{i.metrica ? <><br /><small>{i.metrica}</small></> : null}{i.descriptor ? <><br /><small>{i.descriptor}</small></> : null}</td><td className={styles.resultCell}>{i.revisados !== undefined && i.total ? `${i.revisados.toLocaleString()} de ${i.total.toLocaleString()} (${(100 * i.revisados / i.total).toFixed(1)} %)` : i.terminos ? `términos: ${i.terminos.join(", ")}` : "—"}{i.resultados !== undefined ? <><br /><small>{i.resultados} resultados</small></> : null}</td><td className={styles.resultCell}>{i.ms !== undefined ? `${i.ms} ms` : "—"}</td><td className={styles.resultCell}><code>{i.sql}</code></td></tr>)}</tbody></table> : <div className={styles.empty}><History size={25} /><p>Todavía no hay actividad. Abre la app y navega: cada búsqueda aparece aquí.</p></div>}</div>
      </section>
    </div>
  );
}

function BenchmarkPanel() {
  const [benchmark, setBenchmark] = useState<BenchmarkResponse | null>(null);
  const [gin, setGin] = useState<GinResponse | null>(null);
  const [message, setMessage] = useState("Carga los resultados generados por benchmark_vectorial.py.");
  const [loading, setLoading] = useState(false);
  const [galleryKey, setGalleryKey] = useState("sift_hnsw_coseno");

  async function loadBenchmark() {
    setLoading(true);
    try {
      const response = await fetch(`${API}/api/multimedia/benchmarks`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? "No se pudo cargar el benchmark");
      setBenchmark(data);
      setMessage("");
    } catch (error) {
      setBenchmark(null);
      setMessage(error instanceof Error ? error.message : "Error de benchmark");
    } finally {
      setLoading(false);
    }

  }

  async function loadGin() {
    setLoading(true);
    try {
      const response = await fetch(`${API}/api/text/benchmarks`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? "No se pudo cargar GIN");
      setGin(data);
      setMessage("");
    } catch (error) {
      setGin(null);
      setMessage(error instanceof Error ? error.message : "GIN todavía no está disponible");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className={`${styles.workspace} ${styles.searchWorkspace}`}>
      <section className={`${styles.panel} ${styles.queryPanel}`}>
        <div className={styles.panelHeader}><div><span className={styles.eyebrow}>09 / EXPERIMENT</span><h2 className={styles.panelTitle}>Comparación experimental</h2></div><div className={styles.searchBadge}><Zap size={14} /> BENCHMARK</div></div>
        <div className={styles.panelBody}>
          <div className={styles.catalogHint}><span>IVF vs HNSW · Recall@10 · tiempos · espacio en disco</span></div>
          <div className={styles.queryActions}><button className={styles.runButton} onClick={loadBenchmark} disabled={loading}><RefreshCw size={14} /> {loading ? "CARGANDO" : "CARGAR MULTIMEDIA"}</button><button className={styles.secondaryButton} onClick={loadGin} disabled={loading}><Database size={14} /> CARGAR GIN</button></div>
          {message && <div className={styles.catalogHint}><AlertCircle size={15} /><span>{message}</span></div>}
          {benchmark && <div className={styles.planBody}><div className={styles.nodeDetails}><b>Tamaños:</b> {Array.isArray(benchmark.tamanos) ? benchmark.tamanos.join(", ") : Object.entries(benchmark.tamanos).map(([kind, sizes]) => `${kind}: ${sizes.join(", ")}`).join(" · ")} <b>Consultas:</b> {benchmark.nq} <b>k:</b> {benchmark.k}</div></div>}
        </div>
      </section>
      <section className={`${styles.panel} ${styles.resultsPanel}`}>
        <div className={styles.panelHeader}><div><span className={styles.eyebrow}>10 / EVIDENCE</span><h2 className={styles.panelTitle}>Resultados del benchmark</h2></div><div className={styles.searchBadge}>{benchmark ? `${benchmark.filas.length} filas` : "WAITING"}</div></div>
        <div className={styles.resultTableWrap}>{benchmark?.filas.length ? <table className={styles.resultTable}><thead><tr className={styles.resultRow}><th className={styles.resultHeader}>Tipo</th><th className={styles.resultHeader}>N</th><th className={styles.resultHeader}>Índice</th><th className={styles.resultHeader}>Métrica</th><th className={styles.resultHeader}>Build (s)</th><th className={styles.resultHeader}>Consulta (ms)</th><th className={styles.resultHeader}>Recall@10</th><th className={styles.resultHeader}>KB</th></tr></thead><tbody>{benchmark.filas.map((row, i) => <tr className={styles.resultRow} key={`${row.tipo}-${row.n}-${row.indice}-${row.metrica}-${i}`}><td className={styles.resultCell}>{row.tipo}</td><td className={styles.resultCell}>{row.n}</td><td className={styles.resultCell}>{row.indice}</td><td className={styles.resultCell}>{row.metrica}</td><td className={styles.resultCell}>{row.construir_s.toFixed(2)}</td><td className={styles.resultCell}>{row.consulta_ms.toFixed(2)}</td><td className={styles.resultCell}>{row.recall10.toFixed(3)}</td><td className={styles.resultCell}>{row.espacio_kb.toFixed(0)}</td></tr>)}</tbody></table> : <div className={styles.empty}><Zap size={25} /><p>Ejecuta el benchmark multimedia para mostrar evidencia experimental.</p></div>}</div>
      </section>
      <section className={`${styles.panel} ${styles.resultsPanel}`} style={{ gridColumn: "1 / -1" }}>
        <div className={styles.panelHeader}><div><span className={styles.eyebrow}>12 / TEXT BASELINE</span><h2 className={styles.panelTitle}>PostgreSQL GIN</h2></div><div className={styles.searchBadge}>{gin ? `${gin.filas.length} filas` : "NO CARGADO"}</div></div>
        <div className={styles.resultTableWrap}>{gin?.filas.length ? <table className={styles.resultTable}><thead><tr className={styles.resultRow}><th className={styles.resultHeader}>N</th><th className={styles.resultHeader}>Consulta</th><th className={styles.resultHeader}>Build (s)</th><th className={styles.resultHeader}>Consulta (ms)</th><th className={styles.resultHeader}>Resultados</th><th className={styles.resultHeader}>Índice (KB)</th></tr></thead><tbody>{gin.filas.map((row, i) => <tr className={styles.resultRow} key={`${row.n}-${row.consulta_tipo}-${i}`}><td className={styles.resultCell}>{row.n}</td><td className={styles.resultCell}>{row.consulta_tipo}<br /><small>{row.consulta}</small></td><td className={styles.resultCell}>{row.construir_s.toFixed(2)}</td><td className={styles.resultCell}>{row.consulta_ms.toFixed(2)}</td><td className={styles.resultCell}>{row.resultados}</td><td className={styles.resultCell}>{(row.espacio_bytes / 1024).toFixed(1)}</td></tr>)}</tbody></table> : <div className={styles.empty}><Database size={25} /><p>Ejecuta el benchmark GIN y carga sus resultados para compararlo con SPIMI.</p></div>}</div>
      </section>
      <section className={`${styles.panel} ${styles.resultsPanel}`} style={{ gridColumn: "1 / -1" }}>
        <div className={styles.panelHeader}><div><span className={styles.eyebrow}>11 / GALLERY</span><h2 className={styles.panelTitle}>Aciertos y errores</h2></div><div className={styles.searchBadge}>RECALL@10</div></div>
        <div className={styles.queryMeta}><label>Configuración <select value={galleryKey} onChange={(event) => setGalleryKey(event.target.value)}>{benchmark && Object.keys(benchmark.galeria).map((key) => <option value={key} key={key}>{key}</option>)}</select></label></div>
        <div className={styles.resultTableWrap}>{benchmark?.galeria[galleryKey]?.length ? <table className={styles.resultTable}><thead><tr className={styles.resultRow}><th className={styles.resultHeader}>Cambio</th><th className={styles.resultHeader}>Original</th><th className={styles.resultHeader}>Rank</th><th className={styles.resultHeader}>Top 5 recuperados</th></tr></thead><tbody>{benchmark.galeria[galleryKey].slice(0, 20).map((item, i) => <tr className={styles.resultRow} key={`${item.cambio}-${i}`}><td className={styles.resultCell}>{item.cambio}</td><td className={styles.resultCell}>{item.objetivo}</td><td className={styles.resultCell}>{item.rank ? `#${item.rank}` : "NO ENCONTRADO"}</td><td className={styles.resultCell}>{item.resultados.join(", ")}</td></tr>)}</tbody></table> : <div className={styles.empty}><Zap size={25} /><p>Carga el benchmark para mostrar los casos de galería.</p></div>}</div>
      </section>
    </div>
  );
}

export default function Home() {
  const [tables, setTables] = useState<TableInfo[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [sql, setSql] = useState(starterSql);
  const [result, setResult] = useState<QueryResult>({ message: "Ejecuta una consulta para ver sus resultados.", columns: [], rows: [], plan: [], row_count: 0 });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [view, setView] = useState<"queries" | "transactions" | "text" | "multimedia" | "benchmarks" | "actividad">("queries");
  const [toast, setToast] = useState<{ tone: "success" | "error" | "info"; text: string } | null>(null);
  const [duration, setDuration] = useState<number | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [demoScenario, setDemoScenario] = useState("commit");
  const [demoStep, setDemoStep] = useState(0);
  const [demoPlaying, setDemoPlaying] = useState(false);
  const [liveStatus, setLiveStatus] = useState<LiveTransactionStatus | null>(null);
  const [sessionId, setSessionId] = useState("browser-demo");
  const [selectedTransactionId, setSelectedTransactionId] = useState<number | null>(null);
  const [transactionDetail, setTransactionDetail] = useState<TransactionDetail | null>(null);
  const [spatialPoints, setSpatialPoints] = useState<SpatialPoint[]>([]);
  const [demoRunning, setDemoRunning] = useState(false);
  const [csvDragging, setCsvDragging] = useState(false);
  const [csvUploading, setCsvUploading] = useState(false);
  const editorRef = useRef<HTMLTextAreaElement>(null);
  const gutterRef = useRef<HTMLDivElement>(null);
  const controllerRef = useRef<AbortController | null>(null);
  const [page, setPage] = useState(0);
  const pageSize = 12;

  async function loadTables() {
    try {
      const response = await fetch(`${API}/api/tables`);
      if (!response.ok) throw new Error("No se pudo cargar el catálogo");
      const data = await response.json();
      setTables(data.tables);
      setSelected((current) => current ?? data.tables[0]?.name ?? null);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Error de conexión");
    }
  }

  async function runQuery(query = sql) {
    if (!query.trim()) {
      setError("Escribe una consulta antes de ejecutar.");
      setToast({ tone: "error", text: "No hay una consulta para ejecutar" });
      return;
    }
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    const started = performance.now();
    setLoading(true);
    setError("");
    setPage(0);
    try {
      const response = await fetch(`${API}/api/query`, { method: "POST", headers: { "Content-Type": "application/json", "X-Session-ID": sessionId }, body: JSON.stringify({ sql: query }), signal: controller.signal });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? "La consulta no pudo ejecutarse");
      const elapsed = data.duration_ms ?? performance.now() - started;
      const affected = data.affected_rows ?? data.row_count ?? 0;
      setResult(data);
      setDuration(elapsed);
      setHistory((items) => [{ sql: query, status: "ok" as const, duration: elapsed, rows: affected, at: new Date().toLocaleTimeString() }, ...items].slice(0, 8));
      setToast({ tone: "success", text: `${data.message ?? "Consulta ejecutada"} · ${elapsed.toFixed(0)} ms` });
      loadTables();
    } catch (queryError) {
      if (queryError instanceof DOMException && queryError.name === "AbortError") {
        setToast({ tone: "info", text: "Consulta cancelada" });
        return;
      }
      const elapsed = performance.now() - started;
      setError(queryError instanceof Error ? queryError.message : "Error de consulta");
      setDuration(elapsed);
      setHistory((items) => [{ sql: query, status: "error" as const, duration: elapsed, rows: 0, at: new Date().toLocaleTimeString() }, ...items].slice(0, 8));
      setToast({ tone: "error", text: queryError instanceof Error ? queryError.message : "Error de consulta" });
    } finally {
      setLoading(false);
      controllerRef.current = null;
    }
  }

  function cancelQuery() { controllerRef.current?.abort(); }
  function syncEditorScroll() {
    if (gutterRef.current && editorRef.current) gutterRef.current.scrollTop = editorRef.current.scrollTop;
  }
  function copySql() { void navigator.clipboard?.writeText(sql); setToast({ tone: "info", text: "SQL copiado al portapapeles" }); }

  async function uploadCsv(file: File) {
    const isCsv = file.name.toLowerCase().endsWith(".csv");
    const isJson = /\.(json|jsonl)$/i.test(file.name);
    const isMedia = /\.(jpg|jpeg|png|webp|bmp|mp3|wav|ogg|flac|m4a)$/i.test(file.name);
    if (!isCsv && !isJson && !isMedia) {
      setToast({ tone: "error", text: "Carga CSV, JSON, JSONL, imágenes o audio" });
      return;
    }
    setCsvUploading(true);
    try {
      const content = await file.arrayBuffer();
      let binary = "";
      new Uint8Array(content).forEach((byte) => { binary += String.fromCharCode(byte); });
      const response = await fetch(`${API}/api/import/csv`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename: file.name, content_base64: btoa(binary) }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(apiErrorMessage(data.detail, "No se pudo cargar el archivo"));
      if (data.kind === "text_corpus") {
        if (typeof data.sql === "string") setSql(data.sql);
        setToast({ tone: "success", text: `${file.name} cargado: ${data.rows} documentos. SQL de tabla e importación generado; pulsa EJECUTAR.` });
        setView("queries");
        return;
      }
      if (data.kind === "media_asset") {
        if (typeof data.sql === "string") setSql(data.sql);
        setToast({ tone: "success", text: `${file.name} guardado. Se generó SQL para registrarlo; pulsa EJECUTAR.` });
        setView("queries");
        return;
      }
      const stem = file.name.replace(/\.(csv|jsonl?)$/i, "").replace(/[^A-Za-z0-9_]+/g, "_").toLowerCase();
      const tableName = `demo_${stem}`;
      let primaryKeyAssigned = false;
      const columns = (data.columns as { name: string; type: string }[])
        .map((column) => {
          const safeName = column.name.replace(/[^A-Za-z0-9_]/g, "_").toLowerCase();
          const isPrimaryKey = !primaryKeyAssigned && (safeName === "id" || safeName.endsWith("_id"));
          if (isPrimaryKey) primaryKeyAssigned = true;
          return `  ${safeName} ${column.type}${isPrimaryKey ? " PRIMARY KEY" : ""}`;
        })
        .join(",\n");
      setSql(`CREATE TABLE ${tableName} (\n${columns}\n) USING HEAP;\n\nIMPORT INTO ${tableName} FROM '${data.path}';`);
      setToast({ tone: "success", text: `${file.name} analizado: ${data.columns.length} columnas y ${data.rows} filas. SQL generado.` });
    } catch (uploadError) {
      setToast({ tone: "error", text: uploadError instanceof Error ? uploadError.message : "Error al cargar CSV" });
    } finally {
      setCsvUploading(false);
      setCsvDragging(false);
    }
  }

  function handleCsvDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setCsvDragging(false);
    const file = event.dataTransfer.files[0];
    if (file) void uploadCsv(file);
  }

  useEffect(() => {
    loadTables();
    const saved = window.localStorage.getItem("chuta-query-history");
    if (saved) setHistory(JSON.parse(saved));
    const savedSession = window.sessionStorage.getItem("chuta-session-id");
    const nextSession = savedSession ?? `tab-${crypto.randomUUID()}`;
    setSessionId(nextSession);
    window.sessionStorage.setItem("chuta-session-id", nextSession);
  }, []);
  useEffect(() => { window.localStorage.setItem("chuta-query-history", JSON.stringify(history)); }, [history]);
  
  useEffect(() => {
    // Si la consulta actual tiene resultados, intentamos extraer los puntos directamente del resultado
    if (result.columns.length > 0 && result.rows.length > 0) {
      const lonIdx = result.columns.findIndex(c => ["longitude", "lon", "lng", "x"].includes(c.toLowerCase()));
      const latIdx = result.columns.findIndex(c => ["latitude", "lat", "y"].includes(c.toLowerCase()));
      
      let points: SpatialPoint[] = [];
      // Usar la tabla seleccionada para que MapPanel no oculte los puntos
      const currentTable = selected ?? "Resultado de consulta";

      if (lonIdx !== -1 && latIdx !== -1) {
        points = result.rows.map((row, i) => ({
          table: currentTable,
          rid: [0, i] as [number, number],
          longitude: Number(row[lonIdx]),
          latitude: Number(row[latIdx]),
          label: String(row[1] ?? row[0] ?? `Punto ${i + 1}`), // Intenta usar la columna de nombre
          values: row
        })).filter(p => !isNaN(p.longitude) && !isNaN(p.latitude));
      } else {
        // Buscar la columna nativa POINT (un arreglo [lon, lat] en JSON)
        const pointIdx = result.rows[0].findIndex(val => Array.isArray(val) && val.length === 2 && typeof val[0] === "number");
        if (pointIdx !== -1) {
          points = result.rows.map((row, i) => {
            const pt = row[pointIdx] as [number, number];
            return {
              table: currentTable,
              rid: [0, i] as [number, number],
              longitude: pt[0], // Longitud
              latitude: pt[1],  // Latitud
              label: String(row[1] ?? row[0] ?? `Punto ${i + 1}`), // Intenta usar la columna de nombre
              values: row
            };
          });
        }
      }

      // Si encontramos puntos en la query actual, actualizar mapa e ignorar el endpoint global
      if (points.length > 0) {
        setSpatialPoints(points);
        return; 
      }
    }

    fetch(`${API}/api/spatial/points${selected ? `?table=${encodeURIComponent(selected)}` : ""}`)
      .then((response) => response.ok ? response.json() : { points: [] })
      .then((data) => setSpatialPoints(data.points ?? []))
      .catch(() => setSpatialPoints([]));
  }, [selected, result]);
  

  useEffect(() => { if (!demoPlaying) return; const timer = window.setInterval(() => setDemoStep((step) => step + 1), 1100); return () => window.clearInterval(timer); }, [demoPlaying]);
  useEffect(() => {
    if (view !== "transactions") return;
    let cancelled = false;
    async function loadLiveStatus() {
      try {
        const responses = await Promise.all([fetch(`${API}/api/transactions`), fetch(`${API}/api/locks`), fetch(`${API}/api/wal`), fetch(`${API}/api/recovery/status`)]);
        if (responses.some((response) => !response.ok)) return;
        const [transactions, locks, wal, recovery] = await Promise.all(responses.map((response) => response.json()));
        if (!cancelled) setLiveStatus({ transactions: transactions.transactions, locks: locks.locks, wal: wal.records, recovery });
      } catch { if (!cancelled) setLiveStatus(null); }
    }
    loadLiveStatus();
    const timer = window.setInterval(loadLiveStatus, 1500);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [view]);
  useEffect(() => {
    if (selectedTransactionId === null) { setTransactionDetail(null); return; }
    let cancelled = false;
    fetch(`${API}/api/transactions/${selectedTransactionId}`).then((response) => response.ok ? response.json() : null).then((data) => { if (!cancelled) setTransactionDetail(data); }).catch(() => { if (!cancelled) setTransactionDetail(null); });
    return () => { cancelled = true; };
  }, [selectedTransactionId]);
  const activeTable = tables.find((table) => table.name === selected);
  const visibleRows = result.rows.slice(page * pageSize, (page + 1) * pageSize);
  const pageCount = Math.max(1, Math.ceil(result.rows.length / pageSize));
  const demoScenarios: Record<string, { title: string; sql: string; events: DemoEvent[]; runnable: boolean; requirement?: string }> = {
    commit: { title: "Commit normal", sql: "BEGIN TRANSACTION;\nINSERT INTO demo_ventas VALUES (7, 'Demo', 450.0);\nEND TRANSACTION;", runnable: true, events: [{ type: "BEGIN", label: "Transacción iniciada", detail: "TX-07 · ACTIVE", tone: "active" }, { type: "EXCLUSIVE", label: "Lock adquirido", detail: "demo_ventas · escritura", tone: "warning" }, { type: "UPDATE", label: "Mutación registrada", detail: "WAL · before/after", tone: "neutral" }, { type: "COMMIT", label: "Commit durable", detail: "force() · locks liberados", tone: "success" }] },
    rollback: { title: "Rollback", sql: "BEGIN TRANSACTION;\nUPDATE demo_ventas SET monto = 999.0 WHERE id = 1;\nROLLBACK;", runnable: true, events: [{ type: "BEGIN", label: "Transacción iniciada", detail: "TX-08 · ACTIVE", tone: "active" }, { type: "UREAD", label: "Intención de actualización", detail: "demo_ventas · lector compatible", tone: "warning" }, { type: "EXCLUSIVE", label: "Promoción a escritura", detail: "before/after registrados", tone: "warning" }, { type: "CLR", label: "Undo aplicado", detail: "fila restaurada · ABORTED", tone: "success" }] },
    timeout: { title: "Timeout de lock", sql: "TX A: SELECT * FROM demo_ventas;\nTX B: UPDATE demo_ventas SET monto = 110.0 WHERE id = 1;", runnable: false, requirement: "requiere dos sesiones del navegador", events: [{ type: "SHARED", label: "TX-A lee demo_ventas", detail: "lectura compartida", tone: "active" }, { type: "UREAD", label: "TX-B espera", detail: "intención de actualización", tone: "warning" }, { type: "EXCLUSIVE", label: "Escritura bloqueada", detail: "TX-A conserva SHARED", tone: "warning" }, { type: "TIMEOUT", label: "LockTimeoutError", detail: "TX-B abortada y limpiada", tone: "success" }] },
    recovery: { title: "Crash y recovery", sql: "BEGIN TRANSACTION;\nINSERT INTO demo_ventas VALUES (8, 'Uncommitted', 700.0);", runnable: false, requirement: "requiere detener y reiniciar el backend", events: [{ type: "WAL", label: "WAL durable", detail: "UPDATE · before/after", tone: "neutral" }, { type: "CRASH", label: "Cierre inesperado", detail: "transacción sin COMMIT", tone: "warning" }, { type: "REDO", label: "Redo de confirmadas", detail: "páginas e índices", tone: "active" }, { type: "UNDO", label: "Undo de incompletas", detail: "CLR · estado consistente", tone: "success" }] },
  };
  const demo = demoScenarios[demoScenario];
  const activeDemoStep = Math.min(demoStep, demo.events.length - 1);
  function chooseDemo(value: string) { setDemoScenario(value); setDemoStep(0); setDemoPlaying(false); }
  async function runDemo() {
    if (!demo.runnable) {
      setToast({ tone: "info", text: `${demo.title}: ${demo.requirement}` });
      return;
    }
    setDemoRunning(true);
    setDemoStep(0);
    setToast({ tone: "info", text: `Ejecutando demo real: ${demo.title}` });
    try {
      for (const statement of demo.sql.split("\n").filter(Boolean)) {
        await runQuery(statement);
      }
      const transactionResponse = await fetch(`${API}/api/transactions`);
      if (transactionResponse.ok) {
        const transactionData = await transactionResponse.json();
        const latest = [...transactionData.transactions].sort((a, b) => b.transaction_id - a.transaction_id)[0];
        if (latest) setSelectedTransactionId(latest.transaction_id);
      }
      setDemoStep(demo.events.length - 1);
      setToast({ tone: "success", text: `Demo real completada: ${demo.title}` });
    } finally {
      setDemoRunning(false);
    }
  }
  const liveTimeline = liveStatus?.wal.slice(-12).reverse() ?? [];
  const selectedTimeline = transactionDetail?.events.map((event) => ({
    lsn: event.lsn,
    prev_lsn: 0,
    transaction_id: transactionDetail.transaction_id,
    type: event.type,
    operation: event.operation,
  })).reverse() ?? [];
  const eventExplanation: Record<string, string> = { BEGIN: "La transacción comienza", UPDATE: "Se registra una modificación en el WAL", COMMIT: "Los cambios quedan confirmados y durables", ABORT: "La transacción se cancela", CLR: "Se compensa una operación durante undo", CHECKPOINT: "Se guarda un punto de recuperación" };

  return (
    <main className={styles.shell}>
      <header className={styles.topbar}>
        <div className={styles.brand}><div className={styles.brandMark}><Database size={19} /></div><div><strong>CHUTA DB</strong><span>QUERY TERMINAL / 01</span></div></div>
        <nav className={styles.viewTabs} aria-label="Vistas de la consola">
          <button className={view === "queries" ? styles.viewTabActive : styles.viewTab} onClick={() => setView("queries")}><Terminal size={14} /> Consultas</button>
          <button className={view === "text" ? styles.viewTabActive : styles.viewTab} onClick={() => setView("text")}><Search size={14} /> Texto</button>
          <button className={view === "multimedia" ? styles.viewTabActive : styles.viewTab} onClick={() => setView("multimedia")}><FileBox size={14} /> Multimedia</button>
          <button className={view === "benchmarks" ? styles.viewTabActive : styles.viewTab} onClick={() => setView("benchmarks")}><Zap size={14} /> Benchmarks</button>
          <button className={view === "transactions" ? styles.viewTabActive : styles.viewTab} onClick={() => setView("transactions")}><LockKeyhole size={14} /> Transacciones</button>
          <button className={view === "actividad" ? styles.viewTabActive : styles.viewTab} onClick={() => setView("actividad")}><History size={14} /> Actividad de la app</button>
          <a className={styles.appButton} href="/como-funciona">Cómo funciona</a>
          <a className={styles.appButton} href="/musica">Entrar a la app ▸</a>
        </nav>
        <div className={styles.status}><span className={styles.statusDot} /> STORAGE ONLINE <span className={styles.divider} /> LOCAL INSTANCE</div>
      </header>

      {toast && <div className={`${styles.toast} ${styles[`toast${toast.tone[0].toUpperCase()}${toast.tone.slice(1)}`]}`} role="status"><CheckCircle2 size={16} /><span>{toast.text}</span><button onClick={() => setToast(null)} aria-label="Cerrar notificación"><X size={14} /></button></div>}

      {view === "transactions" ? (
        <div className={styles.demoShell}>
          <section className={styles.demoHero}><div><span className={styles.eyebrow}>05 / TRANSACTION LAB</span><h1>¿Qué está pasando?</h1><p>Lee la transacción de arriba hacia abajo: cada punto representa un evento del motor y explica cómo cambian sus locks y su estado.</p></div><div className={styles.demoHeroStatus}><span className={styles.simulationBadge}><Zap size={14} /> {liveTimeline.length ? "DATOS REALES DEL WAL" : "GUION DE DEMO"}</span><span className={styles.liveBadge}>{liveStatus ? "BACKEND ONLINE" : "BACKEND OFFLINE"}</span></div></section>
          <div className={styles.demoGrid}>
            <aside className={styles.demoScenarios}><span className={styles.eyebrow}>ESCENARIOS</span>{Object.entries(demoScenarios).map(([key, item]) => <button key={key} className={key === demoScenario ? styles.scenarioActive : styles.scenario} onClick={() => chooseDemo(key)}><span>{item.title}</span><small>{item.runnable ? "ejecutable" : "requiere preparación"}</small></button>)}<button className={styles.runDemoButton} onClick={runDemo} disabled={demoRunning}><Play size={13} /> {demoRunning ? "EJECUTANDO SQL..." : demo.runnable ? "EJECUTAR DEMO REAL" : "VER REQUISITOS DE DEMO"}</button><div className={styles.liveTransactions}><span className={styles.eyebrow}>LIVE TRANSACTIONS</span>{liveStatus?.transactions.length ? liveStatus.transactions.slice(-4).map((transaction) => <button className={styles.liveTransactionButton} key={transaction.transaction_id} onClick={() => setSelectedTransactionId(transaction.transaction_id)}><b>TX-{transaction.transaction_id}</b><span>{transaction.status}</span></button>) : <small>Sin transacciones activas</small>}</div><div className={styles.demoControls}><button onClick={() => setDemoStep((step) => Math.max(0, step - 1))} disabled={activeDemoStep === 0}>PREV</button><button onClick={() => setDemoPlaying((playing) => !playing)}>{demoPlaying ? "PAUSAR" : "REPRODUCIR"}</button><button onClick={() => { setDemoStep(0); setDemoPlaying(false); }}><RotateCcw size={13} /> RESET</button><button onClick={() => setDemoStep((step) => Math.min(demo.events.length - 1, step + 1))} disabled={activeDemoStep === demo.events.length - 1}>NEXT</button></div></aside>
            <section className={styles.demoTimeline}><div className={styles.demoSectionHeader}><div><span className={styles.eyebrow}>PASO A PASO</span><h2>{selectedTimeline.length ? transactionDetail ? `TX-${transactionDetail.transaction_id}` : "Eventos recientes" : demo.title}</h2></div><span className={styles.stepCounter}>{selectedTimeline.length ? `${selectedTimeline.length} eventos` : `${String(activeDemoStep + 1).padStart(2, "0")} / ${String(demo.events.length).padStart(2, "0")}`}</span></div><div className={styles.timelineLegend}><span className={styles.liveDot} /> {selectedTimeline.length ? transactionDetail ? "EVENTOS DE LA TX SELECCIONADA" : "EVENTOS REALES DEL WAL" : "SIN EVENTOS REALES AÚN"}</div><div className={styles.eventList}>{selectedTimeline.length ? selectedTimeline.map((event) => <div className={`${styles.event} ${styles.eventVisible} ${styles.liveEvent}`} key={`${event.lsn}-${event.transaction_id}`}><span className={`${styles.eventMarker} ${styles.eventActive}`}>{event.type.slice(0, 3)}</span><div><strong>{eventExplanation[event.type] ?? event.type}</strong><p>TX-{event.transaction_id} · LSN {event.lsn} · {event.operation ?? "cambio de estado"}</p></div></div>) : demo.events.map((event, index) => <div className={`${styles.event} ${index <= activeDemoStep ? styles.eventVisible : ""} ${index === activeDemoStep ? styles.eventCurrent : ""}`} key={`${event.type}-${index}`}><span className={`${styles.eventMarker} ${styles[`event${event.tone[0].toUpperCase()}${event.tone.slice(1)}`]}`}>{index + 1}</span><div><strong>{event.label}</strong><p>{event.type} · {event.detail} · GUION, NO EVENTO REAL</p></div></div>)}</div><div className={styles.explainBox}><strong>Cómo leerlo</strong><span>La transacción empieza en `BEGIN`, toma locks para trabajar y termina en `COMMIT` o `ROLLBACK`. Selecciona una TX para aislar sus eventos.</span></div><pre className={styles.demoSql}>{demo.sql}</pre></section>
            <aside className={styles.demoInspector}><div className={styles.inspectorTitle}><span className={styles.eyebrow}>RESUMEN</span><strong>{transactionDetail ? `TX-${transactionDetail.transaction_id}` : "Sin TX seleccionada"}</strong></div><div className={styles.demoCard}><span className={styles.eyebrow}>LOCK ACTUAL</span><div className={styles.lockRow}><b>{transactionDetail?.locks[0]?.resource ?? liveStatus?.locks[0]?.resource ?? "Ninguno"}</b><span className={styles.lockMode}>{transactionDetail?.locks[0]?.mode ?? liveStatus?.locks[0]?.mode ?? "-"}</span></div><small>{transactionDetail ? `Estado: ${transactionDetail.status}` : "Selecciona una TX real"}</small></div><div className={styles.demoCard}><span className={styles.eyebrow}>ÚLTIMO WAL</span><div className={styles.walLine}><b>LSN {transactionDetail?.last_lsn ?? liveStatus?.wal.at(-1)?.lsn ?? "-"}</b><span>{transactionDetail?.events.at(-1)?.type ?? liveStatus?.wal.at(-1)?.type ?? "-"}</span></div><small>{transactionDetail ? `${transactionDetail.events.length} eventos de esta TX` : "Los eventos aparecen en la timeline"}</small></div><div className={styles.demoCard}><span className={styles.eyebrow}>RECOVERY</span><div className={styles.recoveryFacts}><b>{liveStatus?.recovery.clr_records ?? 0}</b><span>CLR generados</span><b>{liveStatus?.recovery.last_checkpoint ?? "-"}</b><span>último checkpoint</span></div></div></aside>
          </div>
        </div>
      ) : view === "text" ? (
        <TextSearchPanel />
      ) : view === "multimedia" ? (
        <MultimediaSearchPanel />
      ) : view === "actividad" ? (
        <ActividadPanel />
      ) : view === "benchmarks" ? (
        <BenchmarkPanel />
      ) : (
        <div className={styles.workspace}>
          <section className={`${styles.panel} ${styles.filesPanel}`}>
            <div className={styles.panelHeader}><div><span className={styles.eyebrow}>01 / CATALOG</span><h2 className={styles.panelTitle}>Archivos</h2></div><button className={styles.iconButton} onClick={loadTables} title="Actualizar catálogo"><RefreshCw size={15} /></button></div>
            <div className={styles.panelBody}><div className={styles.catalogHint}><FileBox size={15} /><span>{tables.length} tablas registradas</span></div><div className={styles.tableList}>{tables.map((table) => <div key={table.name} className={`${styles.tableItem} ${selected === table.name ? styles.selected : ""}`}><button onClick={() => setSelected(selected === table.name ? null : table.name)}><span className={styles.tableIcon}><Table2 size={15} /></span><span className={styles.tableName}>{table.name}</span>{selected === table.name ? <ChevronDown size={14} /> : <ChevronRight size={14} />}</button>{selected === table.name && <div className={styles.columns}>{table.columns.map((column) => <div className={styles.column} key={column.name}><span>{column.primary_key ? "PK" : "  "}</span><b>{column.name}</b><small>{column.type}</small></div>)}<div className={styles.indexLabel}><Zap size={12} /> {table.indexes.length} índices</div></div>}</div>)}</div></div>
            {activeTable && <footer className={styles.panelFooter}><span>FILE TYPE</span><b>{activeTable.file_type.toUpperCase()}</b></footer>}
          </section>

          <section className={`${styles.panel} ${styles.planPanel}`}><div className={styles.panelHeader}><div><span className={styles.eyebrow}>02 / ANALYSIS</span><h2 className={styles.panelTitle}>Plan de ejecución</h2></div><span className={styles.costTag}>{result.plan.length ? `${result.plan.length} NODOS` : "IDLE"}</span></div><div className={styles.planBody}>{result.plan.length ? <div className={styles.planTree}>{result.plan.map((node, index) => <div className={styles.planNode} key={`${node.node}-${index}`}><div className={styles.nodeLine}><span className={styles.nodeIndex}>{String(index + 1).padStart(2, "0")}</span><strong>{node.node}</strong><span className={styles.nodeOp}>{node.operation}</span></div><div className={styles.nodeDetails}>{node.table && <span>TABLE <b>{node.table}</b></span>}{node.column && <span>COLUMN <b>{node.column}</b></span>}{node.index && <span>ACCESS <b>{node.index} / {node.access}</b></span>}{node.rows !== undefined && <span>ROWS <b>{node.rows}</b></span>}</div></div>)}</div> : <div className={styles.empty}><Zap size={25} /><p>El plan aparecerá después de ejecutar una consulta.</p></div>}</div></section>

          <section className={`${styles.panel} ${styles.queryPanel}`}><div className={styles.panelHeader}><div><span className={styles.eyebrow}>03 / WORKSPACE</span><h2 className={styles.panelTitle}>Consultas</h2></div><div className={styles.queryActions}><button className={styles.iconButton} onClick={copySql} title="Copiar SQL"><Copy size={14} /></button>{loading && <button className={styles.stopButton} onClick={cancelQuery}><CircleStop size={14} /> PARAR</button>}<button className={styles.runButton} onClick={() => runQuery()} disabled={loading}><Play size={14} fill="currentColor" /> {loading ? "EJECUTANDO" : "EJECUTAR"}<kbd className={styles.shortcut}>⌘ ↵</kbd></button></div></div><div className={styles.editorWrap}><div className={styles.editorGutter} ref={gutterRef}>{sql.split("\n").map((_, index) => <span key={index}>{String(index + 1).padStart(2, "0")}</span>)}</div><textarea ref={editorRef} className={styles.editorInput} aria-label="Editor de consulta SQL" spellCheck={false} value={sql} onChange={(event) => setSql(event.target.value)} onScroll={syncEditorScroll} onKeyDown={(event) => { if ((event.metaKey || event.ctrlKey) && event.key === "Enter") runQuery(); }} /></div><div className={styles.csvDropzone} onDragOver={(event) => { event.preventDefault(); setCsvDragging(true); }} onDragLeave={() => setCsvDragging(false)} onDrop={handleCsvDrop}><input id="csv-upload" type="file" accept=".csv,.json,.jsonl,.jpg,.jpeg,.png,.webp,.bmp,.mp3,.wav,.ogg,.flac,.m4a,text/csv,application/json,audio/*,image/*" hidden onChange={(event) => { const file = event.target.files?.[0]; if (file) void uploadCsv(file); event.currentTarget.value = ""; }} /><label htmlFor="csv-upload"><FileBox size={15} /> {csvUploading ? "CARGANDO ARCHIVO..." : csvDragging ? "SUELTA EL ARCHIVO AQUÍ" : "ARRASTRA CSV, JSON, JSONL, IMAGEN O AUDIO"}</label></div><div className={styles.queryMeta}><span><Terminal size={13} /> SQL / ANSI-LIKE</span><span>{sql.length} caracteres {duration !== null && `· ${duration.toFixed(0)} ms`}</span></div>{error && <div className={styles.error}><AlertCircle size={15} /><span>{error}</span></div>}{history.length > 0 && <div className={styles.historyStrip}><History size={13} /><span>HISTORIAL</span>{history.slice(0, 3).map((item, index) => <button key={`${item.at}-${index}`} onClick={() => setSql(item.sql)} className={item.status === "error" ? styles.historyError : ""}>{item.at} · {item.rows} filas</button>)}</div>}</section>

          <section className={`${styles.panel} ${styles.resultsPanel}`}><div className={styles.panelHeader}><div><span className={styles.eyebrow}>04 / OUTPUT</span><h2 className={styles.panelTitle}>Resultados <em className={styles.resultCount}>{result.row_count ? `${result.row_count} filas` : ""}</em></h2></div><div className={styles.searchBadge}><Search size={14} /> <span>{result.columns.length ? "QUERY COMPLETE" : "WAITING"}</span></div></div><div className={styles.resultTableWrap}>{result.columns.length ? <table className={styles.resultTable}><thead><tr className={styles.resultRow}>{result.columns.map((column) => <th className={styles.resultHeader} key={column}>{column}</th>)}</tr></thead><tbody>{visibleRows.map((row, rowIndex) => <tr className={styles.resultRow} key={rowIndex}>{row.map((value, valueIndex) => <td className={styles.resultCell} key={valueIndex}>{String(value ?? "NULL")}</td>)}</tr>)}</tbody></table> : <div className={styles.empty}><Table2 size={25} /><p>{result.message}</p></div>}</div>{result.columns.length > 0 && <div className={styles.pagination}><span>ROWS {page * pageSize + 1}-{Math.min((page + 1) * pageSize, result.rows.length)} / {result.rows.length}</span><div><button disabled={page === 0} onClick={() => setPage(page - 1)}>PREV</button><b>{String(page + 1).padStart(2, "0")} / {String(pageCount).padStart(2, "0")}</b><button disabled={page >= pageCount - 1} onClick={() => setPage(page + 1)}>NEXT</button></div></div>}</section>
          <section className={`${styles.panel} ${styles.spatialPanel}`}><div className={styles.panelHeader}><div><span className={styles.eyebrow}>05 / SPATIAL</span><h2 className={styles.panelTitle}>Mapa de puntos</h2></div><span className={styles.costTag}>{spatialPoints.length ? `${spatialPoints.length} PUNTOS` : "SIN DATOS"}</span></div><MapPanel points={spatialPoints} selectedTable={selected} /></section>
        </div>
      )}
    </main>
  );
}
