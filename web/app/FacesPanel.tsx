"use client";

import { useEffect, useRef, useState } from "react";
import { AlertCircle, Camera, Play, ScanFace, Upload, UserPlus, Zap } from "lucide-react";

import styles from "./page.module.css";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Match = {
  rank: number;
  face_id: number;
  photo_id: number;
  person_id: number;
  name: string;
  distance: number;
  similarity: number;
  similarity_pct: number;
};

type Decision = {
  status: "known" | "unknown";
  person_id: number | null;
  name: string | null;
  closest_name: string | null;
  similarity_pct: number;
  threshold: number;
  threshold_pct: number;
};

type FaceResult = {
  model: string;
  index: string;
  metric: string;
  k: number;
  search_ms: number;
  candidates: number;
  gallery_size: number;
  decision: Decision;
  persons: Match[];
  matches: Match[];
  source?: string;
  cache_ms?: number;
  db_ms?: number;
};

type DetectedFace = {
  x: number;
  y: number;
  w: number;
  h: number;
  det_score?: number;
  result: FaceResult;
};

type RecognizeResponse = {
  image: { width: number; height: number };
  params: Record<string, unknown>;
  faces: DetectedFace[];
  timings: { detect_ms: number; embed_ms: Record<string, number>; total_ms: number };
};

type Person = { person_id: number; name: string; photos: number };

type CacheStats = {
  enabled: boolean;
  backend: string;
  detail: string;
  max_hot: number;
  margin: number;
  hits: number;
  misses: number;
  hit_rate: number;
};

export default function FacesPanel() {
  const [model, setModel] = useState("arcface");
  const [index, setIndex] = useState("hnsw");
  const [metric, setMetric] = useState("coseno");
  const [k, setK] = useState(5);

  const [preview, setPreview] = useState<string>("");
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<RecognizeResponse | null>(null);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);

  const [persons, setPersons] = useState<Person[]>([]);
  const [totalPersons, setTotalPersons] = useState(0);
  const [cache, setCache] = useState<CacheStats | null>(null);

  // alta de personas
  const [newName, setNewName] = useState("");
  const [newPhotos, setNewPhotos] = useState<FileList | null>(null);
  const [registering, setRegistering] = useState(false);

  // cámara
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [cameraOn, setCameraOn] = useState(false);

  useEffect(() => {
    void refreshPersons();
    void refreshCache();
    return () => stopCamera();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function refreshPersons() {
    try {
      const response = await fetch(`${API}/api/faces/persons`);
      if (!response.ok) return;
      const data = await response.json();
      // la API pagina: `persons` es la página actual y `total` el conteo real
      const page: Person[] = Array.isArray(data) ? data : (data.persons ?? []);
      setPersons(page);
      setTotalPersons(Array.isArray(data) ? page.length : (data.total ?? page.length));
    } catch {
      setPersons([]);
      setTotalPersons(0);
    }
  }

  async function refreshCache() {
    try {
      const response = await fetch(`${API}/api/faces/cache`);
      if (response.ok) setCache(await response.json());
    } catch {
      setCache(null);
    }
  }

  function useFile(picked: File | null) {
    setFile(picked);
    setResult(null);
    setMessage("");
    if (preview) URL.revokeObjectURL(preview);
    setPreview(picked ? URL.createObjectURL(picked) : "");
  }

  // ---------------- cámara ----------------

  async function startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
      streamRef.current = stream;
      setCameraOn(true);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
    } catch {
      setMessage("No se pudo abrir la cámara. Revisa los permisos del navegador.");
    }
  }

  function stopCamera() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setCameraOn(false);
  }

  async function capture() {
    const video = videoRef.current;
    if (!video) return;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d")?.drawImage(video, 0, 0);
    const blob: Blob | null = await new Promise((resolve) =>
      canvas.toBlob((b) => resolve(b), "image/jpeg", 0.92));
    if (blob) {
      useFile(new File([blob], "captura.jpg", { type: "image/jpeg" }));
      stopCamera();
    }
  }

  // ---------------- reconocimiento ----------------

  async function recognize() {
    if (!file) return;
    setLoading(true);
    setMessage("");
    try {
      const form = new FormData();
      form.append("image", file);
      form.append("model", model);
      form.append("index", index);
      form.append("metric", metric);
      form.append("k", String(k));
      const response = await fetch(`${API}/api/faces/recognize`, { method: "POST", body: form });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? "No se pudo reconocer el rostro");
      setResult(data);
      void refreshCache();
    } catch (error) {
      setResult(null);
      setMessage(error instanceof Error ? error.message : "Error de reconocimiento");
    } finally {
      setLoading(false);
    }
  }

  async function registerPerson() {
    if (!newName.trim() || !newPhotos?.length) return;
    setRegistering(true);
    setMessage("");
    try {
      const form = new FormData();
      form.append("name", newName.trim());
      Array.from(newPhotos).forEach((photo) => form.append("photos", photo));
      const response = await fetch(`${API}/api/faces/persons`, { method: "POST", body: form });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? "No se pudo registrar a la persona");
      setNewName("");
      setNewPhotos(null);
      await refreshPersons();
      setMessage(`Registrado: ${data.name ?? newName}`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Error al registrar");
    } finally {
      setRegistering(false);
    }
  }

  async function clearCache() {
    try {
      const response = await fetch(`${API}/api/faces/cache/clear`, { method: "POST" });
      if (response.ok) setCache(await response.json());
    } catch {
      /* el caché es opcional: si falla, la vista sigue igual */
    }
  }

  const first = result?.faces?.[0]?.result;

  return (
    <div className={`${styles.workspace} ${styles.searchWorkspace}`}>
      <section className={`${styles.panel} ${styles.queryPanel}`}>
        <div className={styles.panelHeader}>
          <div>
            <span className={styles.eyebrow}>13 / FACE RECOGNITION</span>
            <h2 className={styles.panelTitle}>Reconocer un rostro</h2>
          </div>
          <div className={styles.searchBadge}><ScanFace size={14} /> {totalPersons} personas</div>
        </div>

        <div className={styles.panelBody}>
          <label className={styles.eyebrow}>IMAGEN DE CONSULTA</label>
          <p className={styles.helpText}>
            Sube una foto o toma una con la cámara. Se detectan los rostros, se extrae el
            embedding y se busca el más parecido en la galería.
          </p>

          <div className={styles.exampleChoices}>
            <input
              id="face-upload"
              type="file"
              accept="image/*"
              hidden
              onChange={(event) => useFile(event.target.files?.[0] ?? null)}
            />
            <label htmlFor="face-upload" className={styles.secondaryButton}>
              <Upload size={14} /> SUBIR FOTO
            </label>
            {cameraOn ? (
              <>
                <button className={styles.secondaryButton} onClick={capture}>
                  <Camera size={14} /> CAPTURAR
                </button>
                <button className={styles.secondaryButton} onClick={stopCamera}>CERRAR CÁMARA</button>
              </>
            ) : (
              <button className={styles.secondaryButton} onClick={startCamera}>
                <Camera size={14} /> USAR CÁMARA
              </button>
            )}
          </div>

          {cameraOn && (
            <video ref={videoRef} className={styles.exampleThumbnail}
                   style={{ width: "100%", maxHeight: 280, objectFit: "cover" }}
                   playsInline muted />
          )}

          {!cameraOn && preview && (
            <img className={styles.exampleThumbnail} src={preview} alt="Rostro de consulta"
                 style={{ width: "100%", maxHeight: 280, objectFit: "contain" }} />
          )}

          <div className={styles.queryMeta}>
            <label>Modelo <select value={model} onChange={(event) => setModel(event.target.value)}>
              <option value="arcface">ArcFace</option>
              <option value="facenet">FaceNet</option>
            </select></label>
            <label>Índice <select value={index} onChange={(event) => setIndex(event.target.value)}>
              <option value="hnsw">HNSW</option>
              <option value="ivf">IVF</option>
              <option value="exact">Exacto</option>
            </select></label>
            <label>Métrica <select value={metric} onChange={(event) => setMetric(event.target.value)}>
              <option value="coseno">Coseno</option>
              <option value="euclidiana">Euclidiana</option>
              <option value="producto_punto">Producto punto</option>
            </select></label>
            <label>k <select value={k} onChange={(event) => setK(Number(event.target.value))}>
              {[1, 3, 5, 10].map((n) => <option key={n} value={n}>{n}</option>)}
            </select></label>
          </div>

          <div className={styles.queryActions}>
            <button className={styles.runButton} onClick={recognize} disabled={loading || !file}>
              <Play size={14} fill="currentColor" /> {loading ? "RECONOCIENDO..." : "RECONOCER"}
            </button>
          </div>

          {message && <div className={styles.catalogHint}><AlertCircle size={15} /><span>{message}</span></div>}

          {/* ---- alta de personas ---- */}
          <label className={styles.eyebrow} style={{ marginTop: 18 }}>REGISTRAR PERSONA</label>
          <p className={styles.helpText}>
            Varias fotos de la misma persona mejoran el reconocimiento: cada una aporta un
            embedding a la galería.
          </p>
          <input className={styles.editorInput} placeholder="Nombre y apellido"
                 value={newName} onChange={(event) => setNewName(event.target.value)} />
          <div className={styles.exampleChoices}>
            <input id="face-register" type="file" accept="image/*" multiple hidden
                   onChange={(event) => setNewPhotos(event.target.files)} />
            <label htmlFor="face-register" className={styles.secondaryButton}>
              <Upload size={14} /> {newPhotos?.length ? `${newPhotos.length} FOTOS` : "ELEGIR FOTOS"}
            </label>
            <button className={styles.secondaryButton} onClick={registerPerson}
                    disabled={registering || !newName.trim() || !newPhotos?.length}>
              <UserPlus size={14} /> {registering ? "REGISTRANDO..." : "REGISTRAR"}
            </button>
          </div>
        </div>
      </section>

      <section className={`${styles.panel} ${styles.resultsPanel}`}>
        <div className={styles.panelHeader}>
          <div>
            <span className={styles.eyebrow}>14 / IDENTIDAD</span>
            <h2 className={styles.panelTitle}>Resultado</h2>
          </div>
          <div className={styles.searchBadge}>
            <Zap size={14} /> {result ? `${result.timings.total_ms.toFixed(0)} ms` : "WAITING"}
          </div>
        </div>

        {first && (
          <div className={styles.queryMeta}>
            <span>
              {first.index.toUpperCase()} · {first.metric} · {first.gallery_size} rostros
              {first.source === "cache" && " · CACHÉ"}
            </span>
            <span>
              detección {result!.timings.detect_ms.toFixed(0)} ms · búsqueda{" "}
              {first.source === "cache"
                ? `${(first.cache_ms ?? 0).toFixed(2)} ms (caché)`
                : `${first.search_ms.toFixed(2)} ms`}
            </span>
          </div>
        )}

        <div className={styles.resultTableWrap}>
          {result?.faces?.length ? (
            <>
              {result.faces.map((face, i) => (
                <div key={i} className={styles.planBody}>
                  <div className={styles.nodeDetails}>
                    <b>
                      {face.result.decision.status === "known"
                        ? face.result.decision.name
                        : "Desconocido"}
                    </b>{" "}
                    · {face.result.decision.similarity_pct.toFixed(1)}% de confianza
                    {face.result.decision.status === "unknown" &&
                      ` (el más parecido era ${face.result.decision.closest_name ?? "—"}, umbral ${face.result.decision.threshold_pct}%)`}
                  </div>
                </div>
              ))}
              <table className={styles.resultTable}>
                <thead>
                  <tr className={styles.resultRow}>
                    <th className={styles.resultHeader}>#</th>
                    <th className={styles.resultHeader}>Persona</th>
                    <th className={styles.resultHeader}>Similitud</th>
                    <th className={styles.resultHeader}>Distancia</th>
                    <th className={styles.resultHeader}>Foto</th>
                  </tr>
                </thead>
                <tbody>
                  {(first?.matches ?? []).map((match) => (
                    <tr className={styles.resultRow} key={`${match.face_id}-${match.rank}`}>
                      <td className={styles.resultCell}>{match.rank}</td>
                      <td className={styles.resultCell}>{match.name}</td>
                      <td className={styles.resultCell}>{match.similarity_pct.toFixed(1)}%</td>
                      <td className={styles.resultCell}>{match.distance.toFixed(6)}</td>
                      <td className={styles.resultCell}>
                        <img className={styles.multimediaThumbnail}
                             src={`${API}/api/faces/images/${match.photo_id}`}
                             alt={match.name} loading="lazy" />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          ) : (
            <div className={styles.empty}>
              <ScanFace size={25} />
              <p>{totalPersons
                ? "Sube o captura una foto y pulsa RECONOCER."
                : "Registra al menos una persona para poder reconocer rostros."}</p>
            </div>
          )}
        </div>

        {cache && (
          <footer className={styles.panelFooter}>
            <span>CACHÉ ({cache.backend})</span>
            <b>{cache.hits} aciertos · {cache.misses} fallos · {(cache.hit_rate * 100).toFixed(0)}%</b>
            <button className={styles.secondaryButton} onClick={clearCache}>VACIAR</button>
          </footer>
        )}
      </section>
    </div>
  );
}
