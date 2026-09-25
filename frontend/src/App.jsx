import { useEffect, useState } from "react";
import {
  analyzeSlide,
  artifactUrl,
  clearToken,
  getHealth,
  getJob,
  getJobEvents,
  getMe,
  getToken,
  fetchArtifact,
  listArtifacts,
  listJobs,
  login,
} from "./api";

const POLL_MS = 1500;

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function App() {
  const [health, setHealth] = useState(null);
  const [user, setUser] = useState(null);
  const [authChecking, setAuthChecking] = useState(true);
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("pathovision");
  const [loginBusy, setLoginBusy] = useState(false);

  const [file, setFile] = useState(null);
  const [mode, setMode] = useState("classify");
  const [tileSize, setTileSize] = useState(256);
  const [minTissue, setMinTissue] = useState(0.25);
  const [idempotencyKey, setIdempotencyKey] = useState(() => crypto.randomUUID());

  const [jobId, setJobId] = useState(null);
  const [job, setJob] = useState(null);
  const [events, setEvents] = useState([]);
  const [history, setHistory] = useState([]);
  const [artifacts, setArtifacts] = useState([]);
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    getHealth()
      .then(async (nextHealth) => {
        setHealth(nextHealth);
        if (!nextHealth.auth_required) {
          setUser({ username: "local-dev", role: "admin" });
          setAuthChecking(false);
          return;
        }
        if (!getToken()) {
          setAuthChecking(false);
          return;
        }
        try {
          setUser(await getMe());
        } catch (err) {
          setError(err.message);
        } finally {
          setAuthChecking(false);
        }
      })
      .catch((err) => {
        setError(err.message);
        setAuthChecking(false);
      });
  }, []);

  useEffect(() => {
    if (!user) return;
    listJobs(25).then(setHistory).catch((err) => setError(err.message));
  }, [user]);

  useEffect(() => {
    if (!jobId || !user) return undefined;

    let cancelled = false;
    const poll = async () => {
      try {
        const next = await getJob(jobId);
        if (cancelled) return;
        setJob(next);
        if (next.status === "completed" || next.status === "failed") {
          getJobEvents(jobId).then(setEvents).catch(() => {});
          listArtifacts(jobId).then((data) => setArtifacts(data.artifacts || [])).catch(() => {});
          listJobs(25).then(setHistory).catch(() => {});
        } else {
          window.setTimeout(poll, POLL_MS);
        }
      } catch (err) {
        if (!cancelled) setError(err.message);
      }
    };

    poll();
    return () => {
      cancelled = true;
    };
  }, [jobId, user]);

  async function submitLogin(event) {
    event.preventDefault();
    setLoginBusy(true);
    setError("");
    try {
      await login(username, password);
      setUser(await getMe());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoginBusy(false);
    }
  }

  function logout() {
    clearToken();
    setUser(null);
    setJob(null);
    setJobId(null);
    setHistory([]);
  }

  async function submit() {
    if (!file) {
      setError("Please select a WSI/image first.");
      return;
    }

    setError("");
    setIsSubmitting(true);
    setJob(null);
    setEvents([]);
    setArtifacts([]);

    try {
      const created = await analyzeSlide(file, {
        mode,
        tileSize,
        minTissue,
        idempotencyKey,
      });
      setJobId(created.job_id);
      setIdempotencyKey(crypto.randomUUID());
    } catch (err) {
      setError(err.message);
    } finally {
      setIsSubmitting(false);
    }
  }

  const summary = job?.result || null;
  const heatmapUrl = jobId && artifacts.includes("tumor_probability_heatmap.png") ? artifactUrl(jobId, "tumor_probability_heatmap.png") : null;
  const overlayUrl = jobId && artifacts.includes("tumor_heatmap_overlay.png") ? artifactUrl(jobId, "tumor_heatmap_overlay.png") : null;
  const segmentationUrl = jobId && artifacts.includes("tumor_segmentation_overlay.png") ? artifactUrl(jobId, "tumor_segmentation_overlay.png") : null;

  if (authChecking) {
    return <div className="app-shell"><div className="content"><section className="card upload-card"><h2>Connecting to PathoVision…</h2></section></div></div>;
  }

  if (!user && health?.auth_required) {
    return (
      <div className="app-shell">
        <main className="content auth-layout">
          <section className="card auth-card">
            <div className="eyebrow">PROTECTED WORKSPACE</div>
            <h1>PathoVision AI</h1>
            <p className="muted">Sign in to submit and inspect whole-slide analyses.</p>
            <form onSubmit={submitLogin} className="login-form">
              <label><span>Username</span><input value={username} onChange={(e) => setUsername(e.target.value)} /></label>
              <label><span>Password</span><input type="password" value={password} onChange={(e) => setPassword(e.target.value)} /></label>
              <button className="primary-button" disabled={loginBusy}>{loginBusy ? "Signing in…" : "Sign in"}</button>
            </form>
            <p className="hint">Development compose uses the bootstrap administrator configured in <code>docker-compose.yml</code>.</p>
            {error && <div className="error-box">{error}</div>}
          </section>
        </main>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">DIGITAL PATHOLOGY · PRODUCTION HARDENED</div>
          <h1>PathoVision AI</h1>
          <p>Whole-slide analysis with CNN classification, WSI heatmaps, U-Net segmentation, and quantitative outputs.</p>
        </div>
        <div className="header-actions">
          <div className="health-pill">
            <span className={health?.status === "ok" ? "health-dot" : "health-dot offline"} />
            <span>API {health?.status || "checking"}</span>
          </div>
          {health?.auth_required && <button className="secondary-button" onClick={logout}>{user?.username} · Logout</button>}
        </div>
      </header>

      <main className="content">
        <section className="upload-card card">
          <div className="section-heading">
            <div><div className="eyebrow">ANALYSIS</div><h2>Upload a pathology slide</h2></div>
            <span className="badge">Local-first</span>
          </div>

          <label className="dropzone">
            <input type="file" accept=".svs,.tif,.tiff,.ndpi,.mrxs,.scn,.png,.jpg,.jpeg" onChange={(event) => { setFile(event.target.files?.[0] || null); setError(""); setJob(null); }} />
            <div className="drop-content"><div className="upload-icon">↑</div><strong>{file ? file.name : "Choose a WSI or image"}</strong><span>SVS, TIFF, NDPI, MRXS, SCN, PNG, JPG</span></div>
          </label>

          <div className="controls">
            <label><span>Analysis mode</span><select value={mode} onChange={(event) => setMode(event.target.value)}><option value="classify">Classification + heatmap</option><option value="full">Full: heatmap + U-Net segmentation</option></select></label>
            <label><span>Tile size</span><input type="number" min="64" max="1024" step="64" value={tileSize} onChange={(event) => setTileSize(Number(event.target.value))} /></label>
            <label><span>Minimum tissue</span><input type="number" min="0" max="1" step="0.05" value={minTissue} onChange={(event) => setMinTissue(Number(event.target.value))} /></label>
          </div>

          <button className="primary-button" onClick={submit} disabled={!file || isSubmitting || job?.status === "running"}>{isSubmitting ? "Uploading…" : "Analyze slide"}</button>
          <p className="hint">Each submission gets an idempotency key so accidental retries do not create duplicate analysis jobs.</p>
          {error && <div className="error-box">{error}</div>}
        </section>

        <section className="history-card card">
          <div className="section-heading"><div><div className="eyebrow">PERSISTENT HISTORY</div><h2>Recent analyses</h2></div><button className="secondary-button" onClick={() => listJobs(25).then(setHistory).catch((err) => setError(err.message))}>Refresh</button></div>
          {history.length === 0 ? <div className="empty-history">No analysis jobs yet.</div> : <div className="history-list">{history.map((item) => <button className="history-item" key={item.job_id} onClick={() => { setJobId(item.job_id); setEvents([]); setArtifacts([]); }}><span className="history-main"><strong>{item.filename}</strong><small>{item.mode} · {item.stage}</small></span><span className={`status-badge ${item.status}`}>{item.status}</span></button>)}</div>}
        </section>

        {job && (
          <section className="status-card card">
            <div className="section-heading"><div><div className="eyebrow">JOB {job.job_id}</div><h2>{job.status === "completed" ? "Analysis complete" : job.status === "failed" ? "Analysis failed" : "Analysis running"}</h2></div><span className={`status-badge ${job.status}`}>{job.status}</span></div>
            <div className="progress-track"><div className="progress-value" style={{ width: `${job.progress}%` }} /></div>
            <div className="status-row"><strong>{job.progress}%</strong><span>{job.stage}</span></div>
            <p className="muted">{job.message}</p>
            {job.error && <div className="error-box">{job.error}</div>}
          </section>
        )}

        {events.length > 0 && <section className="audit-card card"><div className="section-heading"><div><div className="eyebrow">AUDIT TRAIL</div><h2>Job timeline</h2></div></div><div className="timeline">{events.map((event) => <div className="timeline-item" key={event.id}><span className="timeline-dot" /><div><strong>{event.message}</strong><small>{new Date(event.created_at).toLocaleString()}</small></div></div>)}</div></section>}

        {summary && (
          <>
            <section className="metrics-grid">
              <Metric label="Tiles inferred" value={summary.tiles_inferred} />
              <Metric label="Positive tiles" value={summary.positive_tiles} />
              <Metric label="Positive tile fraction" value={formatPercent(summary.positive_tile_fraction)} />
              <Metric label="Mean tumor score" value={summary.mean_tumor_probability?.toFixed(3)} />
              <Metric label="Max tumor score" value={summary.max_tumor_probability?.toFixed(3)} />
              <Metric label="Decision threshold" value={summary.threshold?.toFixed(2)} />
            </section>

            <section className="results-grid">
              <ResultImage title="Probability heatmap" subtitle="Patch-level tumor probability" src={heatmapUrl} />
              <ResultImage title="Classification overlay" subtitle="Probability map over the WSI thumbnail" src={overlayUrl} />
            </section>

            {summary.segmentation && <section className="segmentation-card card"><div className="section-heading"><div><div className="eyebrow">PIXEL-LEVEL SEGMENTATION</div><h2>U-Net tumor region map</h2></div><span className="badge">Full workflow</span></div><div className="segmentation-grid"><ResultImage title="Tumor segmentation" subtitle="Cleaned binary mask over the thumbnail" src={segmentationUrl} /><div className="detail-card card"><div className="detail-grid"><Detail label="Tumor %" value={`${summary.segmentation.tumor_percentage_of_analyzable_area?.toFixed(2)}%`} /><Detail label="Regions" value={summary.segmentation.region_count} /><Detail label="Tumor pixels" value={summary.segmentation.tumor_area_pixels_thumbnail?.toLocaleString()} /><Detail label="Analyzable pixels" value={summary.segmentation.analyzable_area_pixels_thumbnail?.toLocaleString()} /></div><p className="disclaimer">Current quantities are expressed on the thumbnail grid. Physical mm² requires valid slide calibration.</p></div></div></section>}

            <section className="detail-card card"><div className="section-heading"><div><div className="eyebrow">SLIDE METADATA</div><h2>Analysis details</h2></div></div><div className="detail-grid"><Detail label="Dimensions" value={`${summary.slide_dimensions?.[0]?.toLocaleString()} × ${summary.slide_dimensions?.[1]?.toLocaleString()}`} /><Detail label="Pyramid levels" value={summary.pyramid_levels} /><Detail label="Thumbnail" value={`${summary.thumbnail_size?.[0]} × ${summary.thumbnail_size?.[1]}`} /><Detail label="Tile size" value={tileSize} /></div><p className="disclaimer">Research/engineering output only. These model outputs are not a clinically validated diagnosis.</p></section>
          </>
        )}
      </main>

      <footer className="footer"><span>PathoVision AI</span><span>CNN · WSI · OpenSlide · U-Net · React · FastAPI · Celery · PostgreSQL · RabbitMQ · Redis</span></footer>
    </div>
  );
}

function Metric({ label, value }) { return <div className="metric-card"><span>{label}</span><strong>{value ?? "—"}</strong></div>; }
function Detail({ label, value }) { return <div className="detail-item"><span>{label}</span><strong>{value ?? "—"}</strong></div>; }
function ResultImage({ title, subtitle, src }) {
  const [resolvedSrc, setResolvedSrc] = useState(null);

  useEffect(() => {
    let cancelled = false;
    let objectUrl = null;

    async function load() {
      if (!src) {
        setResolvedSrc(null);
        return;
      }
      try {
        const blob = await fetchArtifactFromUrl(src);
        objectUrl = URL.createObjectURL(blob);
        if (!cancelled) setResolvedSrc(objectUrl);
      } catch {
        if (!cancelled) setResolvedSrc(null);
      }
    }

    load();
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [src]);

  return <div className="result-card card"><div className="result-heading"><div><h3>{title}</h3><p>{subtitle}</p></div></div><div className="image-frame">{resolvedSrc ? <img src={resolvedSrc} alt={title} /> : <div className="empty-image">{src ? "Loading result…" : "No result yet"}</div>}</div></div>;
}

async function fetchArtifactFromUrl(url) {
  const pathname = new URL(url).pathname;
  const marker = "/api/jobs/";
  const start = pathname.indexOf(marker);
  const pieces = pathname.slice(start + marker.length).split("/");
  const job = pieces.shift();
  const filename = pieces.join("/");
  return fetchArtifact(job, filename);
}

export default App;
