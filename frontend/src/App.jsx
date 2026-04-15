import { useState, useEffect, useRef, useCallback } from "react";
import { io } from "socket.io-client";

import "./App.css";
import StatsPanel from "./components/StatsPanel";
import Waveform from "./components/Waveform";
import HRVCard from "./components/HRVCard";

// ── Throttle constants ────────────────────────────────────────────────────
// How often (ms) to flush the latest measurement data into React state.
// The backend sends ~30 results/s; updating React state at that rate causes
// 30 re-renders/s of the entire tree. 150 ms (~7 fps for data) is smooth
// enough for numeric displays and waveforms while cutting CPU by ~75%.
const DATA_UPDATE_INTERVAL_MS = 150;
// ─────────────────────────────────────────────────────────────────────────

function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [selectedCamera, setSelectedCamera] = useState(0);
  const [cameras, setCameras] = useState([]);
  const [data, setData] = useState(null);
  const [history, setHistory] = useState([]);
  const [isDarkTheme, setIsDarkTheme] = useState(true);

  const socketRef = useRef(null);
  const latestVitalsRef = useRef({ hr: 0, rr: 0 });
  const imgRef = useRef(null);

  // Pending result buffer — written from WS callback, flushed by interval
  const pendingResultRef = useRef(null);

  /* ── Enumerate cameras ── */
  useEffect(() => {
    navigator.mediaDevices.enumerateDevices()
      .then((devices) => {
        const vids = devices.filter((d) => d.kind === "videoinput");
        setCameras(
          vids.length > 0
            ? vids.map((d, i) => ({ label: d.label || `Camera ${i}`, index: i }))
            : [{ label: "Camera 0 (Default)", index: 0 }]
        );
      })
      .catch(() => setCameras([{ label: "Camera 0 (Default)", index: 0 }]));
  }, []);

  /* ── Theme Toggle ── */
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', isDarkTheme ? "dark" : "light");
  }, [isDarkTheme]);

  /* ── WebSocket ── */
  useEffect(() => {
    const socket = io("http://localhost:5000", {
      transports: ["websocket"],
      reconnectionAttempts: 5,
    });
    socketRef.current = socket;

    socket.on("connect", () => { setIsConnected(true); });
    socket.on("disconnect", () => { setIsConnected(false); setIsRunning(false); });
    socket.on("connect_error", () => setIsConnected(false));

    socket.on("measurement_result", (result) => {
      // ── Frame update: direct DOM write, no React re-render ──
      if (imgRef.current && result.frame_b64) {
        imgRef.current.src = `data:image/jpeg;base64,${result.frame_b64}`;
      }
      // ── Data update: buffer it, flush on interval ──
      pendingResultRef.current = result;
    });

    socket.on("status", (msg) => console.log("[Backend]", msg));
    socket.on("error", (err) => alert(err.message || "An error occurred"));

    return () => socket.disconnect();
  }, []);

  /* ── Throttled data flush to React state ── */
  useEffect(() => {
    const id = setInterval(() => {
      const r = pendingResultRef.current;
      if (r !== null) {
        pendingResultRef.current = null;
        setData(r);
      }
    }, DATA_UPDATE_INTERVAL_MS);
    return () => clearInterval(id);
  }, []);

  const handleStart = useCallback(() => {
    if (socketRef.current?.connected) {
      socketRef.current.emit("start_measurement", { camera_index: selectedCamera });
      setIsRunning(true);
    } else {
      alert("Not connected to server. Please wait or refresh.");
    }
  }, [selectedCamera]);

  const handleStop = useCallback(() => {
    if (socketRef.current?.connected) socketRef.current.emit("stop_measurement");
    setIsRunning(false);
  }, []);

  /* ── Derived values ── */
  const hr = data?.heart_rate ?? 0;
  const rr = data?.respiratory_rate ?? 0;

  useEffect(() => { latestVitalsRef.current = { hr, rr }; }, [hr, rr]);

  useEffect(() => {
    let interval = null;
    if (isRunning) {
      interval = setInterval(() => {
        const { hr: currentHr, rr: currentRr } = latestVitalsRef.current;
        if (currentHr > 0) {
          setHistory((prev) => {
            const now = new Date();
            const t = `${now.getHours().toString().padStart(2, "0")}:${now.getMinutes().toString().padStart(2, "0")}:${now.getSeconds().toString().padStart(2, "0")}`;
            return [{ time: t, hr: Math.round(currentHr), rr: Math.round(currentRr) }, ...prev].slice(0, 8);
          });
        }
      }, 10000);
    } else {
      setHistory([]);
    }
    return () => clearInterval(interval);
  }, [isRunning]);

  /* ── HR / RR vitals ── */
  const bpmMin = data?.bpm_min ?? 0;
  const bpmMax = data?.bpm_max ?? 0;
  const rrRange = data?.rr_range ?? [0, 0];
  const rrStatus = data?.rr_status ?? "";
  const hrStatus = data?.status ?? "Idle";
  const faceBox = data?.face_bbox ?? null;
  const faceDetect = data?.face_detected ?? false;
  const pulseSignal = data?.pulse_signal ?? null;
  const respSignal = data?.resp_signal ?? null;
  const signalQuality = Math.round((data?.signal_quality ?? 0) * 100);
  const hrConfidence = Math.round((data?.confidence ?? 0) * 100);
  const rrConfidence = Math.round((data?.rr_confidence ?? 0) * 100);
  const avgBpm = data?.avg_bpm ?? null;
  const elapsed = data?.elapsed ?? 0;
  const emotion = data?.emotion ?? "Unknown";
  const emotionConf = Math.round((data?.emotion_conf ?? 0) * 100);

  /* ── HRV values ── */
  const sdnn = data?.sdnn ?? 0;
  const rmssd = data?.rmssd ?? 0;
  const pnn50 = data?.pnn50 ?? 0;
  const sd1 = data?.sd1 ?? 0;
  const sd2 = data?.sd2 ?? 0;
  const lfHf = data?.lf_hf ?? 0;
  const lfNu = data?.lf_nu ?? 0;
  const hfNu = data?.hf_nu ?? 0;
  const wtLfHf = data?.wt_lf_hf ?? 0;
  const wtTrend = data?.wt_lf_hf_trend ?? [];
  const hrvConf = Math.round((data?.hrv_confidence ?? 0) * 100);
  const hrvStatusMsg = data?.hrv_status_msg ?? "";
  const interpretation = data?.interpretation ?? null;
  const rrIntervals = data?.rr_intervals ?? [];

  const vitalsForPanel = { signalQuality, hrConfidence, rrConfidence, hrvConf };

  const getBpmColor = (v) => {
    if (!v || v === 0) return "#94A3B8";
    if (v < 60) return "#3B82F6";
    if (v < 100) return "#22C55E";
    if (v < 140) return "#F97316";
    return "#EF4444";
  };
  const getRrColor = (v) => {
    if (!v || v === 0) return "#94A3B8";
    if (v < 12) return "#3B82F6";
    if (v < 20) return "#22C55E";
    if (v < 30) return "#F97316";
    return "#EF4444";
  };

  const getEmotionEmoji = (emo) => {
    switch (emo.toLowerCase()) {
      case 'happy': return '😊';
      case 'sad': return '😢';
      case 'angry': return '😠';
      case 'surprise': return '😲';
      case 'fear': return '😨';
      case 'disgust': return '🤢';
      case 'neutral': return '😐';
      default: return '⏳';
    }
  };

  const hrColor = getBpmColor(hr);
  const rrColor = getRrColor(rr);

  const formatElapsed = (s) =>
    `${Math.floor(s / 60).toString().padStart(2, "0")}:${(s % 60).toString().padStart(2, "0")}`;

  return (
    <div className="app-container">

      {/* ── Header ── */}
      <header className="app-header">
        <div className="logo" style={{ gap: "12px" }}>
          <svg viewBox="0 0 100 60" style={{ width: 44, height: 26, flexShrink: 0 }}>
            <defs>
              <linearGradient id="topLid" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#f59e0b" />
                <stop offset="60%" stopColor="#ef4444" />
                <stop offset="100%" stopColor="#be123c" />
              </linearGradient>
              <linearGradient id="botLid" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#a3e635" />
                <stop offset="50%" stopColor="#22c55e" />
                <stop offset="100%" stopColor="#0f766e" />
              </linearGradient>
              <linearGradient id="irisGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#c084fc" />
                <stop offset="50%" stopColor="#7e22ce" />
                <stop offset="100%" stopColor="#3730a3" />
              </linearGradient>
            </defs>
            <path d="M 5 30 C 25 -5, 75 -5, 95 25 C 70 10, 30 10, 5 30 Z" fill="url(#topLid)" />
            <path d="M 20 35 C 45 65, 85 60, 100 35 C 75 50, 45 45, 20 35 Z" fill="url(#botLid)" />
            <circle cx="50" cy="30" r="14" fill="url(#irisGrad)" />
            <circle cx="58" cy="22" r="3.5" fill="var(--bg-panel)" />
            <circle cx="43" cy="33" r="1.5" fill="#ffffff" opacity="0.8" />
          </svg>
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <h1 style={{ margin: 0, padding: 0, fontSize: "22px", fontWeight: 900, letterSpacing: "1.5px", textShadow: "0 0 10px rgba(255, 255, 255, 0.2)" }}>
              CARDIAC VISION
            </h1>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          {avgBpm && (
            <span style={{ fontSize: 11, color: "#8b949e" }}>
              AVG: <strong style={{ color: "#e6edf3" }}>{avgBpm} BPM</strong>
            </span>
          )}
          {elapsed > 0 && (
            <span style={{ fontSize: 11, color: "#8b949e" }}>{formatElapsed(elapsed)}</span>
          )}
          <div className="connection-status">
            <div className="dot" style={{ background: isConnected ? "#3fb950" : "#f85149" }} />
            <span style={{ color: isConnected ? "#3fb950" : "#f85149" }}>
              {isConnected ? "Connected" : "Disconnected"}
            </span>
          </div>
          <button
            onClick={() => setIsDarkTheme(!isDarkTheme)}
            style={{
              background: "var(--bg-card)",
              border: "1px solid var(--border-main)",
              color: "var(--text-main)",
              padding: "4px 8px",
              borderRadius: "4px",
              cursor: "pointer",
              fontSize: "11px",
              fontWeight: "600"
            }}
          >
            {isDarkTheme ? "🌙 DARK" : "☀️ LIGHT"}
          </button>
        </div>
      </header>

      {/* ── 3-column body ── */}
      <div className="app-body">

        {/* CENTER: Camera + Waveform + Controls */}
        <main className="center-panel">

          <div className="camera-wrapper">
            {/* Always render <img> so the ref is always valid; hide with CSS when no feed */}
            <img
              ref={imgRef}
              alt="Live feed"
              className="camera-img"
              style={{ display: data?.frame_b64 ? 'block' : 'none' }}
            />
            {!data?.frame_b64 && (
              <div className="camera-placeholder">
                <svg viewBox="0 0 24 24" fill="none" style={{ width: 44, height: 44, opacity: 0.2 }}>
                  <path d="M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2z"
                    stroke="#94A3B8" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                  <circle cx="12" cy="13" r="4" stroke="#94A3B8" strokeWidth="1.5" />
                </svg>
                <span>Camera inactive — press START</span>
              </div>
            )}

            {faceDetect && faceBox && data?.frame_b64 && (
              <div className="face-box" style={{
                left: `${(faceBox[0] / 640) * 100}%`,
                top: `${(faceBox[1] / 480) * 100}%`,
                width: `${(faceBox[2] / 640) * 100}%`,
                height: `${(faceBox[3] / 480) * 100}%`,
                borderColor: hrColor,
                boxShadow: `0 0 10px ${hrColor}55`,
              }}>
                {(emotion !== "Unknown" && emotion !== "No face") && (
                  <div style={{
                    position: 'absolute',
                    top: '-32px',
                    left: '50%',
                    transform: 'translateX(-50%)',
                    background: 'rgba(9, 13, 19, 0.85)',
                    backdropFilter: 'blur(4px)',
                    color: '#fff',
                    padding: '4px 10px',
                    borderRadius: '6px',
                    fontSize: '12px',
                    fontWeight: 700,
                    whiteSpace: 'nowrap',
                    border: `1px solid ${hrColor}88`,
                    boxShadow: '0 2px 8px rgba(0,0,0,0.5)',
                    display: 'flex',
                    gap: '6px',
                    alignItems: 'center'
                  }}>
                    <span style={{ fontSize: '14px' }}>{getEmotionEmoji(emotion)}</span>
                    <span style={{ letterSpacing: '0.5px' }}>{emotion.toUpperCase()}</span>
                    {emotionConf > 0 && <span style={{ color: '#8b949e', fontSize: '10px' }}>{emotionConf}%</span>}
                  </div>
                )}
              </div>
            )}

            {isRunning && !faceDetect && (
              <div className="camera-status-overlay">{hrStatus}</div>
            )}
          </div>

          {/* Waveform strip */}
          <div className="waveform-section">
            <Waveform
              pulseSignal={pulseSignal}
              respSignal={respSignal}
              hrColor={hrColor}
              rrColor={rrColor}
              isRunning={isRunning}
            />
          </div>

          {/* Camera selector + start/stop */}
          <div className="camera-controls">
            <select
              value={selectedCamera}
              onChange={(e) => setSelectedCamera(Number(e.target.value))}
              disabled={isRunning}
            >
              {cameras.map((cam) => (
                <option key={cam.index} value={cam.index}>{cam.label}</option>
              ))}
            </select>
            <button
              className={`btn-start${isRunning ? " btn-stop" : ""}`}
              onClick={isRunning ? handleStop : handleStart}
            >
              {isRunning ? "⏹ STOP" : "▶ START"}
            </button>
          </div>
        </main>

        {/* RIGHT: Vital cards + HRV card */}
        <aside className="right-panel">

          {/* Heart Rate */}
          <div className="vital-card hrv-metric-hoverable">
            <svg viewBox="0 0 24 24" fill="none" className="vital-svg-icon" style={{
              animation: hr > 0 ? `heartbeat ${(60 / hr).toFixed(2)}s ease-in-out infinite` : "none",
            }}>
              <path
                d="M12 21C12 21 3 14.5 3 8.5C3 5.42 5.42 3 8.5 3C10.24 3 11.91 3.81 13 5.08C14.09 3.81 15.76 3 17.5 3C20.58 3 23 5.42 23 8.5C23 14.5 12 21 12 21Z"
                stroke={hrColor} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"
                fill={hr > 0 ? hrColor + "22" : "none"}
              />
            </svg>
            <div className="vital-value" style={{ color: hrColor }}>
              {hr > 0 ? Math.round(hr) : "–"}
              {hr > 0 && <span className="vital-unit">BPM</span>}
            </div>
            <div className="vital-label">HEART RATE</div>
            {bpmMin > 0 && (
              <div className="vital-range" style={{ color: hrColor, background: hrColor + "18" }}>
                {bpmMin}–{bpmMax} BPM
              </div>
            )}
            <div className="vital-status">
              {hr > 0 ? hrStatus : (isRunning ? "Calibrating…" : "—")}
            </div>
            <div className="custom-tooltip" style={{ top: '80%', left: '50%', transform: 'translateX(-50%)' }}>
              <div className="zone-row" style={{ marginBottom: 4 }}>
                <div className="zone-dot" style={{ background: "#79c0ff" }} />
                <span className="zone-name" style={{ fontSize: 10 }}>Resting</span>
                <span className="zone-range" style={{ fontSize: 10 }}>&lt; 60</span>
              </div>
              <div className="zone-row" style={{ marginBottom: 4 }}>
                <div className="zone-dot" style={{ background: "#3fb950" }} />
                <span className="zone-name" style={{ fontSize: 10 }}>Normal</span>
                <span className="zone-range" style={{ fontSize: 10 }}>60–100</span>
              </div>
              <div className="zone-row" style={{ marginBottom: 4 }}>
                <div className="zone-dot" style={{ background: "#e3b341" }} />
                <span className="zone-name" style={{ fontSize: 10 }}>Elevated</span>
                <span className="zone-range" style={{ fontSize: 10 }}>100–140</span>
              </div>
              <div className="zone-row" style={{ marginBottom: 0 }}>
                <div className="zone-dot" style={{ background: "#f85149" }} />
                <span className="zone-name" style={{ fontSize: 10 }}>High</span>
                <span className="zone-range" style={{ fontSize: 10 }}>&gt; 140</span>
              </div>
            </div>
          </div>

          {/* Respiratory Rate */}
          <div className="vital-card hrv-metric-hoverable">
            <svg viewBox="0 0 24 24" fill="none" className="vital-svg-icon">
              <path d="M12 3v4M12 7c-1.5 0-3 1-3.5 2.5L7 14c-.5 1.5.5 3 2 3h1v2h4v-2h1c1.5 0 2.5-1.5 2-3l-1.5-4.5C15 8 13.5 7 12 7z"
                stroke={rrColor} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"
                fill={rrColor + "22"} />
              <path d="M8.5 17c-1.5.5-3 0-3.5-1.5L4 13c-.5-1.5.5-3 2-3"
                stroke={rrColor} strokeWidth="1.5" strokeLinecap="round" />
              <path d="M15.5 17c1.5.5 3 0 3.5-1.5L20 13c.5-1.5-.5-3-2-3"
                stroke={rrColor} strokeWidth="1.5" strokeLinecap="round" />
            </svg>
            <div className="vital-value" style={{ color: rrColor }}>
              {rr > 0 ? Math.round(rr) : "–"}
              {rr > 0 && <span className="vital-unit">BRPM</span>}
            </div>
            <div className="vital-label">RESP. RATE</div>
            {rrRange[0] > 0 && (
              <div className="vital-range" style={{ color: rrColor, background: rrColor + "18" }}>
                {rrRange[0]}–{rrRange[1]} BRPM
              </div>
            )}
            <div className="vital-status">
              {rr > 0 ? "breaths / min" : (isRunning ? rrStatus : "—")}
            </div>
            <div className="custom-tooltip" style={{ top: '80%', left: '50%', transform: 'translateX(-50%)' }}>
              <div className="zone-row" style={{ marginBottom: 4 }}>
                <div className="zone-dot" style={{ background: "#79c0ff" }} />
                <span className="zone-name" style={{ fontSize: 10 }}>Low</span>
                <span className="zone-range" style={{ fontSize: 10 }}>&lt; 12</span>
              </div>
              <div className="zone-row" style={{ marginBottom: 4 }}>
                <div className="zone-dot" style={{ background: "#3fb950" }} />
                <span className="zone-name" style={{ fontSize: 10 }}>Normal</span>
                <span className="zone-range" style={{ fontSize: 10 }}>12–20</span>
              </div>
              <div className="zone-row" style={{ marginBottom: 4 }}>
                <div className="zone-dot" style={{ background: "#e3b341" }} />
                <span className="zone-name" style={{ fontSize: 10 }}>High</span>
                <span className="zone-range" style={{ fontSize: 10 }}>20–30</span>
              </div>
              <div className="zone-row" style={{ marginBottom: 0 }}>
                <div className="zone-dot" style={{ background: "#f85149" }} />
                <span className="zone-name" style={{ fontSize: 10 }}>Critical</span>
                <span className="zone-range" style={{ fontSize: 10 }}>&gt; 30</span>
              </div>
            </div>
          </div>

          {/* HRV Card */}
          <HRVCard
            sdnn={sdnn}
            rmssd={rmssd}
            pnn50={pnn50}
            sd1={sd1}
            sd2={sd2}
            lfHf={lfHf}
            lfNu={lfNu}
            hfNu={hfNu}
            wtLfHf={wtLfHf}
            hrvConf={hrvConf}
            hrvStatusMsg={hrvStatusMsg}
            interpretation={interpretation}
            isRunning={isRunning}
          />

          {/* History Log */}
          {history.length > 0 && (
            <div className="history-card">
              <div className="vital-label" style={{ marginBottom: 6, color: "#e6edf3" }}>LOG</div>
              <div className="history-table-container">
                <table className="history-table">
                  <thead>
                    <tr>
                      <th>Time</th>
                      <th>HR</th>
                      <th>Resp</th>
                    </tr>
                  </thead>
                  <tbody>
                    {history.map((entry, idx) => (
                      <tr key={idx}>
                        <td style={{ color: "#8b949e" }}>{entry.time}</td>
                        <td style={{ color: getBpmColor(entry.hr), fontWeight: "bold" }}>{entry.hr}</td>
                        <td style={{ color: getRrColor(entry.rr), fontWeight: "bold" }}>{entry.rr > 0 ? entry.rr : "–"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

        </aside>
      </div>
    </div>
  );
}

export default App;