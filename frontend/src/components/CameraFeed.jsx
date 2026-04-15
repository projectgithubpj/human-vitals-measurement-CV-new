import { useEffect, useRef, useState } from "react";

/* ── Animated heart icon ── */
function HeartIcon({ bpm, color }) {
  const scale = bpm > 0 ? 1 : 1;
  return (
    <svg
      viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"
      style={{
        width: 40, height: 40,
        animation: bpm > 0 ? `heartbeat ${60 / bpm}s ease-in-out infinite` : "none",
      }}
    >
      <path
        d="M12 21C12 21 3 14.5 3 8.5C3 5.42 5.42 3 8.5 3C10.24 3 11.91 3.81 13 5.08C14.09 3.81 15.76 3 17.5 3C20.58 3 23 5.42 23 8.5C23 14.5 12 21 12 21Z"
        stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"
        fill={bpm > 0 ? color + "22" : "none"}
      />
    </svg>
  );
}

/* ── Lung / breath icon ── */
function LungIcon({ color }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"
      style={{ width: 36, height: 36 }}>
      <path d="M12 3v4M12 7c-1.5 0-3 1-3.5 2.5L7 14c-.5 1.5.5 3 2 3h1v2h4v-2h1c1.5 0 2.5-1.5 2-3l-1.5-4.5C15 8 13.5 7 12 7z"
        stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"
        fill={color + "22"} />
      <path d="M8.5 17c-1.5.5-3 0-3.5-1.5L4 13c-.5-1.5.5-3 2-3"
        stroke={color} strokeWidth="1.5" strokeLinecap="round" />
      <path d="M15.5 17c1.5.5 3 0 3.5-1.5L20 13c.5-1.5-.5-3-2-3"
        stroke={color} strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

function VitalCard({ icon, title, value, unit, subText, color, range, rangeLabel }) {
  return (
    <div style={{
      background: "#fff",
      borderRadius: 16,
      padding: "20px 20px 16px",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      gap: 6,
      boxShadow: "0 1px 3px rgba(0,0,0,0.07)",
      border: "1px solid #E2E8F0",
      minWidth: 160,
      flex: 1,
    }}>
      <div style={{ marginBottom: 4 }}>{icon}</div>

      <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
        {value > 0 ? (
          <>
            <span style={{ fontSize: 42, fontWeight: 800, color, lineHeight: 1, fontVariantNumeric: "tabular-nums" }}>
              {Math.round(value)}
            </span>
            <span style={{ fontSize: 14, color: "#94A3B8", fontWeight: 500 }}>{unit}</span>
          </>
        ) : (
          <span style={{ fontSize: 32, fontWeight: 700, color: "#CBD5E1" }}>--</span>
        )}
      </div>

      <span style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.1em", color: "#94A3B8" }}>{title}</span>

      {range && range[0] > 0 && (
        <span style={{ fontSize: 10, color: color + "BB", background: color + "15", borderRadius: 20, padding: "2px 10px" }}>
          {range[0]}–{range[1]} {unit}
        </span>
      )}

      <span style={{
        fontSize: 11, color: "#64748B", textAlign: "center",
        minHeight: 14, marginTop: 2,
      }}>
        {subText || ""}
      </span>
    </div>
  );
}

function getBpmColor(bpm) {
  if (!bpm || bpm === 0) return "#94A3B8";
  if (bpm < 60) return "#3B82F6";
  if (bpm < 100) return "#22C55E";
  if (bpm < 140) return "#F97316";
  return "#EF4444";
}

function getRrColor(rr) {
  if (!rr || rr === 0) return "#94A3B8";
  if (rr < 12) return "#3B82F6";
  if (rr < 20) return "#22C55E";
  if (rr < 30) return "#F97316";
  return "#EF4444";
}

export default function CameraFeed({
  data,
  isRunning,
  onStart,
  onStop,
  selectedCamera,
  onCameraChange,
  cameras,
}) {
  const imgRef = useRef(null);

  const frameB64 = data?.frame_b64 ?? null;
  const status = data?.status ?? "Camera inactive";
  const faceBox = data?.face_bbox ?? null;
  const faceDetect = data?.face_detected ?? false;
  const hr = data?.heart_rate ?? 0;
  const rr = data?.respiratory_rate ?? 0;
  const bpmMin = data?.bpm_min ?? 0;
  const bpmMax = data?.bpm_max ?? 0;
  const rrRange = data?.rr_range ?? [0, 0];
  const rrStatus = data?.rr_status ?? "";
  const hrStatus = data?.status ?? "";

  useEffect(() => {
    if (imgRef.current && frameB64) {
      imgRef.current.src = `data:image/jpeg;base64,${frameB64}`;
    }
  }, [frameB64]);

  const hrColor = getBpmColor(hr);
  const rrColor = getRrColor(rr);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

      {/* ── Live Feed Card ── */}
      <div className="card" style={{ padding: 0, overflow: "hidden" }}>
        <div style={{ padding: "14px 18px 10px", borderBottom: "1px solid #F1F5F9" }}>
          <span className="card-title">LIVE FEED</span>
        </div>

        {/* Video container */}
        <div style={{ position: "relative", background: "#0F172A", aspectRatio: "4/3", overflow: "hidden" }}>
          {frameB64 ? (
            <img
              ref={imgRef}
              alt="Live feed"
              style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }}
            />
          ) : (
            <div style={{
              width: "100%", height: "100%",
              display: "flex", flexDirection: "column",
              alignItems: "center", justifyContent: "center", gap: 12,
            }}>
              <svg viewBox="0 0 24 24" fill="none" style={{ width: 48, height: 48, opacity: 0.3 }}>
                <path d="M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2z"
                  stroke="#94A3B8" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                <circle cx="12" cy="13" r="4" stroke="#94A3B8" strokeWidth="1.5" />
              </svg>
              <span style={{ color: "#475569", fontSize: 13 }}>Camera inactive</span>
            </div>
          )}

          {/* Face bounding box overlay */}
          {faceDetect && faceBox && frameB64 && (
            <div style={{
              position: "absolute",
              left: `${(faceBox[0] / 640) * 100}%`,
              top: `${(faceBox[1] / 480) * 100}%`,
              width: `${(faceBox[2] / 640) * 100}%`,
              height: `${(faceBox[3] / 480) * 100}%`,
              border: `2px solid ${hrColor}`,
              borderRadius: 6,
              boxShadow: `0 0 12px ${hrColor}55`,
              pointerEvents: "none",
            }} />
          )}

          {/* Status overlay */}
          {isRunning && !faceDetect && (
            <div style={{
              position: "absolute", bottom: 12, left: "50%", transform: "translateX(-50%)",
              background: "rgba(15,23,42,0.75)", borderRadius: 20,
              padding: "4px 14px", fontSize: 12, color: "#CBD5E1",
              backdropFilter: "blur(4px)",
            }}>
              {status}
            </div>
          )}
        </div>

        {/* Camera selector + controls */}
        <div style={{ padding: "12px 14px", display: "flex", gap: 10, alignItems: "center" }}>
          <select
            value={selectedCamera}
            onChange={(e) => onCameraChange(e.target.value)}
            disabled={isRunning}
            style={{
              flex: 1, padding: "8px 12px", borderRadius: 8,
              border: "1px solid #E2E8F0", fontSize: 13,
              background: isRunning ? "#F8FAFC" : "#fff",
              color: "#334155", outline: "none", cursor: isRunning ? "not-allowed" : "pointer",
            }}
          >
            {cameras.length === 0
              ? <option value="0">Camera 0 (Default)</option>
              : cameras.map((c) => (
                <option key={c.index} value={c.index}>{c.label}</option>
              ))
            }
          </select>

          <button
            onClick={isRunning ? onStop : onStart}
            style={{
              padding: "8px 22px", borderRadius: 8, border: "none",
              background: isRunning ? "#EF4444" : "#3B82F6",
              color: "#fff", fontWeight: 700, fontSize: 13,
              cursor: "pointer", letterSpacing: "0.04em",
              transition: "background 0.2s",
              display: "flex", alignItems: "center", gap: 6,
            }}
          >
            {isRunning ? (
              <>
                <span style={{ width: 8, height: 8, background: "#fff", borderRadius: 1, display: "inline-block" }} />
                STOP
              </>
            ) : (
              <>
                <span style={{
                  width: 0, height: 0,
                  borderTop: "5px solid transparent",
                  borderBottom: "5px solid transparent",
                  borderLeft: "8px solid #fff",
                  display: "inline-block",
                }} />
                START
              </>
            )}
          </button>
        </div>
      </div>

      {/* ── Vitals Row: Heart Rate + Respiratory Rate ── */}
      <div style={{ display: "flex", gap: 14 }}>
        <VitalCard
          icon={<HeartIcon bpm={hr} color={hrColor} />}
          title="HEART RATE"
          value={hr}
          unit="BPM"
          color={hrColor}
          range={bpmMin > 0 ? [bpmMin, bpmMax] : null}
          subText={hr > 0 ? hrStatus : (isRunning ? "Calibrating…" : "—")}
        />
        <VitalCard
          icon={<LungIcon color={rrColor} />}
          title="RESP. RATE"
          value={rr}
          unit="BRPM"
          color={rrColor}
          range={rrRange[0] > 0 ? rrRange : null}
          subText={rr > 0 ? "breaths / min" : (isRunning ? rrStatus : "—")}
        />
      </div>
    </div>
  );
}