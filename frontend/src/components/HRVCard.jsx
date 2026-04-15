/**
 * HRVCard — compact HRV display for the right panel
 * Shows: SDNN, RMSSD, pNN50, SD1/SD2, LF/HF ratio, ANS interpretation
 */
export default function HRVCard({
  sdnn = 0, rmssd = 0, pnn50 = 0,
  sd1 = 0, sd2 = 0,
  lfHf = 0, lfNu = 0, hfNu = 0,
  wtLfHf = 0, hrvConf = 0,
  hrvStatusMsg = "", interpretation = null,
  isRunning = false,
}) {
  const hasData = sdnn > 0 || rmssd > 0;

  /* Badge color for HRV status */
  const statusColor = () => {
    const s = interpretation?.hrv_status;
    if (!s || s === "—") return { bg: "#30363d", text: "#8b949e" };
    if (s === "High") return { bg: "#14532d", text: "#4ade80" };
    if (s === "Normal") return { bg: "#14532d", text: "#3fb950" };
    if (s === "Low") return { bg: "#451a03", text: "#fb923c" };
    return { bg: "#450a0a", text: "#f87171" };
  };
  const sc = statusColor();

  const getSdnnColor = (v) => {
    if (!v || v === 0) return "#8b949e";
    if (v < 20) return "#f85149";
    if (v < 50) return "#e3b341";
    if (v < 100) return "#3fb950";
    return "#79c0ff";
  };
  const getRmssdColor = (v) => {
    if (!v || v === 0) return "#8b949e";
    if (v < 20) return "#f85149";
    if (v < 40) return "#e3b341";
    if (v < 70) return "#3fb950";
    return "#79c0ff";
  };
  const getPnn50Color = (v) => {
    if (!v || v === 0) return "#8b949e";
    if (v < 3) return "#f85149";
    if (v < 10) return "#e3b341";
    if (v < 25) return "#3fb950";
    return "#79c0ff";
  };

  return (
    <div className="hrv-card">
      {/* Title row */}
      <div className="hrv-card-title">
        <span>HRV</span>
        <span
          className="hrv-badge"
          style={{ background: sc.bg, color: sc.text }}
        >
          {interpretation?.hrv_emoji ?? "⏳"} {interpretation?.hrv_status ?? (isRunning ? "…" : "—")}
        </span>
      </div>

      {!hasData ? (
        <div style={{ fontSize: 10, color: "#8b949e", textAlign: "center", paddingTop: 6 }}>
          {isRunning ? (hrvStatusMsg || "Collecting beats…") : "—"}
        </div>
      ) : (
        <>
          {/* 2×2 metric grid */}
          <div className="hrv-metric-grid">
            <div className="hrv-metric hrv-metric-hoverable">
              <span className="hrv-metric-val" style={{ color: getSdnnColor(sdnn) }}>
                {sdnn > 0 ? Math.round(sdnn) : "–"}
              </span>
              <span className="hrv-metric-lbl">SDNN ms</span>
              <div className="custom-tooltip">
                <div className="zone-row" style={{ marginBottom: 4 }}>
                  <div className="zone-dot" style={{ background: "#f85149" }} />
                  <span className="zone-name" style={{ fontSize: 10 }}>Very Low</span>
                  <span className="zone-range" style={{ fontSize: 10 }}>&lt; 20</span>
                </div>
                <div className="zone-row" style={{ marginBottom: 4 }}>
                  <div className="zone-dot" style={{ background: "#e3b341" }} />
                  <span className="zone-name" style={{ fontSize: 10 }}>Low</span>
                  <span className="zone-range" style={{ fontSize: 10 }}>20–50</span>
                </div>
                <div className="zone-row" style={{ marginBottom: 4 }}>
                  <div className="zone-dot" style={{ background: "#3fb950" }} />
                  <span className="zone-name" style={{ fontSize: 10 }}>Normal</span>
                  <span className="zone-range" style={{ fontSize: 10 }}>50–100</span>
                </div>
                <div className="zone-row" style={{ marginBottom: 0 }}>
                  <div className="zone-dot" style={{ background: "#79c0ff" }} />
                  <span className="zone-name" style={{ fontSize: 10 }}>Very Good</span>
                  <span className="zone-range" style={{ fontSize: 10 }}>&gt; 100</span>
                </div>
              </div>
            </div>
            <div className="hrv-metric hrv-metric-hoverable">
              <span className="hrv-metric-val" style={{ color: getRmssdColor(rmssd) }}>
                {rmssd > 0 ? Math.round(rmssd) : "–"}
              </span>
              <span className="hrv-metric-lbl">RMSSD ms</span>
              <div className="custom-tooltip">
                <div className="zone-row" style={{ marginBottom: 4 }}>
                  <div className="zone-dot" style={{ background: "#f85149" }} />
                  <span className="zone-name" style={{ fontSize: 10 }}>Very Low</span>
                  <span className="zone-range" style={{ fontSize: 10 }}>&lt; 20</span>
                </div>
                <div className="zone-row" style={{ marginBottom: 4 }}>
                  <div className="zone-dot" style={{ background: "#e3b341" }} />
                  <span className="zone-name" style={{ fontSize: 10 }}>Low</span>
                  <span className="zone-range" style={{ fontSize: 10 }}>20–40</span>
                </div>
                <div className="zone-row" style={{ marginBottom: 4 }}>
                  <div className="zone-dot" style={{ background: "#3fb950" }} />
                  <span className="zone-name" style={{ fontSize: 10 }}>Normal/Healthy</span>
                  <span className="zone-range" style={{ fontSize: 10 }}>40–70</span>
                </div>
                <div className="zone-row" style={{ marginBottom: 0 }}>
                  <div className="zone-dot" style={{ background: "#79c0ff" }} />
                  <span className="zone-name" style={{ fontSize: 10 }}>Very Good</span>
                  <span className="zone-range" style={{ fontSize: 10 }}>&gt; 70</span>
                </div>
              </div>
            </div>
            <div className="hrv-metric hrv-metric-hoverable">
              <span className="hrv-metric-val" style={{ color: getPnn50Color(pnn50) }}>
                {pnn50 > 0 ? pnn50.toFixed(1) : "–"}
              </span>
              <span className="hrv-metric-lbl">pNN50 %</span>
              <div className="custom-tooltip">
                <div className="zone-row" style={{ marginBottom: 4 }}>
                  <div className="zone-dot" style={{ background: "#f85149" }} />
                  <span className="zone-name" style={{ fontSize: 10 }}>Very Low</span>
                  <span className="zone-range" style={{ fontSize: 10 }}>&lt; 3</span>
                </div>
                <div className="zone-row" style={{ marginBottom: 4 }}>
                  <div className="zone-dot" style={{ background: "#e3b341" }} />
                  <span className="zone-name" style={{ fontSize: 10 }}>Low</span>
                  <span className="zone-range" style={{ fontSize: 10 }}>3–10</span>
                </div>
                <div className="zone-row" style={{ marginBottom: 4 }}>
                  <div className="zone-dot" style={{ background: "#3fb950" }} />
                  <span className="zone-name" style={{ fontSize: 10 }}>Normal/Healthy</span>
                  <span className="zone-range" style={{ fontSize: 10 }}>10–25</span>
                </div>
                <div className="zone-row" style={{ marginBottom: 0 }}>
                  <div className="zone-dot" style={{ background: "#79c0ff" }} />
                  <span className="zone-name" style={{ fontSize: 10 }}>Very Good</span>
                  <span className="zone-range" style={{ fontSize: 10 }}>&gt; 25</span>
                </div>
              </div>
            </div>
          </div>

          <div className="hrv-divider" />

          {/* SD1 / SD2 row */}
          <div style={{ display: "flex", gap: 4 }}>
            <div className="hrv-metric" style={{ flex: 1 }}>
              <span className="hrv-metric-val" style={{ fontSize: 13, color: "#e6edf3" }}>
                {sd1 > 0 ? Math.round(sd1) : "–"}
              </span>
              <span className="hrv-metric-lbl">SD1 ms</span>
            </div>
            <div className="hrv-metric" style={{ flex: 1 }}>
              <span className="hrv-metric-val" style={{ fontSize: 13, color: "#e6edf3" }}>
                {sd2 > 0 ? Math.round(sd2) : "–"}
              </span>
              <span className="hrv-metric-lbl">SD2 ms</span>
            </div>
          </div>



          {/* ANS interpretation */}
          <div className="hrv-ans-row">
            <span className="hrv-ans-val">ANS: </span>
            {interpretation?.ans_state ?? "—"}
          </div>
          <div className="hrv-ans-row">
            <span className="hrv-ans-val">Vagal: </span>
            {interpretation?.vagal_tone ?? "—"}
          </div>

          {/* Confidence */}
          {hrvConf > 0 && (
            <div style={{ marginTop: 2 }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: "#8b949e", marginBottom: 2 }}>
                <span>HRV confidence</span>
                <span>{hrvConf}%</span>
              </div>
              <div className="signal-bar-track">
                <div className="signal-bar-fill" style={{ width: `${hrvConf}%`, background: "#58a6ff" }} />
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}