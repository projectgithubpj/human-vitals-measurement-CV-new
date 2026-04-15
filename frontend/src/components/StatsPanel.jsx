function StatsPanel({ vitals }) {
  const {
    hrConfidence = 0,
    rrConfidence = 0,
    signalQuality = 0,
    hrvConf = 0,
  } = vitals || {};







  return (
    <>






      {/* Signal Quality */}
      <div className="stats-section">
        <div className="stats-section-title">Signal Quality</div>

        <div className="signal-bar-row">
          <div className="signal-bar-header">
            <span>HR Signal</span>
            <span>{signalQuality}%</span>
          </div>
          <div className="signal-bar-track">
            <div className="signal-bar-fill" style={{ width: `${signalQuality}%` }} />
          </div>
        </div>

        <div className="signal-bar-row">
          <div className="signal-bar-header">
            <span>HR Confidence</span>
            <span>{hrConfidence}%</span>
          </div>
          <div className="signal-bar-track">
            <div className="signal-bar-fill" style={{ width: `${hrConfidence}%`, background: "#58a6ff" }} />
          </div>
        </div>

        <div className="signal-bar-row">
          <div className="signal-bar-header">
            <span>RR Confidence</span>
            <span>{rrConfidence}%</span>
          </div>
          <div className="signal-bar-track">
            <div className="signal-bar-fill" style={{ width: `${rrConfidence}%`, background: "#a78bfa" }} />
          </div>
        </div>

        <div className="signal-bar-row">
          <div className="signal-bar-header">
            <span>HRV Confidence</span>
            <span>{hrvConf}%</span>
          </div>
          <div className="signal-bar-track">
            <div className="signal-bar-fill" style={{ width: `${hrvConf}%`, background: "#fbbf24" }} />
          </div>
        </div>
      </div>
    </>
  );
}

export default StatsPanel;