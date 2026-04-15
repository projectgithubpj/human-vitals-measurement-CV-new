import { useEffect, useRef } from "react";

function WaveCanvas({ signal, color, label, height = 76 }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.offsetWidth;
    const H = height;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.height = `${H}px`;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    if (!signal || signal.length < 2) {
      ctx.beginPath();
      ctx.moveTo(0, H / 2);
      ctx.lineTo(W, H / 2);
      ctx.strokeStyle = color + "40";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
      return;
    }

    const arr = signal;
    const min = Math.min(...arr);
    const max = Math.max(...arr);
    const span = max - min || 1;
    const pad = 5;

    const x = (i) => (i / (arr.length - 1)) * W;
    const y = (v) => pad + ((1 - (v - min) / span) * (H - 2 * pad));

    /* gradient fill */
    const gradFill = ctx.createLinearGradient(0, 0, 0, H);
    gradFill.addColorStop(0, color + "22");
    gradFill.addColorStop(1, color + "00");

    ctx.beginPath();
    ctx.moveTo(x(0), y(arr[0]));
    for (let i = 1; i < arr.length; i++) {
      const x0 = x(i - 1), x1 = x(i);
      const y0 = y(arr[i - 1]), y1 = y(arr[i]);
      ctx.bezierCurveTo((x0 + x1) / 2, y0, (x0 + x1) / 2, y1, x1, y1);
    }
    ctx.lineTo(x(arr.length - 1), H);
    ctx.lineTo(x(0), H);
    ctx.closePath();
    ctx.fillStyle = gradFill;
    ctx.fill();

    /* line */
    ctx.beginPath();
    ctx.moveTo(x(0), y(arr[0]));
    for (let i = 1; i < arr.length; i++) {
      const x0 = x(i - 1), x1 = x(i);
      const y0 = y(arr[i - 1]), y1 = y(arr[i]);
      ctx.bezierCurveTo((x0 + x1) / 2, y0, (x0 + x1) / 2, y1, x1, y1);
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.lineJoin = "round";
    ctx.stroke();

    /* leading dot */
    const lx = x(arr.length - 1);
    const ly = y(arr[arr.length - 1]);
    ctx.beginPath();
    ctx.arc(lx, ly, 2.5, 0, Math.PI * 2);
    ctx.fillStyle = "#fff";
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.2;
    ctx.stroke();
  }, [signal, color, height]);

  return (
    <div style={{ position: "relative", width: "100%", flex: 1 }}>
      <canvas ref={canvasRef} style={{ width: "100%", display: "block" }} />
      <span style={{
        position: "absolute", top: 2, left: 5,
        fontSize: 7, fontWeight: 700, letterSpacing: "0.8px",
        color: color + "99",
        textTransform: "uppercase",
        pointerEvents: "none",
      }}>
        {label}
      </span>
    </div>
  );
}

/* RR tachogram — bar chart style, each beat = one bar */
function RRCanvas({ rrIntervals, height = 76 }) {
  const canvasRef = useRef(null);
  const COLOR = "#f59e0b";

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.offsetWidth;
    const H = height;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.height = `${H}px`;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    if (!rrIntervals || rrIntervals.length < 2) {
      ctx.beginPath();
      ctx.moveTo(0, H / 2);
      ctx.lineTo(W, H / 2);
      ctx.strokeStyle = COLOR + "40";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
      return;
    }

    const arr = rrIntervals.slice(-40);   /* last 40 beats */
    const min = Math.min(...arr);
    const max = Math.max(...arr);
    const span = max - min || 1;
    const pad = 5;

    const barW = W / arr.length;
    const gap = Math.max(1, barW * 0.2);

    arr.forEach((v, i) => {
      const bh = ((v - min) / span) * (H - pad * 2 - 4) + 4;
      const bx = i * barW + gap / 2;
      const by = H - pad - bh;
      const bw = barW - gap;

      ctx.fillStyle = COLOR + "33";
      ctx.beginPath();
      ctx.roundRect(bx, by, bw, bh, 1);
      ctx.fill();

      ctx.fillStyle = COLOR;
      ctx.beginPath();
      ctx.roundRect(bx, by, bw, 2, 1);
      ctx.fill();
    });

    /* mean line */
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    const my = H - pad - ((mean - min) / span) * (H - pad * 2 - 4) - 4;
    ctx.beginPath();
    ctx.moveTo(0, my);
    ctx.lineTo(W, my);
    ctx.strokeStyle = COLOR + "55";
    ctx.lineWidth = 0.8;
    ctx.setLineDash([3, 3]);
    ctx.stroke();
    ctx.setLineDash([]);
  }, [rrIntervals, height]);

  return (
    <div style={{ position: "relative", width: "100%", flex: 1 }}>
      <canvas ref={canvasRef} style={{ width: "100%", display: "block" }} />
      <span style={{
        position: "absolute", top: 2, left: 5,
        fontSize: 7, fontWeight: 700, letterSpacing: "0.8px",
        color: COLOR + "99",
        textTransform: "uppercase",
        pointerEvents: "none",
      }}>
        RR tachogram
      </span>
    </div>
  );
}

export default function Waveform({ pulseSignal, respSignal, rrIntervals }) {
  return (
    <div style={{
      display: "flex",
      height: "100%",
      padding: "4px 8px",
      gap: 0,
    }}>
      {/* Pulse */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
        <WaveCanvas
          signal={pulseSignal}
          color="#06B6D4"
          label="Pulse · rPPG CHROM"
          height={76}
        />
      </div>

      <div style={{ width: 1, background: "#30363d", margin: "4px 8px" }} />

      {/* Respiratory */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
        <WaveCanvas
          signal={respSignal}
          color="#8B5CF6"
          label="Respiratory · rPPG mod"
          height={76}
        />
      </div>

      <div style={{ width: 1, background: "#30363d", margin: "4px 8px" }} />

      {/* RR tachogram */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
        <RRCanvas rrIntervals={rrIntervals} height={76} />
      </div>
    </div>
  );
}