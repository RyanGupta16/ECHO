"use client";

import { useMemo, useRef, useState } from "react";

export type Series = { id: string; label: string; color: string; points: { x: number; y: number; lo?: number; hi?: number }[] };

type Props = {
  series: Series[];
  xLabel: string;
  yLabel: string;
  logY?: boolean;
  height?: number;
  format?: (y: number) => string;
  ariaLabel: string;
};

const W = 720;
const PAD = { l: 52, r: 120, t: 16, b: 40 };

function niceTicks(min: number, max: number, log: boolean) {
  if (log) {
    const out: number[] = [];
    for (const base of [1, 2, 5]) for (let e = -1; e < 6; e++) {
      const v = base * 10 ** e;
      if (v >= min * 0.98 && v <= max * 1.02) out.push(v);
    }
    return out.sort((a, b) => a - b);
  }
  const span = max - min || 1;
  const step = 10 ** Math.floor(Math.log10(span / 4));
  const mult = [1, 2, 5, 10].find((m) => span / (m * step) <= 5) ?? 10;
  const s = mult * step;
  const out = [];
  for (let v = Math.ceil(min / s) * s; v <= max + 1e-9; v += s) out.push(+v.toFixed(6));
  return out;
}

export function LineChart({ series, xLabel, yLabel, logY, height = 340, format = (y) => y.toFixed(1), ariaLabel }: Props) {
  const ref = useRef<SVGSVGElement>(null);
  const [hx, setHx] = useState<number | null>(null);
  const H = height;

  const { xs, x, y, yTicks, xMax } = useMemo(() => {
    const all = series.flatMap((s) => s.points);
    const xs = [...new Set(all.map((p) => p.x))].sort((a, b) => a - b);
    const xMin = xs[0], xMax = xs[xs.length - 1];
    const ys = all.flatMap((p) => [p.y, p.lo ?? p.y, p.hi ?? p.y]);
    let yMin = Math.min(...ys), yMax = Math.max(...ys);
    if (!logY) { const pad = (yMax - yMin) * 0.08; yMin -= pad; yMax += pad; } else { yMin *= 0.92; yMax *= 1.06; }
    const tf = (v: number) => (logY ? Math.log(v) : v);
    const x = (v: number) => PAD.l + ((v - xMin) / (xMax - xMin || 1)) * (W - PAD.l - PAD.r);
    const y = (v: number) => PAD.t + (1 - (tf(v) - tf(yMin)) / (tf(yMax) - tf(yMin))) * (H - PAD.t - PAD.b);
    return { xs, x, y, yTicks: niceTicks(yMin, yMax, !!logY), xMax };
  }, [series, logY, H]);

  const onMove = (e: React.PointerEvent) => {
    const r = ref.current!.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    let best = xs[0], d = Infinity;
    for (const v of xs) if (Math.abs(x(v) - px) < d) { d = Math.abs(x(v) - px); best = v; }
    setHx(best);
  };

  // Direct end labels, nudged apart so they never collide.
  const ends = series
    .map((s) => ({ s, p: s.points[s.points.length - 1] }))
    .map((e) => ({ ...e, ly: y(e.p.y) }))
    .sort((a, b) => a.ly - b.ly);
  for (let i = 1; i < ends.length; i++) if (ends[i].ly - ends[i - 1].ly < 16) ends[i].ly = ends[i - 1].ly + 16;

  return (
    <div className="relative">
      <div className="mb-3 flex flex-wrap gap-x-5 gap-y-2" aria-hidden>
        {series.map((s) => (
          <span key={s.id} className="flex items-center gap-2 text-xs text-ink-2">
            <span className="h-[2px] w-4 rounded" style={{ background: s.color }} />
            {s.label}
          </span>
        ))}
      </div>
      <svg
        ref={ref}
        viewBox={`0 0 ${W} ${H}`}
        className="w-full touch-none select-none"
        role="img"
        aria-label={ariaLabel}
        onPointerMove={onMove}
        onPointerLeave={() => setHx(null)}
      >
        {yTicks.map((t) => (
          <g key={t}>
            <line x1={PAD.l} x2={W - PAD.r} y1={y(t)} y2={y(t)} stroke="var(--line)" strokeWidth="1" />
            <text x={PAD.l - 10} y={y(t) + 4} textAnchor="end" fontSize="11" fill="var(--muted)" className="tabular">{format(t)}</text>
          </g>
        ))}
        {xs.map((v) => (
          <text key={v} x={x(v)} y={H - PAD.b + 20} textAnchor="middle" fontSize="11" fill="var(--muted)">{v}</text>
        ))}
        <text x={(PAD.l + W - PAD.r) / 2} y={H - 4} textAnchor="middle" fontSize="11" fill="var(--ink-2)">{xLabel}</text>
        <text x={12} y={PAD.t + (H - PAD.t - PAD.b) / 2} textAnchor="middle" fontSize="11" fill="var(--ink-2)" transform={`rotate(-90 12 ${PAD.t + (H - PAD.t - PAD.b) / 2})`}>{yLabel}</text>

        {series.map((s) => {
          const band = s.points.filter((p) => p.lo !== undefined);
          return (
            <g key={s.id}>
              {band.length > 1 && (
                <path
                  d={`M${band.map((p) => `${x(p.x)},${y(p.hi!)}`).join("L")}L${[...band].reverse().map((p) => `${x(p.x)},${y(p.lo!)}`).join("L")}Z`}
                  fill={s.color}
                  opacity={0.1}
                />
              )}
              <path d={`M${s.points.map((p) => `${x(p.x)},${y(p.y)}`).join("L")}`} fill="none" stroke={s.color} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
            </g>
          );
        })}
        {ends.map(({ s, p, ly }) => (
          <g key={s.id}>
            <circle cx={x(p.x)} cy={y(p.y)} r="4" fill={s.color} stroke="var(--panel-solid)" strokeWidth="2" />
            <text x={x(xMax) + 10} y={ly + 4} fontSize="11" fill="var(--ink-2)">{s.label} <tspan className="tabular" fill="var(--ink)">{format(p.y)}</tspan></text>
          </g>
        ))}

        {hx !== null && (
          <g pointerEvents="none">
            <line x1={x(hx)} x2={x(hx)} y1={PAD.t} y2={H - PAD.b} stroke="var(--line-2)" strokeWidth="1" />
            {series.map((s) => {
              const p = s.points.find((q) => q.x === hx);
              return p ? <circle key={s.id} cx={x(p.x)} cy={y(p.y)} r="4.5" fill={s.color} stroke="var(--panel-solid)" strokeWidth="2" /> : null;
            })}
          </g>
        )}
      </svg>
      {hx !== null && (
        <div
          className="pointer-events-none absolute top-8 rounded-xl border border-line-2 bg-[var(--panel-solid)] px-3 py-2 text-xs shadow-2xl"
          style={{ left: `${(x(hx) / W) * 100}%`, transform: x(hx) > W * 0.6 ? "translateX(calc(-100% - 12px))" : "translateX(12px)" }}
        >
          <p className="mb-1 font-mono text-[10px] uppercase tracking-[0.16em] text-muted">{xLabel} {hx}</p>
          {series.map((s) => {
            const p = s.points.find((q) => q.x === hx);
            return p ? (
              <p key={s.id} className="flex items-center justify-between gap-5">
                <span className="flex items-center gap-1.5 text-ink-2"><span className="h-2 w-2 rounded-full" style={{ background: s.color }} />{s.label}</span>
                <span className="tabular text-ink">{format(p.y)}</span>
              </p>
            ) : null;
          })}
        </div>
      )}
    </div>
  );
}
