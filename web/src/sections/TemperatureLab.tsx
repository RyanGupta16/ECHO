"use client";

import { useMemo, useState } from "react";

import { COLOR, MIND_BY_KIND } from "@/lib/content";
import type { EchoData, TQRow } from "@/lib/data";
import { KINDS, type Kind } from "@/lib/engine/protocol";

const METRICS = [
  { key: "gpt2_ppl", label: "Coherence", sub: "GPT-2 perplexity · lower = more natural", lowBetter: true, log: true, fmt: (v: number) => v.toFixed(0) },
  { key: "distinct_2", label: "Diversity", sub: "distinct-2 · higher = more varied", lowBetter: false, log: false, fmt: (v: number) => v.toFixed(2) },
  { key: "repetition", label: "Repetition", sub: "repeated 3-grams · lower = fewer loops", lowBetter: true, log: false, fmt: (v: number) => `${(v * 100).toFixed(0)}%` },
  { key: "consistency", label: "On-topic", sub: "seed↔text similarity · higher = stays on topic", lowBetter: false, log: false, fmt: (v: number) => v.toFixed(2) },
] as const;

const tLabel = (t: number) => (t === 0 ? "greedy" : t.toString());

function Frontier({ rows, human }: { rows: TQRow[]; human: TQRow }) {
  const [hov, setHov] = useState<TQRow | null>(null);
  const W = 720, H = 380, P = { l: 56, r: 24, t: 20, b: 46 };
  const xs = [...rows.map((r) => r.distinct_2), human.distinct_2];
  const ys = [...rows.map((r) => r.gpt2_ppl), human.gpt2_ppl];
  const x0 = Math.min(...xs) - 0.03, x1 = Math.max(...xs) + 0.03;
  const y0 = Math.log(Math.min(...ys) * 0.8), y1 = Math.log(Math.max(...ys) * 1.2);
  const X = (v: number) => P.l + ((v - x0) / (x1 - x0)) * (W - P.l - P.r);
  const Y = (v: number) => P.t + (1 - (Math.log(v) - y0) / (y1 - y0)) * (H - P.t - P.b);
  const yTicks = [5, 10, 20, 50, 100, 200, 500, 1000, 2000].filter((v) => Math.log(v) > y0 && Math.log(v) < y1);
  const xTicks = Array.from({ length: 6 }, (_, i) => x0 + ((x1 - x0) * (i + 0.5)) / 6);

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Diversity versus coherence for each model across temperatures">
        {yTicks.map((t) => (
          <g key={t}>
            <line x1={P.l} x2={W - P.r} y1={Y(t)} y2={Y(t)} stroke="var(--line)" />
            <text x={P.l - 10} y={Y(t) + 4} textAnchor="end" fontSize="11" fill="var(--muted)">{t}</text>
          </g>
        ))}
        {xTicks.map((t) => (
          <text key={t} x={X(t)} y={H - P.b + 20} textAnchor="middle" fontSize="11" fill="var(--muted)">{t.toFixed(2)}</text>
        ))}
        <text x={(P.l + W - P.r) / 2} y={H - 6} textAnchor="middle" fontSize="11" fill="var(--ink-2)">diversity (distinct-2) →</text>
        <text x={14} y={H / 2} textAnchor="middle" fontSize="11" fill="var(--ink-2)" transform={`rotate(-90 14 ${H / 2})`}>← more natural · GPT-2 perplexity (log)</text>
        {KINDS.map((k) => {
          const pts = rows.filter((r) => r.kind === k).sort((a, b) => (a.temperature ?? 0) - (b.temperature ?? 0));
          return (
            <g key={k}>
              <path d={`M${pts.map((p) => `${X(p.distinct_2)},${Y(p.gpt2_ppl)}`).join("L")}`} fill="none" stroke={COLOR[k]} strokeWidth="2" strokeLinejoin="round" />
              {pts.map((p) => (
                <circle
                  key={p.temperature}
                  cx={X(p.distinct_2)}
                  cy={Y(p.gpt2_ppl)}
                  r={hov === p ? 7 : 5}
                  fill={COLOR[k]}
                  stroke="var(--panel-solid)"
                  strokeWidth="2"
                  onPointerEnter={() => setHov(p)}
                  onPointerLeave={() => setHov(null)}
                  className="cursor-pointer"
                />
              ))}
            </g>
          );
        })}
        <path
          d={`M${X(human.distinct_2)},${Y(human.gpt2_ppl) - 11} l3.2,7 7.6,.8 -5.7,5.2 1.6,7.5 -6.7,-3.8 -6.7,3.8 1.6,-7.5 -5.7,-5.2 7.6,-.8z`}
          fill="var(--ink)"
          stroke="var(--panel-solid)"
          strokeWidth="2"
        />
        <text x={X(human.distinct_2) + 14} y={Y(human.gpt2_ppl) + 4} fontSize="12" fill="var(--ink)">human-written</text>
      </svg>
      {hov && (
        <div
          className="pointer-events-none absolute rounded-xl border border-line-2 bg-[var(--panel-solid)] px-3 py-2 text-xs shadow-2xl"
          style={{ left: `${(X(hov.distinct_2) / W) * 100}%`, top: `${(Y(hov.gpt2_ppl) / H) * 100}%`, transform: "translate(12px,-50%)" }}
        >
          <p className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full" style={{ background: COLOR[hov.kind as Kind] }} />{hov.model} · T {tLabel(hov.temperature ?? 0)}</p>
          <p className="tabular mt-1 text-ink-2">GPT-2 ppl {hov.gpt2_ppl.toFixed(1)} · distinct-2 {hov.distinct_2.toFixed(2)}</p>
        </div>
      )}
      <div className="mt-2 flex flex-wrap gap-x-5 gap-y-2">
        {KINDS.map((k) => (
          <span key={k} className="flex items-center gap-2 text-xs text-ink-2">
            <span className="h-[2px] w-4" style={{ background: COLOR[k] }} />{MIND_BY_KIND[k].name}
          </span>
        ))}
        <span className="text-xs text-muted">· each line runs greedy (left) → T 1.3 (right)</span>
      </div>
    </div>
  );
}

export function TemperatureLab({ data }: { data: EchoData }) {
  const temps = data.config.temperatures;
  const [seedId, setSeedId] = useState(0);
  const [temp, setTemp] = useState(0.8);
  const [variant, setVariant] = useState(0);
  const [metric, setMetric] = useState<(typeof METRICS)[number]["key"]>("gpt2_ppl");
  const [cell, setCell] = useState<{ k: Kind; t: number } | null>(null);

  const human = data.textQuality.find((r) => r.kind === "human")!;
  const rows = data.textQuality.filter((r) => r.kind !== "human");
  const m = METRICS.find((x) => x.key === metric)!;

  const grid = useMemo(() => {
    const vals = rows.map((r) => r[metric] as number);
    const f = (v: number) => (m.log ? Math.log(v) : v);
    const lo = Math.min(...vals.map(f)), hi = Math.max(...vals.map(f));
    return (v: number) => {
      const n = (f(v) - lo) / (hi - lo || 1);
      return m.lowBetter ? 1 - n : n; // 1 = best
    };
  }, [rows, metric, m]);

  const sample = (k: Kind) => data.samples.find((s) => s.kind === k && s.seed_id === seedId && s.temperature === temp && s.sample === variant);

  return (
    <section id="temperature" aria-labelledby="temp-title" className="section">
      <div className="container-x">
        <p className="eyebrow">Chapter 7 · The creativity dial</p>
        <h2 id="temp-title" className="display mt-4 text-[clamp(2.6rem,5.5vw,4.6rem)]">
          Turn up the heat, <em className="text-violet">watch sense melt.</em>
        </h2>
        <p className="lede mt-5">
          Temperature reshapes the model&apos;s probabilities before a word is drawn. Below are {data.samples.length.toLocaleString("en-US")}
          {" "}pre-generated samples — five seeds, five temperatures, five draws, four models — each scored by an independent judge.
        </p>

        {/* Sample browser */}
        <div className="panel mt-10 p-5 sm:p-6">
          <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
            <div className="flex flex-wrap items-center gap-2" role="group" aria-label="Seed">
              <span className="mr-1 text-xs text-muted">Seed</span>
              {data.seeds.map((s) => (
                <button key={s.id} onClick={() => setSeedId(s.id)} aria-pressed={seedId === s.id}
                  className={`h-9 min-w-9 rounded-full border px-3 text-xs ${seedId === s.id ? "border-cyan/60 text-ink" : "border-line text-muted hover:text-ink"}`}>
                  {s.id + 1}
                </button>
              ))}
            </div>
            <div className="flex flex-wrap items-center gap-2" role="group" aria-label="Temperature">
              <span className="mr-1 text-xs text-muted">Temperature</span>
              {temps.map((t) => (
                <button key={t} onClick={() => setTemp(t)} aria-pressed={temp === t}
                  className={`h-9 rounded-full border px-3 font-mono text-xs ${temp === t ? "border-violet/60 text-ink" : "border-line text-muted hover:text-ink"}`}>
                  {tLabel(t)}
                </button>
              ))}
            </div>
            <div className="flex flex-wrap items-center gap-2" role="group" aria-label="Draw">
              <span className="mr-1 text-xs text-muted">Draw</span>
              {[0, 1, 2, 3, 4].map((v) => (
                <button key={v} onClick={() => setVariant(v)} aria-pressed={variant === v} disabled={temp === 0}
                  className={`h-9 w-9 rounded-full border text-xs disabled:opacity-30 ${variant === v ? "border-ink-2 text-ink" : "border-line text-muted"}`}>
                  {v + 1}
                </button>
              ))}
            </div>
          </div>
          <p className="mt-5 font-serif text-xl text-ink-2">&ldquo;{data.seeds[seedId]?.text}&rdquo; …</p>
          <div className="mt-5 grid gap-3 md:grid-cols-2">
            {KINDS.map((k) => {
              const s = sample(k);
              return (
                <article key={k} className="rounded-2xl border border-line bg-bg/50 p-4">
                  <header className="flex items-center justify-between">
                    <span className="flex items-center gap-2 text-sm"><span className="h-2 w-2 rounded-full" style={{ background: COLOR[k] }} />{MIND_BY_KIND[k].name}</span>
                    {s && (
                      <span className="tabular font-mono text-[11px] text-muted">
                        judge ppl {s.gpt2_ppl.toFixed(0)} · rep {(s.repetition * 100).toFixed(0)}%
                      </span>
                    )}
                  </header>
                  <p key={`${seedId}-${temp}-${variant}`} className="word-in mt-3 font-serif text-[1.05rem] leading-relaxed">{s?.text ?? "—"}</p>
                </article>
              );
            })}
          </div>
        </div>

        {/* Heatmap + frontier */}
        <div className="mt-4 grid gap-4 xl:grid-cols-[1fr_1.25fr]">
          <figure className="panel p-5 sm:p-6">
            <figcaption>
              <div className="flex flex-wrap gap-2" role="group" aria-label="Metric">
                {METRICS.map((x) => (
                  <button key={x.key} onClick={() => setMetric(x.key)} aria-pressed={metric === x.key}
                    className={`h-9 rounded-full px-3.5 text-xs ${metric === x.key ? "bg-white/10 text-ink" : "text-muted hover:text-ink"}`}>
                    {x.label}
                  </button>
                ))}
              </div>
              <p className="mt-3 text-sm text-muted">{m.sub} · brighter = better · human text: <span className="text-ink">{m.fmt(human[metric] as number)}</span></p>
            </figcaption>
            <div className="mt-5 overflow-x-auto">
              <table className="w-full min-w-[26rem] border-separate border-spacing-1 text-xs">
                <thead>
                  <tr>
                    <th className="w-28" />
                    {temps.map((t) => <th key={t} scope="col" className="pb-1 font-mono font-normal text-muted">{tLabel(t)}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {KINDS.map((k) => (
                    <tr key={k}>
                      <th scope="row" className="pr-2 text-left font-normal text-ink-2">
                        <span className="flex items-center gap-2"><span className="h-2 w-2 rounded-full" style={{ background: COLOR[k] }} />{MIND_BY_KIND[k].short}</span>
                      </th>
                      {temps.map((t) => {
                        const r = rows.find((x) => x.kind === k && x.temperature === t)!;
                        const v = r[metric] as number;
                        const g = grid(v);
                        return (
                          <td key={t}
                            onPointerEnter={() => setCell({ k, t })} onPointerLeave={() => setCell(null)}
                            onClick={() => { setTemp(t); }}
                            className={`tabular h-12 cursor-pointer rounded-lg text-center font-mono transition-transform hover:scale-[1.04] ${cell?.k === k && cell.t === t ? "outline outline-1 outline-white/40" : ""}`}
                            style={{ background: `color-mix(in oklab, #3987e5 ${Math.round(12 + g * 78)}%, #0b0e16)`, color: g > 0.62 ? "#05060a" : "var(--ink)" }}
                            title={`${MIND_BY_KIND[k].name} at T ${tLabel(t)}: ${m.fmt(v)}`}
                          >
                            {m.fmt(v)}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-3 text-xs text-muted">Click a cell to load that temperature in the sample browser above.</p>
          </figure>
          <figure className="panel p-5 sm:p-6">
            <figcaption className="mb-4">
              <span className="font-medium">The creativity–coherence trade-off</span>
              <p className="mt-1 text-sm text-muted">Closer to the ★ means text that is both as varied and as natural as a human&apos;s.</p>
            </figcaption>
            <Frontier rows={rows} human={human} />
          </figure>
        </div>
      </div>
    </section>
  );
}
