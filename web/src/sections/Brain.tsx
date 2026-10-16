"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { toDisplay } from "@/lib/display";
import { useEngine } from "@/lib/engine/client";
import type { ProbeResult } from "@/lib/engine/protocol";
import { EOS, tokenize } from "@/lib/tokenizer";

const PRESETS = [
  "Lily and her dog Max went to the park. Lily threw the ball and Max",
  "Tom was very sad because it was raining. Then the sun came out and Tom felt",
  "The little bird could not fly. Mom bird said, \"Do not worry, you will",
];

const GATES = [
  { idx: 1, name: "Forget gate", hue: "155 123 255", hint: "1 = keep the old memory · 0 = erase it" },
  { idx: 0, name: "Input gate", hue: "94 231 255", hint: "1 = write this word into memory · 0 = ignore it" },
  { idx: 3, name: "Output gate", hue: "255 143 177", hint: "1 = reveal memory to the prediction · 0 = keep it hidden" },
];
const UNITS = 14;

export function Brain() {
  const engine = useEngine();
  const root = useRef<HTMLElement>(null);
  const [text, setText] = useState(PRESETS[0]);
  const [layer, setLayer] = useState(1);
  const [probe, setProbe] = useState<{ r: ProbeResult; tokens: string[] } | null>(null);
  const [sel, setSel] = useState<number | null>(null);
  const [visible, setVisible] = useState(false);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    const io = new IntersectionObserver(([e]) => e.isIntersecting && setVisible(true), { rootMargin: "500px" });
    if (root.current) io.observe(root.current);
    return () => io.disconnect();
  }, []);

  const run = useCallback(async () => {
    if (!engine.vocab) return;
    setBusy(true);
    const toks = tokenize(text).slice(-23);
    const ids = engine.encodeSeed(toks.join(" ")).slice(-24);
    const r = await engine.probe(ids);
    setProbe({ r, tokens: [EOS, ...toks].slice(-ids.length) });
    setSel(null);
    setBusy(false);
  }, [engine, text]);

  useEffect(() => {
    if (visible && engine.vocab && !probe) {
      engine.load(["lstm_probe"]).then(run).catch(() => {});
    }
  }, [visible, engine, engine.vocab, probe, run]);

  // Pick the units whose gate value changes the most over the sentence: the "interesting" neurons.
  const view = useMemo(() => {
    if (!probe) return null;
    const { r } = probe;
    const { T, H } = r;
    const at = (g: number, t: number, h: number) => r.gates[((layer * T + t) * 4 + g) * H + h];
    return GATES.map((g) => {
      const scores = Array.from({ length: H }, (_, h) => {
        let m = 0, v = 0;
        for (let t = 0; t < T; t++) m += at(g.idx, t, h);
        m /= T;
        for (let t = 0; t < T; t++) v += (at(g.idx, t, h) - m) ** 2;
        return { h, v };
      });
      const units = scores.sort((a, b) => b.v - a.v).slice(0, UNITS).map((s) => s.h);
      const mean = Array.from({ length: T }, (_, t) => {
        let s = 0;
        for (let h = 0; h < H; h++) s += at(g.idx, t, h);
        return s / H;
      });
      return { ...g, cells: units.map((h) => Array.from({ length: T }, (_, t) => at(g.idx, t, h))), units, mean };
    });
  }, [probe, layer]);

  const T = probe?.r.T ?? 0;
  const focus = sel ?? T - 1;
  const disp = probe ? toDisplay(probe.tokens, engine.proper, null) : [];
  const word = (id: number) => engine.vocab?.itos[id] ?? "";
  const progress = engine.progress["lstm_probe"];

  return (
    <section ref={root} id="brain" aria-labelledby="brain-title" className="section">
      <div className="container-x">
        <p className="eyebrow">Chapter 4 · Inside the brain</p>
        <h2 id="brain-title" className="display mt-4 text-[clamp(2.6rem,5.5vw,4.6rem)]">
          Watch an LSTM <em className="text-violet">decide what to remember.</em>
        </h2>
        <p className="lede mt-5">
          This is the trained LSTM, running live. Each column is a word it reads; each row is one of its memory neurons.
          Bright means a gate is open. Click any word to see what the network expected to come next at that moment.
        </p>

        <div className="panel mt-10 p-5 sm:p-6">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-end">
            <div className="flex-1">
              <label htmlFor="brain-input" className="text-sm text-ink-2">Sentence to read</label>
              <input
                id="brain-input"
                value={text}
                onChange={(e) => setText(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && run()}
                className="mt-2 h-12 w-full rounded-xl border border-line-2 bg-bg/70 px-4 font-mono text-base outline-none focus:border-violet sm:text-sm"
                maxLength={200}
              />
            </div>
            <div className="flex gap-2">
              <div className="flex rounded-full border border-line p-1" role="group" aria-label="LSTM layer">
                {[0, 1].map((l) => (
                  <button
                    key={l}
                    onClick={() => setLayer(l)}
                    aria-pressed={layer === l}
                    className={`h-10 rounded-full px-4 text-xs ${layer === l ? "bg-white/10 text-ink" : "text-muted"}`}
                  >
                    Layer {l + 1}
                  </button>
                ))}
              </div>
              <button
                onClick={run}
                disabled={busy || !engine.ready.has("lstm_probe")}
                className="h-12 rounded-full bg-ink px-6 text-sm font-medium text-bg disabled:opacity-50"
              >
                {engine.ready.has("lstm_probe") ? (busy ? "Reading…" : "Read it") : "Loading LSTM…"}
              </button>
            </div>
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            {PRESETS.map((p) => (
              <button
                key={p}
                onClick={() => setText(p)}
                className={`min-h-9 rounded-full border px-3 text-xs ${text === p ? "border-violet/60 text-ink" : "border-line text-muted hover:text-ink"}`}
              >
                {p.slice(0, 34)}…
              </button>
            ))}
          </div>
          {!engine.ready.has("lstm_probe") && progress && (
            <div className="mt-4 h-1 overflow-hidden rounded-full bg-white/[0.05]">
              <div className="h-full bg-violet transition-[width]" style={{ width: `${(progress.loaded / progress.total) * 100}%` }} />
            </div>
          )}

          {view && probe && (
            <div className="mt-8 grid gap-8 xl:grid-cols-[1fr_17rem]">
              <div className="overflow-x-auto">
                <div className="min-w-[40rem]">
                  {/* token header */}
                  <div className="grid gap-[3px] pl-28" style={{ gridTemplateColumns: `repeat(${T}, minmax(0,1fr))` }}>
                    {disp.map((d, t) => (
                      <button
                        key={t}
                        onClick={() => setSel(t)}
                        className={`h-24 rounded-md text-[11px] transition-colors [writing-mode:vertical-rl] rotate-180 font-mono ${
                          focus === t ? "bg-white/10 text-ink" : "text-ink-2 hover:bg-white/[0.04]"
                        }`}
                        aria-label={`Inspect word ${t + 1}: ${d.text}`}
                        aria-pressed={focus === t}
                      >
                        {d.text}
                      </button>
                    ))}
                  </div>
                  {view.map((g) => (
                    <div key={g.name} className="mt-5">
                      <div className="flex items-baseline justify-between pl-28">
                        <p className="text-sm">{g.name}</p>
                        <p className="hidden font-mono text-[10px] text-muted sm:block">{g.hint}</p>
                      </div>
                      <div className="mt-2 flex">
                        <div className="w-28 shrink-0 pr-3 pt-0.5 text-right font-mono text-[10px] leading-[14px] text-muted">
                          {g.units.slice(0, 3).map((u) => <p key={u}>unit {u}</p>)}
                          <p>…</p>
                        </div>
                        <div className="grid flex-1 gap-[3px]" style={{ gridTemplateColumns: `repeat(${T}, minmax(0,1fr))` }}>
                          {Array.from({ length: T }, (_, t) => (
                            <div key={t} className={`grid gap-[3px] rounded ${focus === t ? "outline outline-1 outline-white/30" : ""}`}>
                              {g.cells.map((row, u) => (
                                <span
                                  key={u}
                                  className="block h-[11px] rounded-[2px]"
                                  style={{ background: `rgb(${g.hue} / ${0.04 + 0.96 * row[t]})` }}
                                  title={`${g.name}, unit ${g.units[u]}, word "${disp[t].text}": ${row[t].toFixed(2)}`}
                                />
                              ))}
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  ))}
                  <p className="mt-4 pl-28 font-mono text-[10px] text-muted">
                    Showing the {UNITS} of 256 neurons per gate whose activity changes most across this sentence.
                  </p>
                </div>
              </div>

              <aside className="rounded-2xl border border-line bg-bg/60 p-5" aria-live="polite">
                <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted">After reading</p>
                <p className="mt-1 font-serif text-2xl">&ldquo;{disp[focus]?.text}&rdquo;</p>
                <p className="mt-5 font-mono text-[10px] uppercase tracking-[0.18em] text-muted">It expects next</p>
                <ul className="mt-3 space-y-2">
                  {probe.r.predictions[focus].map((p, i) => (
                    <li key={p.id} className="grid grid-cols-[1fr_auto] items-center gap-2 text-sm">
                      <span className="relative overflow-hidden rounded-md px-2 py-1 font-mono">
                        <span className="absolute inset-y-0 left-0 rounded-md" style={{ width: `${p.p * 100}%`, background: i === 0 ? "rgb(155 123 255 / .35)" : "rgb(255 255 255 / .06)" }} />
                        <span className="relative">{word(p.id)}</span>
                      </span>
                      <span className="tabular font-mono text-xs text-muted">{(p.p * 100).toFixed(1)}%</span>
                    </li>
                  ))}
                </ul>
                <div className="mt-6 space-y-3 border-t border-line pt-4">
                  {view.map((g) => (
                    <div key={g.name}>
                      <div className="flex justify-between text-xs"><span className="text-ink-2">{g.name}</span><span className="tabular font-mono text-muted">{g.mean[focus].toFixed(2)}</span></div>
                      <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-white/[0.05]">
                        <div className="h-full rounded-full transition-[width] duration-300" style={{ width: `${g.mean[focus] * 100}%`, background: `rgb(${g.hue})` }} />
                      </div>
                    </div>
                  ))}
                  <p className="pt-1 text-[11px] text-muted">Average openness across all 256 neurons in layer {layer + 1}.</p>
                </div>
              </aside>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
