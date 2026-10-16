"use client";

import { useMemo, useState } from "react";

import { LineChart, type Series } from "@/components/charts/LineChart";
import { COLOR, MIND_BY_KIND } from "@/lib/content";
import type { EchoData } from "@/lib/data";
import { KINDS, type Kind } from "@/lib/engine/protocol";

type Col = { key: string; label: string; fmt: (v: number) => string; better: "low" | "high" | null; hint: string };

const COLS: Col[] = [
  { key: "test_ppl", label: "Perplexity", fmt: (v) => v.toFixed(2), better: "low", hint: "How many words it is torn between. Lower is better." },
  { key: "test_top1", label: "Top-1 acc.", fmt: (v) => `${(v * 100).toFixed(1)}%`, better: "high", hint: "First guess is exactly right." },
  { key: "test_top5", label: "Top-5 acc.", fmt: (v) => `${(v * 100).toFixed(1)}%`, better: "high", hint: "Right word is in the top five." },
  { key: "time_per_epoch_s", label: "Sec / epoch", fmt: (v) => v.toFixed(0), better: "low", hint: "Training cost on an Apple M4 GPU." },
  { key: "ms_per_token", label: "ms / word", fmt: (v) => v.toFixed(2), better: "low", hint: "Generation latency per word." },
];

export function Results({ data }: { data: EchoData }) {
  const [metric, setMetric] = useState<"val_perplexity" | "train_perplexity">("val_perplexity");
  const [sort, setSort] = useState<{ key: string; dir: 1 | -1 }>({ key: "test_ppl", dir: 1 });

  const rows = useMemo(() => {
    type Row = Record<string, number> & { kind: Kind };
    const r = KINDS.map((k): Row => {
      const s = data.models[k].summary as Record<string, number>;
      const vals: Record<string, number> = { params: s.params_total };
      for (const c of COLS) {
        vals[c.key] = s[`${c.key}_mean`];
        vals[`${c.key}_std`] = s[`${c.key}_std`];
      }
      return Object.assign(vals, { kind: k }) as Row;
    });
    return r.sort((a, b) => (a[sort.key] - b[sort.key]) * sort.dir);
  }, [data, sort]);

  const best = Object.fromEntries(
    COLS.filter((c) => c.better).map((c) => {
      const vals = rows.map((r) => r[c.key]);
      return [c.key, c.better === "low" ? Math.min(...vals) : Math.max(...vals)];
    }),
  );

  const series: Series[] = KINDS.map((k) => {
    const hist = Object.values(data.models[k].history);
    const L = Math.min(...hist.map((h) => h.length));
    return {
      id: k, label: MIND_BY_KIND[k].name, color: COLOR[k],
      points: Array.from({ length: L }, (_, i) => {
        const v = hist.map((h) => h[i][metric]);
        return { x: i + 1, y: v.reduce((a, b) => a + b, 0) / v.length, lo: Math.min(...v), hi: Math.max(...v) };
      }),
    };
  });

  const seeds = Object.keys(data.models.lstm.history).length;

  return (
    <section id="results" aria-labelledby="results-title" className="section">
      <div className="container-x">
        <p className="eyebrow">Chapter 6 · The scoreboard</p>
        <h2 id="results-title" className="display mt-4 text-[clamp(2.6rem,5.5vw,4.6rem)]">
          Measured, <em className="text-cyan">not eyeballed.</em>
        </h2>
        <p className="lede mt-5">
          Every number is the mean of {seeds} independent training runs with different random seeds, evaluated on
          2,500 stories none of the models ever saw. ± shows the spread between runs.
        </p>

        <div className="panel mt-10 overflow-x-auto">
          <table className="w-full min-w-[46rem] text-sm">
            <caption className="sr-only">Test-set results per model, mean over seeds</caption>
            <thead>
              <tr className="border-b border-line text-left">
                <th scope="col" className="px-5 py-4 font-normal text-muted">Model</th>
                {COLS.map((c) => (
                  <th
                    key={c.key}
                    scope="col"
                    aria-sort={sort.key === c.key ? (sort.dir === 1 ? "ascending" : "descending") : "none"}
                    className="px-4 py-4 text-right font-normal"
                  >
                    <button
                      onClick={() => setSort((s) => ({ key: c.key, dir: s.key === c.key ? (-s.dir as 1 | -1) : c.better === "high" ? -1 : 1 }))}
                      className={`inline-flex items-center gap-1 ${sort.key === c.key ? "text-ink" : "text-muted hover:text-ink"}`}
                      title={c.hint}
                    >
                      {c.label}
                      <span aria-hidden className="text-[10px]">{sort.key === c.key ? (sort.dir === 1 ? "▲" : "▼") : "↕"}</span>
                    </button>
                  </th>
                ))}
                <th scope="col" className="px-5 py-4 text-right font-normal text-muted">Params</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.kind} className="border-b border-line/70 last:border-0">
                  <th scope="row" className="px-5 py-4 text-left font-normal">
                    <span className="flex items-center gap-2.5">
                      <span className="h-2.5 w-2.5 rounded-full" style={{ background: COLOR[r.kind] }} />
                      {MIND_BY_KIND[r.kind].name}
                    </span>
                  </th>
                  {COLS.map((c) => {
                    const isBest = c.better && r[c.key] === best[c.key];
                    return (
                      <td key={c.key} className="tabular px-4 py-4 text-right">
                        <span className={isBest ? "text-ink" : "text-ink-2"}>{c.fmt(r[c.key])}</span>
                        {Number.isFinite(r[`${c.key}_std`]) && c.key.startsWith("test") && (
                          <span className="ml-1 text-[11px] text-muted">±{c.fmt(r[`${c.key}_std`]).replace("%", "")}</span>
                        )}
                        {isBest && (
                          <span className="ml-2 rounded-full border border-cyan/40 px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wider text-cyan">best</span>
                        )}
                      </td>
                    );
                  })}
                  <td className="tabular px-5 py-4 text-right text-ink-2">{(r.params / 1e6).toFixed(2)}M</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <figure className="panel mt-4 p-5 sm:p-6">
          <figcaption className="mb-5 flex flex-wrap items-center justify-between gap-3">
            <span>
              <span className="font-medium">Learning curves</span>
              <span className="ml-2 text-sm text-muted">mean of {seeds} seeds · band = min–max · log scale</span>
            </span>
            <span className="flex rounded-full border border-line p-1" role="group" aria-label="Curve">
              {(["val_perplexity", "train_perplexity"] as const).map((m) => (
                <button
                  key={m}
                  onClick={() => setMetric(m)}
                  aria-pressed={metric === m}
                  className={`h-9 rounded-full px-4 text-xs ${metric === m ? "bg-white/10 text-ink" : "text-muted"}`}
                >
                  {m === "val_perplexity" ? "Validation" : "Training"}
                </button>
              ))}
            </span>
          </figcaption>
          <LineChart
            series={series}
            xLabel="epoch"
            yLabel="perplexity"
            logY
            ariaLabel="Perplexity per epoch for each model"
            format={(v) => v.toFixed(1)}
          />
        </figure>
      </div>
    </section>
  );
}
