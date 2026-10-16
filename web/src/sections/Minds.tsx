"use client";

import * as Dialog from "@radix-ui/react-dialog";
import { gsap } from "gsap";
import { useEffect, useRef } from "react";

import { CellDiagram } from "@/components/CellDiagram";
import { BONUS_MIND, COLOR, MINDS, type Mind } from "@/lib/content";
import type { Kind } from "@/lib/engine/protocol";

type Stat = { params: number; core: number; ppl?: number; top1?: number };
export type MindStats = Record<Exclude<Kind, "tfm">, Stat> & { tfm?: Stat };

const fmt = (n: number) => (n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : `${Math.round(n / 1e3)}K`);

function MindDialog({ mind, stats, children }: { mind: Mind; stats: Stat; children: React.ReactNode }) {
  return (
    <Dialog.Root>
      <Dialog.Trigger asChild>{children}</Dialog.Trigger>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-[70] bg-black/70 backdrop-blur-sm data-[state=open]:animate-[fadeIn_200ms]" />
        <Dialog.Content
          className="panel fixed left-1/2 top-1/2 z-[80] max-h-[88dvh] w-[min(92vw,44rem)] -translate-x-1/2 -translate-y-1/2 overflow-y-auto p-6 sm:p-9"
          style={{ background: "var(--panel-solid)" }}
        >
          <div className="flex items-start justify-between gap-6">
            <div>
              <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-muted">{mind.year}</p>
              <Dialog.Title className="display mt-2 text-5xl">{mind.name}</Dialog.Title>
              <Dialog.Description className="mt-2 text-ink-2">{mind.oneLiner}</Dialog.Description>
            </div>
            <Dialog.Close
              className="grid h-11 w-11 shrink-0 place-items-center rounded-full border border-line-2 text-ink-2 hover:text-ink"
              aria-label="Close"
            >
              <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden>
                <path d="M2 2l10 10M12 2L2 12" stroke="currentColor" strokeWidth="1.6" />
              </svg>
            </Dialog.Close>
          </div>
          <div className="mt-6 h-36 rounded-2xl border border-line bg-bg/60 p-3">
            <CellDiagram kind={mind.kind} color={COLOR[mind.kind]} />
          </div>
          <ol className="mt-6 space-y-3">
            {mind.how.map((h, i) => (
              <li key={i} className="flex gap-3 text-[15px] text-ink-2">
                <span className="tabular font-mono text-xs text-cyan">{String(i + 1).padStart(2, "0")}</span>
                {h}
              </li>
            ))}
          </ol>
          <div className="mt-6 rounded-2xl border border-line bg-bg/60 p-4 font-mono text-sm text-ink">
            {mind.equation.map((e) => (
              <p key={e} className="leading-7">{e}</p>
            ))}
          </div>
          <dl className="mt-6 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
            <div><dt className="text-muted">Parameters</dt><dd className="tabular mt-1 text-lg">{fmt(stats.params)}</dd></div>
            <div><dt className="text-muted">{mind.kind === "tfm" ? "Attention blocks" : "Recurrent core"}</dt><dd className="tabular mt-1 text-lg">{fmt(stats.core)}</dd></div>
            {stats.ppl !== undefined && <div><dt className="text-muted">Test perplexity</dt><dd className="tabular mt-1 text-lg">{stats.ppl.toFixed(1)}</dd></div>}
            {stats.top1 !== undefined && <div><dt className="text-muted">Top-1 accuracy</dt><dd className="tabular mt-1 text-lg">{(stats.top1 * 100).toFixed(1)}%</dd></div>}
          </dl>
          <div className="mt-6 grid gap-3 sm:grid-cols-2">
            <p className="rounded-xl border border-line p-4 text-sm"><span className="text-muted">Strength · </span>{mind.strength}</p>
            <p className="rounded-xl border border-line p-4 text-sm"><span className="text-muted">Weakness · </span>{mind.weakness}</p>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

export function Minds({ stats }: { stats: MindStats }) {
  const root = useRef<HTMLElement>(null);
  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const ctx = gsap.context(() => {
      gsap.from("[data-mind]", {
        y: 60, opacity: 0, duration: 1, ease: "power3.out", stagger: 0.08,
        scrollTrigger: { trigger: "[data-minds-grid]", start: "top 80%" },
      });
    }, root);
    return () => ctx.revert();
  }, []);

  return (
    <section ref={root} id="minds" aria-labelledby="minds-title" className="section">
      <div className="container-x">
        <p className="eyebrow">Chapter 3 · The contenders</p>
        <h2 id="minds-title" className="display mt-4 text-[clamp(2.6rem,5.5vw,4.6rem)]">
          Four minds, <em className="text-violet">one rulebook.</em>
        </h2>
        <p className="lede mt-5">
          Same vocabulary, same 24-word window, same embedding and hidden sizes, same optimiser, same training budget,
          three random seeds each. The only thing that changes is how each network remembers.
        </p>
        <div data-minds-grid className="mt-14 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          {MINDS.map((m) => (
            <MindDialog key={m.kind} mind={m} stats={stats[m.kind as Exclude<Kind, "tfm">]}>
              <button
                data-mind
                className="panel group relative flex flex-col overflow-hidden p-5 text-left transition-[border-color,transform] duration-300 hover:-translate-y-1 hover:border-line-2"
                aria-label={`Open ${m.name} details`}
              >
                <div
                  className="pointer-events-none absolute -right-16 -top-16 h-40 w-40 rounded-full opacity-25 blur-3xl transition-opacity duration-500 group-hover:opacity-50"
                  style={{ background: COLOR[m.kind] }}
                />
                <div className="flex items-center justify-between">
                  <span className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.18em] text-muted">
                    <span className="h-2 w-2 rounded-full" style={{ background: COLOR[m.kind] }} />
                    {m.short}
                  </span>
                  <span className="font-mono text-[11px] text-muted">{m.year.split(" · ")[0]}</span>
                </div>
                <div className="mt-4 h-28">
                  <CellDiagram kind={m.kind} color={COLOR[m.kind]} />
                </div>
                <h3 className="display mt-4 text-3xl">{m.name}</h3>
                <p className="mt-1 font-serif text-lg italic text-ink-2">{m.tagline}</p>
                <p className="mb-5 mt-3 text-sm text-ink-2">{m.oneLiner}</p>
                <div className="mt-auto flex items-end justify-between border-t border-line pt-4">
                  <div>
                    <p className="text-xs text-muted">Parameters</p>
                    <p className="tabular font-mono text-sm">{fmt(stats[m.kind as Exclude<Kind, "tfm">].params)}</p>
                  </div>
                  <span className="text-sm text-ink-2 transition-colors group-hover:text-ink">Explore →</span>
                </div>
              </button>
            </MindDialog>
          ))}
        </div>

        {stats.tfm && (
          <MindDialog mind={BONUS_MIND} stats={stats.tfm}>
            <button
              data-mind
              className="group relative mt-4 grid w-full gap-6 overflow-hidden rounded-[20px] border border-dashed border-line-2 bg-bg-2/60 p-5 text-left transition-colors hover:border-ink-2/50 md:grid-cols-[1fr_1.3fr_auto] md:items-center"
              aria-label="Open Transformer (bonus contender) details"
            >
              <div>
                <span className="inline-flex items-center gap-2 rounded-full border border-tfm/50 px-2.5 py-1 font-mono text-[10px] uppercase tracking-[0.18em] text-ink-2">
                  <span className="h-1.5 w-1.5 rounded-full bg-tfm" /> Bonus contender · not in the brief
                </span>
                <h3 className="display mt-4 text-3xl">{BONUS_MIND.name}</h3>
                <p className="mt-1 font-serif text-lg italic text-ink-2">{BONUS_MIND.tagline}</p>
                <p className="mt-3 max-w-md text-sm text-ink-2">{BONUS_MIND.oneLiner}</p>
              </div>
              <div className="h-28">
                <CellDiagram kind="tfm" color={COLOR.tfm} />
              </div>
              <div className="flex items-end justify-between gap-8 md:flex-col md:items-end">
                <div className="md:text-right">
                  <p className="text-xs text-muted">Parameters</p>
                  <p className="tabular font-mono text-sm">{fmt(stats.tfm.params)}</p>
                </div>
                <span className="text-sm text-ink-2 group-hover:text-ink">Explore →</span>
              </div>
            </button>
          </MindDialog>
        )}
      </div>
    </section>
  );
}
