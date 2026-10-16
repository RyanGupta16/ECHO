"use client";

import { gsap } from "gsap";
import { useEffect, useRef, useState } from "react";

import type { Walkthrough } from "@/lib/data";

const PHASES = [
  { k: "01", title: "Read", body: "Words become numbers. The model looks only at the last 24 — its context window." },
  { k: "02", title: "Predict", body: "It scores all 8,000 words in its vocabulary. Most get almost nothing; a few light up." },
  { k: "03", title: "Pick & repeat", body: "One word is chosen, appended, the window slides forward — and the loop runs again." },
];

export function HowItWrites({ walk }: { walk: Walkthrough }) {
  const root = useRef<HTMLElement>(null);
  const [step, setStep] = useState(0);
  const N = walk.steps.length;

  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    if (!window.matchMedia("(min-width: 1024px)").matches) return;
    const ctx = gsap.context(() => {
      gsap.timeline({
        scrollTrigger: {
          trigger: root.current,
          pin: "[data-pin]",
          start: "top top",
          end: `+=${N * 70}`,
          scrub: 0.4,
          onUpdate: (st) => setStep(Math.min(N - 1, Math.floor(st.progress * N))),
        },
      });
    }, root);
    return () => ctx.revert();
  }, [N]);

  const seedToks = walk.seed;
  const gen = walk.steps.slice(0, step).map((s) => s.token);
  const all = [...seedToks, ...gen];
  const winStart = Math.max(0, all.length - 24);
  const cur = walk.steps[step];
  const phase = step < 2 ? 0 : step < 5 ? 1 : 2;

  return (
    <section ref={root} id="how" aria-labelledby="how-title" className="relative">
      <div data-pin className="section flex min-h-dvh items-center">
        <div className="container-x grid w-full gap-10 lg:grid-cols-[1fr_1.35fr] lg:gap-16">
          <div>
            <p className="eyebrow">Chapter 1 · Autoregression</p>
            <h2 id="how-title" className="display mt-4 text-[clamp(2.6rem,5.5vw,4.6rem)]">
              One word <em className="text-cyan">at a time.</em>
            </h2>
            <p className="lede mt-5">
              Every model here plays the same game: given what has been written so far, guess the next word. Then do it again.
              That loop is what &ldquo;autoregressive&rdquo; means — and it is the same idea behind today&apos;s largest language models.
            </p>
            <ol className="mt-10 space-y-5">
              {PHASES.map((p, i) => (
                <li key={p.k} className={`flex gap-4 transition-opacity duration-300 ${phase === i ? "opacity-100" : "opacity-40"}`}>
                  <span className="tabular font-mono text-xs text-cyan">{p.k}</span>
                  <div>
                    <p className="font-medium">{p.title}</p>
                    <p className="mt-1 text-sm text-ink-2">{p.body}</p>
                  </div>
                </li>
              ))}
            </ol>
          </div>

          <div className="panel relative overflow-hidden p-5 sm:p-7">
            <div className="flex items-center justify-between font-mono text-[11px] uppercase tracking-[0.18em] text-muted">
              <span>{walk.model} · greedy</span>
              <span className="tabular">step {String(step + 1).padStart(2, "0")} / {N}</span>
            </div>

            <p className="mt-5 min-h-[9.5rem] font-mono text-[13.5px] leading-[2.1] sm:text-sm" aria-live="polite">
              {all.map((t, i) => {
                const inWin = i >= winStart;
                const isGen = i >= seedToks.length;
                return (
                  <span
                    key={i}
                    className={`mr-[0.45em] inline-block rounded px-0.5 transition-colors duration-300 ${
                      inWin ? (isGen ? "bg-cyan/12 text-ink" : "bg-white/[0.06] text-ink-2") : "text-muted/50"
                    } ${i === all.length - 1 && isGen ? "word-in text-cyan" : ""}`}
                  >
                    {t}
                  </span>
                );
              })}
              <span className="caret text-cyan" />
            </p>
            <div className="mt-2 flex items-center gap-2 font-mono text-[11px] text-muted">
              <span className="inline-block h-2.5 w-5 rounded-sm bg-white/[0.08]" /> context window (last 24 tokens)
            </div>

            <div className="mt-7 border-t border-line pt-5">
              <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-muted">
                What comes after &ldquo;{all[all.length - 1]}&rdquo;?
              </p>
              <ul className="mt-4 space-y-2.5">
                {cur.top.map(([w, p], i) => (
                  <li key={w + i} className="grid grid-cols-[6.5rem_1fr_3.5rem] items-center gap-3 text-sm">
                    <span className={`truncate font-mono ${i === 0 ? "text-ink" : "text-ink-2"}`}>{w}</span>
                    <span className="h-2 overflow-hidden rounded-full bg-white/[0.05]">
                      <span
                        className="block h-full rounded-full transition-[width] duration-500"
                        style={{
                          width: `${Math.max(2, p * 100)}%`,
                          background: i === 0 ? "linear-gradient(90deg,var(--cyan),var(--violet))" : "var(--line-2)",
                        }}
                      />
                    </span>
                    <span className="tabular text-right font-mono text-xs text-ink-2">{(p * 100).toFixed(1)}%</span>
                  </li>
                ))}
              </ul>
            </div>

            <div className="mt-6 flex items-center gap-2 lg:hidden">
              <button
                onClick={() => setStep((s) => Math.max(0, s - 1))}
                className="h-11 rounded-full border border-line-2 px-4 text-sm"
                aria-label="Previous step"
              >
                ← Prev
              </button>
              <button
                onClick={() => setStep((s) => Math.min(N - 1, s + 1))}
                className="h-11 rounded-full bg-ink px-4 text-sm text-bg"
                aria-label="Next step"
              >
                Next word →
              </button>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
