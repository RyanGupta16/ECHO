"use client";

import { gsap } from "gsap";
import dynamic from "next/dynamic";
import { useEffect, useRef, useState } from "react";

import { SITE } from "@/lib/content";

const LoopField = dynamic(() => import("@/components/hero/LoopField"), { ssr: false });

/** Cycles real model generations word by word under the headline. */
function Typewriter({ lines }: { lines: { model: string; seed: string; text: string }[] }) {
  const [li, setLi] = useState(0);
  const [n, setN] = useState(0);
  const words = lines[li]?.text.split(" ") ?? [];

  useEffect(() => {
    if (!lines.length) return;
    const done = n >= words.length;
    const t = setTimeout(
      () => {
        if (done) {
          setLi((i) => (i + 1) % lines.length);
          setN(0);
        } else setN((x) => x + 1);
      },
      done ? 2600 : 90 + Math.random() * 90,
    );
    return () => clearTimeout(t);
  }, [n, li, words.length, lines.length]);

  const line = lines[li];
  if (!line) return null;
  return (
    <div className="mx-auto max-w-2xl text-left" aria-live="off">
      <div className="mb-2 flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.2em] text-muted">
        <span className="h-1.5 w-1.5 rounded-full bg-cyan shadow-[0_0_10px_var(--cyan)]" />
        live sample · {line.model}
      </div>
      <p className="min-h-[5.2em] font-mono text-[13px] leading-relaxed text-ink-2 sm:text-sm">
        <span className="text-muted">{line.seed} </span>
        {words.slice(0, n).join(" ")}
        <span className="caret text-cyan" />
      </p>
    </div>
  );
}

export function Hero({ lines, stats }: { lines: { model: string; seed: string; text: string }[]; stats: { label: string; value: string }[] }) {
  const root = useRef<HTMLElement>(null);

  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const ctx = gsap.context(() => {
      gsap.from("[data-hero-reveal]", { y: 40, opacity: 0, filter: "blur(10px)", duration: 1.4, ease: "expo.out", stagger: 0.12, delay: 0.2 });
      gsap.to("[data-hero-canvas]", {
        scale: 1.35, opacity: 0, ease: "none",
        scrollTrigger: { trigger: root.current, start: "top top", end: "bottom top", scrub: true },
      });
      gsap.to("[data-hero-copy]", {
        y: -120, opacity: 0, ease: "none",
        scrollTrigger: { trigger: root.current, start: "top top", end: "70% top", scrub: true },
      });
    }, root);
    return () => ctx.revert();
  }, []);

  return (
    <header ref={root} className="relative flex min-h-dvh flex-col items-center justify-center overflow-hidden px-4 pb-16 pt-28">
      <div data-hero-canvas className="absolute inset-0">
        <LoopField />
      </div>
      <div
        className="pointer-events-none absolute inset-0"
        style={{ background: "radial-gradient(ellipse 60% 50% at 50% 50%, transparent 0%, var(--bg) 85%)" }}
      />
      <div data-hero-copy className="relative z-10 flex flex-col items-center text-center">
        <p data-hero-reveal className="eyebrow mb-6">
          {SITE.tagline} · Autoregressive generation
        </p>
        <h1 data-hero-reveal className="display text-[clamp(4.5rem,17vw,13rem)]">
          <span className="bg-gradient-to-b from-ink via-ink to-ink/40 bg-clip-text text-transparent">ECHO</span>
        </h1>
        <p data-hero-reveal className="mt-4 max-w-xl text-balance text-lg text-ink-2 sm:text-xl">
          Four neural networks learned to tell stories from 30,000 children&apos;s books.
          <span className="text-ink"> Watch them think — live, in your browser.</span>
        </p>
        <div data-hero-reveal className="mt-9 flex flex-wrap justify-center gap-3">
          <a
            href="#arena"
            className="group inline-flex h-12 items-center gap-2 rounded-full bg-ink px-6 text-sm font-medium text-bg transition-transform duration-200 hover:scale-[1.03] active:scale-[0.98]"
          >
            Enter the arena
            <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden className="transition-transform group-hover:translate-x-0.5">
              <path d="M3 8h10M9 4l4 4-4 4" stroke="currentColor" strokeWidth="1.6" fill="none" strokeLinecap="round" />
            </svg>
          </a>
          <a
            href="#how"
            className="inline-flex h-12 items-center rounded-full border border-line-2 px-6 text-sm text-ink-2 transition-colors hover:border-ink-2 hover:text-ink"
          >
            How it works
          </a>
        </div>
        <div data-hero-reveal className="mt-14 w-full">
          <Typewriter lines={lines} />
        </div>
      </div>
      <dl data-hero-reveal className="absolute inset-x-0 bottom-6 z-10 mx-auto hidden max-w-4xl grid-cols-4 gap-4 px-6 md:grid">
        {stats.map((s) => (
          <div key={s.label} className="border-t border-line pt-3">
            <dt className="font-mono text-[11px] uppercase tracking-[0.18em] text-muted">{s.label}</dt>
            <dd className="tabular mt-1 text-lg text-ink">{s.value}</dd>
          </div>
        ))}
      </dl>
    </header>
  );
}
