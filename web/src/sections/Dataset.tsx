"use client";

import { useState } from "react";

import type { EchoData } from "@/lib/data";
import { toDisplay } from "@/lib/display";
import { useEngine } from "@/lib/engine/client";
import { tokenize } from "@/lib/tokenizer";

export function Dataset({ d }: { d: EchoData["dataset"] }) {
  const s = d.stats;
  const { counts, edges } = d.lengthHistogram;
  const max = Math.max(...counts);
  const [hover, setHover] = useState<number | null>(null);
  const window = s.window;
  const engine = useEngine();
  const exampleTokens = toDisplay(tokenize(d.example).slice(0, 64), engine.proper, null);

  const tiles = [
    { k: "Stories", v: s.stories.train.toLocaleString("en-US"), sub: `+ ${(s.stories.val + s.stories.test).toLocaleString("en-US")} held out` },
    { k: "Words", v: `${(s.tokens.train / 1e6).toFixed(2)}M`, sub: "training tokens" },
    { k: "Vocabulary", v: s.vocab_size.toLocaleString("en-US"), sub: `${(s.train_coverage * 100).toFixed(2)}% coverage` },
    { k: "Training pairs", v: `${(s.examples.train / 1e6).toFixed(1)}M`, sub: `${window} tokens → next token` },
  ];

  return (
    <section id="dataset" aria-labelledby="data-title" className="section">
      <div className="container-x">
        <p className="eyebrow">Chapter 2 · The library</p>
        <h2 id="data-title" className="display mt-4 text-[clamp(2.6rem,5.5vw,4.6rem)]">
          30,000 bedtime stories, <em className="text-cyan">read word by word.</em>
        </h2>
        <p className="lede mt-5">
          TinyStories is a dataset of short stories written in the vocabulary of a 3–4 year old. Small vocabulary, rich
          plot and dialogue — exactly the setting where small networks can learn to write coherently.{" "}
          <a href={d.source} target="_blank" rel="noreferrer" className="text-ink underline decoration-line-2 underline-offset-4 hover:decoration-cyan">
            Source on Hugging Face
          </a>{" "}
          ·{" "}
          <a href={d.paper} target="_blank" rel="noreferrer" className="text-ink underline decoration-line-2 underline-offset-4 hover:decoration-cyan">
            Eldan &amp; Li, 2023
          </a>
        </p>

        <dl className="mt-12 grid grid-cols-2 gap-4 lg:grid-cols-4">
          {tiles.map((t) => (
            <div key={t.k} className="panel p-5">
              <dt className="font-mono text-[11px] uppercase tracking-[0.18em] text-muted">{t.k}</dt>
              <dd className="tabular display mt-3 text-5xl">{t.v}</dd>
              <dd className="mt-1 text-xs text-ink-2">{t.sub}</dd>
            </div>
          ))}
        </dl>

        <div className="mt-4 grid gap-4 lg:grid-cols-[1.3fr_1fr]">
          <figure className="panel p-5 sm:p-6">
            <figcaption className="flex items-baseline justify-between">
              <span className="font-medium">How long is a story?</span>
              <span className="tabular font-mono text-xs text-muted">
                {hover !== null ? `${edges[hover]}–${edges[hover + 1]} words · ${counts[hover].toLocaleString("en-US")} stories` : `mean ${Math.round(s.words_per_story.mean)} words`}
              </span>
            </figcaption>
            <div className="mt-6 flex h-44 items-end gap-[2px]" role="img" aria-label={`Histogram of story lengths, mean ${Math.round(s.words_per_story.mean)} words`}>
              {counts.map((c, i) => (
                <div
                  key={i}
                  className="relative flex h-full flex-1 items-end"
                  onMouseEnter={() => setHover(i)}
                  onMouseLeave={() => setHover(null)}
                >
                  <div
                    className="w-full rounded-t-[3px] transition-colors"
                    style={{ height: `${(c / max) * 100}%`, background: hover === i ? "var(--cyan)" : "var(--rnn)" }}
                  />
                </div>
              ))}
            </div>
            <div className="mt-2 flex justify-between font-mono text-[10px] text-muted">
              <span>0</span><span>150</span><span>300</span><span>450</span><span>600 words</span>
            </div>
          </figure>

          <figure className="panel p-5 sm:p-6">
            <figcaption className="flex items-baseline justify-between">
              <span className="font-medium">Most common tokens</span>
              <span className="font-mono text-xs text-muted">count</span>
            </figcaption>
            <ul className="mt-5 space-y-1.5">
              {s.top_words.slice(0, 10).map(([w, c]) => (
                <li key={w} className="grid grid-cols-[3.5rem_1fr_4.5rem] items-center gap-3 text-sm">
                  <span className="font-mono text-ink">{w}</span>
                  <span className="h-1.5 rounded-full bg-white/[0.05]">
                    <span className="block h-full rounded-full bg-gru" style={{ width: `${(c / s.top_words[0][1]) * 100}%` }} />
                  </span>
                  <span className="tabular text-right font-mono text-xs text-muted">{c.toLocaleString("en-US")}</span>
                </li>
              ))}
            </ul>
          </figure>
        </div>

        <figure className="panel mt-4 p-5 sm:p-6">
          <figcaption className="flex flex-wrap items-baseline justify-between gap-2">
            <span className="font-medium">Turning a story into training pairs</span>
            <span className="font-mono text-xs text-muted">sliding window · {window} tokens in, 1 token out</span>
          </figcaption>
          <p className="mt-5 font-serif text-lg leading-relaxed text-ink-2">
            {exampleTokens.map((t, i) => (
              <span key={i}>
                {t.space && i > 0 ? " " : ""}
                <span className={i < window ? "rounded bg-cyan/10 text-ink" : i === window ? "rounded bg-violet/30 px-0.5 text-ink" : ""}>
                  {t.text}
                </span>
              </span>
            ))}
            …
          </p>
          <p className="mt-4 text-sm text-muted">
            <span className="text-cyan">Highlighted</span> = input window,{" "}
            <span className="text-violet">violet</span> = the target word. Punctuation counts as a token. Slide one step right and repeat — 6.2 million times.
          </p>
        </figure>
      </div>
    </section>
  );
}
