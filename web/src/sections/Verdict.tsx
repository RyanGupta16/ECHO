import { COLOR, FINDINGS, MIND_BY_KIND, SITE } from "@/lib/content";
import type { EchoData } from "@/lib/data";
import { KINDS, type Kind } from "@/lib/engine/protocol";

export function Verdict({ data }: { data: EchoData }) {
  const ppl = (k: Kind) => Number(data.models[k].summary.test_ppl_mean);
  const ranked = [...KINDS].sort((a, b) => ppl(a) - ppl(b));
  const winner = ranked[0];
  const base = ppl("rnn");

  return (
    <section id="verdict" aria-labelledby="verdict-title" className="section overflow-hidden">
      <div
        className="pointer-events-none absolute left-1/2 top-24 h-[36rem] w-[36rem] -translate-x-1/2 rounded-full opacity-20 blur-[120px]"
        style={{ background: COLOR[winner] }}
      />
      <div className="container-x relative">
        <p className="eyebrow">Chapter 8 · The verdict</p>
        <h2 id="verdict-title" className="display mt-4 text-[clamp(3rem,8vw,7rem)]">
          The {MIND_BY_KIND[winner].name} <em style={{ color: COLOR[winner] }}>remembers best.</em>
        </h2>

        <ol className="mt-12 grid gap-3 md:grid-cols-4">
          {ranked.map((k, i) => (
            <li key={k} className={`panel p-5 ${i === 0 ? "border-line-2" : ""}`}>
              <p className="tabular font-mono text-xs text-muted">#{i + 1}</p>
              <p className="mt-2 flex items-center gap-2 text-lg">
                <span className="h-2.5 w-2.5 rounded-full" style={{ background: COLOR[k] }} />
                {MIND_BY_KIND[k].name}
              </p>
              <p className="tabular display mt-3 text-5xl">{ppl(k).toFixed(1)}</p>
              <p className="text-xs text-muted">test perplexity</p>
              {k !== "rnn" && (
                <p className="tabular mt-3 text-sm text-ink-2">
                  {ppl(k) < base ? "−" : "+"}{Math.abs(((ppl(k) - base) / base) * 100).toFixed(1)}% vs Simple RNN
                </p>
              )}
            </li>
          ))}
        </ol>

        <div className="mt-12 grid gap-4 lg:grid-cols-2">
          {FINDINGS.map((f, i) => (
            <div key={i} className="panel p-6">
              <p className="tabular font-mono text-xs text-cyan">{String(i + 1).padStart(2, "0")}</p>
              <h3 className="mt-2 text-lg">{f.title}</h3>
              <p className="mt-2 text-ink-2">{f.body}</p>
            </div>
          ))}
        </div>

        <div className="mt-16 flex flex-wrap gap-3">
          <a href="#arena" className="inline-flex h-12 items-center rounded-full bg-ink px-6 text-sm font-medium text-bg transition-transform hover:scale-[1.03]">
            Try them yourself
          </a>
          <a href={SITE.repo} target="_blank" rel="noreferrer" className="inline-flex h-12 items-center rounded-full border border-line-2 px-6 text-sm text-ink-2 hover:text-ink">
            Read the code &amp; report
          </a>
        </div>
      </div>
    </section>
  );
}

export function Footer({ data }: { data: EchoData }) {
  return (
    <footer className="border-t border-line px-4 py-12 sm:px-6">
      <div className="container-x flex flex-col justify-between gap-8 text-sm text-muted md:flex-row">
        <div>
          <p className="font-mono tracking-[0.3em] text-ink">{SITE.name}</p>
          <p className="mt-2 max-w-sm">
            {SITE.tagline}. Built by {SITE.author} for {SITE.course}. All four models run locally in your browser via ONNX Runtime Web.
          </p>
        </div>
        <ul className="space-y-2">
          <li>Dataset: <a className="text-ink-2 underline decoration-line-2 underline-offset-4 hover:text-ink" href={data.dataset.source} target="_blank" rel="noreferrer">TinyStories (Eldan &amp; Li, 2023)</a></li>
          <li>Code &amp; report: <a className="text-ink-2 underline decoration-line-2 underline-offset-4 hover:text-ink" href={SITE.repo} target="_blank" rel="noreferrer">GitHub</a></li>
          <li>Stack: PyTorch · ONNX · Next.js · Three.js · GSAP</li>
        </ul>
      </div>
    </footer>
  );
}
