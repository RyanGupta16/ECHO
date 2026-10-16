"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { COLOR, MIND_BY_KIND, PRESET_SEEDS, SITE } from "@/lib/content";
import { repetitionRate, toDisplay } from "@/lib/display";
import { rng, useEngine } from "@/lib/engine/client";
import { BONUS, KINDS, type Kind, type TopK } from "@/lib/engine/protocol";
import { shareCard } from "@/lib/shareCard";
import { useSound } from "@/lib/sound";
import { tokenize } from "@/lib/tokenizer";

const LENGTH = 48;

type Lane = { ids: number[]; tokens: string[]; tops: TopK[]; ms: number[]; chosenP: number[] };
const emptyLane = (): Lane => ({ ids: [], tokens: [], tops: [], ms: [], chosenP: [] });

function tempLabel(t: number) {
  if (t === 0) return "Greedy — always the top word";
  if (t < 0.65) return "Cautious";
  if (t <= 1.05) return "Balanced";
  return "Wild";
}

function ProbBars({ top, chosen, word, color }: { top: TopK | undefined; chosen?: number; word: (id: number) => string; color: string }) {
  return (
    <ul className="space-y-1.5" aria-label="Top five next-word probabilities">
      {Array.from({ length: 5 }, (_, i) => {
        const t = top?.[i];
        const isChosen = t && t.id === chosen;
        return (
          <li key={i} className="grid grid-cols-[5.5rem_1fr_3rem] items-center gap-2 text-xs">
            <span className={`truncate font-mono ${isChosen ? "text-ink" : "text-muted"}`}>{t ? word(t.id) : "—"}</span>
            <span className="h-1.5 overflow-hidden rounded-full bg-white/[0.05]">
              <span
                className="block h-full rounded-full transition-[width] duration-300"
                style={{ width: t ? `${Math.max(1.5, t.p * 100)}%` : "0%", background: isChosen ? color : "var(--line-2)" }}
              />
            </span>
            <span className="tabular text-right font-mono text-muted">{t ? `${(t.p * 100).toFixed(1)}%` : ""}</span>
          </li>
        );
      })}
    </ul>
  );
}

export function Arena({ hasBonus }: { hasBonus: boolean }) {
  const engine = useEngine();
  const sound = useSound();
  const root = useRef<HTMLElement>(null);
  const track = useRef<HTMLDivElement>(null);
  const [seed, setSeed] = useState(PRESET_SEEDS[0]);
  const [temp, setTemp] = useState(0.8);
  const [withBonus, setWithBonus] = useState(false);
  const lanesKinds: Kind[] = useMemo(() => (withBonus ? [...KINDS, BONUS] : KINDS), [withBonus]);
  const [lanes, setLanes] = useState<Partial<Record<Kind, Lane>>>({});
  const [running, setRunning] = useState(false);
  const [turbo, setTurbo] = useState(false);
  const [dice, setDice] = useState(7);
  const [hover, setHover] = useState<{ kind: Kind; i: number } | null>(null);
  const [mobileIdx, setMobileIdx] = useState(0);
  const [shareMsg, setShareMsg] = useState<string | null>(null);
  const stopRef = useRef(false);
  const [wantLoad, setWantLoad] = useState(false);

  // Download models only when the arena comes into view.
  useEffect(() => {
    const io = new IntersectionObserver(([e]) => e.isIntersecting && setWantLoad(true), { rootMargin: "600px" });
    if (root.current) io.observe(root.current);
    return () => io.disconnect();
  }, []);
  useEffect(() => {
    if (wantLoad) engine.load(lanesKinds).catch(() => {});
  }, [wantLoad, engine, lanesKinds]);

  const ready = lanesKinds.every((k) => engine.ready.has(k)) && !!engine.vocab;
  const word = useCallback((id: number) => engine.vocab?.itos[id] ?? "", [engine.vocab]);

  const unknown = useMemo(
    () => (engine.vocab ? [...new Set(tokenize(seed).filter((t) => !engine.vocab!.has(t)))] : []),
    [seed, engine.vocab],
  );

  const run = useCallback(async () => {
    if (!ready || running) return;
    stopRef.current = false;
    setRunning(true);
    const start = engine.encodeSeed(seed);
    const state = Object.fromEntries(lanesKinds.map((k) => [k, { ...emptyLane(), ids: [...start] }])) as Record<Kind, Lane>;
    setLanes(structuredClone(state));
    const u = rng(dice);
    for (let s = 0; s < LENGTH && !stopRef.current; s++) {
      const t0 = performance.now();
      const shared = u(); // the same dice roll for every model
      const results = await Promise.all(lanesKinds.map((k) => engine.step(k, state[k].ids, temp, shared)));
      lanesKinds.forEach((k, i) => {
        const r = results[i];
        const L = state[k];
        L.ids.push(r.token);
        L.tokens.push(word(r.token));
        L.tops.push(r.top);
        L.ms.push(r.ms);
        L.chosenP.push(r.chosenP);
        sound.tick(r.chosenP, i);
      });
      setLanes(structuredClone(state));
      const wait = (turbo ? 25 : 140) - (performance.now() - t0);
      if (wait > 0) await new Promise((res) => setTimeout(res, wait));
    }
    setRunning(false);
    // `sound` is a stable singleton wrapper
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, running, engine, seed, dice, temp, turbo, word, lanesKinds]);

  const seedTokens = useMemo(() => tokenize(seed), [seed]);
  const progress = engine.progress;
  const totalBytes = lanesKinds.reduce((a, k) => a + (progress[k]?.total ?? 0), 0);
  const loadedBytes = lanesKinds.reduce((a, k) => a + (progress[k]?.loaded ?? 0), 0);

  const share = async (k: Kind, text: string) => {
    const r = await shareCard({
      model: MIND_BY_KIND[k].name, color: COLOR[k], seed, text, temperature: temp,
      url: typeof window !== "undefined" ? window.location.origin + window.location.pathname : SITE.repo,
    });
    setShareMsg(r === "shared" ? "Shared" : "Image saved");
    setTimeout(() => setShareMsg(null), 2500);
  };

  // Phone carousel: keep tabs and scroll position in sync.
  const goTo = (i: number) => {
    setMobileIdx(i);
    const el = track.current?.children[i] as HTMLElement | undefined;
    el?.scrollIntoView({ behavior: "smooth", inline: "start", block: "nearest" });
  };
  const onTrackScroll = () => {
    const t = track.current;
    if (!t) return;
    const i = Math.round(t.scrollLeft / t.clientWidth);
    if (i !== mobileIdx) setMobileIdx(i);
  };

  return (
    <section ref={root} id="arena" aria-labelledby="arena-title" className="section">
      <div className="container-x">
        <p className="eyebrow">Chapter 5 · The arena</p>
        <h2 id="arena-title" className="display mt-4 text-[clamp(2.6rem,5.5vw,4.6rem)]">
          Same seed. Same dice. <em className="text-cyan">Four minds.</em>
        </h2>
        <p className="lede mt-5">
          All models run right here in your browser. Every step they share one random number, so any difference you
          see comes from the brain — not from luck. Hover any word to see what else the model considered.
        </p>

        {/* Controls */}
        <div className="panel mt-10 grid gap-6 p-5 sm:p-6 lg:grid-cols-[1.4fr_1fr_auto] lg:items-end">
          <div>
            <label htmlFor="seed" className="text-sm text-ink-2">Opening line</label>
            <input
              id="seed"
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
              disabled={running}
              className="mt-2 h-12 w-full rounded-xl border border-line-2 bg-bg/70 px-4 font-mono text-base text-ink outline-none transition-colors focus:border-cyan sm:text-sm"
              maxLength={160}
              autoComplete="off"
            />
            <div className="mt-3 flex flex-wrap gap-2">
              {PRESET_SEEDS.map((p) => (
                <button
                  key={p}
                  onClick={() => setSeed(p)}
                  disabled={running}
                  className={`min-h-9 rounded-full border px-3 text-xs transition-colors ${seed === p ? "border-cyan/60 text-ink" : "border-line text-muted hover:text-ink"}`}
                >
                  {p.split(" ").slice(0, 4).join(" ")}…
                </button>
              ))}
            </div>
            {unknown.length > 0 && (
              <p className="mt-2 text-xs text-[#e8a54b]" role="status">
                Not in the 8,000-word vocabulary, read as &lt;unk&gt;: {unknown.join(", ")}
              </p>
            )}
          </div>
          <div>
            <div className="flex items-baseline justify-between">
              <label htmlFor="temp" className="text-sm text-ink-2">Temperature</label>
              <span className="tabular font-mono text-sm">{temp.toFixed(2)} · <span className="text-muted">{tempLabel(temp)}</span></span>
            </div>
            <input
              id="temp"
              type="range"
              min={0}
              max={1.5}
              step={0.05}
              value={temp}
              onChange={(e) => setTemp(Number(e.target.value))}
              disabled={running}
              className="echo-range mt-1"
            />
            <div className="flex justify-between font-mono text-[10px] uppercase tracking-[0.16em] text-muted">
              <span>safe</span><span>creative</span><span>chaos</span>
            </div>
          </div>
          <div className="flex flex-wrap gap-2 lg:w-56 lg:flex-col">
            {!running ? (
              <button
                onClick={run}
                disabled={!ready}
                className="h-12 min-w-36 flex-1 rounded-full bg-ink px-6 text-sm font-medium text-bg transition-transform hover:scale-[1.03] active:scale-[0.98] disabled:cursor-wait disabled:opacity-50 lg:flex-none"
              >
                {ready ? "Generate" : "Loading models…"}
              </button>
            ) : (
              <button onClick={() => (stopRef.current = true)} className="h-12 min-w-36 flex-1 rounded-full border border-line-2 px-6 text-sm lg:flex-none">
                Stop
              </button>
            )}
            <div className="flex flex-wrap gap-2">
              <button
                onClick={() => setDice((d) => d + 1)}
                disabled={running}
                className="h-10 flex-1 rounded-full border border-line px-3 text-xs text-ink-2 hover:text-ink"
                title="Change the shared random stream"
              >
                Re-roll dice #{dice}
              </button>
              <button
                onClick={() => setTurbo((t) => !t)}
                aria-pressed={turbo}
                className={`h-10 rounded-full border px-3 text-xs ${turbo ? "border-cyan/60 text-ink" : "border-line text-ink-2"}`}
              >
                Turbo
              </button>
              <button
                onClick={sound.toggle}
                aria-pressed={sound.enabled}
                aria-label={sound.enabled ? "Mute word sounds" : "Play a sound for each word"}
                className={`grid h-10 w-10 place-items-center rounded-full border ${sound.enabled ? "border-cyan/60 text-ink" : "border-line text-ink-2"}`}
              >
                <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
                  <path d="M2 6h2.5L8 3v10L4.5 10H2z" fill="currentColor" stroke="none" />
                  {sound.enabled ? <path d="M10.5 5.5a3.5 3.5 0 010 5M12.5 3.5a6 6 0 010 9" /> : <path d="M11 6l4 4M15 6l-4 4" />}
                </svg>
              </button>
            </div>
            {hasBonus && (
              <label className="flex min-h-10 cursor-pointer items-center gap-2.5 rounded-full border border-dashed border-line-2 px-3 text-xs text-ink-2">
                <input
                  type="checkbox"
                  checked={withBonus}
                  onChange={(e) => setWithBonus(e.target.checked)}
                  disabled={running}
                  className="h-4 w-4 accent-[var(--tfm)]"
                />
                Add the Transformer <span className="text-muted">(bonus)</span>
              </label>
            )}
          </div>
          {!ready && totalBytes > 0 && (
            <div className="lg:col-span-3">
              <div className="h-1 overflow-hidden rounded-full bg-white/[0.05]">
                <div className="h-full bg-gradient-to-r from-cyan to-violet transition-[width]" style={{ width: `${(loadedBytes / totalBytes) * 100}%` }} />
              </div>
              <p className="mt-2 font-mono text-[11px] text-muted">
                Downloading {lanesKinds.length} neural networks · {(loadedBytes / 1e6).toFixed(1)} / {(totalBytes / 1e6).toFixed(1)} MB · runs fully on your device
              </p>
            </div>
          )}
        </div>

        {/* Phone tabs (lanes swipe sideways below md) */}
        <div className="mt-4 flex gap-1.5 overflow-x-auto md:hidden" role="tablist" aria-label="Model lanes">
          {lanesKinds.map((k, i) => (
            <button
              key={k}
              role="tab"
              aria-selected={mobileIdx === i}
              onClick={() => goTo(i)}
              className={`flex h-10 shrink-0 items-center gap-2 rounded-full border px-3.5 text-xs ${mobileIdx === i ? "border-line-2 bg-white/[0.06] text-ink" : "border-line text-muted"}`}
            >
              <span className="h-2 w-2 rounded-full" style={{ background: COLOR[k] }} />
              {MIND_BY_KIND[k].short}
            </button>
          ))}
        </div>

        {/* Lanes */}
        <div
          ref={track}
          onScroll={onTrackScroll}
          className={`mt-3 flex snap-x snap-mandatory gap-4 overflow-x-auto pb-2 [scrollbar-width:none] md:mt-4 md:grid md:overflow-visible md:pb-0 ${withBonus ? "md:grid-cols-2 xl:grid-cols-3" : "md:grid-cols-2"}`}
        >
          {lanesKinds.map((k) => {
            const m = MIND_BY_KIND[k];
            const L = lanes[k] ?? emptyLane();
            const disp = toDisplay(L.tokens, engine.proper, seedTokens[seedTokens.length - 1] ?? null);
            const h = hover?.kind === k ? hover.i : null;
            const shownTop = h !== null ? L.tops[h] : L.tops[L.tops.length - 1];
            const shownChosen = h !== null ? L.ids[L.ids.length - L.tokens.length + h] : L.ids[L.ids.length - 1];
            const avgMs = L.ms.length ? L.ms.reduce((a, b) => a + b, 0) / L.ms.length : 0;
            const rep = repetitionRate(L.tokens);
            const plain = disp.map((d, i) => (d.space && i > 0 ? " " : "") + d.text).join("");
            return (
              <article
                key={k}
                className={`panel flex w-[88%] shrink-0 snap-start flex-col p-5 md:w-auto ${k === BONUS ? "border-dashed" : ""}`}
                aria-label={`${m.name} output`}
              >
                <header className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2.5">
                    <span className="h-2.5 w-2.5 rounded-full" style={{ background: COLOR[k], boxShadow: `0 0 12px ${COLOR[k]}` }} />
                    <h3 className="font-medium">{m.name}</h3>
                    {k === BONUS && <span className="rounded-full border border-line px-1.5 font-mono text-[9px] uppercase tracking-wider text-muted">bonus</span>}
                  </div>
                  <div className="tabular flex gap-4 font-mono text-[11px] text-muted">
                    <span title="Average milliseconds per generated word">{avgMs ? `${avgMs.toFixed(1)} ms` : "— ms"}</span>
                    <span title="Share of repeated 3-word phrases">rep {(rep * 100).toFixed(0)}%</span>
                  </div>
                </header>
                <p className="mt-4 min-h-[9.5rem] font-serif text-[1.15rem] leading-relaxed" aria-live="polite">
                  <span className="text-muted">{seed} </span>
                  {disp.map((d, i) => (
                    <span key={i}>
                      {d.space && i > 0 ? " " : ""}
                      <span
                        className={`word-in cursor-default rounded transition-colors ${h === i ? "bg-white/10 text-ink" : "text-ink"}`}
                        onMouseEnter={() => setHover({ kind: k, i })}
                        onMouseLeave={() => setHover(null)}
                        style={i === disp.length - 1 && running ? { color: COLOR[k] } : undefined}
                      >
                        {d.text}
                      </span>
                    </span>
                  ))}
                  {running && <span className="caret" style={{ color: COLOR[k] }} />}
                </p>
                <div className="mt-4 border-t border-line pt-4">
                  <p className="mb-2 font-mono text-[10px] uppercase tracking-[0.18em] text-muted">
                    {h !== null ? `candidates for word ${h + 1}` : "next-word candidates"}
                  </p>
                  <ProbBars top={shownTop} chosen={shownChosen} word={word} color={COLOR[k]} />
                </div>
                {!running && L.tokens.length > 0 && (
                  <button
                    onClick={() => share(k, plain)}
                    className="mt-4 inline-flex h-10 items-center justify-center gap-2 self-start rounded-full border border-line px-4 text-xs text-ink-2 transition-colors hover:border-ink-2 hover:text-ink"
                  >
                    <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" aria-hidden>
                      <path d="M8 10V2M5 5l3-3 3 3M3 9v4h10V9" />
                    </svg>
                    Share this story
                  </button>
                )}
              </article>
            );
          })}
        </div>
        <p className="sr-only" role="status" aria-live="polite">{shareMsg ?? ""}</p>
        {shareMsg && (
          <div className="fixed inset-x-0 bottom-6 z-50 mx-auto w-fit rounded-full border border-line-2 bg-[var(--panel-solid)] px-4 py-2 text-sm shadow-2xl" style={{ marginBottom: "env(safe-area-inset-bottom, 0px)" }}>
            {shareMsg}
          </div>
        )}
      </div>
    </section>
  );
}
