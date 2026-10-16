import { existsSync } from "node:fs";
import path from "node:path";

import { Nav } from "@/components/Nav";
import { MIND_BY_KIND } from "@/lib/content";
import { loadEcho } from "@/lib/data";
import { BONUS, KINDS, type Kind } from "@/lib/engine/protocol";
import { Arena } from "@/sections/Arena";
import { Brain } from "@/sections/Brain";
import { Dataset } from "@/sections/Dataset";
import { Hero } from "@/sections/Hero";
import { HowItWrites } from "@/sections/HowItWrites";
import { Minds, type MindStats } from "@/sections/Minds";
import { Results } from "@/sections/Results";
import { TemperatureLab } from "@/sections/TemperatureLab";
import { Footer, Verdict } from "@/sections/Verdict";

// Parameter counts are fixed by the shared config (embed 128, hidden 256, 2 layers, 8k vocab).
const PARAMS: Record<Kind, { params: number; core: number }> = {
  rnn: { params: 3_310_400, core: 230_400 },
  birnn: { params: 5_719_872, core: 591_872 },
  lstm: { params: 4_001_600, core: 921_600 },
  gru: { params: 3_771_200, core: 691_200 },
  tfm: { params: 3_336_128, core: 1_779_456 },
};

export default async function Home() {
  const data = await loadEcho();
  const s = data.dataset.stats;
  const hasBonus = existsSync(path.join(process.cwd(), "public", "models", "tfm.onnx"));
  const hasResults = KINDS.every((k) => data.models?.[k]?.runs?.length);

  const mindStats = Object.fromEntries(
    [...KINDS, BONUS].map((k) => {
      const sum = data.models?.[k]?.summary;
      return [k, {
        ...PARAMS[k],
        ppl: sum ? Number(sum.test_ppl_mean) : undefined,
        top1: sum ? Number(sum.test_top1_mean) : undefined,
      }];
    }),
  ) as MindStats;

  const heroLines = data.samples.length
    ? data.samples
        .filter((x) => x.kind === "lstm" && x.temperature === 0.8 && x.sample === 0)
        .map((x) => ({ model: `${MIND_BY_KIND[x.kind].name} · T 0.8`, seed: data.seeds[x.seed_id].text, text: x.text }))
    : [{
        model: data.walkthrough.model,
        seed: data.walkthrough.seed.join(" ").replace(/ ([.,!?])/g, "$1"),
        text: data.walkthrough.steps.map((x) => x.token).join(" ").replace(/ ([.,!?])/g, "$1"),
      }];

  const heroStats = [
    { label: "Stories read", value: s.stories.train.toLocaleString("en-US") },
    { label: "Words seen", value: `${(s.tokens.train / 1e6).toFixed(1)}M` },
    { label: "Vocabulary", value: s.vocab_size.toLocaleString("en-US") },
    { label: "Models racing", value: "4" },
  ];

  return (
    <>
      <Nav />
      <main id="main">
        <div id="top" />
        <Hero lines={heroLines} stats={heroStats} />
        <HowItWrites walk={data.walkthrough} />
        <Dataset d={data.dataset} />
        <Minds stats={mindStats} />
        <Brain />
        <Arena hasBonus={hasBonus} />
        {hasResults && <Results data={data} />}
        {hasResults && data.textQuality.length > 0 && <TemperatureLab data={data} />}
        {hasResults && <Verdict data={data} />}
      </main>
      <Footer data={data} />
    </>
  );
}
