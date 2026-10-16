import { EOS } from "./tokenizer";

export type DisplayToken = { text: string; space: boolean; raw: string };

const NO_SPACE_BEFORE = new Set([".", ",", "!", "?", ";", ":"]);
const SENTENCE_END = new Set([".", "!", "?", EOS]);

/** Token-by-token display (keeps one span per token for animation/hover). */
export function toDisplay(tokens: string[], proper: Set<string>, prevRaw: string | null): DisplayToken[] {
  const out: DisplayToken[] = [];
  let prev = prevRaw;
  let quoteOpen = false;
  for (const t of tokens) {
    let text = t;
    if (t === EOS) text = "¶";
    else if (t === "i" || proper.has(t) || prev === null || SENTENCE_END.has(prev) || (prev === '"' && quoteOpen))
      text = t[0].toUpperCase() + t.slice(1);
    let space = !NO_SPACE_BEFORE.has(t);
    if (t === '"') {
      space = !quoteOpen;
      quoteOpen = !quoteOpen;
    } else if (prev === '"' && quoteOpen) space = false;
    out.push({ text, space, raw: t });
    prev = t;
  }
  return out;
}

export function repetitionRate(tokens: string[], n = 3): number {
  if (tokens.length < n) return 0;
  const seen = new Set<string>();
  let total = 0;
  for (let i = 0; i + n <= tokens.length; i++) {
    seen.add(tokens.slice(i, i + n).join(" "));
    total++;
  }
  return 1 - seen.size / total;
}
