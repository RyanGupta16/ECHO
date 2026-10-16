// Port of src/echo/data.py: same tokenisation the models were trained with.

export const PAD = "<pad>";
export const UNK = "<unk>";
export const EOS = "<eos>";

const TOKEN_RE = /[a-z]+(?:'[a-z]+)?|[0-9]+|[.,!?;:"]/g;
const QUOTES: Record<string, string> = {
  "‘": "'", "’": "'", "“": '"', "”": '"', "—": " , ", "–": " , ",
};

export function tokenize(text: string): string[] {
  let t = text;
  for (const [a, b] of Object.entries(QUOTES)) t = t.split(a).join(b);
  return t.normalize("NFKC").toLowerCase().match(TOKEN_RE) ?? [];
}

const PUNCT = new Set([".", ",", "!", "?", ";", ":"]);

/** Join tokens into display text. `continuation` = don't capitalise the first word. */
export function detokenize(tokens: string[], proper: Set<string>, continuation = false): string {
  let out = "";
  let quoteOpen = false;
  for (const tok of tokens) {
    if (tok === EOS) {
      out = out.trimEnd() + "\n\n";
      continue;
    }
    if (tok === '"') {
      if (quoteOpen) out = out.trimEnd() + '"';
      else out += (out === "" || out.endsWith("\n") || out.endsWith(" ") ? "" : " ") + '"';
      quoteOpen = !quoteOpen;
      continue;
    }
    if (PUNCT.has(tok)) out = out.trimEnd() + tok;
    else out += (out === "" || ((out.endsWith("\n") || out.endsWith('"')) && quoteOpen) ? "" : " ") + tok;
  }
  out = out.replace(/\b[a-z]+\b/g, (w) => (proper.has(w) ? w[0].toUpperCase() + w.slice(1) : w));
  out = out.replace(/\bi\b(?!')|\bi(?='[a-z])/g, "I");
  const start = continuation ? "" : "^|";
  const re = new RegExp("(" + start + '[.!?]"?\\s+"?|\\n\\s*"?|[,:]\\s")([a-z])', "g");
  return out.trim().replace(re, (_, a: string, b: string) => a + b.toUpperCase());
}

/** Display form of a single token inside a token stream (for word-by-word UIs). */
export function displayToken(tok: string, proper: Set<string>): string {
  if (tok === EOS) return "¶";
  if (tok === "i") return "I";
  if (proper.has(tok)) return tok[0].toUpperCase() + tok.slice(1);
  return tok;
}

export class Vocab {
  itos: string[];
  stoi: Map<string, number>;
  constructor(itos: string[]) {
    this.itos = itos;
    this.stoi = new Map(itos.map((w, i) => [w, i]));
  }
  encode(tokens: string[]): number[] {
    const unk = this.stoi.get(UNK)!;
    return tokens.map((t) => this.stoi.get(t) ?? unk);
  }
  has(tok: string) {
    return this.stoi.has(tok);
  }
}
