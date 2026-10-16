import type { Kind } from "./engine/protocol";

export const SITE = {
  name: "ECHO",
  tagline: "Recurrent Text Lab",
  author: "Ryan Gupta",
  repo: "https://github.com/RyanGupta16/echo-recurrent-text-lab",
  course: "Foundations of Generative AI · ICT4403",
};

export const NAV = [
  { href: "#how", label: "How it writes" },
  { href: "#minds", label: "The four minds" },
  { href: "#brain", label: "Inside the brain" },
  { href: "#arena", label: "Arena" },
  { href: "#results", label: "Results" },
];

export interface Mind {
  kind: Kind;
  name: string;
  short: string;
  year: string;
  tagline: string;
  oneLiner: string;
  how: string[];
  strength: string;
  weakness: string;
  equation: string[];
}

export const MINDS: Mind[] = [
  {
    kind: "rnn",
    name: "Simple RNN",
    short: "RNN",
    year: "1990 · Elman",
    tagline: "The goldfish.",
    oneLiner: "Reads word by word, squeezing everything it has seen into one running memory vector.",
    how: [
      "At every word it mixes the new word with its previous memory and passes the result through tanh.",
      "That single vector is all it has — so old information is constantly overwritten.",
      "Gradients shrink as they travel back through time, so it struggles to learn long-range links.",
    ],
    strength: "Fastest and smallest. A clean baseline.",
    weakness: "Forgets quickly — vanishing gradients.",
    equation: ["hₜ = tanh(Wₓ xₜ + Wₕ hₜ₋₁ + b)"],
  },
  {
    kind: "birnn",
    name: "Bidirectional RNN",
    short: "BiRNN",
    year: "1997 · Schuster & Paliwal",
    tagline: "Reads the room twice.",
    oneLiner: "Two RNNs read the context window — one forwards, one backwards — and pool what they saw.",
    how: [
      "One RNN reads the 24-word window left → right, a second reads it right → left.",
      "Their final states are concatenated, giving every word a view of its neighbours on both sides.",
      "It never sees the word it must predict: the target sits outside the window, so there is no leakage.",
    ],
    strength: "Richer understanding of the context it is given.",
    weakness: "Twice the cost; still a forgetful tanh cell inside.",
    equation: ["h→ = RNN(x₁ … x_W)", "h← = RNN(x_W … x₁)", "h = [h→ ; h←]"],
  },
  {
    kind: "lstm",
    name: "LSTM",
    short: "LSTM",
    year: "1997 · Hochreiter & Schmidhuber",
    tagline: "The librarian.",
    oneLiner: "Adds a protected memory lane and three gates that decide what to write, keep and reveal.",
    how: [
      "A cell state cₜ runs alongside the hidden state like a conveyor belt that information can ride unchanged.",
      "The forget gate decides what to erase, the input gate what to write, the output gate what to show.",
      "Because the belt is additive, gradients survive long distances — it remembers names and plot.",
    ],
    strength: "Long memory, most stable training.",
    weakness: "Largest recurrent core; slowest per step.",
    equation: ["fₜ, iₜ, oₜ = σ(W [hₜ₋₁, xₜ])", "cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ tanh(W_c [hₜ₋₁, xₜ])", "hₜ = oₜ ⊙ tanh(cₜ)"],
  },
  {
    kind: "gru",
    name: "GRU",
    short: "GRU",
    year: "2014 · Cho et al.",
    tagline: "The minimalist.",
    oneLiner: "An LSTM with the fat trimmed: two gates, no separate cell state.",
    how: [
      "An update gate blends old memory with a new candidate in one move.",
      "A reset gate decides how much of the past to consult when forming that candidate.",
      "About 25% fewer recurrent weights than an LSTM, often with similar quality.",
    ],
    strength: "Near-LSTM memory at lower cost.",
    weakness: "Less explicit control than LSTM's separate cell.",
    equation: ["zₜ, rₜ = σ(W [hₜ₋₁, xₜ])", "h̃ₜ = tanh(W [rₜ ⊙ hₜ₋₁, xₜ])", "hₜ = (1 − zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ"],
  },
];

/** Bonus contender — outside the assignment's four, same data/window/optimiser/budget. */
export const BONUS_MIND: Mind = {
  kind: "tfm",
  name: "Transformer",
  short: "TFM",
  year: "2017 · Vaswani et al.",
  tagline: "The bonus round.",
  oneLiner: "No recurrence at all: every word looks directly at every earlier word through attention.",
  how: [
    "Instead of passing a memory vector along, each word asks every earlier word how relevant it is (self-attention).",
    "A causal mask stops it from looking ahead, so — like the others — it never sees the word it must predict.",
    "This is the architecture behind GPT and ChatGPT. Here it is shrunk to the same parameter budget as the RNNs.",
  ],
  strength: "Direct access to any word in the window; trains in parallel.",
  weakness: "Not part of the assignment brief — shown for context.",
  equation: ["Attention(Q, K, V) = softmax(QKᵀ / √d + mask) V", "4 layers · 6 heads · d = 192 · tied embeddings"],
};

export const MIND_BY_KIND = Object.fromEntries([...MINDS, BONUS_MIND].map((m) => [m.kind, m])) as Record<Kind, Mind>;

export const COLOR: Record<Kind, string> = {
  rnn: "var(--rnn)",
  birnn: "var(--birnn)",
  lstm: "var(--lstm)",
  gru: "var(--gru)",
  tfm: "var(--tfm)",
};

export const PRESET_SEEDS = [
  "Once upon a time, there was a little girl named Lily. She",
  "One day, a big dog found a red ball in the park. The dog",
  "Tom wanted to help his mom bake a cake. He",
  "The sun was hot, so the children went to the",
  "Sam was sad because he lost his favorite toy. His friend",
];

export const GLOSSARY = [
  { term: "Autoregressive", def: "Writes one word, feeds it back in, writes the next — over and over." },
  { term: "Context window", def: "The last 24 words the model is allowed to look at." },
  { term: "Temperature", def: "A creativity dial: low is safe and repetitive, high is wild and random." },
  { term: "Perplexity", def: "How many words the model is torn between at each step. Lower is smarter." },
  { term: "Top-1 accuracy", def: "How often the model's first guess is exactly the real next word." },
];

/** Key findings for the verdict — finalised from the measured results. */
export const FINDINGS: { title: string; body: string }[] = [
  { title: "Pending results", body: "Findings are written once all training runs complete." },
];
