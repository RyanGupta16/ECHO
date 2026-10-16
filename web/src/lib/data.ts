import "server-only";

import { readFile } from "node:fs/promises";
import path from "node:path";

import type { Kind } from "./engine/protocol";

export type Walkthrough = {
  model: string;
  seed: string[];
  steps: { token: string; top: [string, number][] }[];
};

export type RunRow = {
  run: string; kind: Kind; seed: number; hidden_dim: number; params_total: number; params_core: number;
  epochs_run: number; best_epoch: number; val_loss: number; val_ppl: number; test_loss: number; test_ppl: number;
  test_top1: number; test_top5: number; train_time_min: number; time_per_epoch_s: number; time_to_best_min: number;
  ms_per_token: number;
};

export type HistoryRow = {
  epoch: number; train_perplexity: number; val_perplexity: number; val_top1_acc: number; epoch_time_s: number;
};

export type ModelBlock = {
  name: string;
  summary: Record<string, number | string>;
  runs: RunRow[];
  history: Record<string, HistoryRow[]>;
  bestRun: string;
  matched: RunRow[];
};

export type TQRow = {
  kind: Kind | "human"; model: string; temperature: number | null; gpt2_ppl: number; consistency: number;
  distinct_1: number; distinct_2: number; distinct_3: number; self_bleu: number | null; repetition: number; length: number;
};

export type Sample = {
  kind: Kind; seed_id: number; temperature: number; sample: number; text: string;
  gpt2_ppl: number; repetition: number; consistency: number;
};

export type EchoData = {
  dataset: {
    name: string; source: string; paper: string; example: string;
    stats: {
      stories: Record<string, number>; tokens: Record<string, number>; unique_words_train: number; vocab_size: number;
      train_coverage: number; words_per_story: { mean: number; min: number; max: number }; top_words: [string, number][];
      window: number; examples: Record<string, number>;
    };
    lengthHistogram: { counts: number[]; edges: number[] };
  };
  config: { window: number; temperatures: number[]; model: Record<string, number>; train: Record<string, unknown> };
  models: Record<Kind, ModelBlock>;
  textQuality: TQRow[];
  seeds: { id: number; text: string }[];
  samples: Sample[];
  walkthrough: Walkthrough;
};

export async function loadEcho(): Promise<EchoData> {
  const file = path.join(process.cwd(), "public", "data", "echo.json");
  return JSON.parse(await readFile(file, "utf8"));
}
