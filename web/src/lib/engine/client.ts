"use client";

import { useEffect, useSyncExternalStore } from "react";

import { EOS, Vocab, tokenize } from "../tokenizer";
import type { Kind, ProbeResult, StepResult, WorkerRequest, WorkerResponse } from "./protocol";

type NoId<T> = T extends unknown ? Omit<T, "id"> : never;
type Pending = { resolve: (v: never) => void; reject: (e: Error) => void };
type Progress = Record<string, { loaded: number; total: number }>;

export const WINDOW = 24;
/** Deploy sub-path (e.g. "/echo-recurrent-text-lab" on GitHub Pages), baked in at build time. */
export const BASE = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

class Engine {
  private worker: Worker | null = null;
  private nextId = 1;
  private pending = new Map<number, Pending>();
  private listeners = new Set<() => void>();
  progress: Progress = {};
  version = 0;
  ready = new Set<string>();
  vocab: Vocab | null = null;
  proper = new Set<string>();
  private vocabPromise: Promise<void> | null = null;

  private ensureWorker() {
    if (this.worker) return this.worker;
    this.worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    this.worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
      const m = e.data;
      if (m.type === "progress") {
        this.progress = { ...this.progress, [m.name]: { loaded: m.loaded, total: m.total } };
        this.emit();
        return;
      }
      if (m.type === "loaded") {
        m.names.forEach((n) => this.ready.add(n));
        this.ready = new Set(this.ready);
        this.emit();
      }
      const p = m.id !== undefined ? this.pending.get(m.id) : undefined;
      if (!p) return;
      this.pending.delete(m.id!);
      if (m.type === "error") p.reject(new Error(m.message));
      else p.resolve(m as never);
    };
    return this.worker;
  }

  private call<T>(req: NoId<WorkerRequest>): Promise<T> {
    const id = this.nextId++;
    const w = this.ensureWorker();
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (v: never) => void, reject });
      w.postMessage({ ...req, id });
    });
  }

  subscribe = (fn: () => void) => {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  };
  private emit() {
    this.version++;
    this.listeners.forEach((fn) => fn());
  }

  loadVocab() {
    this.vocabPromise ??= Promise.all([
      fetch(`${BASE}/data/vocab.json`).then((r) => r.json()),
      fetch(`${BASE}/data/proper_nouns.json`).then((r) => r.json()),
    ]).then(([itos, proper]: [string[], string[]]) => {
      this.vocab = new Vocab(itos);
      this.proper = new Set(proper);
      this.emit();
    });
    return this.vocabPromise;
  }

  load(names: string[]) {
    return this.call<void>({ type: "load", names });
  }

  /** Encode a seed the way generation did in training: <eos> + tokens. */
  encodeSeed(text: string): number[] {
    const v = this.vocab!;
    return [v.stoi.get(EOS)!, ...v.encode(tokenize(text))];
  }

  step(kind: Kind, ids: number[], temperature: number, u: number, topK = 5) {
    return this.call<StepResult>({ type: "step", kind, ids: ids.slice(-WINDOW), temperature, u, topK });
  }

  probe(ids: number[]) {
    return this.call<ProbeResult>({ type: "probe", ids: ids.slice(-WINDOW) });
  }
}

let engine: Engine | null = null;
export function getEngine() {
  engine ??= new Engine();
  return engine;
}

/** Re-render on engine progress / readiness changes. */
export function useEngine() {
  const e = getEngine();
  useSyncExternalStore(e.subscribe, () => e.version, () => 0);
  useEffect(() => {
    e.loadVocab();
  }, [e]);
  return e;
}

/** Seeded uniform RNG (mulberry32) so all four models share one random stream. */
export function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
