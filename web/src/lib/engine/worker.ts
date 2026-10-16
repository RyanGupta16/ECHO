/// <reference lib="webworker" />
// Runs the four ONNX language models off the main thread.
import * as ort from "onnxruntime-web/wasm";

import type { Kind, WorkerRequest, WorkerResponse } from "./protocol";

// The worker lives at <base>/_next/static/..., so the deploy sub-path is everything before /_next/.
const BASE = self.location.pathname.split("/_next/")[0];
ort.env.wasm.wasmPaths = `${self.location.origin}${BASE}/ort/`;
ort.env.wasm.numThreads = 1;

const sessions = new Map<string, Promise<ort.InferenceSession>>();
const BANNED = [0, 1]; // <pad>, <unk>

function post(msg: WorkerResponse) {
  (self as unknown as Worker).postMessage(msg);
}

async function fetchWithProgress(name: string): Promise<Uint8Array> {
  const res = await fetch(`${BASE}/models/${name}.onnx`);
  if (!res.ok || !res.body) throw new Error(`failed to load ${name}`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    if (total) post({ type: "progress", name, loaded: got, total });
  }
  const buf = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) {
    buf.set(c, o);
    o += c.length;
  }
  return buf;
}

function session(name: string) {
  let s = sessions.get(name);
  if (!s) {
    s = fetchWithProgress(name).then((bytes) =>
      ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" }),
    );
    sessions.set(name, s);
  }
  return s;
}

function idsTensor(ids: number[]) {
  return new ort.Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, ids.length]);
}

/** Temperature softmax + sampling with an externally supplied uniform number u. */
function sample(logits: Float32Array, temperature: number, u: number, k = 5) {
  for (const b of BANNED) logits[b] = -Infinity;
  const V = logits.length;
  const t = Math.max(temperature, 1e-6);
  let max = -Infinity;
  for (let i = 0; i < V; i++) if (logits[i] > max) max = logits[i];
  const p = new Float32Array(V);
  let sum = 0;
  for (let i = 0; i < V; i++) {
    p[i] = Math.exp((logits[i] - max) / t);
    sum += p[i];
  }
  for (let i = 0; i < V; i++) p[i] /= sum;

  // top-k for display
  const top: { id: number; p: number }[] = [];
  for (let i = 0; i < V; i++) {
    if (top.length < k || p[i] > top[top.length - 1].p) {
      top.push({ id: i, p: p[i] });
      top.sort((a, b) => b.p - a.p);
      if (top.length > k) top.pop();
    }
  }
  let token = top[0].id;
  if (temperature > 0) {
    let acc = 0;
    for (let i = 0; i < V; i++) {
      acc += p[i];
      if (u < acc) {
        token = i;
        break;
      }
    }
  }
  return { token, top, chosenP: p[token] };
}

self.onmessage = async (e: MessageEvent<WorkerRequest>) => {
  const msg = e.data;
  try {
    if (msg.type === "load") {
      await Promise.all(msg.names.map((n) => session(n)));
      post({ type: "loaded", id: msg.id, names: msg.names });
    } else if (msg.type === "step") {
      const s = await session(msg.kind);
      const t0 = performance.now();
      const out = await s.run({ ids: idsTensor(msg.ids) });
      const logits = out.logits.data as Float32Array;
      const ms = performance.now() - t0;
      const r = sample(new Float32Array(logits), msg.temperature, msg.u, msg.topK ?? 5);
      post({ type: "step", id: msg.id, kind: msg.kind as Kind, ...r, ms });
    } else if (msg.type === "probe") {
      const s = await session("lstm_probe");
      const out = await s.run({ ids: idsTensor(msg.ids) });
      const T = msg.ids.length;
      const V = out.step_logits.dims[1] as number;
      const H = out.hidden.dims[2] as number;
      const L = out.hidden.dims[0] as number;
      const logits = out.step_logits.data as Float32Array;
      // Expected next word at each step (top-5 at T=1).
      const predictions = Array.from({ length: T }, (_, t) =>
        sample(logits.slice(t * V, (t + 1) * V), 1, 0, 5).top,
      );
      post({
        type: "probe", id: msg.id, T, H, L,
        gates: out.gates.data as Float32Array, hidden: out.hidden.data as Float32Array, predictions,
      });
    }
  } catch (err) {
    post({ type: "error", id: (msg as { id?: number }).id, message: String(err) });
  }
};
