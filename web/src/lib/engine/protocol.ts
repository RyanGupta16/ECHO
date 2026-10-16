export type Kind = "rnn" | "birnn" | "lstm" | "gru" | "tfm";
/** The four assignment models. */
export const KINDS: Kind[] = ["rnn", "birnn", "lstm", "gru"];
/** Bonus contender, shown separately. */
export const BONUS: Kind = "tfm";

export type TopK = { id: number; p: number }[];

export type WorkerRequest =
  | { type: "load"; id: number; names: string[] }
  | { type: "step"; id: number; kind: Kind; ids: number[]; temperature: number; u: number; topK?: number }
  | { type: "probe"; id: number; ids: number[] };

export type StepResult = { token: number; top: TopK; chosenP: number; ms: number };

export type ProbeResult = {
  T: number;
  H: number;
  L: number;
  /** (L, T, 4, H): input, forget, cell-candidate, output */
  gates: Float32Array;
  /** (L, T, H) */
  hidden: Float32Array;
  predictions: TopK[];
};

export type WorkerResponse =
  | { type: "progress"; name: string; loaded: number; total: number }
  | { type: "loaded"; id: number; names: string[] }
  | ({ type: "step"; id: number; kind: Kind } & StepResult)
  | ({ type: "probe"; id: number } & ProbeResult)
  | { type: "error"; id?: number; message: string };
