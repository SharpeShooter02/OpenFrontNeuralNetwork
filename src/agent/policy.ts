// A tiny feed-forward neural network (MLP) written by hand - no libraries.
// Maps the observation vector -> a probability over actions. "Training" nudges these
// weights toward choices that led to reward.

export const ACTIONS = [
  "expand", "attackWeakest", "attackStrongest", "wait", "requestAlliance",
  "buildCity", "buildDefensePost", "buildMissileSilo", "buildSAMLauncher", "launchNuke",
  "boatAttack", "buildPort",
] as const;

function makeRng(seed: number) {
  let s = seed >>> 0;
  return () => { s = (Math.imul(s, 1103515245) + 12345) >>> 0; return s / 0xffffffff; };
}
function randn(rng: () => number) {
  return Math.sqrt(-2 * Math.log(1 - rng())) * Math.cos(2 * Math.PI * rng());
}

export class Policy {
  W1: number[][]; b1: number[];
  W2: number[][]; b2: number[];
  constructor(public nIn: number, public nHidden = 8, public nOut = 3, seed = 42) {
    const rng = makeRng(seed);
    const mat = (r: number, c: number) => Array.from({ length: r }, () => Array.from({ length: c }, () => randn(rng) * 0.5));
    const vec = (n: number) => Array.from({ length: n }, () => 0);
    this.W1 = mat(nHidden, nIn); this.b1 = vec(nHidden);
    this.W2 = mat(nOut, nHidden); this.b2 = vec(nOut);
  }
  forward(x: number[]): { probs: number[]; action: number } {
    const h = this.W1.map((row, i) => Math.tanh(row.reduce((a, w, j) => a + w * x[j], 0) + this.b1[i]));
    const logits = this.W2.map((row, i) => row.reduce((a, w, j) => a + w * h[j], 0) + this.b2[i]);
    const m = Math.max(...logits);
    const exps = logits.map((l) => Math.exp(l - m));
    const sum = exps.reduce((a, v) => a + v, 0);
    const probs = exps.map((e) => e / sum);
    const action = probs.indexOf(Math.max(...probs));
    return { probs, action };
  }
}

// --- weight helpers for training: flatten to / from a single list of numbers ---
export function getFlat(p: Policy): number[] {
  return [...p.W1.flat(), ...p.b1, ...p.W2.flat(), ...p.b2];
}
export function setFlat(p: Policy, v: number[]): void {
  let k = 0;
  for (const row of p.W1) for (let j = 0; j < row.length; j++) row[j] = v[k++];
  for (let i = 0; i < p.b1.length; i++) p.b1[i] = v[k++];
  for (const row of p.W2) for (let j = 0; j < row.length; j++) row[j] = v[k++];
  for (let i = 0; i < p.b2.length; i++) p.b2[i] = v[k++];
}
