// A tiny hand-written MLP. Outputs BOTH an action choice AND a troop-commitment fraction,
// so the agent learns not just *what* to do but *how much* of its army to commit.

export const ACTIONS = [
  "expand", "attackWeakest", "attackStrongest", "diplomacy", "wait",
  "buildCity", "buildDefensePost", "buildMissileSilo", "buildSAMLauncher", "launchNuke",
  "boatAttack", "buildPort", "buildFactory",
] as const;

function makeRng(seed: number) { let s = seed >>> 0; return () => { s = (Math.imul(s, 1103515245) + 12345) >>> 0; return s / 0xffffffff; }; }
function randn(rng: () => number) { return Math.sqrt(-2 * Math.log(1 - rng())) * Math.cos(2 * Math.PI * rng()); }

export class Policy {
  W1: number[][]; b1: number[];
  W2: number[][]; b2: number[];
  constructor(public nIn: number, public nHidden = 16, public nActions = 12, seed = 42) {
    const nOut = nActions + 1; // last output = troop-commitment fraction
    const rng = makeRng(seed);
    const mat = (r: number, c: number) => Array.from({ length: r }, () => Array.from({ length: c }, () => randn(rng) * 0.5));
    const vec = (n: number) => Array.from({ length: n }, () => 0);
    this.W1 = mat(nHidden, nIn); this.b1 = vec(nHidden);
    this.W2 = mat(nOut, nHidden); this.b2 = vec(nOut);
  }
  forward(x: number[]): { probs: number[]; action: number; troopFraction: number } {
    const h = this.W1.map((row, i) => Math.tanh(row.reduce((a, w, j) => a + w * x[j], 0) + this.b1[i]));
    const out = this.W2.map((row, i) => row.reduce((a, w, j) => a + w * h[j], 0) + this.b2[i]);
    const al = out.slice(0, this.nActions);
    const m = Math.max(...al);
    const exps = al.map((l) => Math.exp(l - m));
    const sum = exps.reduce((a, v) => a + v, 0);
    const probs = exps.map((e) => e / sum);
    const action = probs.indexOf(Math.max(...probs));
    const troopFraction = 1 / (1 + Math.exp(-out[this.nActions])); // sigmoid of the extra output
    return { probs, action, troopFraction };
  }
}

export function getFlat(p: Policy): number[] { return [...p.W1.flat(), ...p.b1, ...p.W2.flat(), ...p.b2]; }
export function setFlat(p: Policy, v: number[]): void {
  let k = 0;
  for (const row of p.W1) for (let j = 0; j < row.length; j++) row[j] = v[k++];
  for (let i = 0; i < p.b1.length; i++) p.b1[i] = v[k++];
  for (const row of p.W2) for (let j = 0; j < row.length; j++) row[j] = v[k++];
  for (let i = 0; i < p.b2.length; i++) p.b2[i] = v[k++];
}
