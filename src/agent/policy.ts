// A tiny feed-forward neural network (MLP) written by hand - no libraries.
// It maps the 7-number scalar observation -> a probability over 3 actions.
//
// RIGHT NOW THE WEIGHTS ARE RANDOM, so it plays badly on purpose. "Training" (a later
// step) is just nudging these weight numbers toward choices that led to winning.
//
// A "neural network" here is literally: multiply inputs by weights, add them up, bend
// the result with a nonlinearity, repeat. That's the whole thing.

export const ACTIONS = ["expand", "attackWeakest", "wait"] as const;

// a seeded random number generator so runs are reproducible
function makeRng(seed: number) {
  let s = seed >>> 0;
  return () => { s = (Math.imul(s, 1103515245) + 12345) >>> 0; return s / 0xffffffff; };
}
// standard-normal sample (Box-Muller) - used to initialize weights to small random values
function randn(rng: () => number) {
  return Math.sqrt(-2 * Math.log(1 - rng())) * Math.cos(2 * Math.PI * rng());
}

export class Policy {
  W1: number[][]; b1: number[];   // input layer -> hidden layer weights + biases
  W2: number[][]; b2: number[];   // hidden layer -> output layer (one score per action)

  constructor(public nIn: number, public nHidden = 8, public nOut = 3, seed = 42) {
    const rng = makeRng(seed);
    const mat = (r: number, c: number) => Array.from({ length: r }, () => Array.from({ length: c }, () => randn(rng) * 0.5));
    const vec = (n: number) => Array.from({ length: n }, () => 0);
    this.W1 = mat(nHidden, nIn); this.b1 = vec(nHidden);
    this.W2 = mat(nOut, nHidden); this.b2 = vec(nOut);
  }

  // the forward pass: observation numbers in -> action out
  forward(x: number[]): { probs: number[]; action: number } {
    // hidden layer:  h = tanh(W1 . x + b1)
    const h = this.W1.map((row, i) => Math.tanh(row.reduce((a, w, j) => a + w * x[j], 0) + this.b1[i]));
    // output layer:  logits = W2 . h + b2   (one raw score per action)
    const logits = this.W2.map((row, i) => row.reduce((a, w, j) => a + w * h[j], 0) + this.b2[i]);
    // softmax: turn the raw scores into probabilities that sum to 1
    const m = Math.max(...logits);
    const exps = logits.map((l) => Math.exp(l - m));
    const sum = exps.reduce((a, v) => a + v, 0);
    const probs = exps.map((e) => e / sum);
    // pick the highest-probability action
    const action = probs.indexOf(Math.max(...probs));
    return { probs, action };
  }
}
