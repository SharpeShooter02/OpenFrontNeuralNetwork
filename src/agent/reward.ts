// The REWARD scores a game. Territory is weighted heavily so the agent is pushed to EXPAND
// (grab wilderness), with a survival-time term for a continuous gradient.

export interface EpisodeStats {
  peakLandShare: number;     // largest fraction of land ever controlled (0..1)
  survivalFraction: number;  // ticks alive / max ticks (0..1)
  survived: boolean;
  won: boolean;
}
export function computeReward(s: EpisodeStats): number {
  return (
    5 * s.peakLandShare +           // TERRITORY dominates — expand!
    0.3 * s.survivalFraction +      // continuous "stay alive" gradient
    (s.survived ? 0.3 : 0) +
    (s.won ? 5 : 0)
  );
}
