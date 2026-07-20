// The REWARD scores how well one game went - the single number training maximizes.
// Includes a SURVIVAL-TIME term so the signal varies continuously (a policy that lives
// longer scores higher even before it can win) - this fixes the "flat 0.005" problem.

export interface EpisodeStats {
  peakLandShare: number;     // largest fraction of the map we ever controlled (0..1)
  survivalFraction: number;  // ticks we stayed alive / max ticks (0..1)  <-- dense signal
  survived: boolean;         // still alive at the end?
  won: boolean;              // did we win?
}

export function computeReward(s: EpisodeStats): number {
  return (
    s.peakLandShare +               // territory
    0.5 * s.survivalFraction +      // reward living longer (continuous gradient)
    (s.survived ? 0.5 : 0) +        // bonus for surviving to the end
    (s.won ? 5 : 0)                 // big bonus for winning
  );
}
