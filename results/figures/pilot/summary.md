# Tabular pilot summary

New reconstruction. Pilot runs, not a confirmatory study.

Thin curves: individual training seeds. Thick curves: seed means. Vertical bars: ±1 sample standard deviation across training seeds, clipped to [0,1] for display; these are not confidence intervals. Each greedy point is one deterministic rollout on one fixed layout. The 20 exploring-policy episodes estimate within-policy behavior; they are not independent training seeds.

| Condition | Training seeds | Final greedy success (mean ± SD) | Final exploring success (mean ± SD) | First threshold, interactions (each seed) | Normalized greedy-return AUC (mean ± SD) | Total wall seconds |
|---|---:|---:|---:|---|---:|---:|
| decaying_epsilon | 3 | 1.000 ± 0.000 | 1.000 ± 0.000 | seed 0: 4000, seed 1: 3000, seed 2: 5000 | 0.883 ± 0.033 | 1.67 |
| fixed_epsilon | 3 | 1.000 ± 0.000 | 1.000 ± 0.000 | seed 0: 4000, seed 1: 3000, seed 2: 5000 | 0.883 ± 0.033 | 1.87 |
| softmax | 3 | 1.000 ± 0.000 | 1.000 ± 0.000 | seed 0: 4000, seed 1: 3000, seed 2: 5000 | 0.883 ± 0.033 | 2.34 |

Threshold: greedy success and route length ≤1.25× BFS shortest path at an evaluation checkpoint. First crossing is quantized by the evaluation interval and is not a sustained-convergence measure. Non-crossing runs are reported explicitly. AUC uses trapezoidal integration of evaluation return from 0 to the shared interaction horizon, divided by that horizon; it is an interpolation summary, not a measurement between evaluations.

Do not select the final study conditions solely to favor a pilot winner. No claim here establishes generalization, reward-sparsity effects, entropy-regularized optimization, or a DQN/PPO comparison.
