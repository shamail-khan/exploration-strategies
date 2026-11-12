# Tabular protocol v0.1 — pilot

Written before running the pilot, 2026-11-02. Configuration: `configs/tabular_pilot.json`. All current results are new reconstruction work.

## Purpose and controlled factors

Can one shared Q-learning implementation learn under three action-selection rules? Do logging, evaluation and policy diagnostics agree? This pilot checks the pipeline and informs a future protocol; it does not confirm an exploration ranking.

One deterministic grid, fixed start and goal, four cardinal actions; blocked actions leave the agent in place and consume a step. State is the integer index of a traversable cell. Reward is +1 on entering the goal and 0 otherwise. Episodes end on success or at 60 interactions. BFS computes the shortest feasible route before training. There is no reward shaping or stochastic transition noise. The custom environment has Gymnasium-style reset/step return values but is not yet a `gymnasium.Env` and must not be passed directly to SB3.

Hold Q initialization (zero), alpha (0.15), gamma (0.97), update rule, layout, horizon, interaction budget (30,000), evaluation times and training seed set constant. Conditions:

| Condition | Action rule | Schedule in training interactions |
|---|---|---|
| fixed_epsilon | epsilon-greedy | epsilon = 0.2 |
| decaying_epsilon | epsilon-greedy | linear 1.0 → 0.05 over 18,000 steps, then fixed |
| softmax | Boltzmann over Q values | linear temperature 1.0 → 0.05 over 18,000 steps, then fixed |

Epsilon-greedy splits exploitation probability across all exactly tied maximizing actions and mixes uniform random actions across all four actions. Softmax subtracts the largest Q value before exponentiation for numerical stability. Initial Q values are tied, so all three strategies initially sample uniformly.

Temperature is measured relative to Q-value scale; its numerical value cannot be equated to epsilon. A matched initial/final number does not imply matched exploration or an equally tuned condition. Record actual action-distribution entropy in nats at every training step. Softmax is action selection, not an entropy penalty added to the Q-learning target.

## Learning target and time limits

Q(s,a) ← Q(s,a) + alpha × [r + gamma × max Q(s',a') − Q(s,a)]. The bootstrap is zero only on true goal termination. A time-limit truncation bootstraps because the learned value represents the underlying continuing navigation task with an external rollout cap. The policy does not observe remaining time. This is deliberately not a finite-horizon optimal-control formulation; evaluation still measures success within the cap. Goal arrival exactly on the last step counts as success, with termination taking precedence.

The environment is deterministic and does not use its reset seed. The agent uses its own seeded Python RNG. Each training run starts a fresh environment and Q table. Matching seed IDs across conditions does not imply matched action histories. Reset states and transition destination states are counted in visitation totals; no evaluation visits enter training maps.

## Evaluation

- Training seeds 0, 1, 2 are tuning/pilot seeds. Final experiments will use distinct seeds selected before their results are inspected.
- Evaluate before training, every 1,000 training interactions, and at the final budget. Evaluation interactions are excluded from the training budget and included in elapsed wall time.
- Primary: one deterministic greedy rollout, with action-index order breaking ties. The task has a single fixed start and deterministic dynamics, so repeating this identical rollout provides no extra evidence. Success is consequently 0 or 1 per trained policy. Return equals success in this reward scheme.
- Secondary: sample actions using the current training exploration value over 20 episodes with a separate evaluation RNG, seeds 10000–10019. These estimate behavior with exploration enabled. They are not the main learned-policy metric and are not 20 independent training runs.
- Both evaluations create separate environment instances and never call updates or consume training RNG state.
- Threshold fixed before the pilot: greedy success = 1 and greedy route length ≤1.25 × shortest path. Report first observed crossing and runs that never cross. This is not proof of stable convergence; inspect later checkpoints for regressions.
- Save all evaluation checkpoints and use the final budget for final comparisons; do not choose each seed's best checkpoint after seeing the curves.
- Plot each training seed and its mean. Descriptive sample standard deviation across training seeds is not a confidence interval. With three seeds, estimates of reliability are preliminary.
- Optional normalized return AUC: trapezoidal integration on common evaluation interactions, including step zero, divided by the common horizon. Do not compare different training horizons with this summary.
- Wall time includes logging, evaluation and checkpoint serialization, so it measures this pipeline rather than pure algorithm update throughput.

## Diagnostics and limitations

Environment validation saves five random trajectories and a BFS reference before training. Inspect the offline demo. Final diagnostics show all condition/seed trajectories and cumulative visitation maps, avoiding favorable seed selection. The automatic failure label is only `time_limit`; distinguish blocked actions, loops and poor route choice through inspection before making specific claims.

Coverage includes the start and goal and is divided by all traversable states. Coverage alone does not establish useful exploration. Episode length alone is ambiguous because failed capped episodes can be longer than successful episodes, and both optimal and suboptimal successful policies receive return 1. Always interpret length alongside success and the BFS reference.

One small deterministic task can saturate quickly and mask strategy differences. It cannot establish generalization, difficult-exploration performance, sensitivity to reward sparsity or a universal best strategy. No held-out layout results exist yet.

## Before main experiments

Review pilot diagnostics with Shamail. Freeze the layout(s), conditions, budget, threshold, evaluation modes and uncertainty procedure in a separate main configuration/protocol. Target five new training seeds per condition. Predeclare any held-out-layout procedure: tabular Q tables do not transfer automatically across layouts, and training fresh agents on held-out layouts assesses cross-task robustness rather than zero-shot generalization.

Do not run DQN versus PPO as the principal causal comparison. For the later deep study, use maintained implementations and within-DQN epsilon schedules and within-PPO entropy coefficients. Fix other settings, log actual epsilon/entropy, verify discrete-action categorical distributions and separate tuning from final evaluation. Select the meaningful main task only after the installation checks and pilot behavior are reviewed. A balancing task may serve as an installation check only.
