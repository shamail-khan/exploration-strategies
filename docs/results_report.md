# Results report — reconstruction in progress

No historical results are available or verified. The original CV description is not a source of numerical findings.

Current scope: a newly implemented tabular pipeline, environment validation and three-seed pilot. Generated numerical results are in `results/figures/pilot/summary.md`, with per-seed scalars and plots beside it and raw runs in `results/pilot/`.

## Executed pilot — 2 November 2026

Three strategies × three independent training seeds × 30,000 interactions: 270,000 training interactions in total. All runs completed under Python 3.13.9 on CPU. Aggregate elapsed time was approximately 5.9 seconds including evaluations, logging and checkpoints. Per-run hardware/interpreter information is in metadata; physical RAM and exact CPU model remain unverified. Seven focused automated tests passed. Environment validation and a visual inspection of the BFS/random trajectory contact sheet preceded training.

All nine final greedy policies succeeded in 12 steps, matching the pretraining BFS reference. Every run covered all 36 traversable states during training. At the 1,000-interaction evaluation spacing, first threshold crossings were 4,000, 3,000 and 5,000 interactions for seeds 0, 1 and 2, respectively, for each of the three strategies. Normalized greedy-return AUC was 0.883 ± 0.033 (sample SD across three training seeds) for every strategy.

These identical aggregate outcomes do not imply identical learned Q tables or trajectories. Independent inspection found distinct final Q-table fingerprints for each strategy at seed 0. Before the first reward, zero-valued tied Q tables induce identical uniform action probabilities and therefore identical action histories under a matched RNG seed. After reward discovery the strategies diverge. Coarse evaluations and a quickly saturated deterministic task obscure some learning differences.

### Behavior with exploration enabled

All final sampled-policy evaluations succeeded in all 20 episodes for every trained seed. Mean episode lengths still differed. Here each trained policy's 20-episode mean is calculated first, then those three means are summarized:

| Condition | Final episode length, mean ± sample SD across training seeds |
|---|---:|
| fixed epsilon 0.2 | 15.42 ± 0.46 |
| decaying epsilon, final 0.05 | 12.95 ± 0.09 |
| softmax, final temperature 0.05 | 12.98 ± 1.70 |

The exploring-policy success curve is lower for softmax during much of the middle of training, while its greedy policy is already successful. This supports separating acquired greedy-policy quality from ongoing exploratory behavior on this task. It does not establish that Boltzmann selection learns more slowly. The schedules were not matched for action entropy, and temperature is sensitive to reward/Q scale.

All final trajectories and visitation maps are available in `results/figures/pilot/policy_diagnostics.html`. Seed 0's fixed/decaying-epsilon policies take the top/right route, while its softmax policy takes a different equally short route. The early untrained greedy policy chooses the first tied action (up), repeatedly hits the top boundary, and reaches the time limit. This illustrates a diagnostic failure caused by tie-breaking before learning, rather than evidence of a learned poor local optimum. The other random examples show unsuccessful navigation within the cap; five examples are not a random-policy performance estimate.

### What the pilot supports

The shared implementation can learn under all three action-selection rules on this grid. Greedy and exploration-enabled metrics answer different questions. The outputs expose seed variation and alternative successful routes. There is no evidence here for a universal winning strategy, robust generalization, a shaping advantage or a PPO entropy effect. The softmax seed variation in exploring episode length deserves inspection rather than removal.

The generated curves were visually inspected through a raster preview of the same raw evaluation data, and final policy trajectories were inspected. Main SVG plots are reproduced by the documented plot command; PNG contact sheets are supplementary previews.

## Next stage

Review the environment, Q update and evaluation distinction using `docs/learning_notes.md`. Before main experiments, freeze a separate protocol with five new training seeds, consider finer evaluation spacing, and decide whether this grid is only a sanity check or part of a broader task set. A deliberately high fixed epsilon could help measure persistent exploration, but that would be a newly specified condition. Do not silently revise these pilot runs or tune a task just to manufacture a winner.

No main experiments, held-out-layout evaluation or deep-RL results have been completed yet. These remain work to do; this report is not the final project report.

## Accurate interim CV wording

**Exploration Strategies in Reinforcement Learning — ongoing reconstruction, November 2026**

- Implemented a reproducible tabular Q-learning study comparing fixed and decaying epsilon-greedy with scheduled Boltzmann action selection on a sparse-reward navigation task.
- Evaluated three training seeds per condition, separating greedy-policy quality from exploratory behavior and inspecting visitation, route length and learning curves against a shortest-path reference.

These bullets describe completed new work. Do not retain a claim of completed entropy-regularization experiments yet. Code and documentation were developed with Codex assistance; Shamail should review and understand them before using the implementation as evidence of independent technical ability.
