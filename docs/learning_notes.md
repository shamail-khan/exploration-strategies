# Decisions Shamail should be able to explain

## First checkpoint: environment and causal question

Our question is how the action-selection rule changes learning when the Q-learning update and task are held fixed. DQN versus PPO would change many things simultaneously. Within-algorithm controls make attribution narrower, although changing a schedule changes the visited data and therefore the learned Q values too.

Read `environment.py` first: state encoding, action ordering, blocked movement, terminal reward, reset and truncation. Follow the BFS reference in the demo before interpreting any learned behavior. Be able to explain why the shortest path is a reference, not a trained baseline.

## Second checkpoint: learning versus behavior

Read `agent.py`. Derive one Q update by hand. Zero initialization and sparse rewards mean most updates are initially zero; positive information propagates backward only after discovering the goal. Gamma below one prefers shorter successful paths even though undiscounted evaluation return is identical for all successes.

Epsilon-greedy explores uniformly regardless of how bad an action looks. Boltzmann makes action probabilities depend on differences in Q values divided by temperature. Changing reward scale therefore changes softmax behavior even with an unchanged temperature. Neither strategy is the same as PPO's entropy regularization.

The entropy we log is −Σ p(a|s) log p(a|s) of the behavior action distribution at the state visited during training. Its average can change because of the schedule, learned values, or which states are visited. It is not a separate causal measure of exploration quality.

## Third checkpoint: evaluation and uncertainty

Read `evaluation.py`. A learner can acquire a good greedy policy while its training behavior still makes random mistakes. Conversely, exploration can visit many cells without producing a successful greedy policy. Report these as different questions.

A training seed gives one trained policy. Repeating evaluation of it does not create new independent training outcomes. Here the deterministic greedy evaluation repeats exactly on the fixed layout, so we evaluate once. We need multiple independently trained seeds to discuss learning reliability.

Explain the distinction between a true terminal state and an externally stopped trajectory. Our truncation bootstraps; an actual finite-horizon problem would require a different target/state formulation. Explain why a first threshold crossing does not establish convergence and why a mean curve can hide seed failures.

## Suggested hands-on review

1. Predict the reward and flags when entering G on the final allowed step; check the test.
2. Calculate epsilon at interactions 0, 9000, 18000 and 30000.
3. Compare softmax probabilities for Q values [1, 0, 0, 0] at temperatures 1 and 0.05.
4. Open the final diagnostics. Explain any success/length difference and any discrepancy between greedy and exploring success.
5. Identify what evidence is still needed before claiming reward-sparsity sensitivity, stability or generalization.

The next stage should follow this review, rather than hiding the tabular implementation inside a finished deep-RL pipeline.
