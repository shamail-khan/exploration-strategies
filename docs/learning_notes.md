# Method notes

## Experimental question

The tabular study compares action-selection rules while holding the Q-learning update and task fixed. Changing an exploration schedule changes the data the agent visits and therefore the Q values it learns.

A comparison between DQN and PPO would also change the learning algorithm, optimization, and data reuse. The planned deep-RL study therefore uses within-algorithm comparisons.

The environment implementation in `src/exploration/environment.py` defines state encoding, action ordering, blocked movement, terminal reward, reset, and time-limit handling. Breadth-first search supplies a shortest-path reference.

## Learning and action selection

The update is:

```text
Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
```

The bootstrap term is zero on goal termination. Time-limit truncations retain bootstrapping under the task formulation in the protocol.

With zero initialization and sparse rewards, early updates are zero until the agent finds the goal. Positive values then propagate backward through later updates. A discount factor below one favors shorter successful routes, even though undiscounted evaluation return is identical for every success.

Epsilon-greedy mixes greedy action selection with uniform random actions. Boltzmann selection assigns probabilities according to Q-value differences divided by temperature. Reward scaling can therefore change Boltzmann behavior even when temperature remains fixed.

The implementation logs the entropy of the behavior action distribution at each visited state. Average entropy depends on the schedule, learned values, and state distribution. Boltzmann action selection is distinct from adding an entropy term to a policy-optimization objective.

## Greedy and exploratory behavior

A trained agent can have a successful greedy policy while exploratory actions produce longer routes. State coverage and task success also measure different aspects of behavior.

The primary evaluation uses greedy actions. The secondary evaluation retains exploration. Both run in separate environments without learning or consuming the training random-number generator.

The fixed layout and deterministic dynamics produce the same greedy trajectory each time a particular policy is evaluated. Repeating that rollout supplies no additional variation. Independent training seeds measure variation in learned policies.

## Time limits and summary metrics

Goal termination and time-limit truncation have different meanings. The current implementation treats the horizon as an external rollout cap. A finite-horizon formulation would require a corresponding value target and state representation.

The first observed performance-threshold crossing measures an evaluation checkpoint. Later checkpoints are needed to assess whether performance persists. Mean learning curves should be read alongside individual seeds.

## Worked examples

The implementation supports several small checks:

1. Entering the goal on the final allowed step counts as successful termination.
2. The epsilon schedule reaches 0.05 at interaction 18,000 and remains fixed afterward.
3. For Q values `[1, 0, 0, 0]`, reducing softmax temperature concentrates probability on the first action.
4. Final greedy and exploration-enabled trajectories can have different lengths despite both succeeding.

The [protocol](experimental_protocol.md) provides the complete definitions, and the [results report](results_report.md) connects them to the pilot findings.
