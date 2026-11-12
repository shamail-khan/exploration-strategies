# Exploration Strategies in Reinforcement Learning


How do exploration schedules affect the speed and reliability of learning, and when does continued exploration interfere with successful behavior? This is a staged empirical study, starting with an interpretable tabular task and later adding controlled deep-RL comparisons.

## First stage

A sparse-reward grid with obstacles, a fixed start/goal and a 60-step cap. A shared Q-learning implementation compares fixed epsilon-greedy, decaying epsilon-greedy and scheduled Boltzmann action selection. BFS supplies the shortest-path reference. Boltzmann action selection is not entropy-regularized policy optimization.

The current pilot uses three training seeds and one layout. It checks the pipeline and informs the main study; it does not establish a best exploration strategy. PPO entropy regularization and DQN schedules have not yet been implemented or compared.

![Tabular pilot learning curves and example final routes](results/figures/pilot/pilot_preview.png)

All nine final greedy policies found a shortest 12-step route. Read the [pilot results and limitations](docs/results_report.md) and [per-condition summary](results/figures/pilot/summary.md) before interpreting the curves. Raw runs, checkpoints and the original run source snapshot are included.

## Run locally on Windows

Python 3.12 or newer; **no third-party dependencies** for this stage. Clone the repository and work from its directory:

```powershell
git clone https://github.com/shamail-khan/exploration-strategies.git
cd exploration-strategies
```

Set `$python` to your Python executable. If `python` is available on PATH, use:

```powershell
$python = 'python'
& $python --version
& $python scripts/test.py
& $python scripts/run.py check --output results/environment_check_repeat
```

Open `results/environment_check_repeat/trajectories.html` to inspect the shortest-path reference and five random trajectories. Check uses a fresh output directory to preserve existing evidence. On Linux/macOS, run the same script arguments with `python3` instead of PowerShell's `& $python` prefix.

Short training example (one pilot condition/seed, 30,000 inexpensive tabular interactions):

```powershell
& $python scripts/run.py train --condition decaying_epsilon --seed 0 --output results/example
& $python scripts/run.py evaluate --checkpoint results/example/decaying_epsilon/seed_0/checkpoint_0030000.json
```

Full pilot procedure (three conditions × three seeds):

```powershell
& $python scripts/run.py train --config configs/tabular_pilot.json --output results/pilot_repeat
& $python scripts/run.py plot --input results/pilot_repeat --output results/figures/pilot_repeat
```

The original pilot output already occupies `results/pilot`; the commands above write a repeat separately. Choose another fresh directory for subsequent repeats. Training refuses to overwrite an existing run. Optional installation: `python -m pip install -e .`; not required by the source runner.

To reproduce the plots from the included original runs without training:

```powershell
& $python scripts/run.py plot --input results/pilot --output results/figures/pilot
```

## Outputs and interpretation

Each run saves the complete configuration, training episodes, per-interaction exploration value and action entropy, evaluation metrics, all evaluation checkpoints, visitation counts, source fingerprint, interpreter/platform information, seed, timestamp and elapsed time. Checkpoints support loading for evaluation, **not training resumption**. Raw run files are the evidence; generated summaries can be rebuilt.

Primary evaluation disables exploration and runs a deterministic greedy policy in a separate environment. Secondary sampled evaluation retains the current exploration schedule; report it separately. No evaluation updates the agent or consumes the training RNG. Success and return are equivalent in this sparse task; route length and visitation supply additional behavior information.

Plots show individual training seeds, means and descriptive ±1 sample SD bars. Evaluation episodes are not treated as independent training runs. Threshold and AUC definitions, time-limit handling, seed roles and limitations are specified in the [protocol](docs/experimental_protocol.md).

## Project map and next stages

- `src/exploration/`: environment, shared Q learner, evaluation, CLI and portable SVG/HTML diagnostics.
- `configs/`: complete experimental definitions.
- `tests/`: boundary, action-probability, evaluation-isolation, reproducibility and checkpoint checks.
- `results/`: raw pilots, environment checks and generated figures.
- `docs/`: [audit](docs/provenance.md), [protocol](docs/experimental_protocol.md), [report](docs/results_report.md), [learning notes](docs/learning_notes.md).

Next: inspect this stage with Shamail, then freeze a five-seed main tabular protocol; add a small set of layouts if justified; install and pilot established DQN/PPO implementations within the CPU budget. Keep within-algorithm comparisons as the main evidence. Choose a deep-RL main environment where exploration matters. Reward shaping, if added, is a separate experimental factor.


