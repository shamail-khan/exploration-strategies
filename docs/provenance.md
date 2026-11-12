# Project history and reproducibility

## Scope

This repository contains a new implementation of an exploration-strategy study, developed from 2 November 2026 onward. Earlier project files and experiment records are unavailable, so the included runs are documented as new experiments.

The initial release covers tabular Q-learning with fixed epsilon-greedy, decaying epsilon-greedy, and scheduled Boltzmann action selection. Deep-RL comparisons and additional layouts are planned.

This is a standalone project, separate from the Fraunhofer thesis and IPIN paper.

## Recorded experiments

The tabular pilot was executed on 2 November 2026 using Python 3.13.9 on CPU. The implementation uses the Python standard library. Each run records its configuration, seed, source fingerprint, available platform information, timestamp, and elapsed time.

The pilot contains three strategies, three training seeds per strategy, and 30,000 interactions per run. Raw outputs are stored in `results/pilot/`. The [experimental protocol](experimental_protocol.md) defines evaluation and aggregation, and the [results report](results_report.md) discusses findings and limitations.

## Development

Code and documentation were developed with Codex assistance.

## Extensions

Future experiments will use separate configurations and output directories so the recorded pilot remains reproducible. New results will identify their task, training seeds, interaction budget, and evaluation procedure.
