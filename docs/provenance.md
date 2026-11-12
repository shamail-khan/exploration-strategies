# Provenance and audit — 2026-11-02

Owner: Mohammad Shamail Jamal Khan (Shamail).

## A. Original work

The user reports that all original code, logs, notebooks and reports are lost, and does not remember the environments, algorithms or settings. The workspace was empty at the start of this chat. No historical implementation, experiment or numerical result has been verified.

The surviving CV description mentions epsilon-greedy versus entropy-based regularization, sample efficiency, convergence, stability and policy failure analysis. These statements establish the advertised scope, not evidence that those experiments were completed. No search of private accounts or recovery of lost files has been performed.

## B. New reconstruction

All files and runs in this repository were created from 2026-11-02 onward. No attempt is made to claim they reproduce a known original implementation. The tabular study is a newly designed reconstruction of the empirical question. Run metadata explicitly records this status and distinguishes pilot runs.

## C. New extensions

None established yet. Deep RL, reward-structure comparisons and held-out layouts remain planned. Once a reconstruction baseline is frozen, record subsequent extensions here with dates, configurations and run directories.

This project is separate from the Fraunhofer thesis and IPIN paper. No technical or experimental connection has been established.

## Machine audit

- Windows 11, build 26300; 8 logical processors reported to the process.
- Local Python 3.13.9 works by absolute executable path. The `py` launcher reports no installed Python; `python` on PATH points to a WindowsApps alias. This is a launcher/PATH problem, not proof that Python is absent.
- Bundled Codex Python 3.12.14 is also available. Its environment contains NumPy, but PyTorch, Gymnasium, Stable-Baselines3 and Matplotlib were not found.
- Registry reports Intel Iris Xe graphics. `nvidia-smi` was not found. CUDA availability has not been independently tested.
- Physical RAM could not be queried because CIM access was denied in the sandbox; do not infer a RAM value.
- User budget: CPU only, up to 30 minutes per experiment batch. Start with lightweight pilots before selecting deep-RL budgets.
- Tabular implementation needs only the Python standard library. No global environment changes were made.

## Claims policy

Use new results with their dates and limitations. Do not claim an entropy-regularization comparison until PPO experiments actually run. Boltzmann action selection is not entropy-regularized optimization. Do not claim statistical reliability from three tuning seeds or generalization from one layout.
