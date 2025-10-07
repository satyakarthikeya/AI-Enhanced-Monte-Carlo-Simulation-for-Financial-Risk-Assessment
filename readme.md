# AI-Enhanced Monte Carlo Simulation for Financial Risk Assessment

High-performance credit risk analytics that blend modern machine learning with a C++/Python Monte Carlo engine optimized for millions of simulations per run.

## Table of contents

- [Why this project](#why-this-project)
- [Key capabilities](#key-capabilities)
- [Recent results](#recent-results)
- [System architecture](#system-architecture)
- [Repository layout](#repository-layout)
- [Getting started](#getting-started)
  - [Quick start (recommended)](#quick-start-recommended)
  - [Manual installation](#manual-installation)
- [Workflow: from data to risk metrics](#workflow-from-data-to-risk-metrics)
- [Monte Carlo engine highlights](#monte-carlo-engine-highlights)
- [Generated artefacts and dashboards](#generated-artefacts-and-dashboards)
- [Configuration knobs](#configuration-knobs)
- [Performance and benchmarks](#performance-and-benchmarks)
- [Troubleshooting](#troubleshooting)
- [Next steps](#next-steps)

## Why this project

Financial institutions need accurate, timely insight into changing credit exposure and portfolio volatility. Static scorecards and spreadsheet-driven simulations are too slow and brittle. This project delivers an end-to-end, GPU/CPU-friendly workflow that:

- predicts credit default probability using explainable ML (XGBoost);
- captures correlated default shocks through a Monte Carlo engine written in modern C++ and exposed to Python;
- scales to 1M+ simulation paths with OpenMP parallelism; and
- generates dashboards tailored to credit risk teams and quantitative researchers.

## Key capabilities

- **End-to-end credit pipeline** – purpose-built for credit-card default scoring with reusable components.
- **Production-ready credit training stack** – feature engineering, hyper-parameter tuning, evaluation reports, and SHAP explanations.
- **High-performance Monte Carlo** – compiled extension with vectorized kernels, random number batching, and variance-reduction options.
- **Automated reporting** – confusion matrices, ROC/PR curves, VaR/CVaR summaries, and threshold optimisation reports.
- **Experiment tracking** – timestamped JSON summaries, logs, and artefact directories for reproducibility.

## Recent results

Latest end-to-end run (2025-10-07) produced the following headline metrics:

- **XGBoost credit model**
   - Validation set (n = 4,800) – Accuracy **0.803**, ROC AUC **0.778**, PR AUC **0.538**, MCC **0.294**
   - Test set (n = 6,000) – Accuracy **0.806**, ROC AUC **0.774**, PR AUC **0.543**, MCC **0.310**
   - Confusion matrix (test): TN **4,559**, FP **114**, FN **1,050**, TP **277**
- **Monte Carlo simulation** (`num_simulations=10,000`, `threads=8`, portfolio size 1,000)
   - Expected loss **USD 1.59M**, 95% VaR **USD 1.83M**, 99% VaR **USD 1.95M**
   - 95% CVaR **USD 1.90M**, 99% CVaR **USD 2.00M**, max simulated loss **USD 2.26M**
   - Throughput **160K iterations/sec**, total runtime **0.125 s**, memory footprint ~**100 MB**

## System architecture

Credit card default pipeline:

![Credit Card Default Risk Assessment Pipeline](Images/pipeline/credit.png)

## Repository layout

```text
├── archive/                 # Raw datasets (UCI Credit Card)
├── artifacts/               # Saved models, metrics, viz, threshold studies
├── build/                   # Generated CMake build outputs
├── configs/                 # JSON configs for preprocessing/training
├── processed_data/          # Cleaned CSV/Parquet artefacts
├── scripts/                 # CLI entrypoints (training, simulations)
├── src/
│   ├── main_training.py     # Core CreditRiskTrainer pipeline
│   ├── monte_carlo/         # C++ engine source + headers
│   ├── monte_carlo_wrapper.py
│   ├── performance_benchmark.py
│   └── ...                  # Data loader, evaluator, tuner, validators
├── setup_monte_carlo.sh     # Automated bootstrap script (Linux)
└── readme.md
```

## Getting started

### Quick start (recommended)

1. **Clone** the repository and move into the project root.
2. **Run the bootstrap script** (auto-detects your Linux package manager, installs Python deps, builds the C++ engine, and performs a smoke test):

   ```bash
   ./setup_monte_carlo.sh
   ```

   Pass `--skip-system-deps` if you already manage compilers/CMake/OpenMP.

3. **Activate your environment** (if you work inside a virtual environment) and execute the demo:

   ```bash
   python3 src/monte_carlo_demo.py --quick
   ```

### Manual installation

If you prefer to control every step or work on macOS/Windows WSL:

1. **Install system requirements**
   - Python 3.9+
   - CMake ≥ 3.18
   - GCC/Clang with OpenMP support
   - `python3-dev`, `libomp-dev`, `pkg-config`

2. **Create a virtual environment (optional but recommended)**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   ```

3. **Install Python dependencies**

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn xgboost pybind11 psutil joblib
   ```

4. **Build the Monte Carlo extension**

   ```bash
   cmake -S src/monte_carlo -B build/monte_carlo -DCMAKE_BUILD_TYPE=Release
   cmake --build build/monte_carlo -- -j $(nproc)
   cp build/monte_carlo/monte_carlo_engine*.so src/
   ```

5. **Verify the installation**

   ```bash
   python3 -c "import sys; sys.path.append('src'); import monte_carlo_engine"
   ```

## Workflow: from data to risk metrics

1. **Prepare data**
   - The UCI credit card dataset ships under `archive/UCI_Credit_Card.csv`.
   - `processed_data/` contains engineered features and parquet exports created by the training pipeline.

2. **Train or refresh the credit XGBoost model**


   ```bash
   python3 scripts/train_xgboost_model.py \
       --data-dir processed_data \
       --artifacts-dir artifacts \
       --visualize
   ```

   Helpful flags:
   - `--quick` limits samples for smoke tests.
   - `--tune --strategy randomized --n-iter 50` enables hyper-parameter search.
   - `--evaluate-cv` runs stratified cross-validation.

3. **Execute the Monte Carlo simulation**


   ```bash
   python3 scripts/run_monte_carlo_simulation.py \
       --num-simulations 20000 \
       --threads 4 \
       --plot --save-json
   ```

   The script auto-discovers the latest trained model in `artifacts/models/` unless you pass `--model-path`.

4. **Review reports and dashboards**
   - Metrics: `artifacts/metrics/*.json` and `evaluation_report_*.txt`
   - Visuals: `artifacts/visualizations/` and `artifacts/models/xgboost_evaluation.png`
   - Threshold optimisation: `artifacts/threshold_optimization/`

5. **(Optional) Performance benchmarking**

   ```bash
   python3 src/performance_benchmark.py --quick
   ```

## Monte Carlo engine highlights

- Modern C++17 core with deterministic seeding, vectorised random draws, and OpenMP parallel loops.
- Python bindings exposed through `monte_carlo_wrapper.MonteCarloSimulator` for ergonomic usage.
- Supports millions of scenarios, VaR/CVaR aggregation, expected loss, and portfolio summaries.
- Designed to plug into any probability model that outputs per-account default probabilities.

Minimal usage example:

```python
import sys
sys.path.append("src")
import numpy as np
import pandas as pd
from monte_carlo_wrapper import MonteCarloSimulator

portfolio = pd.DataFrame({
   "account_id": range(1000),
   "balance": np.random.lognormal(9, 1, 1000),
   "limit": np.random.lognormal(10, 0.8, 1000),
   "exposure_at_default": np.random.lognormal(9.2, 1.2, 1000),
   "loss_given_default": np.random.beta(2, 3, 1000) * 0.6 + 0.2,
})

default_probs = np.clip(np.random.beta(1, 12, len(portfolio)), 1e-4, 0.5)

simulator = MonteCarloSimulator(num_simulations=50_000, num_threads=4)
results = simulator.run_simulation(portfolio, default_probs)
print(results["risk_metrics"])
```

## Generated artefacts and dashboards

- `artifacts/metrics/` – evaluation summaries, Monte Carlo run metadata, pipeline stats.
- `artifacts/visualizations/` – ROC/PR curves, confusion matrices, threshold charts.
- `artifacts/models/` – serialized XGBoost models and feature importance plots.
- `monte_carlo_results/` – archived simulation outputs from scripted runs.

## Configuration knobs

- `configs/credit_preprocessing_config.json` – feature engineering, scaling, and split parameters.
- CLI flags on `scripts/train_xgboost_model.py` and `scripts/run_monte_carlo_simulation.py` override defaults for quick experiments.
- Environment variables: set `OMP_NUM_THREADS` to control parallelism when benchmarking.

## Performance and benchmarks

- The compiled extension logs iterations-per-second and thread utilisation after each run.
- `src/performance_benchmark.py` sweeps simulation counts, thread counts, and reports median runtime.
- Use `python3 src/performance_benchmark.py --num-simulations 250000 --threads 8` to profile your hardware.

## Troubleshooting

- **`ImportError: monte_carlo_engine not found`** – ensure the `.so` file lives in `src/` (rerun the build step).
- **`ModuleNotFoundError: xgboost`** – verify the Python environment matches the one used during training or rerun the setup script.
- **Slow simulations** – export `OMP_NUM_THREADS=<cores>` or adjust `--threads` when launching the simulator.
- **Missing dataset** – download the UCI Credit Card dataset and place it in `archive/`, or update `--data-dir` to point at your own CSV/Parquet files.

## Next steps

- Extend credit feature engineering with behavioural time-series aggregations.
- Wire the artefacts into a web dashboard (Streamlit/Plotly Dash) for live portfolio reporting.
- Integrate experiment tracking (Weights & Biases or MLflow) for hyper-parameter studies.
