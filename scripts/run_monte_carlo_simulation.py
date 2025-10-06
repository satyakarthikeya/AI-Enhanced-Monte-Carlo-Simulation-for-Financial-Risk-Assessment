#!/usr/bin/env python3
"""Run Monte Carlo risk simulation using a trained XGBoost model.

This script loads an existing trained model, prepares a portfolio sample,
executes the high-performance Monte Carlo engine, and surfaces key risk
metrics alongside optional artefacts (JSON summary, plots).

Example usage (quick smoke run):
    python scripts/run_monte_carlo_simulation.py --portfolio-size 500 --num-simulations 5000

Run with a specific model and export artefacts:
    python scripts/run_monte_carlo_simulation.py \
        --model-path artifacts/models/trained_xgboost_credit_risk_20251006_203500.pkl \
        --num-simulations 20000 --threads 4 --plot --save-json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:  # Optional dependency for some model artefacts
    import joblib
except ImportError:  # pragma: no cover - optional
    joblib = None

# --------------------------------------------------------------------------------------
# Project import setup
# --------------------------------------------------------------------------------------

def _resolve_project_root() -> Path:
    current = Path(__file__).resolve()
    for _ in range(5):
        if (current.parent / "src").exists():
            return current.parent
        current = current.parent
    return Path(__file__).resolve().parents[1]


PROJECT_ROOT = _resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src"))

from data_loader import DataLoader  # type: ignore  # noqa: E402
from xgboost_model import XGBoostModel  # type: ignore  # noqa: E402
from monte_carlo_wrapper import MonteCarloSimulator, format_currency, format_percentage  # type: ignore  # noqa: E402

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _discover_latest_model(models_dir: Path) -> Path:
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    candidates = sorted(models_dir.glob("*.pkl")) + sorted(models_dir.glob("*.joblib"))
    if not candidates:
        raise FileNotFoundError(f"No model artefacts found in {models_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_model(model_path: Optional[Path]) -> XGBoostModel:
    if model_path is None:
        model_path = _discover_latest_model(PROJECT_ROOT / "artifacts" / "models")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logging.info("Loading model from %s", model_path)
    if model_path.suffix == '.pkl':
        with open(model_path, 'rb') as f:
            obj = pickle.load(f)
    else:
        if joblib is None:
            raise RuntimeError("joblib is required to load .joblib artefacts")
        obj = joblib.load(model_path)

    if isinstance(obj, XGBoostModel) and getattr(obj, 'is_trained', False):
        return obj

    from xgboost.sklearn import XGBClassifier  # imported lazily to avoid overhead

    if isinstance(obj, dict) and 'model' in obj and isinstance(obj['model'], XGBClassifier):
        wrapper = XGBoostModel()
        wrapper.model = obj['model']
        wrapper.is_trained = True
        wrapper.feature_names = obj.get('feature_names')
        wrapper.optimal_threshold = obj.get('optimal_threshold', 0.5)
        return wrapper

    if isinstance(obj, XGBClassifier):
        wrapper = XGBoostModel()
        wrapper.model = obj
        wrapper.is_trained = True
        wrapper.feature_names = getattr(obj, 'feature_names_in_', None)
        return wrapper

    raise TypeError(f"Unsupported model artefact type: {type(obj)!r}")


def _prepare_features(args: argparse.Namespace) -> pd.DataFrame:
    loader = DataLoader(data_dir=str(args.data_dir), random_state=args.random_state)
    loader.load_and_prepare_data(val_size=args.val_size, use_parquet=not args.use_csv)
    X_train, X_val, X_test, y_train, y_val, y_test = loader.get_data_splits()

    if args.feature_source == 'train':
        X_source, y_source = X_train, y_train
    elif args.feature_source == 'val':
        X_source, y_source = X_val, y_val
    else:
        X_source, y_source = X_test, y_test

    if args.max_feature_samples > 0 and len(X_source) > args.max_feature_samples:
        X_source = X_source.sample(n=args.max_feature_samples, random_state=args.random_state)

    return X_source.reset_index(drop=True)


def _build_portfolio(features: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(features)
    portfolio = pd.DataFrame({'account_id': np.arange(n)})

    balances = rng.lognormal(mean=9.5, sigma=0.9, size=n)
    utilisation = np.clip(rng.normal(loc=0.45, scale=0.18, size=n), 0.05, 0.95)
    limits = balances / utilisation
    exposure = balances * rng.uniform(0.85, 1.2, size=n)
    lgd = np.clip(rng.beta(2, 5, size=n) * 0.6 + 0.2, 0.1, 0.95)

    portfolio['balance'] = balances
    portfolio['limit'] = limits
    portfolio['exposure_at_default'] = exposure
    portfolio['loss_given_default'] = lgd

    return portfolio


def _generate_default_probabilities(model: XGBoostModel, features: pd.DataFrame) -> np.ndarray:
    feature_names = model.feature_names or list(features.columns)
    missing = [col for col in feature_names if col not in features.columns]
    if missing:
        raise ValueError(f"Feature set missing required columns: {missing[:10]}")

    X = features[feature_names].values
    probs = model.model.predict_proba(X)[:, 1]
    return np.clip(probs, 1e-6, 1 - 1e-6)


def _summarise_probabilities(probs: np.ndarray) -> Dict[str, float]:
    return {
        'mean': float(np.mean(probs)),
        'std': float(np.std(probs)),
        'min': float(np.min(probs)),
        'max': float(np.max(probs)),
    }


def _print_section(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))


def _display_results(portfolio: pd.DataFrame, probs_summary: Dict[str, float], results: Dict[str, Any]) -> None:
    metrics = results.get('risk_metrics', {})
    performance = results.get('performance', {})
    portfolio_summary = results.get('portfolio_summary', {})

    _print_section("Portfolio Summary")
    print(f"Accounts: {portfolio_summary.get('num_accounts', len(portfolio)):,}")
    print(f"Total Exposure: {format_currency(portfolio_summary.get('total_exposure', portfolio['exposure_at_default'].sum()))}")
    print(f"Average PD: {format_percentage(probs_summary['mean'])}")
    print(f"PD Range: {format_percentage(probs_summary['min'])} – {format_percentage(probs_summary['max'])}")

    _print_section("Risk Metrics")
    if not metrics:
        print("No risk metrics returned from simulation.")
    else:
        print(f"Expected Loss: {format_currency(metrics['expected_loss'])}")
        print(f"95% VaR:       {format_currency(metrics['var_95'])}")
        print(f"99% VaR:       {format_currency(metrics['var_99'])}")
        print(f"95% CVaR:      {format_currency(metrics['cvar_95'])}")
        print(f"99% CVaR:      {format_currency(metrics['cvar_99'])}")

    _print_section("Performance")
    if performance:
        its = performance.get('iterations_per_second')
        print(f"Iterations per second: {its:,.0f}" if its else "Iterations per second: N/A")
        print(f"Simulation time (s): {performance.get('simulation_time_seconds', 'N/A')}")
        print(f"Total time (s): {performance.get('total_time_seconds', 'N/A')}")
        print(f"Memory usage (MB): {performance.get('memory_usage_mb', 'N/A')}")
        print(f"Threads used: {performance.get('num_threads_used', 'N/A')}")
    else:
        print("No performance metrics available.")


def _save_plot(metrics: Dict[str, Any], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required for plotting. Install it or omit --plot.") from exc

    labels = [
        'Expected Loss',
        'VaR 95%',
        'VaR 99%',
        'CVaR 95%',
        'CVaR 99%',
    ]
    values = [
        metrics.get('expected_loss', 0.0),
        metrics.get('var_95', 0.0),
        metrics.get('var_99', 0.0),
        metrics.get('cvar_95', 0.0),
        metrics.get('cvar_99', 0.0),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values, color=['#4caf50', '#ff9800', '#f44336', '#9c27b0', '#3f51b5'])
    ax.set_ylabel('Loss Amount (USD)')
    ax.set_title('Monte Carlo Risk Metrics')
    ax.tick_params(axis='x', rotation=30)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Core routine
# --------------------------------------------------------------------------------------


def run_simulation(args: argparse.Namespace) -> Dict[str, Any]:
    model = _load_model(args.model_path)
    if not getattr(model, 'is_trained', False):
        raise RuntimeError("Loaded model is not marked as trained.")

    features = _prepare_features(args)
    if len(features) < args.portfolio_size:
        raise ValueError(f"Requested portfolio size {args.portfolio_size} exceeds available samples {len(features)}")

    features_subset = features.sample(n=args.portfolio_size, random_state=args.random_state).reset_index(drop=True)
    portfolio = _build_portfolio(features_subset, np.random.default_rng(args.random_state))
    default_probs = _generate_default_probabilities(model, features_subset)

    logging.info("Running Monte Carlo simulation (n_iter=%d, threads=%d)", args.num_simulations, args.threads)
    simulator = MonteCarloSimulator(
        num_simulations=args.num_simulations,
        num_threads=args.threads,
        use_antithetic_variates=not args.disable_antithetic,
        enable_correlation=not args.disable_correlation,
        random_seed=args.random_state,
    )

    results = simulator.run_simulation(
        portfolio_data=portfolio,
        default_probabilities=default_probs,
        stress_factor=args.stress_factor if args.stress_factor else None,
    )

    if not results.get('success', False):
        raise RuntimeError(f"Monte Carlo simulation failed: {results.get('error_message', 'Unknown error')}")

    probs_summary = _summarise_probabilities(default_probs)
    _display_results(portfolio, probs_summary, results)

    artefacts = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.save_json:
        output_dir = args.output_dir.resolve()
        _ensure_dir(output_dir)
        json_path = output_dir / f"monte_carlo_run_{timestamp}.json"
        payload = {
            'timestamp': timestamp,
            'model_path': str(args.model_path) if args.model_path else '<auto>',
            'config': {
                'portfolio_size': args.portfolio_size,
                'num_simulations': args.num_simulations,
                'threads': args.threads,
                'stress_factor': args.stress_factor,
            },
            'probability_summary': probs_summary,
            'results': results,
        }
        with open(json_path, 'w') as f:
            json.dump(payload, f, indent=2)
        artefacts['json'] = str(json_path.relative_to(PROJECT_ROOT))
        logging.info("Saved simulation summary to %s", json_path)

    if args.plot:
        output_dir = args.output_dir.resolve() / "visualizations"
        _ensure_dir(output_dir)
        plot_path = output_dir / f"monte_carlo_run_{timestamp}.png"
        _save_plot(results['risk_metrics'], plot_path)
        artefacts['plot'] = str(plot_path.relative_to(PROJECT_ROOT))
        logging.info("Saved plot to %s", plot_path)

    if artefacts:
        print("\nArtefacts generated:")
        for key, value in artefacts.items():
            print(f"  {key}: {value}")

    results['artefacts'] = artefacts
    results['probability_summary'] = probs_summary
    return results


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Monte Carlo credit risk simulation")
    parser.add_argument("--model-path", type=Path, default=None, help="Path to trained model artefact (auto-detect latest if omitted)")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "processed_data", help="Processed data directory")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation split (for reproducible feature preparation)")
    parser.add_argument("--feature-source", choices=['train', 'val', 'test'], default='test', help="Dataset split to sample features from")
    parser.add_argument("--max-feature-samples", type=int, default=0, help="Optional cap on source sample count before slicing")
    parser.add_argument("--portfolio-size", type=int, default=1000, help="Number of accounts in simulated portfolio")
    parser.add_argument("--num-simulations", type=int, default=10000, help="Monte Carlo iteration count")
    parser.add_argument("--threads", type=int, default=4, help="Number of OpenMP threads for simulation")
    parser.add_argument("--stress-factor", type=float, default=0.0, help="Optional stress multiplier for default probabilities")
    parser.add_argument("--disable-antithetic", action="store_true", help="Disable antithetic variates")
    parser.add_argument("--disable-correlation", action="store_true", help="Disable correlated scenario generation")
    parser.add_argument("--save-json", action="store_true", help="Persist simulation summary to JSON under --output-dir")
    parser.add_argument("--plot", action="store_true", help="Generate a bar chart of risk metrics")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "metrics", help="Directory for optional artefacts")
    parser.add_argument("--random-state", type=int, default=99, help="Random seed")
    parser.add_argument("--use-csv", action="store_true", help="Load CSV files instead of parquet")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        run_simulation(args)
    except Exception as exc:  # pragma: no cover - CLI error handling
        logging.exception("Monte Carlo simulation failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
