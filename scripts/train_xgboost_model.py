#!/usr/bin/env python3
"""Train the credit risk XGBoost model using the existing CreditRiskTrainer pipeline.

This script wraps the existing modular pipeline (CreditRiskTrainer from main_training.py)
to provide a convenient command-line interface for model training, evaluation, and artifact export.

Example usage (quick smoke run):
    python scripts/train_xgboost_model.py --quick --max-train-samples 5000

Full training using the entire dataset and optimized hyperparameters:
    python scripts/train_xgboost_model.py

With hyperparameter tuning:
    python scripts/train_xgboost_model.py --tune --strategy randomized --n-iter 50

With cross-validation evaluation:
    python scripts/train_xgboost_model.py --evaluate-cv --cv-folds 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, roc_auc_score

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

from main_training import CreditRiskTrainer  # type: ignore  # noqa: E402

# --------------------------------------------------------------------------------------
# Core routine using CreditRiskTrainer
# --------------------------------------------------------------------------------------


def train_model(args: argparse.Namespace) -> Dict[str, Any]:
    """Train model using the existing CreditRiskTrainer class."""
    
    logging.info("Initializing CreditRiskTrainer...")
    trainer = CreditRiskTrainer(
        data_path=str(args.data_dir),
        artifacts_path=str(args.artifacts_dir),
        random_state=args.random_state
    )
    
    # Load data
    logging.info("Loading and preparing data...")
    data_summary = trainer.load_data(val_split=args.val_split)
    
    # Optionally downsample for quick runs
    if args.max_train_samples and args.max_train_samples > 0:
        logging.info(f"Downsampling training data to {args.max_train_samples} samples")
        sample_indices = np.random.RandomState(args.random_state).choice(
            len(trainer.X_train), 
            size=min(args.max_train_samples, len(trainer.X_train)), 
            replace=False
        )
        trainer.X_train = trainer.X_train.iloc[sample_indices]
        trainer.y_train = trainer.y_train.iloc[sample_indices]
    
    results = {}
    
    # Hyperparameter tuning if requested
    if args.tune:
        logging.info(f"Starting hyperparameter tuning with {args.strategy} strategy...")
        tuning_results = trainer.tune_hyperparameters(
            strategy=args.strategy,
            n_iter=args.n_iter,
            scoring=args.scoring,
            save_results=True
        )
        results['tuning'] = tuning_results
        logging.info(f"Best {args.scoring}: {tuning_results['best_score']:.4f}")
        logging.info(f"Best params: {tuning_results['best_params']}")
    else:
        # Standard training with optional XGBoost native CV
        logging.info("Training XGBoost model...")
        training_results = trainer.train_model(
            use_optimized_params=not args.quick,
            optimize_threshold=True,
            save_model=True,
            use_cv=args.use_xgb_cv,  # Use XGBoost native CV
            cv_folds=args.cv_folds,
            early_stopping_rounds=args.early_stopping
        )
        results['training'] = training_results
    
    # Cross-validation evaluation if requested
    if args.evaluate_cv:
        logging.info(f"Performing {args.cv_folds}-fold cross-validation...")
        cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)
        
        # Combine train and validation for CV
        X_combined = trainer.X_train
        y_combined = trainer.y_train
        
        if trainer.X_val is not None:
            import pandas as pd
            X_combined = pd.concat([trainer.X_train, trainer.X_val])
            y_combined = pd.concat([trainer.y_train, trainer.y_val])
        
        # Get the underlying estimator for CV
        estimator = trainer.model.model if hasattr(trainer.model, 'model') else trainer.model
        
        cv_scores = cross_val_score(
            estimator,
            X_combined,
            y_combined,
            cv=cv,
            scoring=make_scorer(roc_auc_score, needs_proba=True),
            n_jobs=-1
        )
        
        cv_results = {
            'cv_scores': cv_scores.tolist(),
            'mean_score': float(cv_scores.mean()),
            'std_score': float(cv_scores.std()),
            'n_folds': args.cv_folds
        }
        results['cross_validation'] = cv_results
        
        logging.info(f"CV {args.cv_folds}-fold AUC: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']:.4f})")
    
    # Evaluation on test set
    logging.info("Evaluating model on test set...")
    test_results = trainer.evaluate_model(
        dataset='test',
        create_visualizations=args.visualize,
        save_report=True
    )
    results['test_evaluation'] = test_results
    
    # Also evaluate on validation set
    logging.info("Evaluating model on validation set...")
    val_results = trainer.evaluate_model(
        dataset='val',
        create_visualizations=args.visualize,
        save_report=True
    )
    results['val_evaluation'] = val_results
    
    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = args.artifacts_dir / "metrics" / f"training_summary_{timestamp}.json"
    
    # Prepare serializable results - extract only key metrics
    summary = {
        'config': {
            'quick_mode': args.quick,
            'max_train_samples': args.max_train_samples,
            'val_split': args.val_split,
            'random_state': args.random_state,
            'tuning_enabled': args.tune,
            'cv_enabled': args.evaluate_cv
        },
        'data_summary': {
            'train_samples': data_summary.get('train_samples'),
            'val_samples': data_summary.get('val_samples'),
            'test_samples': data_summary.get('test_samples'),
            'n_features': data_summary.get('n_features')
        }
    }
    
    # Add training/tuning results
    if 'training' in results:
        summary['training'] = {
            'train_accuracy': results['training'].get('train_accuracy'),
            'val_accuracy': results['training'].get('val_accuracy')
        }
    
    if 'tuning' in results:
        summary['tuning'] = {
            'best_score': results['tuning'].get('best_score'),
            'best_params': results['tuning'].get('best_params')
        }
    
    # Add evaluation results
    if 'test_evaluation' in results:
        summary['test_metrics'] = results['test_evaluation'].get('basic_metrics', {})
    
    if 'val_evaluation' in results:
        summary['val_metrics'] = results['val_evaluation'].get('basic_metrics', {})
    
    if 'cross_validation' in results:
        summary['cross_validation'] = results['cross_validation']
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"Training summary saved to: {summary_path}")
    
    return results


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the credit risk XGBoost model using CreditRiskTrainer")
    
    # Data arguments
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "processed_data", 
                       help="Directory containing processed data")
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts", 
                       help="Directory to store trained models and metrics")
    parser.add_argument("--val-split", type=float, default=0.2, 
                       help="Validation set fraction")
    
    # Training arguments
    parser.add_argument("--random-state", type=int, default=42, 
                       help="Random seed")
    parser.add_argument("--quick", action="store_true", 
                       help="Use lighter-weight training configuration")
    parser.add_argument("--max-train-samples", type=int, default=0, 
                       help="Optional cap on training samples for quick runs")
    
    # Hyperparameter tuning arguments
    parser.add_argument("--tune", action="store_true", 
                       help="Perform hyperparameter tuning")
    parser.add_argument("--strategy", type=str, default="randomized", 
                       choices=["randomized", "grid", "progressive", "bayesian"],
                       help="Tuning strategy")
    parser.add_argument("--n-iter", type=int, default=50, 
                       help="Number of iterations for randomized/bayesian search")
    parser.add_argument("--scoring", type=str, default="roc_auc", 
                       help="Scoring metric for tuning")
    
    # Cross-validation arguments
    parser.add_argument("--evaluate-cv", action="store_true", 
                       help="Perform cross-validation evaluation (sklearn)")
    parser.add_argument("--use-xgb-cv", action="store_true",
                       help="Use XGBoost native cross-validation during training")
    parser.add_argument("--cv-folds", type=int, default=5, 
                       help="Number of CV folds")
    parser.add_argument("--early-stopping", type=int, default=50,
                       help="Early stopping rounds for XGBoost training")
    
    # Other arguments
    parser.add_argument("--visualize", action="store_true", 
                       help="Create evaluation visualizations")
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       help="Logging verbosity")
    
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        results = train_model(args)
        
        print("\n" + "="*80)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*80)
        
        # Print key metrics
        if 'test_evaluation' in results:
            test_metrics = results['test_evaluation']['basic_metrics']
            print(f"\nTest Set Performance:")
            print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
            print(f"  Precision: {test_metrics['precision']:.4f}")
            print(f"  Recall:    {test_metrics['recall']:.4f}")
            print(f"  F1-Score:  {test_metrics['f1_score']:.4f}")
            if 'roc_auc' in test_metrics:
                print(f"  ROC AUC:   {test_metrics['roc_auc']:.4f}")
        
        if 'cross_validation' in results:
            cv = results['cross_validation']
            print(f"\nCross-Validation ({cv['n_folds']}-fold):")
            print(f"  Mean AUC: {cv['mean_score']:.4f} (+/- {cv['std_score']:.4f})")
        
        if 'tuning' in results:
            tuning = results['tuning']
            print(f"\nHyperparameter Tuning:")
            print(f"  Best Score: {tuning['best_score']:.4f}")
            print(f"  Best Params: {tuning['best_params']}")
        
        print("\n" + "="*80)
        
        return 0
        
    except Exception as exc:
        logging.exception("Training failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
