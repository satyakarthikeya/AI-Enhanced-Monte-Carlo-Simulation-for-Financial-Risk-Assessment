#!/usr/bin/env python3
"""
XGBoost Model Threshold Optimizer - CRITICAL FIX for Low Recall
================================================================

This module provides threshold optimization to fix the critical issue of 
20.9% recall (missing 79% of defaults). Implements:

1. Threshold sweep from 0.05 to 0.9
2. Target minimum recall of 70%
3. Maintain acceptable precision (30%+)
4. Find optimal F1 score within constraints

Usage:
    from threshold_optimizer import optimize_model_threshold
    
    optimal_threshold = optimize_model_threshold(
        model, X_val, y_val, 
        min_recall=0.70
    )
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import logging

logger = logging.getLogger(__name__)


def optimize_model_threshold(model,
                            X_val: Union[pd.DataFrame, np.ndarray],
                            y_val: Union[pd.Series, np.ndarray],
                            min_recall: float = 0.70,
                            min_precision: float = 0.30,
                            verbose: bool = True) -> Tuple[float, Dict]:
    """
    Find optimal threshold to achieve target recall while maintaining precision.
    
    CRITICAL FIX: Default 0.5 threshold causes 20.9% recall (catastrophic for risk assessment).
    Optimal threshold is typically 0.15-0.30 for imbalanced credit risk data.
    
    Args:
        model: Trained XGBoost model with predict_proba method
        X_val: Validation features
        y_val: Validation targets
        min_recall: Minimum acceptable recall (default: 70%)
        min_precision: Minimum acceptable precision (default: 30%)
        verbose: Print detailed results
        
    Returns:
        Tuple of (optimal_threshold, metrics_dict)
    """
    
    if verbose:
        logger.info("=" * 70)
        logger.info("THRESHOLD OPTIMIZATION - Fixing Low Recall Issue")
        logger.info("=" * 70)
    
    # Get probabilities
    y_proba = model.predict_proba(X_val)[:, 1]
    
    # Test thresholds from 0.05 to 0.9
    thresholds = np.linspace(0.05, 0.9, 200)  # Fine-grained search
    
    best_threshold = 0.5
    best_f1 = 0
    best_metrics = {}
    all_results = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        cm = confusion_matrix(y_val, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        result = {
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        }
        all_results.append(result)
        
        # Check if meets minimum requirements
        if recall >= min_recall and precision >= min_precision:
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = result
    
    # If no threshold meets strict requirements, find best recall with acceptable precision
    if best_f1 == 0:
        if verbose:
            logger.warning(f"⚠️ Could not find threshold meeting min_recall={min_recall:.1%} "
                         f"and min_precision={min_precision:.1%}")
            logger.info("Finding best available threshold...")
        
        for result in sorted(all_results, key=lambda x: x['recall'], reverse=True):
            if result['precision'] >= min_precision * 0.8:  # Relax precision slightly
                best_threshold = result['threshold']
                best_metrics = result
                best_f1 = result['f1']
                if verbose:
                    logger.info(f"Using relaxed precision requirement: {min_precision*0.8:.1%}")
                break
    
    # If still no solution, find best F1
    if best_f1 == 0:
        best_result = max(all_results, key=lambda x: x['f1'])
        best_threshold = best_result['threshold']
        best_metrics = best_result
        if verbose:
            logger.warning("Using threshold with best F1 score (no constraints met)")
    
    # Print detailed results
    if verbose:
        logger.info("\\n" + "=" * 70)
        logger.info("OPTIMIZATION RESULTS")
        logger.info("=" * 70)
        logger.info(f"\\n✅ Optimal Threshold: {best_threshold:.4f}")
        logger.info(f"\\n📊 Performance Metrics:")
        logger.info(f"   Precision: {best_metrics['precision']:.4f} ({best_metrics['precision']:.1%})")
        logger.info(f"   Recall:    {best_metrics['recall']:.4f} ({best_metrics['recall']:.1%})")
        logger.info(f"   F1-Score:  {best_metrics['f1']:.4f}")
        logger.info(f"\\n📈 Confusion Matrix:")
        logger.info(f"   True Positives:  {best_metrics['tp']: >6} (Correctly identified defaults)")
        logger.info(f"   False Positives: {best_metrics['fp']: >6} (False alarms)")
        logger.info(f"   True Negatives:  {best_metrics['tn']: >6} (Correctly identified non-defaults)")
        logger.info(f"   False Negatives: {best_metrics['fn']: >6} (Missed defaults - CRITICAL)")
        
        # Calculate improvement
        y_pred_default = (y_proba >= 0.5).astype(int)
        default_recall = recall_score(y_val, y_pred_default)
        improvement = (best_metrics['recall'] - default_recall) / default_recall * 100
        
        logger.info(f"\\n🎯 Improvement vs Default Threshold (0.5):")
        logger.info(f"   Default threshold recall: {default_recall:.1%}")
        logger.info(f"   Optimized threshold recall: {best_metrics['recall']:.1%}")
        logger.info(f"   Improvement: {improvement:+.1f}%")
        logger.info("=" * 70)
    
    return best_threshold, best_metrics


def apply_threshold(model, X, threshold: float = 0.5) -> np.ndarray:
    """
    Apply custom threshold to model predictions.
    
    Args:
        model: Trained model with predict_proba
        X: Features
        threshold: Decision threshold
        
    Returns:
        Binary predictions
    """
    y_proba = model.predict_proba(X)[:, 1]
    return (y_proba >= threshold).astype(int)


def evaluate_thresholds_range(model,
                              X_val: Union[pd.DataFrame, np.ndarray],
                              y_val: Union[pd.Series, np.ndarray],
                              thresholds: np.ndarray = None) -> pd.DataFrame:
    """
    Evaluate model performance across range of thresholds.
    
    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation targets
        thresholds: Array of thresholds to test
        
    Returns:
        DataFrame with metrics for each threshold
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 100)
    
    y_proba = model.predict_proba(X_val)[:, 1]
    
    results = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("\\n" + "=" * 70)
    print("Threshold Optimizer - Fixing 20.9% Recall Issue")
    print("=" * 70)
    print("\\nThis module optimizes decision thresholds to fix low recall.")
    print("\\nDefault threshold (0.5) causes catastrophic miss rate for defaults.")
    print("Optimal threshold for credit risk is typically 0.15-0.30.")
    print("\\nUsage:")
    print("  from threshold_optimizer import optimize_model_threshold")
    print("  threshold, metrics = optimize_model_threshold(model, X_val, y_val)")
    print("=" * 70)
