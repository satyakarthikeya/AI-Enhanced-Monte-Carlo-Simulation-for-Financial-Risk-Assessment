#!/usr/bin/env python3
"""
Hyperparameter Tuning Module for Credit Risk Assessment
=======================================================

This module provides comprehensive hyperparameter tuning capabilities for XGBoost
models using various optimization strategies including RandomizedSearch, GridSearch,
and custom optimization approaches.

Classes:
- HyperparameterTuner: Main hyperparameter tuning class
- TuningConfig: Configuration class for tuning parameters
- OptimizationStrategy: Enum for different optimization approaches
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from enum import Enum
from sklearn.model_selection import (
    RandomizedSearchCV, GridSearchCV, cross_val_score,
    StratifiedKFold, train_test_split
)
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, make_scorer
import xgboost as xgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class OptimizationStrategy(Enum):
    """Enumeration of optimization strategies."""
    RANDOMIZED = "randomized"
    GRID = "grid"
    BAYESIAN = "bayesian"
    CUSTOM = "custom"


class TuningConfig:
    """Configuration class for hyperparameter tuning."""
    
    def __init__(self):
        """Initialize tuning configuration."""
        self.param_distributions = {
            'n_estimators': [500, 800, 1200, 1500, 2000],
            'max_depth': stats.randint(3, 12),
            'learning_rate': stats.uniform(0.01, 0.15),
            'reg_alpha': stats.uniform(0.0, 2.0),
            'reg_lambda': stats.uniform(0.5, 3.0),
            'gamma': stats.uniform(0.0, 0.2),
            'subsample': stats.uniform(0.7, 0.3),
            'colsample_bytree': stats.uniform(0.6, 0.4),
            'colsample_bylevel': stats.uniform(0.7, 0.3),
            'min_child_weight': stats.randint(1, 8),
            'max_delta_step': stats.randint(0, 5)
        }
        
        self.grid_params = {
            'n_estimators': [800, 1200, 1500],
            'max_depth': [4, 6, 8, 10],
            'learning_rate': [0.01, 0.02, 0.05, 0.1],
            'reg_alpha': [0.0, 0.3, 0.7, 1.5],
            'reg_lambda': [0.5, 1.0, 1.8, 2.5],
            'gamma': [0.0, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'min_child_weight': [1, 2, 4]
        }
        
        self.quick_params = {
            'n_estimators': [500, 1000, 1500],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.02, 0.05, 0.1],
            'reg_alpha': [0.0, 0.5, 1.0],
            'reg_lambda': [1.0, 1.5, 2.0]
        }


class HyperparameterTuner:
    """Comprehensive hyperparameter tuning class for XGBoost models."""
    
    def __init__(self, 
                 random_state: int = 42,
                 n_jobs: int = -1,
                 cv_folds: int = 5):
        """
        Initialize hyperparameter tuner.
        
        Args:
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs
            cv_folds: Number of cross-validation folds
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.cv_folds = cv_folds
        self.logger = logging.getLogger(__name__)
        
        # Initialize configuration
        self.config = TuningConfig()
        
        # Tuning results
        self.best_params = None
        self.best_score = None
        self.best_estimator = None
        self.tuning_history = {}
        self.cv_results = None
        
        # Cross-validation setup
        self.cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    def _create_base_model(self, scale_pos_weight: Optional[float] = None) -> xgb.XGBClassifier:
        """
        Create base XGBoost model with fixed parameters.
        
        Args:
            scale_pos_weight: Class balance weight
            
        Returns:
            Base XGBoost classifier
        """
        base_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
            'verbosity': 0,
            'enable_categorical': False
        }
        
        if scale_pos_weight is not None:
            base_params['scale_pos_weight'] = scale_pos_weight
        
        return xgb.XGBClassifier(**base_params)
    
    def _calculate_scale_pos_weight(self, y: Union[pd.Series, np.ndarray]) -> float:
        """
        Calculate scale_pos_weight for class imbalance.
        
        Args:
            y: Target values
            
        Returns:
            Scale positive weight value
        """
        if isinstance(y, pd.Series):
            y = y.values
        
        neg_count = np.sum(y == 0)
        pos_count = np.sum(y == 1)
        
        if pos_count == 0:
            return 1.0
        
        return neg_count / pos_count
    
    def randomized_search(self,
                         X_train: Union[pd.DataFrame, np.ndarray],
                         y_train: Union[pd.Series, np.ndarray],
                         n_iter: int = 100,
                         scoring: str = 'roc_auc',
                         param_distributions: Optional[Dict[str, Any]] = None,
                         verbose: int = 1) -> Dict[str, Any]:
        """
        Perform randomized hyperparameter search.
        
        Args:
            X_train: Training features
            y_train: Training targets
            n_iter: Number of parameter combinations to try
            scoring: Scoring metric
            param_distributions: Custom parameter distributions
            verbose: Verbosity level
            
        Returns:
            Dictionary containing tuning results
        """
        self.logger.info(f"Starting randomized search with {n_iter} iterations...")
        
        # Calculate class balance
        scale_pos_weight = self._calculate_scale_pos_weight(y_train)
        
        # Create base model
        base_model = self._create_base_model(scale_pos_weight=scale_pos_weight)
        
        # Use provided or default parameter distributions
        params = param_distributions if param_distributions else self.config.param_distributions
        
        # Setup RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=params,
            n_iter=n_iter,
            cv=self.cv,
            scoring=scoring,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=verbose,
            return_train_score=True
        )
        
        # Perform search
        import time
        start_time = time.time()
        random_search.fit(X_train, y_train)
        search_time = time.time() - start_time
        
        # Store results
        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_
        self.best_estimator = random_search.best_estimator_
        self.cv_results = random_search.cv_results_
        
        results = {
            'strategy': 'randomized',
            'best_params': self.best_params,
            'best_score': self.best_score,
            'search_time': search_time,
            'n_iter': n_iter,
            'scoring': scoring,
            'scale_pos_weight': scale_pos_weight
        }
        
        self.tuning_history['randomized'] = results
        
        self.logger.info(f"✅ Randomized search completed in {search_time:.2f} seconds")
        self.logger.info(f"✓ Best {scoring}: {self.best_score:.4f}")
        self.logger.info(f"✓ Best parameters: {self.best_params}")
        
        return results
    
    def grid_search(self,
                   X_train: Union[pd.DataFrame, np.ndarray],
                   y_train: Union[pd.Series, np.ndarray],
                   param_grid: Optional[Dict[str, List]] = None,
                   scoring: str = 'roc_auc',
                   use_quick_params: bool = False,
                   verbose: int = 1) -> Dict[str, Any]:
        """
        Perform grid search hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training targets
            param_grid: Custom parameter grid
            scoring: Scoring metric
            use_quick_params: Whether to use quick parameter set
            verbose: Verbosity level
            
        Returns:
            Dictionary containing tuning results
        """
        self.logger.info("Starting grid search...")
        
        # Calculate class balance
        scale_pos_weight = self._calculate_scale_pos_weight(y_train)
        
        # Create base model
        base_model = self._create_base_model(scale_pos_weight=scale_pos_weight)
        
        # Select parameter grid
        if param_grid is not None:
            params = param_grid
        elif use_quick_params:
            params = self.config.quick_params
            self.logger.info("Using quick parameter grid")
        else:
            params = self.config.grid_params
            self.logger.info("Using full parameter grid")
        
        # Calculate total combinations
        total_combinations = np.prod([len(v) for v in params.values()])
        self.logger.info(f"Testing {total_combinations} parameter combinations")
        
        # Setup GridSearchCV
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=params,
            cv=self.cv,
            scoring=scoring,
            n_jobs=self.n_jobs,
            verbose=verbose,
            return_train_score=True
        )
        
        # Perform search
        import time
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        search_time = time.time() - start_time
        
        # Store results
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        self.best_estimator = grid_search.best_estimator_
        self.cv_results = grid_search.cv_results_
        
        results = {
            'strategy': 'grid',
            'best_params': self.best_params,
            'best_score': self.best_score,
            'search_time': search_time,
            'total_combinations': total_combinations,
            'scoring': scoring,
            'scale_pos_weight': scale_pos_weight
        }
        
        self.tuning_history['grid'] = results
        
        self.logger.info(f"✅ Grid search completed in {search_time:.2f} seconds")
        self.logger.info(f"✓ Best {scoring}: {self.best_score:.4f}")
        self.logger.info(f"✓ Best parameters: {self.best_params}")
        
        return results
    
    def bayesian_optimization(self,
                             X_train: Union[pd.DataFrame, np.ndarray],
                             y_train: Union[pd.Series, np.ndarray],
                             n_calls: int = 50,
                             scoring: str = 'roc_auc') -> Dict[str, Any]:
        """
        Perform Bayesian optimization (requires scikit-optimize).
        
        Args:
            X_train: Training features
            y_train: Training targets
            n_calls: Number of optimization calls
            scoring: Scoring metric
            
        Returns:
            Dictionary containing tuning results
        """
        try:
            from skopt import gp_minimize, space
            from skopt.utils import use_named_args
        except ImportError:
            raise ImportError("scikit-optimize required for Bayesian optimization. Install with: pip install scikit-optimize")
        
        self.logger.info(f"Starting Bayesian optimization with {n_calls} calls...")
        
        # Calculate class balance
        scale_pos_weight = self._calculate_scale_pos_weight(y_train)
        
        # Define search space
        search_space = [
            space.Integer(500, 2000, name='n_estimators'),
            space.Integer(3, 12, name='max_depth'),
            space.Real(0.01, 0.2, name='learning_rate'),
            space.Real(0.0, 2.0, name='reg_alpha'),
            space.Real(0.5, 3.0, name='reg_lambda'),
            space.Real(0.0, 0.3, name='gamma'),
            space.Real(0.7, 1.0, name='subsample'),
            space.Real(0.6, 1.0, name='colsample_bytree'),
            space.Integer(1, 8, name='min_child_weight')
        ]
        
        # Define objective function
        @use_named_args(search_space)
        def objective(**params):
            model = self._create_base_model(scale_pos_weight=scale_pos_weight)
            model.set_params(**params)
            
            # Cross-validation score
            scores = cross_val_score(model, X_train, y_train, cv=self.cv, scoring=scoring, n_jobs=self.n_jobs)
            return -scores.mean()  # Minimize negative score
        
        # Perform optimization
        import time
        start_time = time.time()
        
        result = gp_minimize(objective, search_space, n_calls=n_calls, random_state=self.random_state)
        
        search_time = time.time() - start_time
        
        # Extract best parameters
        param_names = [dim.name for dim in search_space]
        best_params = dict(zip(param_names, result.x))
        best_score = -result.fun  # Convert back to positive
        
        # Train best model
        best_model = self._create_base_model(scale_pos_weight=scale_pos_weight)
        best_model.set_params(**best_params)
        best_model.fit(X_train, y_train)
        
        # Store results
        self.best_params = best_params
        self.best_score = best_score
        self.best_estimator = best_model
        
        results = {
            'strategy': 'bayesian',
            'best_params': best_params,
            'best_score': best_score,
            'search_time': search_time,
            'n_calls': n_calls,
            'scoring': scoring,
            'scale_pos_weight': scale_pos_weight
        }
        
        self.tuning_history['bayesian'] = results
        
        self.logger.info(f"✅ Bayesian optimization completed in {search_time:.2f} seconds")
        self.logger.info(f"✓ Best {scoring}: {best_score:.4f}")
        self.logger.info(f"✓ Best parameters: {best_params}")
        
        return results
    
    def progressive_tuning(self,
                          X_train: Union[pd.DataFrame, np.ndarray],
                          y_train: Union[pd.Series, np.ndarray],
                          stages: Optional[List[Dict[str, Any]]] = None,
                          scoring: str = 'roc_auc') -> Dict[str, Any]:
        """
        Perform progressive hyperparameter tuning with multiple stages.
        
        Args:
            X_train: Training features
            y_train: Training targets
            stages: List of tuning stages with parameters
            scoring: Scoring metric
            
        Returns:
            Dictionary containing tuning results
        """
        self.logger.info("Starting progressive hyperparameter tuning...")
        
        # Default stages if none provided
        if stages is None:
            stages = [
                {
                    'name': 'coarse',
                    'method': 'randomized',
                    'params': {
                        'n_estimators': [300, 500, 800, 1200],
                        'max_depth': stats.randint(3, 10),
                        'learning_rate': stats.uniform(0.02, 0.15)
                    },
                    'n_iter': 30
                },
                {
                    'name': 'fine',
                    'method': 'grid',
                    'params': None,  # Will be derived from coarse stage
                    'refinement_factor': 0.3
                }
            ]
        
        all_results = {}
        current_best = None
        
        for i, stage in enumerate(stages):
            self.logger.info(f"Stage {i+1}: {stage['name']}")
            
            # Prepare parameters for this stage
            if stage.get('params') is None and current_best is not None:
                # Refine parameters around current best
                stage_params = self._refine_parameters(current_best, stage.get('refinement_factor', 0.3))
            else:
                stage_params = stage.get('params', self.config.param_distributions)
            
            # Run tuning for this stage
            if stage['method'] == 'randomized':
                stage_results = self.randomized_search(
                    X_train, y_train, 
                    n_iter=stage.get('n_iter', 50),
                    scoring=scoring,
                    param_distributions=stage_params
                )
            elif stage['method'] == 'grid':
                stage_results = self.grid_search(
                    X_train, y_train,
                    param_grid=stage_params,
                    scoring=scoring
                )
            
            all_results[stage['name']] = stage_results
            current_best = self.best_params
        
        results = {
            'strategy': 'progressive',
            'stages': all_results,
            'final_best_params': self.best_params,
            'final_best_score': self.best_score,
            'scoring': scoring
        }
        
        self.tuning_history['progressive'] = results
        
        self.logger.info(f"✅ Progressive tuning completed")
        self.logger.info(f"✓ Final best {scoring}: {self.best_score:.4f}")
        
        return results
    
    def _refine_parameters(self, best_params: Dict[str, Any], factor: float = 0.3) -> Dict[str, List]:
        """
        Refine parameter ranges around best values.
        
        Args:
            best_params: Best parameters from previous stage
            factor: Refinement factor (proportion of range to explore)
            
        Returns:
            Refined parameter grid
        """
        refined_params = {}
        
        for param, value in best_params.items():
            if param == 'n_estimators':
                # Integer parameter
                range_size = int(value * factor)
                refined_params[param] = list(range(
                    max(100, value - range_size),
                    value + range_size + 1,
                    max(50, range_size // 3)
                ))
            elif param == 'max_depth':
                # Integer parameter with bounds
                refined_params[param] = list(range(
                    max(3, value - 2),
                    min(15, value + 3)
                ))
            elif isinstance(value, (int, float)):
                # Continuous parameter
                range_size = value * factor
                refined_params[param] = [
                    max(0.001, value - range_size),
                    value,
                    value + range_size
                ]
        
        return refined_params
    
    def evaluate_best_model(self,
                           X_test: Union[pd.DataFrame, np.ndarray],
                           y_test: Union[pd.Series, np.ndarray],
                           threshold_optimization: bool = True) -> Dict[str, Any]:
        """
        Evaluate the best model on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            threshold_optimization: Whether to optimize threshold
            
        Returns:
            Dictionary containing evaluation results
        """
        if self.best_estimator is None:
            raise ValueError("No best model found. Run tuning first.")
        
        self.logger.info("Evaluating best model on test data...")
        
        # Make predictions
        y_pred = self.best_estimator.predict(X_test)
        y_proba = self.best_estimator.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'auc': auc,
            'f1': f1,
            'threshold': 0.5
        }
        
        # Threshold optimization
        if threshold_optimization:
            best_threshold = self._optimize_threshold(y_test, y_proba, metric='accuracy')
            y_pred_optimized = (y_proba >= best_threshold).astype(int)
            
            optimized_accuracy = accuracy_score(y_test, y_pred_optimized)
            optimized_f1 = f1_score(y_test, y_pred_optimized)
            
            results.update({
                'optimized_threshold': best_threshold,
                'optimized_accuracy': optimized_accuracy,
                'optimized_f1': optimized_f1
            })
            
            self.logger.info(f"✓ Optimized threshold: {best_threshold:.3f}")
            self.logger.info(f"✓ Optimized accuracy: {optimized_accuracy:.4f}")
        
        self.logger.info(f"✓ Test accuracy: {accuracy:.4f}")
        self.logger.info(f"✓ Test AUC: {auc:.4f}")
        self.logger.info(f"✓ Test F1: {f1:.4f}")
        
        return results
    
    def _optimize_threshold(self, y_true: np.ndarray, y_proba: np.ndarray, 
                           metric: str = 'accuracy') -> float:
        """
        Optimize decision threshold.
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            metric: Optimization metric
            
        Returns:
            Optimal threshold
        """
        thresholds = np.linspace(0.1, 0.8, 70)
        best_score = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            if metric == 'accuracy':
                score = accuracy_score(y_true, y_pred)
            elif metric == 'f1':
                score = f1_score(y_true, y_pred)
            else:
                continue
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold
    
    def get_tuning_summary(self) -> pd.DataFrame:
        """
        Get summary of all tuning attempts.
        
        Returns:
            DataFrame with tuning results summary
        """
        if not self.tuning_history:
            return pd.DataFrame()
        
        summary_data = []
        for strategy, results in self.tuning_history.items():
            summary_data.append({
                'strategy': strategy,
                'best_score': results['best_score'],
                'search_time': results.get('search_time', 0),
                'n_iterations': results.get('n_iter', results.get('total_combinations', 0))
            })
        
        return pd.DataFrame(summary_data).sort_values('best_score', ascending=False)


# Utility functions
def quick_tune_xgboost(X_train: Union[pd.DataFrame, np.ndarray],
                      y_train: Union[pd.Series, np.ndarray],
                      strategy: str = 'randomized',
                      n_iter: int = 50,
                      scoring: str = 'roc_auc') -> HyperparameterTuner:
    """
    Quick hyperparameter tuning for XGBoost.
    
    Args:
        X_train: Training features
        y_train: Training targets
        strategy: Tuning strategy ('randomized', 'grid', 'progressive')
        n_iter: Number of iterations
        scoring: Scoring metric
        
    Returns:
        Trained HyperparameterTuner instance
    """
    tuner = HyperparameterTuner()
    
    if strategy == 'randomized':
        tuner.randomized_search(X_train, y_train, n_iter=n_iter, scoring=scoring)
    elif strategy == 'grid':
        tuner.grid_search(X_train, y_train, scoring=scoring, use_quick_params=True)
    elif strategy == 'progressive':
        tuner.progressive_tuning(X_train, y_train, scoring=scoring)
    
    return tuner


if __name__ == "__main__":
    # Example usage
    import logging
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    # Create sample data
    X, y = make_classification(n_samples=2000, n_features=20, n_classes=2, 
                              n_informative=15, n_redundant=3, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Quick tuning example
    tuner = quick_tune_xgboost(X_train, y_train, strategy='randomized', n_iter=30)
    
    # Evaluate best model
    results = tuner.evaluate_best_model(X_test, y_test)
    
    # Get summary
    summary = tuner.get_tuning_summary()
    print("\n📊 Tuning Summary:")
    print("=" * 50)
    print(summary)