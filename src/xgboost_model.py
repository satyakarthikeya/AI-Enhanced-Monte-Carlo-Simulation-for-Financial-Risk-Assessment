#!/usr/bin/env python3
"""
XGBoost Model Module for Credit Risk Assessment
================================================

This module handles XGBoost model creation, training, and prediction operations
for the credit risk assessment system.

Classes:
- XGBoostModel: Main XGBoost model class with training and prediction
- ModelConfig: Configuration class for XGBoost parameters
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Tuple, Optional, Union
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb

class ModelConfig:
    """Configuration class for XGBoost parameters."""
    
    def __init__(self):
        """Initialize default XGBoost configuration."""
        self.default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'n_jobs': -1,
            'random_state': 42,
            'verbosity': 0,
            'enable_categorical': False
        }
        
        self.optimized_params = {
            'n_estimators': 1500,
            'max_depth': 6,
            'learning_rate': 0.02,
            'reg_alpha': 0.3,
            'reg_lambda': 1.8,
            'gamma': 0.05,
            'subsample': 0.9,
            'colsample_bytree': 0.85,
            'colsample_bylevel': 0.9,
            'min_child_weight': 2,
            'max_delta_step': 1
        }
    
    def get_base_config(self, use_class_weights: bool = True, scale_pos_weight: float = None) -> Dict[str, Any]:
        """
        Get base XGBoost configuration.
        
        Args:
            use_class_weights: Whether to use class weighting
            scale_pos_weight: Custom scale_pos_weight value
            
        Returns:
            Dictionary of XGBoost parameters
        """
        config = self.default_params.copy()
        
        if use_class_weights and scale_pos_weight is not None:
            config['scale_pos_weight'] = scale_pos_weight
        
        return config
    
    def get_optimized_config(self, use_class_weights: bool = True, scale_pos_weight: float = None) -> Dict[str, Any]:
        """
        Get optimized XGBoost configuration.
        
        Args:
            use_class_weights: Whether to use class weighting
            scale_pos_weight: Custom scale_pos_weight value
            
        Returns:
            Dictionary of optimized XGBoost parameters
        """
        config = {**self.default_params, **self.optimized_params}
        
        if use_class_weights and scale_pos_weight is not None:
            config['scale_pos_weight'] = scale_pos_weight
        
        return config


class XGBoostModel:
    """XGBoost model class for credit risk assessment."""
    
    def __init__(self, config: Optional[ModelConfig] = None, random_state: int = 42):
        """
        Initialize XGBoost model.
        
        Args:
            config: Model configuration instance
            random_state: Random seed for reproducibility
        """
        self.config = config if config is not None else ModelConfig()
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        
        # Model components
        self.model = None
        self.is_trained = False
        self.optimal_threshold = 0.5
        self.feature_names = None
        self.class_weights = None
        self.scale_pos_weight = None
        
        # Training history
        self.training_history = {}
        self.validation_scores = {}
    
    def calculate_class_weights(self, y_train: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """
        Calculate class weights for imbalanced dataset.
        
        Args:
            y_train: Training target values
            
        Returns:
            Dictionary containing class weight information
        """
        self.logger.info("Calculating class weights for imbalanced dataset...")
        
        # Convert to numpy array if pandas Series
        if isinstance(y_train, pd.Series):
            y_array = y_train.values
        else:
            y_array = y_train
        
        # Calculate balanced class weights
        classes = np.unique(y_array)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_array)
        
        # Create class weight dictionary
        self.class_weights = dict(zip(classes, class_weights))
        
        # Calculate scale_pos_weight (for XGBoost)
        self.scale_pos_weight = self.class_weights[0] / self.class_weights[1]
        
        self.logger.info(f"✓ Class weights calculated: {self.class_weights}")
        self.logger.info(f"✓ Scale pos weight: {self.scale_pos_weight:.4f}")
        
        return {
            'class_weights': self.class_weights,
            'scale_pos_weight': self.scale_pos_weight
        }
    
    def create_model(self, 
                    params: Optional[Dict[str, Any]] = None,
                    use_optimized: bool = True,
                    y_train: Optional[Union[pd.Series, np.ndarray]] = None) -> xgb.XGBClassifier:
        """
        Create XGBoost model with specified parameters.
        
        Args:
            params: Custom parameters (overrides default/optimized)
            use_optimized: Whether to use optimized parameters
            y_train: Training target for class weight calculation
            
        Returns:
            Configured XGBoost classifier
        """
        self.logger.info("Creating XGBoost model...")
        
        # Calculate class weights if training data provided
        if y_train is not None:
            self.calculate_class_weights(y_train)
        
        # Get base configuration
        if params is not None:
            # Use custom parameters
            model_params = params.copy()
            self.logger.info("Using custom parameters")
        elif use_optimized:
            # Use optimized parameters
            model_params = self.config.get_optimized_config(
                use_class_weights=True, 
                scale_pos_weight=self.scale_pos_weight
            )
            self.logger.info("Using optimized parameters")
        else:
            # Use base parameters
            model_params = self.config.get_base_config(
                use_class_weights=True,
                scale_pos_weight=self.scale_pos_weight
            )
            self.logger.info("Using base parameters")
        
        # Create model
        self.model = xgb.XGBClassifier(**model_params)
        
        # Log key parameters
        self.logger.info(f"✓ Model created with key parameters:")
        self.logger.info(f"   • n_estimators: {model_params.get('n_estimators', 100)}")
        self.logger.info(f"   • max_depth: {model_params.get('max_depth', 6)}")
        self.logger.info(f"   • learning_rate: {model_params.get('learning_rate', 0.3)}")
        self.logger.info(f"   • scale_pos_weight: {model_params.get('scale_pos_weight', 1.0):.4f}")
        
        return self.model
    
    def train(self,
              X_train: Union[pd.DataFrame, np.ndarray],
              y_train: Union[pd.Series, np.ndarray],
              X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
              y_val: Optional[Union[pd.Series, np.ndarray]] = None,
              verbose: bool = False) -> Dict[str, Any]:
        """
        Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            verbose: Whether to show training progress
            
        Returns:
            Dictionary containing training results
        """
        if self.model is None:
            # Create model if not already created
            self.create_model(y_train=y_train)
        
        self.logger.info("Starting XGBoost model training...")
        
        # Store feature names
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)
        
        # Prepare evaluation set
        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        
        # Train model
        import time
        start_time = time.time()
        
        if eval_set is not None:
            self.model.fit(X_train, y_train, eval_set=eval_set, verbose=verbose)
        else:
            self.model.fit(X_train, y_train, verbose=verbose)
        
        training_time = time.time() - start_time
        self.is_trained = True
        
        # Calculate training metrics
        train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        
        results = {
            'training_time': training_time,
            'train_accuracy': train_accuracy,
            'n_features': X_train.shape[1],
            'n_samples': X_train.shape[0]
        }
        
        # Calculate validation metrics if available
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_proba = self.model.predict_proba(X_val)[:, 1]
            
            val_accuracy = accuracy_score(y_val, val_pred)
            val_auc = roc_auc_score(y_val, val_proba)
            
            results.update({
                'val_accuracy': val_accuracy,
                'val_auc': val_auc
            })
            
            self.validation_scores = {
                'accuracy': val_accuracy,
                'auc': val_auc
            }
            
            self.logger.info(f"✓ Validation accuracy: {val_accuracy:.4f}")
            self.logger.info(f"✓ Validation AUC: {val_auc:.4f}")
        
        # Store training history
        self.training_history = results
        
        self.logger.info(f"✅ Training completed in {training_time:.2f} seconds")
        self.logger.info(f"✓ Training accuracy: {train_accuracy:.4f}")
        
        return results
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray], 
                use_threshold: bool = False, 
                threshold: Optional[float] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions with the trained model.
        
        Args:
            X: Input features
            use_threshold: Whether to use custom threshold
            threshold: Custom threshold value
            
        Returns:
            Predictions (and probabilities if use_threshold=True)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if use_threshold:
            # Return both predictions and probabilities
            probabilities = self.model.predict_proba(X)[:, 1]
            threshold_val = threshold if threshold is not None else self.optimal_threshold
            predictions = (probabilities >= threshold_val).astype(int)
            return predictions, probabilities
        else:
            # Return standard predictions
            return self.model.predict(X)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, importance_type: str = 'weight') -> pd.DataFrame:
        """
        Get feature importance from the trained model.
        
        Args:
            importance_type: Type of importance ('weight', 'gain', 'cover')
            
        Returns:
            DataFrame with feature importance
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        if importance_type == 'weight':
            importance = self.model.feature_importances_
        else:
            # Get importance by type
            importance = self.model.get_booster().get_score(importance_type=importance_type)
            # Convert to array matching feature order
            if self.feature_names:
                importance = [importance.get(f, 0.0) for f in self.feature_names]
            else:
                importance = list(importance.values())
        
        # Create DataFrame
        feature_names = self.feature_names if self.feature_names else [f'feature_{i}' for i in range(len(importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def optimize_threshold(self, 
                          X_val: Union[pd.DataFrame, np.ndarray],
                          y_val: Union[pd.Series, np.ndarray],
                          metric: str = 'accuracy',
                          threshold_range: Tuple[float, float] = (0.1, 0.8),
                          n_thresholds: int = 70) -> Dict[str, Any]:
        """
        Optimize decision threshold for better performance.
        
        Args:
            X_val: Validation features
            y_val: Validation targets
            metric: Optimization metric ('accuracy', 'f1', 'precision', 'recall')
            threshold_range: Range of thresholds to test
            n_thresholds: Number of thresholds to test
            
        Returns:
            Dictionary containing optimization results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before threshold optimization")
        
        self.logger.info(f"Optimizing threshold using {metric} metric...")
        
        # Get probabilities
        y_proba = self.predict_proba(X_val)[:, 1]
        
        # Test thresholds
        thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)
        results = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            if metric == 'accuracy':
                from sklearn.metrics import accuracy_score
                score = accuracy_score(y_val, y_pred)
            elif metric == 'f1':
                from sklearn.metrics import f1_score
                score = f1_score(y_val, y_pred)
            elif metric == 'precision':
                from sklearn.metrics import precision_score
                score = precision_score(y_val, y_pred)
            elif metric == 'recall':
                from sklearn.metrics import recall_score
                score = recall_score(y_val, y_pred)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            
            results.append((threshold, score))
        
        # Find best threshold
        best_threshold, best_score = max(results, key=lambda x: x[1])
        self.optimal_threshold = best_threshold
        
        self.logger.info(f"✓ Optimal threshold: {best_threshold:.3f}")
        self.logger.info(f"✓ Best {metric}: {best_score:.4f}")
        
        return {
            'optimal_threshold': best_threshold,
            'best_score': best_score,
            'metric': metric,
            'all_results': results
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary containing model information
        """
        if not self.is_trained:
            return {'status': 'not_trained'}
        
        info = {
            'status': 'trained',
            'model_type': 'XGBClassifier',
            'is_trained': self.is_trained,
            'optimal_threshold': self.optimal_threshold,
            'n_features': len(self.feature_names) if self.feature_names else None,
            'feature_names': self.feature_names,
            'class_weights': self.class_weights,
            'scale_pos_weight': self.scale_pos_weight,
            'training_history': self.training_history,
            'validation_scores': self.validation_scores
        }
        
        # Add model parameters
        if self.model is not None:
            info['model_params'] = self.model.get_params()
        
        return info


# Utility functions for standalone usage
def create_and_train_xgboost(X_train: Union[pd.DataFrame, np.ndarray],
                             y_train: Union[pd.Series, np.ndarray],
                             X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
                             y_val: Optional[Union[pd.Series, np.ndarray]] = None,
                             use_optimized: bool = True,
                             optimize_threshold: bool = True) -> XGBoostModel:
    """
    Convenience function to create and train XGBoost model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        use_optimized: Whether to use optimized parameters
        optimize_threshold: Whether to optimize decision threshold
        
    Returns:
        Trained XGBoostModel instance
    """
    # Create model
    model = XGBoostModel()
    model.create_model(use_optimized=use_optimized, y_train=y_train)
    
    # Train model
    model.train(X_train, y_train, X_val, y_val)
    
    # Optimize threshold if validation data available
    if optimize_threshold and X_val is not None and y_val is not None:
        model.optimize_threshold(X_val, y_val)
    
    return model


if __name__ == "__main__":
    # Example usage
    import logging
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    # Create sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, 
                              n_informative=10, n_redundant=5, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Create and train model
    model = create_and_train_xgboost(X_train, y_train, X_val, y_val)
    
    # Make predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    # Get model info
    info = model.get_model_info()
    print("\n📋 Model Information:")
    print("=" * 50)
    print(f"Training accuracy: {info['training_history']['train_accuracy']:.4f}")
    print(f"Validation accuracy: {info['validation_scores']['accuracy']:.4f}")
    print(f"Optimal threshold: {info['optimal_threshold']:.3f}")