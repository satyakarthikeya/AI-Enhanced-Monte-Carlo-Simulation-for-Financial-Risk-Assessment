#!/usr/bin/env python3
"""
Main Training Script for Credit Risk Assessment
===============================================

This script orchestrates all modules to provide a clean interface for the 
complete XGBoost training pipeline including data loading, model training,
hyperparameter tuning, and comprehensive evaluation.

Usage:
    python main_training.py --mode train --data-path processed_data/
    python main_training.py --mode tune --strategy randomized --n-iter 100
    python main_training.py --mode evaluate --model-path artifacts/models/best_model.pkl
"""

import os
import sys
import argparse
import logging
import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

# Import our modular components
from data_loader import DataLoader
from xgboost_model import XGBoostModel, ModelConfig
from hyperparameter_tuner import HyperparameterTuner, OptimizationStrategy
from evaluator import ModelEvaluator

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class CreditRiskTrainer:
    """Main trainer class that orchestrates all modules."""
    
    def __init__(self, 
                 data_path: str = "processed_data/",
                 artifacts_path: str = "artifacts/",
                 config_path: str = "configs/",
                 random_state: int = 42):
        """
        Initialize the credit risk trainer.
        
        Args:
            data_path: Path to preprocessed data files
            artifacts_path: Path to save artifacts
            config_path: Path to configuration files
            random_state: Random seed for reproducibility
        """
        self.data_path = Path(data_path)
        self.artifacts_path = Path(artifacts_path)
        self.config_path = Path(config_path)
        self.random_state = random_state
        
        # Create directories if they don't exist
        self.artifacts_path.mkdir(exist_ok=True)
        (self.artifacts_path / "models").mkdir(exist_ok=True)
        (self.artifacts_path / "metrics").mkdir(exist_ok=True)
        (self.artifacts_path / "visualizations").mkdir(exist_ok=True)
        
        # Initialize logger
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_loader = None
        self.model = None
        self.tuner = None
        self.evaluator = None
        
        # Training data storage
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        # Results storage
        self.training_results = {}
        self.evaluation_results = {}
        
        self.logger.info("🚀 Credit Risk Trainer initialized")
        self.logger.info(f"   • Data path: {self.data_path}")
        self.logger.info(f"   • Artifacts path: {self.artifacts_path}")
        self.logger.info(f"   • Random state: {self.random_state}")
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = f"credit_risk_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def load_data(self, 
                  train_file: str = "train/train.parquet",
                  test_file: str = "test/test.parquet",
                  val_split: float = 0.2) -> Dict[str, Any]:
        """
        Load and prepare data using DataLoader.
        
        Args:
            train_file: Training data file name (not used - for compatibility)
            test_file: Test data file name (not used - for compatibility)
            val_split: Validation split ratio
            
        Returns:
            Dictionary containing data loading results
        """
        self.logger.info("📂 Loading and preparing data...")
        
        # Initialize data loader
        self.data_loader = DataLoader(
            data_dir=str(self.data_path),
            random_state=self.random_state
        )
        
        # Load and prepare all data using the complete pipeline
        self.data_loader.load_and_prepare_data(val_size=val_split, use_parquet=True)
        
        # Get data splits
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_loader.get_data_splits()
        
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        # Data summary
        data_summary = {
            'train_samples': len(self.X_train),
            'val_samples': len(self.X_val),
            'test_samples': len(self.X_test) if self.X_test is not None else 0,
            'n_features': self.X_train.shape[1],
            'train_class_dist': np.bincount(self.y_train),
            'val_class_dist': np.bincount(self.y_val),
            'feature_names': list(self.X_train.columns) if hasattr(self.X_train, 'columns') else None
        }
        
        if self.X_test is not None:
            data_summary['test_class_dist'] = np.bincount(self.y_test)
        
        self.logger.info("✅ Data loading completed")
        self.logger.info(f"   • Training samples: {data_summary['train_samples']:,}")
        self.logger.info(f"   • Validation samples: {data_summary['val_samples']:,}")
        self.logger.info(f"   • Test samples: {data_summary['test_samples']:,}")
        self.logger.info(f"   • Features: {data_summary['n_features']}")
        
        return data_summary
    
    def train_model(self, 
                   use_optimized_params: bool = True,
                   optimize_threshold: bool = True,
                   save_model: bool = True) -> Dict[str, Any]:
        """
        Train XGBoost model using XGBoostModel component.
        
        Args:
            use_optimized_params: Whether to use pre-optimized parameters
            optimize_threshold: Whether to optimize decision threshold
            save_model: Whether to save trained model
            
        Returns:
            Dictionary containing training results
        """
        if self.X_train is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        self.logger.info("🎯 Training XGBoost model...")
        
        # Initialize model
        self.model = XGBoostModel(random_state=self.random_state)
        
        # Create and train model
        self.model.create_model(
            use_optimized=use_optimized_params,
            y_train=self.y_train
        )
        
        # Train model
        training_results = self.model.train(
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            verbose=False
        )
        
        # Optimize threshold if requested
        if optimize_threshold and self.X_val is not None:
            self.logger.info("🔧 Optimizing decision threshold...")
            threshold_results = self.model.optimize_threshold(
                X_val=self.X_val,
                y_val=self.y_val,
                metric='accuracy'
            )
            training_results['threshold_optimization'] = threshold_results
        
        # Save model if requested
        if save_model:
            model_path = self.artifacts_path / "models" / "best_xgboost_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            self.logger.info(f"💾 Model saved: {model_path}")
            training_results['model_path'] = str(model_path)
        
        self.training_results = training_results
        
        self.logger.info("✅ Model training completed")
        self.logger.info(f"   • Training accuracy: {training_results['train_accuracy']:.4f}")
        if 'val_accuracy' in training_results:
            self.logger.info(f"   • Validation accuracy: {training_results['val_accuracy']:.4f}")
        
        return training_results
    
    def tune_hyperparameters(self,
                           strategy: str = "randomized",
                           n_iter: int = 100,
                           scoring: str = "roc_auc",
                           save_results: bool = True) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using HyperparameterTuner.
        
        Args:
            strategy: Tuning strategy ('randomized', 'grid', 'progressive', 'bayesian')
            n_iter: Number of iterations for randomized search
            scoring: Scoring metric
            save_results: Whether to save tuning results
            
        Returns:
            Dictionary containing tuning results
        """
        if self.X_train is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        self.logger.info(f"🔍 Starting hyperparameter tuning with {strategy} strategy...")
        
        # Initialize tuner
        self.tuner = HyperparameterTuner(random_state=self.random_state)
        
        # Perform tuning based on strategy
        if strategy == "randomized":
            tuning_results = self.tuner.randomized_search(
                X_train=self.X_train,
                y_train=self.y_train,
                n_iter=n_iter,
                scoring=scoring
            )
        elif strategy == "grid":
            tuning_results = self.tuner.grid_search(
                X_train=self.X_train,
                y_train=self.y_train,
                scoring=scoring,
                use_quick_params=True
            )
        elif strategy == "progressive":
            tuning_results = self.tuner.progressive_tuning(
                X_train=self.X_train,
                y_train=self.y_train,
                scoring=scoring
            )
        elif strategy == "bayesian":
            try:
                tuning_results = self.tuner.bayesian_optimization(
                    X_train=self.X_train,
                    y_train=self.y_train,
                    n_calls=n_iter,
                    scoring=scoring
                )
            except ImportError:
                self.logger.warning("Bayesian optimization requires scikit-optimize. Falling back to randomized search.")
                tuning_results = self.tuner.randomized_search(
                    X_train=self.X_train,
                    y_train=self.y_train,
                    n_iter=n_iter,
                    scoring=scoring
                )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Update model with best parameters
        if self.tuner.best_estimator:
            self.model = XGBoostModel(random_state=self.random_state)
            self.model.model = self.tuner.best_estimator
            self.model.is_trained = True
            self.model.best_params = self.tuner.best_params
        
        # Evaluate best model on validation set
        if self.X_val is not None and self.y_val is not None:
            val_results = self.tuner.evaluate_best_model(
                X_test=self.X_val,
                y_test=self.y_val,
                threshold_optimization=True
            )
            tuning_results['validation_results'] = val_results
        
        # Save results if requested
        if save_results:
            results_path = self.artifacts_path / "metrics" / f"tuning_results_{strategy}.json"
            
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            # Deep convert all numpy types
            serializable_results = json.loads(
                json.dumps(tuning_results, default=convert_numpy)
            )
            
            with open(results_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            self.logger.info(f"💾 Tuning results saved: {results_path}")
            tuning_results['results_path'] = str(results_path)
        
        self.logger.info("✅ Hyperparameter tuning completed")
        self.logger.info(f"   • Best {scoring}: {tuning_results['best_score']:.4f}")
        self.logger.info(f"   • Best parameters: {tuning_results['best_params']}")
        
        return tuning_results
    
    def evaluate_model(self, 
                      dataset: str = "test",
                      create_visualizations: bool = True,
                      save_report: bool = True) -> Dict[str, Any]:
        """
        Evaluate model using ModelEvaluator.
        
        Args:
            dataset: Dataset to evaluate on ('test', 'val', 'train')
            create_visualizations: Whether to create evaluation plots
            save_report: Whether to save evaluation report
            
        Returns:
            Dictionary containing evaluation results
        """
        if self.model is None or not self.model.is_trained:
            raise ValueError("Model not trained. Call train_model() or tune_hyperparameters() first.")
        
        # Select dataset
        if dataset == "test" and self.X_test is not None:
            X_eval, y_eval = self.X_test, self.y_test
        elif dataset == "val" and self.X_val is not None:
            X_eval, y_eval = self.X_val, self.y_val
        elif dataset == "train":
            X_eval, y_eval = self.X_train, self.y_train
        else:
            raise ValueError(f"Dataset '{dataset}' not available")
        
        self.logger.info(f"📊 Evaluating model on {dataset} data...")
        
        # Initialize evaluator
        self.evaluator = ModelEvaluator(
            model_name="XGBoost Credit Risk Model",
            save_plots=create_visualizations,
            plot_dir=str(self.artifacts_path / "visualizations")
        )
        
        # Make predictions
        y_pred = self.model.predict(X_eval)
        y_proba = self.model.predict_proba(X_eval)[:, 1]
        
        # Evaluate model
        evaluation_results = self.evaluator.evaluate_model(
            y_true=y_eval,
            y_pred=y_pred,
            y_proba=y_proba,
            dataset_name=dataset
        )
        
        # Create visualizations if requested
        if create_visualizations:
            self.logger.info("🎨 Creating evaluation visualizations...")
            
            try:
                # Create individual plots
                self.evaluator.plot_confusion_matrix(dataset)
                self.evaluator.plot_roc_curve(dataset)
                self.evaluator.plot_precision_recall_curve(dataset)
                self.evaluator.plot_threshold_analysis(dataset)
                
                # Create comprehensive dashboard
                self.evaluator.create_evaluation_dashboard(dataset)
                
                # Plot feature importance if available
                if hasattr(self.model.model, 'feature_importances_'):
                    importance_df = self.model.get_feature_importance()
                    self.evaluator.plot_feature_importance(importance_df)
                
            except Exception as e:
                self.logger.warning(f"Some visualizations failed: {e}")
        
        # Generate and save report
        if save_report:
            report = self.evaluator.generate_report(dataset)
            report_path = self.artifacts_path / "metrics" / f"evaluation_report_{dataset}.txt"
            
            with open(report_path, 'w') as f:
                f.write(report)
            
            self.logger.info(f"📄 Evaluation report saved: {report_path}")
            evaluation_results['report_path'] = str(report_path)
        
        self.evaluation_results[dataset] = evaluation_results
        
        basic_metrics = evaluation_results['basic_metrics']
        self.logger.info("✅ Model evaluation completed")
        self.logger.info(f"   • Accuracy: {basic_metrics['accuracy']:.4f}")
        self.logger.info(f"   • Precision: {basic_metrics['precision']:.4f}")
        self.logger.info(f"   • Recall: {basic_metrics['recall']:.4f}")
        self.logger.info(f"   • F1-Score: {basic_metrics['f1_score']:.4f}")
        if 'roc_auc' in basic_metrics:
            self.logger.info(f"   • ROC AUC: {basic_metrics['roc_auc']:.4f}")
        
        return evaluation_results
    
    def run_complete_pipeline(self,
                             train_file: str = "train/train.parquet",
                             test_file: str = "test/test.parquet",
                             tune_hyperparams: bool = False,
                             tuning_strategy: str = "randomized",
                             n_iter: int = 50) -> Dict[str, Any]:
        """
        Run the complete training and evaluation pipeline.
        
        Args:
            train_file: Training data file name
            test_file: Test data file name
            tune_hyperparams: Whether to perform hyperparameter tuning
            tuning_strategy: Tuning strategy if tune_hyperparams=True
            n_iter: Number of iterations for tuning
            
        Returns:
            Dictionary containing all pipeline results
        """
        self.logger.info("🔥 Starting complete credit risk training pipeline...")
        
        pipeline_results = {}
        
        # 1. Load data
        data_results = self.load_data(train_file=train_file, test_file=test_file)
        pipeline_results['data_loading'] = data_results
        
        # 2. Train model or tune hyperparameters
        if tune_hyperparams:
            tuning_results = self.tune_hyperparameters(
                strategy=tuning_strategy,
                n_iter=n_iter
            )
            pipeline_results['hyperparameter_tuning'] = tuning_results
        else:
            training_results = self.train_model()
            pipeline_results['model_training'] = training_results
        
        # 3. Evaluate on all available datasets
        evaluation_results = {}
        
        # Evaluate on validation set
        if self.X_val is not None:
            val_results = self.evaluate_model(dataset="val")
            evaluation_results['validation'] = val_results
        
        # Evaluate on test set if available
        if self.X_test is not None:
            test_results = self.evaluate_model(dataset="test")
            evaluation_results['test'] = test_results
        
        pipeline_results['evaluation'] = evaluation_results
        
        # 4. Save complete pipeline results
        pipeline_summary_path = self.artifacts_path / "metrics" / "pipeline_summary.json"
        
        # Prepare summary for JSON serialization
        summary = {
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'data_path': str(self.data_path),
                'random_state': self.random_state,
                'tune_hyperparams': tune_hyperparams,
                'tuning_strategy': tuning_strategy if tune_hyperparams else None,
                'n_iter': n_iter if tune_hyperparams else None
            },
            'data_summary': data_results,
            'best_model_path': pipeline_results.get('model_training', {}).get('model_path', 
                                                   pipeline_results.get('hyperparameter_tuning', {}).get('model_path')),
            'final_metrics': {}
        }
        
        # Add final metrics
        for dataset_name, results in evaluation_results.items():
            summary['final_metrics'][dataset_name] = results['basic_metrics']
        
        with open(pipeline_summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"📋 Pipeline summary saved: {pipeline_summary_path}")
        
        self.logger.info("🎉 Complete pipeline execution finished successfully!")
        
        return pipeline_results


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Credit Risk Assessment XGBoost Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training
    python main_training.py --mode train
    
    # Hyperparameter tuning
    python main_training.py --mode tune --strategy randomized --n-iter 100
    
    # Complete pipeline with tuning
    python main_training.py --mode pipeline --tune --strategy progressive
    
    # Evaluate existing model
    python main_training.py --mode evaluate --model-path artifacts/models/best_xgboost_model.pkl
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['train', 'tune', 'evaluate', 'pipeline'],
        default='pipeline',
        help='Execution mode (default: pipeline)'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default='processed_data/',
        help='Path to preprocessed data files (default: processed_data/)'
    )
    
    parser.add_argument(
        '--train-file',
        type=str,
        default='train/train.parquet',
        help='Training data file name (default: train/train.parquet)'
    )
    
    parser.add_argument(
        '--test-file',
        type=str,
        default='test/test.parquet',
        help='Test data file name (default: test/test.parquet)'
    )
    
    parser.add_argument(
        '--artifacts-path',
        type=str,
        default='artifacts/',
        help='Path to save artifacts (default: artifacts/)'
    )
    
    parser.add_argument(
        '--strategy',
        choices=['randomized', 'grid', 'progressive', 'bayesian'],
        default='randomized',
        help='Hyperparameter tuning strategy (default: randomized)'
    )
    
    parser.add_argument(
        '--n-iter',
        type=int,
        default=50,
        help='Number of iterations for tuning (default: 50)'
    )
    
    parser.add_argument(
        '--scoring',
        type=str,
        default='roc_auc',
        choices=['accuracy', 'roc_auc', 'f1', 'precision', 'recall'],
        help='Scoring metric for tuning (default: roc_auc)'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Enable hyperparameter tuning in pipeline mode'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        help='Path to existing model for evaluation mode'
    )
    
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Disable visualization creation'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = CreditRiskTrainer(
            data_path=args.data_path,
            artifacts_path=args.artifacts_path,
            random_state=args.random_state
        )
        
        if args.mode == 'train':
            # Basic training mode
            trainer.load_data(train_file=args.train_file, test_file=args.test_file)
            results = trainer.train_model()
            
        elif args.mode == 'tune':
            # Hyperparameter tuning mode
            trainer.load_data(train_file=args.train_file, test_file=args.test_file)
            results = trainer.tune_hyperparameters(
                strategy=args.strategy,
                n_iter=args.n_iter,
                scoring=args.scoring
            )
            
        elif args.mode == 'evaluate':
            # Evaluation mode
            if args.model_path:
                # Load existing model
                with open(args.model_path, 'rb') as f:
                    trainer.model = pickle.load(f)
                trainer.load_data(train_file=args.train_file, test_file=args.test_file)
            else:
                # Train new model
                trainer.load_data(train_file=args.train_file, test_file=args.test_file)
                trainer.train_model()
            
            results = trainer.evaluate_model(
                dataset="test" if trainer.X_test is not None else "val",
                create_visualizations=not args.no_viz
            )
            
        elif args.mode == 'pipeline':
            # Complete pipeline mode
            results = trainer.run_complete_pipeline(
                train_file=args.train_file,
                test_file=args.test_file,
                tune_hyperparams=args.tune,
                tuning_strategy=args.strategy,
                n_iter=args.n_iter
            )
        
        print("\n🎉 Execution completed successfully!")
        print("📁 Check artifacts/ directory for saved models, metrics, and visualizations")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()