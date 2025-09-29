
"""
Model Evaluation Module for Credit Risk Assessment
==================================================

This module provides comprehensive evaluation capabilities for machine learning models
including metrics calculation, visualizations, performance analysis, and reporting.

Classes:
- ModelEvaluator: Main evaluation class with comprehensive metrics and visualizations
- EvaluationReport: Class for generating detailed evaluation reports
- MetricsCalculator: Utility class for various metric calculations
"""

import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    average_precision_score, matthews_corrcoef, log_loss, brier_score_loss
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')


class MetricsCalculator:
    """Utility class for calculating various evaluation metrics."""
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, 
                             y_pred: np.ndarray, 
                             y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate comprehensive set of classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            
        Returns:
            Dictionary of calculated metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'specificity': MetricsCalculator._calculate_specificity(y_true, y_pred),
            'matthews_corrcoef': matthews_corrcoef(y_true, y_pred)
        }
        
        if y_proba is not None:
            metrics.update({
                'roc_auc': roc_auc_score(y_true, y_proba),
                'pr_auc': average_precision_score(y_true, y_proba),
                'log_loss': log_loss(y_true, y_proba),
                'brier_score': brier_score_loss(y_true, y_proba)
            })
        
        return metrics
    
    @staticmethod
    def _calculate_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (true negative rate)."""
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            return tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return 0.0
    
    @staticmethod
    def calculate_threshold_metrics(y_true: np.ndarray, 
                                   y_proba: np.ndarray,
                                   thresholds: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Calculate metrics across different thresholds.
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            thresholds: Threshold values to test
            
        Returns:
            DataFrame with metrics for each threshold
        """
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 81)
        
        results = []
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            metrics = MetricsCalculator.calculate_all_metrics(y_true, y_pred, y_proba)
            metrics['threshold'] = threshold
            results.append(metrics)
        
        return pd.DataFrame(results)


class ModelEvaluator:
    """Comprehensive model evaluation class."""
    
    def __init__(self, 
                 model_name: str = "XGBoost Model",
                 save_plots: bool = True,
                 plot_dir: str = "artifacts/visualizations/"):
        """
        Initialize model evaluator.
        
        Args:
            model_name: Name of the model being evaluated
            save_plots: Whether to save plots to disk
            plot_dir: Directory to save plots
            
        """
        self.model_name = model_name
        self.save_plots = save_plots
        self.plot_dir = plot_dir
        self.logger = logging.getLogger(__name__)
        
        # Results storage
        self.evaluation_results = {}
        self.plots_created = []
        
        # Ensure plot directory exists
        if save_plots:
            import os
            os.makedirs(plot_dir, exist_ok=True)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'seaborn')
        sns.set_palette("husl")
    
    def evaluate_model(self,
                      y_true: Union[pd.Series, np.ndarray],
                      y_pred: Union[pd.Series, np.ndarray],
                      y_proba: Optional[Union[pd.Series, np.ndarray]] = None,
                      dataset_name: str = "test") -> Dict[str, Any]:
        """
        Perform comprehensive model evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            dataset_name: Name of the dataset being evaluated
            
        Returns:
            Dictionary containing evaluation results
        """
        self.logger.info(f"Evaluating {self.model_name} on {dataset_name} data...")
        
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if y_proba is not None:
            y_proba = np.array(y_proba)
            if y_proba.ndim > 1 and y_proba.shape[1] > 1:
                y_proba = y_proba[:, 1]  # Use positive class probabilities
        
        # Calculate basic metrics
        basic_metrics = MetricsCalculator.calculate_all_metrics(y_true, y_pred, y_proba)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Generate classification report
        class_report = classification_report(y_true, y_pred, output_dict=True)
        
        # Store results
        results = {
            'dataset': dataset_name,
            'basic_metrics': basic_metrics,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'sample_size': len(y_true),
            'class_distribution': np.bincount(y_true),
            'prediction_distribution': np.bincount(y_pred)
        }
        
        # Add probability-based metrics if available
        if y_proba is not None:
            # ROC curve data
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
            results['roc_curve'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': roc_thresholds}
            
            # Precision-Recall curve data
            precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
            results['pr_curve'] = {'precision': precision, 'recall': recall, 'thresholds': pr_thresholds}
            
            # Calibration curve data
            fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_proba, n_bins=10)
            results['calibration'] = {
                'fraction_of_positives': fraction_of_positives,
                'mean_predicted_value': mean_predicted_value
            }
            
            # Threshold analysis
            threshold_metrics = MetricsCalculator.calculate_threshold_metrics(y_true, y_proba)
            results['threshold_analysis'] = threshold_metrics
        
        # Store results
        self.evaluation_results[dataset_name] = results
        
        # Log key metrics
        self.logger.info(f"✓ {dataset_name.title()} Results:")
        self.logger.info(f"   • Accuracy: {basic_metrics['accuracy']:.4f}")
        self.logger.info(f"   • Precision: {basic_metrics['precision']:.4f}")
        self.logger.info(f"   • Recall: {basic_metrics['recall']:.4f}")
        self.logger.info(f"   • F1-Score: {basic_metrics['f1_score']:.4f}")
        if y_proba is not None:
            self.logger.info(f"   • AUC-ROC: {basic_metrics['roc_auc']:.4f}")
            self.logger.info(f"   • AUC-PR: {basic_metrics['pr_auc']:.4f}")
        
        return results
    
    def plot_confusion_matrix(self, dataset_name: str = "test", figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            dataset_name: Dataset to plot results for
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if dataset_name not in self.evaluation_results:
            raise ValueError(f"No evaluation results found for {dataset_name}")
        
        cm = self.evaluation_results[dataset_name]['confusion_matrix']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['No Default', 'Default'],
                   yticklabels=['No Default', 'Default'])
        
        ax.set_title(f'Confusion Matrix - {self.model_name}\n({dataset_name.title()} Data)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        # Add performance metrics as text
        results = self.evaluation_results[dataset_name]
        metrics_text = f"Accuracy: {results['basic_metrics']['accuracy']:.3f}\n"
        metrics_text += f"Precision: {results['basic_metrics']['precision']:.3f}\n"
        metrics_text += f"Recall: {results['basic_metrics']['recall']:.3f}\n"
        metrics_text += f"F1-Score: {results['basic_metrics']['f1_score']:.3f}"
        
        ax.text(1.05, 0.5, metrics_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if self.save_plots:
            filename = f"{self.plot_dir}/confusion_matrix_{dataset_name}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            self.plots_created.append(filename)
            self.logger.info(f"✓ Confusion matrix saved: {filename}")
        
        return fig
    
    def plot_roc_curve(self, dataset_name: str = "test", figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Plot ROC curve.
        
        Args:
            dataset_name: Dataset to plot results for
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if dataset_name not in self.evaluation_results:
            raise ValueError(f"No evaluation results found for {dataset_name}")
        
        results = self.evaluation_results[dataset_name]
        
        if 'roc_curve' not in results:
            raise ValueError(f"ROC curve data not available for {dataset_name}")
        
        roc_data = results['roc_curve']
        auc_score = results['basic_metrics']['roc_auc']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot ROC curve
        ax.plot(roc_data['fpr'], roc_data['tpr'], linewidth=2, 
               label=f'{self.model_name} (AUC = {auc_score:.3f})')
        
        # Plot random classifier line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier (AUC = 0.5)')
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curve - {self.model_name}\n({dataset_name.title()} Data)', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots:
            filename = f"{self.plot_dir}/roc_curve_{dataset_name}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            self.plots_created.append(filename)
            self.logger.info(f"✓ ROC curve saved: {filename}")
        
        return fig
    
    def plot_precision_recall_curve(self, dataset_name: str = "test", 
                                   figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Plot Precision-Recall curve.
        
        Args:
            dataset_name: Dataset to plot results for
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if dataset_name not in self.evaluation_results:
            raise ValueError(f"No evaluation results found for {dataset_name}")
        
        results = self.evaluation_results[dataset_name]
        
        if 'pr_curve' not in results:
            raise ValueError(f"PR curve data not available for {dataset_name}")
        
        pr_data = results['pr_curve']
        auc_score = results['basic_metrics']['pr_auc']
        
        # Calculate baseline (random classifier performance)
        n_pos = results['class_distribution'][1]
        n_total = results['sample_size']
        baseline = n_pos / n_total
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot PR curve
        ax.plot(pr_data['recall'], pr_data['precision'], linewidth=2,
               label=f'{self.model_name} (AUC = {auc_score:.3f})')
        
        # Plot baseline
        ax.axhline(y=baseline, color='k', linestyle='--', linewidth=1,
                  label=f'Random Classifier (AUC = {baseline:.3f})')
        
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'Precision-Recall Curve - {self.model_name}\n({dataset_name.title()} Data)', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots:
            filename = f"{self.plot_dir}/precision_recall_curve_{dataset_name}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            self.plots_created.append(filename)
            self.logger.info(f"✓ PR curve saved: {filename}")
        
        return fig
    
    def plot_threshold_analysis(self, dataset_name: str = "test", 
                               metrics: List[str] = None,
                               figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot metrics vs threshold analysis.
        
        Args:
            dataset_name: Dataset to plot results for
            metrics: List of metrics to plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if dataset_name not in self.evaluation_results:
            raise ValueError(f"No evaluation results found for {dataset_name}")
        
        results = self.evaluation_results[dataset_name]
        
        if 'threshold_analysis' not in results:
            raise ValueError(f"Threshold analysis data not available for {dataset_name}")
        
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        threshold_data = results['threshold_analysis']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot metrics vs threshold
        for metric in metrics:
            if metric in threshold_data.columns:
                ax.plot(threshold_data['threshold'], threshold_data[metric], 
                       linewidth=2, label=metric.replace('_', ' ').title())
        
        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'Metrics vs Threshold - {self.model_name}\n({dataset_name.title()} Data)', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Add optimal threshold lines
        for metric in metrics:
            if metric in threshold_data.columns:
                best_idx = threshold_data[metric].idxmax()
                best_threshold = threshold_data.loc[best_idx, 'threshold']
                best_score = threshold_data.loc[best_idx, metric]
                ax.axvline(x=best_threshold, color='red', linestyle=':', alpha=0.7)
                ax.text(best_threshold, best_score, f'{best_threshold:.2f}', 
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if self.save_plots:
            filename = f"{self.plot_dir}/threshold_analysis_{dataset_name}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            self.plots_created.append(filename)
            self.logger.info(f"✓ Threshold analysis saved: {filename}")
        
        return fig
    
    def plot_feature_importance(self, 
                              feature_importance: Union[pd.DataFrame, Dict[str, float]],
                              top_n: int = 20,
                              figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            feature_importance: Feature importance data
            top_n: Number of top features to display
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if isinstance(feature_importance, dict):
            importance_df = pd.DataFrame.from_dict(feature_importance, orient='index', columns=['importance'])
            importance_df.reset_index(inplace=True)
            importance_df.columns = ['feature', 'importance']
        else:
            importance_df = feature_importance.copy()
        
        # Sort and take top N
        importance_df = importance_df.sort_values('importance', ascending=True).tail(top_n)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create horizontal bar plot
        bars = ax.barh(importance_df['feature'], importance_df['importance'])
        
        # Color bars with gradient
        colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Features', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importance - {self.model_name}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if self.save_plots:
            filename = f"{self.plot_dir}/feature_importance.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            self.plots_created.append(filename)
            self.logger.info(f"✓ Feature importance saved: {filename}")
        
        return fig
    
    def plot_calibration_curve(self, dataset_name: str = "test", 
                              figsize: Tuple[int, int] = (8, 8)) -> plt.Figure:
        """
        Plot calibration curve.
        
        Args:
            dataset_name: Dataset to plot results for
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if dataset_name not in self.evaluation_results:
            raise ValueError(f"No evaluation results found for {dataset_name}")
        
        results = self.evaluation_results[dataset_name]
        
        if 'calibration' not in results:
            raise ValueError(f"Calibration data not available for {dataset_name}")
        
        cal_data = results['calibration']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot calibration curve
        ax.plot(cal_data['mean_predicted_value'], cal_data['fraction_of_positives'], 
               'o-', linewidth=2, label=f'{self.model_name}')
        
        # Plot perfectly calibrated line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfectly Calibrated')
        
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_title(f'Calibration Curve - {self.model_name}\n({dataset_name.title()} Data)', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if self.save_plots:
            filename = f"{self.plot_dir}/calibration_curve_{dataset_name}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            self.plots_created.append(filename)
            self.logger.info(f"✓ Calibration curve saved: {filename}")
        
        return fig
    
    def create_evaluation_dashboard(self, dataset_name: str = "test") -> plt.Figure:
        """
        Create comprehensive evaluation dashboard.
        
        Args:
            dataset_name: Dataset to create dashboard for
            
        Returns:
            Matplotlib figure with subplots
        """
        if dataset_name not in self.evaluation_results:
            raise ValueError(f"No evaluation results found for {dataset_name}")
        
        results = self.evaluation_results[dataset_name]
        has_proba = 'roc_curve' in results
        
        if has_proba:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Confusion Matrix
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=['No Default', 'Default'],
                   yticklabels=['No Default', 'Default'])
        ax1.set_title('Confusion Matrix', fontweight='bold')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # 2. Metrics Bar Chart
        metrics = results['basic_metrics']
        metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_values = [metrics[name] for name in metric_names]
        
        bars = ax2.bar(metric_names, metric_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        ax2.set_title('Performance Metrics', fontweight='bold')
        ax2.set_ylabel('Score')
        ax2.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        if has_proba:
            # 3. ROC Curve
            roc_data = results['roc_curve']
            auc_score = metrics['roc_auc']
            ax3.plot(roc_data['fpr'], roc_data['tpr'], linewidth=2,
                    label=f'Model (AUC = {auc_score:.3f})')
            ax3.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
            ax3.set_xlabel('False Positive Rate')
            ax3.set_ylabel('True Positive Rate')
            ax3.set_title('ROC Curve', fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. Precision-Recall Curve
            pr_data = results['pr_curve']
            pr_auc = metrics['pr_auc']
            ax4.plot(pr_data['recall'], pr_data['precision'], linewidth=2,
                    label=f'Model (AUC = {pr_auc:.3f})')
            baseline = results['class_distribution'][1] / results['sample_size']
            ax4.axhline(y=baseline, color='k', linestyle='--', linewidth=1,
                       label=f'Baseline ({baseline:.3f})')
            ax4.set_xlabel('Recall')
            ax4.set_ylabel('Precision')
            ax4.set_title('Precision-Recall Curve', fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'{self.model_name} - Evaluation Dashboard ({dataset_name.title()} Data)',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if self.save_plots:
            filename = f"{self.plot_dir}/evaluation_dashboard_{dataset_name}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            self.plots_created.append(filename)
            self.logger.info(f"✓ Evaluation dashboard saved: {filename}")
        
        return fig
    
    def generate_report(self, dataset_name: str = "test") -> str:
        """
        Generate text-based evaluation report.
        
        Args:
            dataset_name: Dataset to generate report for
            
        Returns:
            Formatted evaluation report
        """
        if dataset_name not in self.evaluation_results:
            raise ValueError(f"No evaluation results found for {dataset_name}")
        
        results = self.evaluation_results[dataset_name]
        metrics = results['basic_metrics']
        
        report = f"""
        
📊 EVALUATION REPORT: {self.model_name}
{'=' * 60}

Dataset: {dataset_name.upper()}
Sample Size: {results['sample_size']:,}
Class Distribution: {results['class_distribution']}

PERFORMANCE METRICS:
{'-' * 30}
• Accuracy:      {metrics['accuracy']:.4f}
• Precision:     {metrics['precision']:.4f}
• Recall:        {metrics['recall']:.4f}
• F1-Score:      {metrics['f1_score']:.4f}
• Specificity:   {metrics['specificity']:.4f}
• MCC:          {metrics['matthews_corrcoef']:.4f}
"""
        
        if 'roc_auc' in metrics:
            report += f"""
PROBABILITY-BASED METRICS:
{'-' * 30}
• ROC AUC:       {metrics['roc_auc']:.4f}
• PR AUC:        {metrics['pr_auc']:.4f}
• Log Loss:      {metrics['log_loss']:.4f}
• Brier Score:   {metrics['brier_score']:.4f}
"""
        
        # Confusion Matrix
        cm = results['confusion_matrix']
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            report += f"""
CONFUSION MATRIX:
{'-' * 30}
                Predicted
                No    Yes
Actual No    {tn:4d}  {fp:4d}
       Yes   {fn:4d}  {tp:4d}

• True Negatives:  {tn:4d}
• False Positives: {fp:4d}
• False Negatives: {fn:4d}
• True Positives:  {tp:4d}
"""
        
        # Classification Report
        class_report = results['classification_report']
        report += f"""
DETAILED CLASSIFICATION REPORT:
{'-' * 30}
"""
        for class_label in ['0', '1']:
            if class_label in class_report:
                cr = class_report[class_label]
                report += f"Class {class_label}: Precision={cr['precision']:.3f}, Recall={cr['recall']:.3f}, F1={cr['f1-score']:.3f}\n"
        
        if self.plots_created:
            report += f"""
GENERATED VISUALIZATIONS:
{'-' * 30}
"""
            for plot in self.plots_created:
                report += f"• {plot.split('/')[-1]}\n"
        
        report += f"\n{'=' * 60}\n"
        
        return report
    
    def get_summary_metrics(self) -> pd.DataFrame:
        """
        Get summary of all evaluation results.
        
        Returns:
            DataFrame with summary metrics
        """
        summary_data = []
        
        for dataset, results in self.evaluation_results.items():
            metrics = results['basic_metrics']
            row = {
                'dataset': dataset,
                'sample_size': results['sample_size'],
                **metrics
            }
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)


if __name__ == "__main__":
    # Example usage
    import logging
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    # Create sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, 
                              n_informative=10, n_redundant=5, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train a simple model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate model
    evaluator = ModelEvaluator(model_name="Random Forest", save_plots=False)
    results = evaluator.evaluate_model(y_test, y_pred, y_proba)
    
    # Generate report
    report = evaluator.generate_report()
    print(report)