#!/usr/bin/env python3
"""
Complete Monte Carlo Integration Example
========================================

This example demonstrates the complete integration of the high-performance
C++ Monte Carlo simulation engine with the existing XGBoost credit risk
assessment pipeline.

This script shows how to:
1. Load and prepare portfolio data
2. Use the existing XGBoost model for default probability prediction
3. Run high-performance Monte Carlo risk simulation
4. Analyze and visualize results
5. Perform stress testing scenarios
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

# Import existing XGBoost components
try:
    from data_loader import DataLoader
    from xgboost_model import XGBoostModel, ModelConfig
    from evaluator import ModelEvaluator
    XGBOOST_AVAILABLE = True
except ImportError as e:
    print(f"Warning: XGBoost components not available: {e}")
    XGBOOST_AVAILABLE = False

# Import Monte Carlo components
try:
    from monte_carlo_wrapper import MonteCarloSimulator, XGBoostMonteCarloIntegrator, create_stress_scenarios
    MONTE_CARLO_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Monte Carlo components not available: {e}")
    MONTE_CARLO_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CreditRiskMonteCarloDemo:
    """
    Complete demonstration of XGBoost + Monte Carlo integration.
    """
    
    def __init__(self, data_path: str = "processed_data", models_path: str = "artifacts/models"):
        """
        Initialize the demo with data and model paths.
        
        Args:
            data_path: Path to processed data directory
            models_path: Path to trained models directory
        """
        self.data_path = Path(data_path)
        self.models_path = Path(models_path)
        self.results_path = Path("monte_carlo_results")
        self.results_path.mkdir(exist_ok=True)
        
        # Initialize components
        self.data_loader = None
        self.xgboost_model = None
        self.monte_carlo_simulator = None
        self.integrator = None
        
        # Data containers
        self.portfolio_data = None
        self.X_test = None
        self.y_test = None
        self.default_probabilities = None
        
        logger.info(f"Demo initialized - Data: {self.data_path}, Models: {self.models_path}")
    
    def load_data_and_model(self):
        """Load processed data and trained XGBoost model."""
        
        if not XGBOOST_AVAILABLE:
            raise RuntimeError("XGBoost components not available")
        
        logger.info("Loading data and trained model...")
        
        # Initialize data loader
        self.data_loader = DataLoader(str(self.data_path))
        
        # Load preprocessed data
        train_df, test_df = self.data_loader.load_preprocessed_data()
        
        # Prepare test data for portfolio simulation
        self.X_test, self.y_test = self.data_loader.prepare_features_target(test_df)
        
        # Convert to portfolio format for Monte Carlo simulation
        self.portfolio_data = self._convert_to_portfolio_format(test_df)
        
        # Load trained XGBoost model
        model_files = list(self.models_path.glob("*xgboost*.pkl")) + list(self.models_path.glob("*xgboost*.joblib"))
        if not model_files:
            raise FileNotFoundError(f"No XGBoost model found in {self.models_path}")
        
        model_file = sorted(model_files)[-1]  # Use the most recent model
        logger.info(f"Loading model from {model_file}")
        
        self.xgboost_model = XGBoostModel()
        self.xgboost_model.load_model(str(model_file))
        
        logger.info(f"Data loaded: {len(self.portfolio_data)} accounts")
        logger.info(f"Total portfolio exposure: ${self.portfolio_data['exposure_at_default'].sum():,.2f}")
    
    def _convert_to_portfolio_format(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert test data to portfolio format for Monte Carlo simulation.
        
        Args:
            test_df: Test dataset
            
        Returns:
            Portfolio DataFrame with required columns
        """
        
        # Create portfolio data with required fields
        portfolio = pd.DataFrame()
        
        # Account identifiers
        portfolio['account_id'] = range(len(test_df))
        
        # Financial data - derive from credit card features
        # Assuming the dataset has LIMIT_BAL, PAY_AMT1, etc.
        if 'LIMIT_BAL' in test_df.columns:
            portfolio['limit'] = test_df['LIMIT_BAL']
            portfolio['balance'] = test_df['LIMIT_BAL'] * 0.3  # Assume 30% utilization
        else:
            # Create synthetic financial data if not available
            portfolio['limit'] = np.random.lognormal(mean=10.0, sigma=0.8, size=len(test_df))
            portfolio['balance'] = portfolio['limit'] * np.random.beta(2, 5, size=len(test_df))
        
        # Risk parameters
        portfolio['exposure_at_default'] = portfolio['balance']  # Conservative assumption
        portfolio['loss_given_default'] = np.random.beta(2, 3, size=len(test_df)) * 0.6 + 0.2  # 20-80% LGD
        
        # Additional portfolio features for analysis
        if 'AGE' in test_df.columns:
            portfolio['age'] = test_df['AGE']
        if 'SEX' in test_df.columns:
            portfolio['gender'] = test_df['SEX']
        if 'EDUCATION' in test_df.columns:
            portfolio['education'] = test_df['EDUCATION']
        
        return portfolio
    
    def generate_predictions(self):
        """Generate default probability predictions using XGBoost model."""
        
        logger.info("Generating XGBoost default probability predictions...")
        
        # Get predictions from XGBoost model
        predictions = self.xgboost_model.predict_proba(self.X_test)
        
        if predictions.ndim == 2:
            # Binary classification - take probability of default class (class 1)
            self.default_probabilities = predictions[:, 1]
        else:
            # Already probability scores
            self.default_probabilities = predictions
        
        # Validate predictions
        assert len(self.default_probabilities) == len(self.portfolio_data)
        assert np.all((self.default_probabilities >= 0) & (self.default_probabilities <= 1))
        
        logger.info(f"Predictions generated: Mean={self.default_probabilities.mean():.4f}, "
                   f"Range=[{self.default_probabilities.min():.4f}, {self.default_probabilities.max():.4f}]")
    
    def run_monte_carlo_simulation(self, num_simulations: int = 100000):
        """
        Run Monte Carlo risk simulation.
        
        Args:
            num_simulations: Number of Monte Carlo iterations
        """
        
        if not MONTE_CARLO_AVAILABLE:
            raise RuntimeError("Monte Carlo components not available")
        
        logger.info(f"Running Monte Carlo simulation with {num_simulations:,} iterations...")
        
        # Initialize Monte Carlo simulator
        self.monte_carlo_simulator = MonteCarloSimulator(
            num_simulations=num_simulations,
            num_threads=0,  # Auto-detect optimal thread count
            scenarios_per_batch=1000,
            use_antithetic_variates=True,
            enable_correlation=True,
            random_seed=42
        )
        
        # Run simulation
        results = self.monte_carlo_simulator.run_simulation(
            portfolio_data=self.portfolio_data,
            default_probabilities=self.default_probabilities
        )
        
        if not results['success']:
            raise RuntimeError(f"Monte Carlo simulation failed: {results['error_message']}")
        
        self.simulation_results = results
        
        # Log key results
        risk_metrics = results['risk_metrics']
        performance = results['performance']
        
        logger.info("Monte Carlo simulation completed!")
        logger.info(f"Performance: {performance['iterations_per_second']:,.0f} iterations/second")
        logger.info(f"Expected Loss: ${risk_metrics['expected_loss']:,.2f}")
        logger.info(f"95% VaR: ${risk_metrics['var_95']:,.2f}")
        logger.info(f"99% VaR: ${risk_metrics['var_99']:,.2f}")
        logger.info(f"95% CVaR: ${risk_metrics['cvar_95']:,.2f}")
        logger.info(f"99% CVaR: ${risk_metrics['cvar_99']:,.2f}")
        
        return results
    
    def run_stress_testing(self):
        """Run stress testing with multiple economic scenarios."""
        
        logger.info("Running stress testing scenarios...")
        
        # Define stress scenarios
        stress_scenarios = [
            {
                'name': 'Severe Recession',
                'params': {
                    'gdp_growth': -0.05,
                    'unemployment_rate': 0.12,
                    'interest_rate': 0.08,
                    'market_volatility': 0.35,
                    'credit_spread': 0.06
                }
            },
            {
                'name': 'Moderate Recession',
                'params': {
                    'gdp_growth': -0.03,
                    'unemployment_rate': 0.10,
                    'interest_rate': 0.06,
                    'market_volatility': 0.25,
                    'credit_spread': 0.04
                }
            },
            {
                'name': 'Economic Boom',
                'params': {
                    'gdp_growth': 0.08,
                    'unemployment_rate': 0.03,
                    'interest_rate': 0.01,
                    'market_volatility': 0.08,
                    'credit_spread': 0.01
                }
            }
        ]
        
        stress_results = []
        
        for scenario in stress_scenarios:
            logger.info(f"Running scenario: {scenario['name']}")
            
            result = self.monte_carlo_simulator.run_simulation(
                portfolio_data=self.portfolio_data,
                default_probabilities=self.default_probabilities,
                economic_scenario=scenario['params']
            )
            
            if result['success']:
                result['scenario_name'] = scenario['name']
                result['scenario_params'] = scenario['params']
                stress_results.append(result)
                
                logger.info(f"{scenario['name']} - 95% VaR: ${result['risk_metrics']['var_95']:,.2f}")
        
        self.stress_results = stress_results
        return stress_results
    
    def analyze_and_visualize_results(self):
        """Create comprehensive analysis and visualizations."""
        
        logger.info("Generating analysis and visualizations...")
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 16))
        
        # Create grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Loss Distribution Histogram
        ax1 = fig.add_subplot(gs[0, :2])
        losses = self.simulation_results['loss_distribution']
        ax1.hist(losses, bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
        ax1.axvline(self.simulation_results['risk_metrics']['var_95'], color='red', linestyle='--', 
                   label=f"95% VaR: ${self.simulation_results['risk_metrics']['var_95']:,.0f}")
        ax1.axvline(self.simulation_results['risk_metrics']['var_99'], color='darkred', linestyle='--',
                   label=f"99% VaR: ${self.simulation_results['risk_metrics']['var_99']:,.0f}")
        ax1.set_xlabel('Portfolio Loss ($)')
        ax1.set_ylabel('Density')
        ax1.set_title('Portfolio Loss Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Risk Metrics Comparison
        ax2 = fig.add_subplot(gs[0, 2:])
        metrics = ['Expected Loss', '95% VaR', '99% VaR', '95% CVaR', '99% CVaR']
        values = [
            self.simulation_results['risk_metrics']['expected_loss'],
            self.simulation_results['risk_metrics']['var_95'],
            self.simulation_results['risk_metrics']['var_99'],
            self.simulation_results['risk_metrics']['cvar_95'],
            self.simulation_results['risk_metrics']['cvar_99']
        ]
        
        colors = ['lightblue', 'orange', 'red', 'lightcoral', 'darkred']
        bars = ax2.bar(metrics, values, color=colors, alpha=0.8)
        ax2.set_ylabel('Risk Value ($)')
        ax2.set_title('Risk Metrics Summary')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'${value:,.0f}', ha='center', va='bottom', fontsize=10)
        
        # 3. Default Probability Distribution
        ax3 = fig.add_subplot(gs[1, :2])
        ax3.hist(self.default_probabilities, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.axvline(self.default_probabilities.mean(), color='red', linestyle='--',
                   label=f'Mean: {self.default_probabilities.mean():.3f}')
        ax3.set_xlabel('Default Probability')
        ax3.set_ylabel('Frequency')
        ax3.set_title('XGBoost Default Probability Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Exposure Distribution
        ax4 = fig.add_subplot(gs[1, 2:])
        exposures = self.portfolio_data['exposure_at_default']
        ax4.hist(exposures, bins=30, alpha=0.7, color='gold', edgecolor='black')
        ax4.axvline(exposures.mean(), color='red', linestyle='--',
                   label=f'Mean: ${exposures.mean():,.0f}')
        ax4.set_xlabel('Exposure at Default ($)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Portfolio Exposure Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Stress Test Results (if available)
        if hasattr(self, 'stress_results') and self.stress_results:
            ax5 = fig.add_subplot(gs[2, :2])
            scenario_names = [r['scenario_name'] for r in self.stress_results]
            var_95_values = [r['risk_metrics']['var_95'] for r in self.stress_results]
            
            # Add baseline
            scenario_names.insert(0, 'Baseline')
            var_95_values.insert(0, self.simulation_results['risk_metrics']['var_95'])
            
            colors = ['blue'] + ['red' if 'Recession' in name else 'green' for name in scenario_names[1:]]
            bars = ax5.bar(scenario_names, var_95_values, color=colors, alpha=0.7)
            ax5.set_ylabel('95% VaR ($)')
            ax5.set_title('Stress Test Results - 95% VaR')
            ax5.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, var_95_values):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'${value:,.0f}', ha='center', va='bottom', fontsize=9)
        
        # 6. Performance Metrics
        ax6 = fig.add_subplot(gs[2, 2:])
        perf = self.simulation_results['performance']
        perf_metrics = ['Iterations/sec', 'Memory (MB)', 'Threads', 'Total Time (s)']
        perf_values = [
            perf['iterations_per_second'],
            perf['memory_usage_mb'],
            perf['num_threads_used'],
            perf['total_time_seconds']
        ]
        
        # Normalize values for display
        normalized_values = [v/max(perf_values) for v in perf_values]
        bars = ax6.bar(perf_metrics, normalized_values, color='purple', alpha=0.7)
        ax6.set_ylabel('Normalized Value')
        ax6.set_title('Performance Metrics')
        ax6.tick_params(axis='x', rotation=45)
        
        # Add actual values as labels
        for bar, actual_value in zip(bars, perf_values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{actual_value:,.0f}', ha='center', va='bottom', fontsize=9)
        
        # 7. Portfolio Summary Table
        ax7 = fig.add_subplot(gs[3, :2])
        ax7.axis('off')
        
        summary_data = [
            ['Portfolio Size', f"{len(self.portfolio_data):,} accounts"],
            ['Total Exposure', f"${self.portfolio_data['exposure_at_default'].sum():,.2f}"],
            ['Average Default Prob', f"{self.default_probabilities.mean():.3f}"],
            ['Monte Carlo Iterations', f"{self.simulation_results['risk_metrics']['expected_loss'] / self.default_probabilities.mean() * len(self.portfolio_data):,.0f}"],
            ['Simulation Time', f"{perf['total_time_seconds']:.2f} seconds"],
            ['Performance Target', "✓ Achieved" if perf['iterations_per_second'] >= 10000 else "✗ Not Met"]
        ]
        
        table = ax7.table(cellText=summary_data, colLabels=['Metric', 'Value'],
                         cellLoc='left', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax7.set_title('Portfolio & Simulation Summary')
        
        # 8. Risk vs Exposure Scatter
        ax8 = fig.add_subplot(gs[3, 2:])
        # Create risk score as product of probability and exposure
        risk_scores = self.default_probabilities * self.portfolio_data['exposure_at_default']
        ax8.scatter(self.portfolio_data['exposure_at_default'], self.default_probabilities, 
                   c=risk_scores, cmap='Reds', alpha=0.6, s=30)
        ax8.set_xlabel('Exposure at Default ($)')
        ax8.set_ylabel('Default Probability')
        ax8.set_title('Risk Profile: Exposure vs Default Probability')
        ax8.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(ax8.collections[0], ax=ax8)
        cbar.set_label('Risk Score', rotation=270, labelpad=20)
        
        plt.suptitle('Monte Carlo Credit Risk Assessment - Complete Analysis', fontsize=16, y=0.98)
        
        # Save plot
        plot_path = self.results_path / f'monte_carlo_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Analysis plots saved to {plot_path}")
        
        plt.show()
    
    def save_results(self):
        """Save detailed results to files."""
        
        logger.info("Saving detailed results...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save simulation results
        results_file = self.results_path / f'simulation_results_{timestamp}.json'
        import json
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            results_copy = self.simulation_results.copy()
            if 'loss_distribution' in results_copy:
                results_copy['loss_distribution'] = results_copy['loss_distribution'][:1000]  # Save first 1000 values
            json.dump(results_copy, f, indent=2, default=str)
        
        # Save portfolio data with predictions
        portfolio_with_predictions = self.portfolio_data.copy()
        portfolio_with_predictions['default_probability'] = self.default_probabilities
        portfolio_with_predictions['risk_score'] = (
            self.default_probabilities * portfolio_with_predictions['exposure_at_default']
        )
        
        portfolio_file = self.results_path / f'portfolio_with_predictions_{timestamp}.csv'
        portfolio_with_predictions.to_csv(portfolio_file, index=False)
        
        # Save stress test results if available
        if hasattr(self, 'stress_results'):
            stress_file = self.results_path / f'stress_test_results_{timestamp}.json'
            with open(stress_file, 'w') as f:
                json.dump(self.stress_results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {self.results_path}")
    
    def run_complete_demo(self, num_simulations: int = 100000):
        """
        Run the complete demonstration workflow.
        
        Args:
            num_simulations: Number of Monte Carlo iterations
        """
        
        logger.info("Starting complete Monte Carlo + XGBoost demonstration...")
        
        try:
            # Step 1: Load data and model
            self.load_data_and_model()
            
            # Step 2: Generate predictions
            self.generate_predictions()
            
            # Step 3: Run Monte Carlo simulation
            self.run_monte_carlo_simulation(num_simulations)
            
            # Step 4: Run stress testing
            self.run_stress_testing()
            
            # Step 5: Analyze and visualize
            self.analyze_and_visualize_results()
            
            # Step 6: Save results
            self.save_results()
            
            logger.info("Complete demonstration finished successfully!")
            
            # Print summary
            print("\n" + "="*80)
            print("MONTE CARLO CREDIT RISK ASSESSMENT SUMMARY")
            print("="*80)
            print(f"Portfolio Size: {len(self.portfolio_data):,} accounts")
            print(f"Total Exposure: ${self.portfolio_data['exposure_at_default'].sum():,.2f}")
            print(f"Monte Carlo Iterations: {num_simulations:,}")
            print(f"Performance: {self.simulation_results['performance']['iterations_per_second']:,.0f} iterations/second")
            print(f"Expected Loss: ${self.simulation_results['risk_metrics']['expected_loss']:,.2f}")
            print(f"95% VaR: ${self.simulation_results['risk_metrics']['var_95']:,.2f}")
            print(f"99% VaR: ${self.simulation_results['risk_metrics']['var_99']:,.2f}")
            print(f"95% CVaR: ${self.simulation_results['risk_metrics']['cvar_95']:,.2f}")
            print(f"99% CVaR: ${self.simulation_results['risk_metrics']['cvar_99']:,.2f}")
            
            target_met = "✓" if self.simulation_results['performance']['iterations_per_second'] >= 10000 else "✗"
            print(f"Performance Target (10,000+ it/s): {target_met}")
            print("="*80)
            
        except Exception as e:
            logger.error(f"Demo failed: {str(e)}")
            raise


def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(description='Monte Carlo + XGBoost Integration Demo')
    parser.add_argument('--data-path', default='processed_data',
                       help='Path to processed data directory')
    parser.add_argument('--models-path', default='artifacts/models',
                       help='Path to trained models directory')
    parser.add_argument('--simulations', type=int, default=100000,
                       help='Number of Monte Carlo simulations')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick demo with fewer simulations')
    
    args = parser.parse_args()
    
    # Check availability of required components
    if not XGBOOST_AVAILABLE:
        print("ERROR: XGBoost components not available. Please ensure the main training pipeline is set up.")
        return 1
    
    if not MONTE_CARLO_AVAILABLE:
        print("ERROR: Monte Carlo components not available. Please build the C++ module first using:")
        print("cd src/monte_carlo && ./build.sh install")
        return 1
    
    # Adjust simulations for quick demo
    num_simulations = 10000 if args.quick else args.simulations
    
    try:
        # Run the complete demo
        demo = CreditRiskMonteCarloDemo(args.data_path, args.models_path)
        demo.run_complete_demo(num_simulations)
        
        return 0
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        return 1


if __name__ == '__main__':
    sys.exit(main())