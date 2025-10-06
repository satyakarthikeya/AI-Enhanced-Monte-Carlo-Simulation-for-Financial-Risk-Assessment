"""
Monte Carlo Engine Python Wrapper
==================================

This module provides a high-level Python interface to the C++ Monte Carlo
simulation engine, designed for seamless integration with XGBoost credit
risk models.

Usage Example:
    from monte_carlo_wrapper import MonteCarloSimulator
    import pandas as pd
    
    # Load your portfolio data
    portfolio_df = pd.read_csv('portfolio.csv')
    
    # Initialize simulator
    simulator = MonteCarloSimulator(
        num_simulations=100000,
        num_threads=8,
        use_antithetic_variates=True
    )
    
    # Run simulation
    results = simulator.run_simulation(
        portfolio_data=portfolio_df,
        default_probabilities=xgboost_predictions
    )
    
    print(f"Portfolio 95% VaR: ${results['risk_metrics']['var_95']:,.2f}")
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Union, List
import warnings
from pathlib import Path
import sys

# Add the monte_carlo directory to Python path for importing the C++ module
monte_carlo_dir = Path(__file__).parent
sys.path.insert(0, str(monte_carlo_dir))

try:
    import monte_carlo_engine
    CPLUS_MODULE_AVAILABLE = True
except ImportError as e:
    CPLUS_MODULE_AVAILABLE = False
    warnings.warn(f"C++ Monte Carlo engine not available: {e}. "
                 "Please build the C++ module first using build.sh")

class MonteCarloSimulator:
    """
    High-level Python interface to the C++ Monte Carlo simulation engine.
    
    This class provides easy integration with pandas DataFrames and XGBoost
    models while leveraging the high-performance C++ backend.
    """
    
    def __init__(self, 
                 num_simulations: int = 100000,
                 num_threads: int = 0,
                 scenarios_per_batch: int = 1000,
                 use_antithetic_variates: bool = True,
                 enable_correlation: bool = True,
                 random_seed: int = 42,
                 **kwargs):
        """
        Initialize the Monte Carlo simulator.
        
        Args:
            num_simulations: Number of Monte Carlo iterations
            num_threads: Number of OpenMP threads (0 = auto-detect)
            scenarios_per_batch: Scenarios processed per batch
            use_antithetic_variates: Enable variance reduction technique
            enable_correlation: Enable correlated economic scenarios
            random_seed: Random seed for reproducibility
            **kwargs: Additional configuration parameters
        """
        
        if not CPLUS_MODULE_AVAILABLE:
            raise RuntimeError("C++ Monte Carlo engine is not available. "
                             "Please build the module first.")
        
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config = {
            'num_simulations': num_simulations,
            'num_threads': num_threads if num_threads > 0 else monte_carlo_engine.MonteCarloEngine.get_optimal_thread_count(),
            'scenarios_per_batch': scenarios_per_batch,
            'use_antithetic_variates': use_antithetic_variates,
            'enable_correlation': enable_correlation,
            'random_seed': random_seed
        }
        
        # Update with any additional parameters
        self.config.update(kwargs)
        
        # Initialize C++ engine
        self.engine = monte_carlo_engine.MonteCarloEngine(self.config)
        
        self.logger.info(f"Monte Carlo Simulator initialized with {self.config['num_threads']} threads")
        
    def validate_portfolio_data(self, portfolio_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate portfolio data format and content.
        
        Args:
            portfolio_data: Portfolio DataFrame
            
        Returns:
            Validation results dictionary
        """
        validation = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Required columns
        required_columns = [
            'account_id', 'balance', 'limit', 'exposure_at_default', 'loss_given_default'
        ]
        
        missing_columns = [col for col in required_columns if col not in portfolio_data.columns]
        if missing_columns:
            validation['valid'] = False
            validation['errors'].append(f"Missing required columns: {missing_columns}")
        
        if not validation['valid']:
            return validation
        
        # Data validation
        if portfolio_data.empty:
            validation['valid'] = False
            validation['errors'].append("Portfolio data is empty")
        
        if portfolio_data['balance'].isna().any():
            validation['errors'].append("Found NaN values in balance column")
            validation['valid'] = False
        
        if (portfolio_data['balance'] < 0).any():
            validation['warnings'].append("Found negative balance values")
        
        if (portfolio_data['exposure_at_default'] < 0).any():
            validation['valid'] = False
            validation['errors'].append("Found negative exposure values")
        
        if (portfolio_data['loss_given_default'] < 0).any() or (portfolio_data['loss_given_default'] > 1).any():
            validation['valid'] = False
            validation['errors'].append("Loss given default must be between 0 and 1")
        
        return validation
    
    def validate_default_probabilities(self, default_probs: np.ndarray) -> Dict[str, Any]:
        """
        Validate XGBoost default probability predictions.
        
        Args:
            default_probs: Array of default probabilities
            
        Returns:
            Validation results dictionary
        """
        return monte_carlo_engine.validate_xgboost_predictions(default_probs)
    
    def run_simulation(self,
                      portfolio_data: pd.DataFrame,
                      default_probabilities: np.ndarray,
                      economic_scenario: Optional[Dict[str, float]] = None,
                      stress_factor: Optional[float] = None) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation on portfolio data.
        
        Args:
            portfolio_data: Portfolio DataFrame with account information
            default_probabilities: XGBoost predicted default probabilities
            economic_scenario: Economic scenario parameters (optional)
            stress_factor: Stress factor to apply to default probabilities (optional)
            
        Returns:
            Simulation results dictionary with risk metrics and performance stats
        """
        
        # Validate inputs
        portfolio_validation = self.validate_portfolio_data(portfolio_data)
        if not portfolio_validation['valid']:
            raise ValueError(f"Invalid portfolio data: {portfolio_validation['errors']}")
        
        prob_validation = self.validate_default_probabilities(default_probabilities)
        if not prob_validation['valid']:
            raise ValueError(f"Invalid default probabilities: {prob_validation['message']}")
        
        if len(default_probabilities) != len(portfolio_data):
            raise ValueError("Default probabilities array must match portfolio data length")
        
        # Apply stress factor if provided
        if stress_factor is not None:
            self.logger.info(f"Applying stress factor: {stress_factor}")
            default_probabilities = monte_carlo_engine.adjust_predictions_for_stress(
                default_probabilities, stress_factor)
        
        # Prepare data arrays for C++ engine
        balances = portfolio_data['balance'].values.astype(np.float64)
        limits = portfolio_data['limit'].values.astype(np.float64)
        exposures = portfolio_data['exposure_at_default'].values.astype(np.float64)
        lgds = portfolio_data['loss_given_default'].values.astype(np.float64)
        account_ids = portfolio_data['account_id'].values.astype(np.int32)
        
        # Ensure default probabilities are float64
        default_probabilities = default_probabilities.astype(np.float64)
        
        self.logger.info(f"Running simulation on {len(portfolio_data)} accounts "
                        f"with {self.config['num_simulations']} iterations")
        
        # Run C++ simulation
        try:
            results = self.engine.run_simulation(
                balances=balances,
                limits=limits,
                default_probs=default_probabilities,
                exposures=exposures,
                lgds=lgds,
                account_ids=account_ids,
                economic_scenario=economic_scenario or {}
            )
            
            # Add portfolio summary to results
            results['portfolio_summary'] = {
                'num_accounts': len(portfolio_data),
                'total_exposure': float(exposures.sum()),
                'average_default_prob': float(default_probabilities.mean()),
                'max_default_prob': float(default_probabilities.max()),
                'min_default_prob': float(default_probabilities.min())
            }
            
            # Log key results
            if results['success']:
                risk_metrics = results['risk_metrics']
                self.logger.info(f"Simulation completed successfully")
                self.logger.info(f"Expected Loss: ${risk_metrics['expected_loss']:,.2f}")
                self.logger.info(f"95% VaR: ${risk_metrics['var_95']:,.2f}")
                self.logger.info(f"99% VaR: ${risk_metrics['var_99']:,.2f}")
                self.logger.info(f"Performance: {results['performance']['iterations_per_second']:,.0f} it/s")
            else:
                self.logger.error(f"Simulation failed: {results['error_message']}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Simulation error: {str(e)}")
            return {
                'success': False,
                'error_message': str(e),
                'risk_metrics': {},
                'performance': {}
            }
    
    def run_stress_test(self,
                       portfolio_data: pd.DataFrame,
                       default_probabilities: np.ndarray,
                       stress_scenarios: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Run stress tests with multiple economic scenarios.
        
        Args:
            portfolio_data: Portfolio DataFrame
            default_probabilities: Base default probabilities
            stress_scenarios: List of stress scenario parameters
            
        Returns:
            Stress test results for all scenarios
        """
        
        stress_results = {
            'scenarios': [],
            'summary': {}
        }
        
        base_results = self.run_simulation(portfolio_data, default_probabilities)
        stress_results['base_case'] = base_results
        
        for i, scenario in enumerate(stress_scenarios):
            self.logger.info(f"Running stress scenario {i+1}/{len(stress_scenarios)}")
            
            scenario_results = self.run_simulation(
                portfolio_data, 
                default_probabilities,
                economic_scenario=scenario
            )
            
            scenario_results['scenario_params'] = scenario
            stress_results['scenarios'].append(scenario_results)
        
        # Calculate stress test summary
        if all(result['success'] for result in stress_results['scenarios']):
            var_95_values = [result['risk_metrics']['var_95'] for result in stress_results['scenarios']]
            cvar_95_values = [result['risk_metrics']['cvar_95'] for result in stress_results['scenarios']]
            
            stress_results['summary'] = {
                'worst_case_var_95': max(var_95_values),
                'worst_case_cvar_95': max(cvar_95_values),
                'var_95_range': (min(var_95_values), max(var_95_values)),
                'cvar_95_range': (min(cvar_95_values), max(cvar_95_values))
            }
        
        return stress_results
    
    def estimate_memory_usage(self, num_accounts: int) -> int:
        """
        Estimate memory usage for a given portfolio size.
        
        Args:
            num_accounts: Number of accounts in portfolio
            
        Returns:
            Estimated memory usage in MB
        """
        return monte_carlo_engine.MonteCarloEngine.estimate_memory_usage(
            num_accounts, self.config['num_simulations'])
    
    def update_config(self, **kwargs) -> None:
        """
        Update simulation configuration.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        self.config.update(kwargs)
        self.engine.update_config(self.config)
        self.logger.info("Configuration updated")
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self.config.copy()
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """
        Get system information for performance tuning.
        
        Returns:
            System information dictionary
        """
        return {
            'optimal_thread_count': monte_carlo_engine.MonteCarloEngine.get_optimal_thread_count(),
            'cplus_module_available': CPLUS_MODULE_AVAILABLE,
            'numpy_version': np.__version__,
            'pandas_version': pd.__version__
        }


class XGBoostMonteCarloIntegrator:
    """
    Integration class for XGBoost models and Monte Carlo simulation.
    
    This class provides utilities for seamless integration between
    XGBoost credit risk models and the Monte Carlo simulation engine.
    """
    
    def __init__(self, xgboost_model=None):
        """
        Initialize the integrator.
        
        Args:
            xgboost_model: Trained XGBoost model (optional)
        """
        self.model = xgboost_model
        self.logger = logging.getLogger(__name__)
    
    def predict_and_simulate(self,
                           portfolio_data: pd.DataFrame,
                           feature_columns: List[str],
                           simulation_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Predict default probabilities and run Monte Carlo simulation.
        
        Args:
            portfolio_data: Portfolio DataFrame with features
            feature_columns: Column names to use as features
            simulation_config: Monte Carlo configuration (optional)
            
        Returns:
            Combined prediction and simulation results
        """
        
        if self.model is None:
            raise ValueError("XGBoost model not provided")
        
        # Prepare feature matrix
        X = portfolio_data[feature_columns].values
        
        # Predict default probabilities
        self.logger.info("Generating XGBoost predictions...")
        default_probs = self.model.predict_proba(X)[:, 1]  # Probability of default class
        
        # Initialize Monte Carlo simulator
        config = simulation_config or {}
        simulator = MonteCarloSimulator(**config)
        
        # Run simulation
        self.logger.info("Running Monte Carlo simulation...")
        results = simulator.run_simulation(portfolio_data, default_probs)
        
        # Add XGBoost information to results
        results['xgboost_info'] = {
            'model_type': str(type(self.model).__name__),
            'num_features': len(feature_columns),
            'feature_columns': feature_columns,
            'prediction_stats': {
                'mean_default_prob': float(default_probs.mean()),
                'std_default_prob': float(default_probs.std()),
                'min_default_prob': float(default_probs.min()),
                'max_default_prob': float(default_probs.max())
            }
        }
        
        return results
    
    def set_model(self, xgboost_model):
        """Set the XGBoost model."""
        self.model = xgboost_model
        self.logger.info("XGBoost model updated")


# Utility functions
def create_stress_scenarios() -> List[Dict[str, float]]:
    """
    Create standard stress testing scenarios.
    
    Returns:
        List of stress scenario dictionaries
    """
    return [
        {
            'gdp_growth': -0.05,
            'unemployment_rate': 0.12,
            'interest_rate': 0.08,
            'market_volatility': 0.35,
            'credit_spread': 0.06
        },
        {
            'gdp_growth': -0.03,
            'unemployment_rate': 0.10,
            'interest_rate': 0.06,
            'market_volatility': 0.25,
            'credit_spread': 0.04
        },
        {
            'gdp_growth': 0.08,
            'unemployment_rate': 0.03,
            'interest_rate': 0.01,
            'market_volatility': 0.08,
            'credit_spread': 0.01
        }
    ]


def format_currency(amount: float) -> str:
    """Format currency amount for display."""
    return f"${amount:,.2f}"


def format_percentage(value: float) -> str:
    """Format percentage for display."""
    return f"{value*100:.2f}%"


# Module-level configuration
__version__ = "1.0.0"
__author__ = "AI-Enhanced Monte Carlo Simulation Engine"

# Export main classes
__all__ = [
    'MonteCarloSimulator',
    'XGBoostMonteCarloIntegrator', 
    'create_stress_scenarios',
    'format_currency',
    'format_percentage'
]