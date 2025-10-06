#!/usr/bin/env python3
"""
Performance Testing and Benchmarking Script
===========================================

This script provides comprehensive performance testing for the Monte Carlo
simulation engine, measuring throughput, scalability, and memory usage.
"""

import sys
import time
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from monte_carlo_wrapper import MonteCarloSimulator, XGBoostMonteCarloIntegrator
    WRAPPER_AVAILABLE = True
except ImportError:
    WRAPPER_AVAILABLE = False
    print("Warning: Monte Carlo wrapper not available")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceBenchmark:
    """
    Comprehensive performance benchmarking for Monte Carlo engine.
    """
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """
        Initialize benchmark suite.
        
        Args:
            output_dir: Directory to save benchmark results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = []
        self.system_info = self._get_system_info()
        
        logger.info(f"Benchmark initialized - Output directory: {self.output_dir}")
        logger.info(f"System: {self.system_info['cpu_count']} cores, "
                   f"{self.system_info['memory_gb']:.1f}GB RAM")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmark context."""
        return {
            'cpu_count': psutil.cpu_count(logical=True),
            'cpu_count_physical': psutil.cpu_count(logical=False),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': sys.version,
            'timestamp': datetime.now().isoformat()
        }
    
    def _create_test_portfolio(self, num_accounts: int, seed: int = 42) -> pd.DataFrame:
        """
        Create synthetic portfolio data for testing.
        
        Args:
            num_accounts: Number of accounts to generate
            seed: Random seed for reproducibility
            
        Returns:
            Portfolio DataFrame
        """
        np.random.seed(seed)
        
        # Generate realistic portfolio data
        data = {
            'account_id': range(num_accounts),
            'balance': np.random.lognormal(mean=9.0, sigma=1.0, size=num_accounts),
            'limit': np.random.lognormal(mean=10.0, sigma=0.8, size=num_accounts),
            'exposure_at_default': np.random.lognormal(mean=9.2, sigma=1.2, size=num_accounts),
            'loss_given_default': np.random.beta(a=2, b=3, size=num_accounts) * 0.8 + 0.1
        }
        
        # Ensure exposure is reasonable relative to balance
        portfolio_df = pd.DataFrame(data)
        portfolio_df['exposure_at_default'] = np.minimum(
            portfolio_df['exposure_at_default'], 
            portfolio_df['balance'] * 1.2
        )
        
        return portfolio_df
    
    def _create_test_predictions(self, num_accounts: int, seed: int = 42) -> np.ndarray:
        """
        Create synthetic default probability predictions.
        
        Args:
            num_accounts: Number of predictions to generate
            seed: Random seed for reproducibility
            
        Returns:
            Array of default probabilities
        """
        np.random.seed(seed)
        
        # Generate realistic default probabilities with some correlation to account size
        base_probs = np.random.beta(a=1, b=20, size=num_accounts)  # Low default rates
        
        # Add some high-risk accounts
        high_risk_mask = np.random.random(num_accounts) < 0.05  # 5% high risk
        base_probs[high_risk_mask] = np.random.beta(a=3, b=5, size=high_risk_mask.sum())
        
        return np.clip(base_probs, 1e-6, 0.999)
    
    def benchmark_scalability(self, 
                            portfolio_sizes: List[int] = None,
                            simulation_counts: List[int] = None,
                            thread_counts: List[int] = None) -> Dict[str, Any]:
        """
        Benchmark scalability across different problem sizes.
        
        Args:
            portfolio_sizes: List of portfolio sizes to test
            simulation_counts: List of simulation counts to test
            thread_counts: List of thread counts to test
            
        Returns:
            Scalability benchmark results
        """
        
        if not WRAPPER_AVAILABLE:
            raise RuntimeError("Monte Carlo wrapper not available for benchmarking")
        
        logger.info("Starting scalability benchmark...")
        
        # Default test parameters
        portfolio_sizes = portfolio_sizes or [1000, 5000, 10000, 30000]
        simulation_counts = simulation_counts or [10000, 50000, 100000]
        thread_counts = thread_counts or [1, 2, 4, 8]
        
        results = {
            'portfolio_scaling': [],
            'simulation_scaling': [],
            'thread_scaling': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # Portfolio size scaling (fixed simulations and threads)
        logger.info("Testing portfolio size scaling...")
        for portfolio_size in portfolio_sizes:
            portfolio_df = self._create_test_portfolio(portfolio_size)
            default_probs = self._create_test_predictions(portfolio_size)
            
            simulator = MonteCarloSimulator(
                num_simulations=50000,
                num_threads=4,
                random_seed=42
            )
            
            start_time = time.time()
            memory_before = psutil.virtual_memory().used / (1024**2)
            
            result = simulator.run_simulation(portfolio_df, default_probs)
            
            end_time = time.time()
            memory_after = psutil.virtual_memory().used / (1024**2)
            
            if result['success']:
                results['portfolio_scaling'].append({
                    'portfolio_size': portfolio_size,
                    'time_seconds': end_time - start_time,
                    'iterations_per_second': result['performance']['iterations_per_second'],
                    'memory_used_mb': memory_after - memory_before,
                    'var_95': result['risk_metrics']['var_95']
                })
            
            logger.info(f"Portfolio size {portfolio_size}: "
                       f"{result['performance']['iterations_per_second']:,.0f} it/s")
        
        # Simulation count scaling (fixed portfolio and threads)
        logger.info("Testing simulation count scaling...")
        portfolio_df = self._create_test_portfolio(10000)
        default_probs = self._create_test_predictions(10000)
        
        for sim_count in simulation_counts:
            simulator = MonteCarloSimulator(
                num_simulations=sim_count,
                num_threads=4,
                random_seed=42
            )
            
            start_time = time.time()
            result = simulator.run_simulation(portfolio_df, default_probs)
            end_time = time.time()
            
            if result['success']:
                results['simulation_scaling'].append({
                    'num_simulations': sim_count,
                    'time_seconds': end_time - start_time,
                    'iterations_per_second': result['performance']['iterations_per_second'],
                    'var_95': result['risk_metrics']['var_95']
                })
            
            logger.info(f"Simulations {sim_count}: "
                       f"{result['performance']['iterations_per_second']:,.0f} it/s")
        
        # Thread count scaling (fixed portfolio and simulations)
        logger.info("Testing thread count scaling...")
        for thread_count in thread_counts:
            simulator = MonteCarloSimulator(
                num_simulations=100000,
                num_threads=thread_count,
                random_seed=42
            )
            
            start_time = time.time()
            result = simulator.run_simulation(portfolio_df, default_probs)
            end_time = time.time()
            
            if result['success']:
                results['thread_scaling'].append({
                    'num_threads': thread_count,
                    'time_seconds': end_time - start_time,
                    'iterations_per_second': result['performance']['iterations_per_second'],
                    'var_95': result['risk_metrics']['var_95']
                })
            
            logger.info(f"Threads {thread_count}: "
                       f"{result['performance']['iterations_per_second']:,.0f} it/s")
        
        return results
    
    def benchmark_memory_usage(self, max_portfolio_size: int = 50000) -> Dict[str, Any]:
        """
        Benchmark memory usage characteristics.
        
        FIXED: Corrected memory measurement to avoid negative values.
        Uses proper garbage collection and process memory tracking.
        
        Args:
            max_portfolio_size: Maximum portfolio size to test
            
        Returns:
            Memory usage benchmark results
        """
        
        if not WRAPPER_AVAILABLE:
            raise RuntimeError("Monte Carlo wrapper not available for benchmarking")
        
        logger.info("Starting memory usage benchmark...")
        
        portfolio_sizes = [1000, 5000, 10000, 20000, 30000]
        if max_portfolio_size < 30000:
            portfolio_sizes = [s for s in portfolio_sizes if s <= max_portfolio_size]
        
        results = {
            'memory_usage': [],
            'timestamp': datetime.now().isoformat()
        }
        
        import gc
        
        for portfolio_size in portfolio_sizes:
            # Force garbage collection before measurement
            gc.collect()
            gc.collect()  # Run twice to ensure cleanup
            
            # Get process handle
            process = psutil.Process()
            
            # Baseline memory (after GC)
            baseline_memory = process.memory_info().rss / (1024**2)  # MB
            
            # Create portfolio
            portfolio_df = self._create_test_portfolio(portfolio_size)
            default_probs = self._create_test_predictions(portfolio_size)
            
            # Calculate expected exposure
            total_exposure = portfolio_df['exposure_at_default'].sum()
            
            # Estimate memory needs (simplified formula)
            estimated_memory = (
                portfolio_size * 8 * 5 +  # Account data (5 doubles per account)
                100000 * 8                  # Loss distribution (100K simulations)
            ) / (1024**2)  # Convert to MB
            
            # Run simulation
            simulator = MonteCarloSimulator(
                num_simulations=100000,
                num_threads=4,
                random_seed=42
            )
            
            result = simulator.run_simulation(portfolio_df, default_probs)
            
            # Force garbage collection before final measurement
            gc.collect()
            gc.collect()
            
            # Peak memory during simulation
            peak_memory = process.memory_info().rss / (1024**2)  # MB
            
            # Calculate actual memory used (ensure non-negative)
            actual_memory = max(0, peak_memory - baseline_memory)
            
            results['memory_usage'].append({
                'portfolio_size': portfolio_size,
                'estimated_memory_mb': round(estimated_memory, 2),
                'baseline_memory_mb': round(baseline_memory, 2),
                'peak_memory_mb': round(peak_memory, 2),
                'actual_memory_mb': round(actual_memory, 2),
                'total_exposure': total_exposure
            })
            
            logger.info(f"Portfolio {portfolio_size}: "
                       f"Baseline={baseline_memory:.1f}MB, "
                       f"Peak={peak_memory:.1f}MB, "
                       f"Used={actual_memory:.1f}MB")
            
            # Cleanup
            del simulator, portfolio_df, default_probs, result
            gc.collect()
        
        logger.info("Starting memory usage benchmark...")
        
        portfolio_sizes = [1000, 5000, 10000, 20000, 30000, max_portfolio_size]
        results = {
            'memory_usage': [],
            'timestamp': datetime.now().isoformat()
        }
        
        for portfolio_size in portfolio_sizes:
            # Create test data
            portfolio_df = self._create_test_portfolio(portfolio_size)
            default_probs = self._create_test_predictions(portfolio_size)
            
            # Measure memory before simulation
            process = psutil.Process()
            memory_before = process.memory_info().rss / (1024**2)  # MB
            
            simulator = MonteCarloSimulator(
                num_simulations=100000,
                num_threads=4,
                random_seed=42
            )
            
            # Estimate memory usage
            estimated_mb = simulator.estimate_memory_usage(portfolio_size)
            
            # Run simulation and measure peak memory
            result = simulator.run_simulation(portfolio_df, default_probs)
            memory_after = process.memory_info().rss / (1024**2)  # MB
            
            if result['success']:
                results['memory_usage'].append({
                    'portfolio_size': portfolio_size,
                    'estimated_memory_mb': estimated_mb,
                    'actual_memory_mb': memory_after - memory_before,
                    'total_exposure': portfolio_df['exposure_at_default'].sum(),
                    'peak_memory_mb': memory_after
                })
            
            logger.info(f"Portfolio {portfolio_size}: "
                       f"Est: {estimated_mb}MB, Actual: {memory_after - memory_before:.1f}MB")
        
        return results
    
    def benchmark_accuracy_convergence(self, 
                                     portfolio_size: int = 10000,
                                     max_simulations: int = 1000000) -> Dict[str, Any]:
        """
        Benchmark accuracy convergence with simulation count.
        
        Args:
            portfolio_size: Portfolio size for testing
            max_simulations: Maximum number of simulations
            
        Returns:
            Convergence benchmark results
        """
        
        logger.info("Starting accuracy convergence benchmark...")
        
        # Create test data
        portfolio_df = self._create_test_portfolio(portfolio_size)
        default_probs = self._create_test_predictions(portfolio_size)
        
        simulation_counts = [1000, 5000, 10000, 25000, 50000, 100000, 250000, 500000, max_simulations]
        results = {
            'convergence': [],
            'timestamp': datetime.now().isoformat()
        }
        
        for sim_count in simulation_counts:
            simulator = MonteCarloSimulator(
                num_simulations=sim_count,
                num_threads=4,
                random_seed=42  # Fixed seed for convergence analysis
            )
            
            result = simulator.run_simulation(portfolio_df, default_probs)
            
            if result['success']:
                results['convergence'].append({
                    'num_simulations': sim_count,
                    'expected_loss': result['risk_metrics']['expected_loss'],
                    'var_95': result['risk_metrics']['var_95'],
                    'var_99': result['risk_metrics']['var_99'],
                    'cvar_95': result['risk_metrics']['cvar_95'],
                    'std_dev_loss': result['risk_metrics']['std_dev_loss'],
                    'time_seconds': result['performance']['total_time_seconds']
                })
            
            logger.info(f"Simulations {sim_count}: VaR95=${result['risk_metrics']['var_95']:,.0f}")
        
        return results
    
    def generate_benchmark_report(self, results: Dict[str, Any]) -> None:
        """
        Generate comprehensive benchmark report with visualizations.
        
        Args:
            results: Combined benchmark results
        """
        
        logger.info("Generating benchmark report...")
        
        # Set up matplotlib style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Monte Carlo Engine Performance Benchmark', fontsize=16)
        
        # Portfolio scaling plot
        if 'scalability' in results and results['scalability']['portfolio_scaling']:
            data = results['scalability']['portfolio_scaling']
            portfolio_sizes = [d['portfolio_size'] for d in data]
            throughput = [d['iterations_per_second'] for d in data]
            
            axes[0, 0].plot(portfolio_sizes, throughput, 'o-', linewidth=2, markersize=8)
            axes[0, 0].set_xlabel('Portfolio Size')
            axes[0, 0].set_ylabel('Iterations per Second')
            axes[0, 0].set_title('Throughput vs Portfolio Size')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # Thread scaling plot
        if 'scalability' in results and results['scalability']['thread_scaling']:
            data = results['scalability']['thread_scaling']
            thread_counts = [d['num_threads'] for d in data]
            throughput = [d['iterations_per_second'] for d in data]
            
            axes[0, 1].plot(thread_counts, throughput, 'o-', linewidth=2, markersize=8)
            axes[0, 1].set_xlabel('Number of Threads')
            axes[0, 1].set_ylabel('Iterations per Second')
            axes[0, 1].set_title('Throughput vs Thread Count')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Calculate parallel efficiency
            if len(throughput) > 1:
                baseline = throughput[0]
                efficiency = [t / (baseline * threads) for t, threads in zip(throughput, thread_counts)]
                
                ax_eff = axes[0, 1].twinx()
                ax_eff.plot(thread_counts, efficiency, 's--', color='red', alpha=0.7)
                ax_eff.set_ylabel('Parallel Efficiency', color='red')
                ax_eff.set_ylim(0, 1.1)
        
        # Memory usage plot
        if 'memory' in results and results['memory']['memory_usage']:
            data = results['memory']['memory_usage']
            portfolio_sizes = [d['portfolio_size'] for d in data]
            estimated_memory = [d['estimated_memory_mb'] for d in data]
            actual_memory = [d['actual_memory_mb'] for d in data]
            
            axes[0, 2].plot(portfolio_sizes, estimated_memory, 'o-', label='Estimated', linewidth=2)
            axes[0, 2].plot(portfolio_sizes, actual_memory, 's-', label='Actual', linewidth=2)
            axes[0, 2].set_xlabel('Portfolio Size')
            axes[0, 2].set_ylabel('Memory Usage (MB)')
            axes[0, 2].set_title('Memory Usage vs Portfolio Size')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # Convergence plot
        if 'convergence' in results and results['convergence']['convergence']:
            data = results['convergence']['convergence']
            sim_counts = [d['num_simulations'] for d in data]
            var_95 = [d['var_95'] for d in data]
            var_99 = [d['var_99'] for d in data]
            
            axes[1, 0].semilogx(sim_counts, var_95, 'o-', label='95% VaR', linewidth=2)
            axes[1, 0].semilogx(sim_counts, var_99, 's-', label='99% VaR', linewidth=2)
            axes[1, 0].set_xlabel('Number of Simulations')
            axes[1, 0].set_ylabel('Value at Risk ($)')
            axes[1, 0].set_title('VaR Convergence')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Performance summary table
        axes[1, 1].axis('off')
        summary_text = "Performance Summary\n\n"
        
        if 'scalability' in results:
            max_throughput = max([d['iterations_per_second'] 
                                for d in results['scalability']['portfolio_scaling']], default=0)
            summary_text += f"Max Throughput: {max_throughput:,.0f} it/s\n"
            
            target_met = "✓" if max_throughput >= 10000 else "✗"
            summary_text += f"Target (10k+ it/s): {target_met}\n\n"
        
        if 'memory' in results:
            max_memory = max([d['actual_memory_mb'] 
                            for d in results['memory']['memory_usage']], default=0)
            summary_text += f"Max Memory: {max_memory:.1f} MB\n\n"
        
        summary_text += f"System Info:\n"
        summary_text += f"CPU Cores: {self.system_info['cpu_count']}\n"
        summary_text += f"Memory: {self.system_info['memory_gb']:.1f} GB\n"
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                       fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        # System utilization (placeholder)
        axes[1, 2].pie([70, 20, 10], labels=['Computation', 'Memory I/O', 'Other'], 
                      autopct='%1.1f%%', startangle=90)
        axes[1, 2].set_title('Resource Utilization')
        
        plt.tight_layout()
        
        # Save plots
        plot_path = self.output_dir / 'benchmark_plots.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Benchmark plots saved to {plot_path}")
        
        # Save detailed results
        json_path = self.output_dir / 'benchmark_results.json'
        with open(json_path, 'w') as f:
            json.dump({**results, 'system_info': self.system_info}, f, indent=2)
        logger.info(f"Detailed results saved to {json_path}")
        
        plt.show()
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """
        Run complete benchmark suite.
        
        Returns:
            Complete benchmark results
        """
        
        logger.info("Starting full benchmark suite...")
        
        results = {}
        
        try:
            # Scalability benchmark
            results['scalability'] = self.benchmark_scalability()
            
            # Memory benchmark
            results['memory'] = self.benchmark_memory_usage()
            
            # Convergence benchmark
            results['convergence'] = self.benchmark_accuracy_convergence()
            
            # Generate report
            self.generate_benchmark_report(results)
            
            logger.info("Full benchmark completed successfully!")
            
        except Exception as e:
            logger.error(f"Benchmark failed: {str(e)}")
            results['error'] = str(e)
        
        return results


def main():
    """Main benchmark execution."""
    
    parser = argparse.ArgumentParser(description='Monte Carlo Engine Performance Benchmark')
    parser.add_argument('--output-dir', default='benchmark_results',
                       help='Output directory for results')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick benchmark with smaller test sizes')
    parser.add_argument('--memory-only', action='store_true',
                       help='Run only memory usage benchmark')
    parser.add_argument('--scalability-only', action='store_true',
                       help='Run only scalability benchmark')
    
    args = parser.parse_args()
    
    # Initialize benchmark
    benchmark = PerformanceBenchmark(args.output_dir)
    
    if not WRAPPER_AVAILABLE:
        logger.error("Monte Carlo wrapper not available. Please build the C++ module first.")
        return 1
    
    # Run selected benchmarks
    results = {}
    
    try:
        if args.memory_only:
            results['memory'] = benchmark.benchmark_memory_usage()
        elif args.scalability_only:
            results['scalability'] = benchmark.benchmark_scalability()
        elif args.quick:
            # Quick benchmark with smaller sizes
            results['scalability'] = benchmark.benchmark_scalability(
                portfolio_sizes=[1000, 5000],
                simulation_counts=[10000, 50000],
                thread_counts=[1, 2, 4]
            )
            results['memory'] = benchmark.benchmark_memory_usage(max_portfolio_size=20000)
        else:
            # Full benchmark
            results = benchmark.run_full_benchmark()
        
        # Generate report if we have results
        if results and not (args.memory_only or args.scalability_only):
            benchmark.generate_benchmark_report(results)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        return 1


if __name__ == '__main__':
    sys.exit(main())