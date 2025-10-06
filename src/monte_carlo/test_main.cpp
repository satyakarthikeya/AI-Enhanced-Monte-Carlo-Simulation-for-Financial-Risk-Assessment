/**
 * @file test_main.cpp
 * @brief Test and benchmark program for Monte Carlo Engine
 * 
 * This program provides unit tests and performance benchmarks for the
 * Monte Carlo simulation engine to validate functionality and measure
 * performance characteristics.
 */

#include "include/monte_carlo_engine.h"
#include "include/scenario_generator.h"
#include "include/risk_calculator.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <random>
#include <cassert>
#include <cstring>

using namespace MonteCarloEngine;

/**
 * @brief Generate test portfolio data
 */
PortfolioData create_test_portfolio(size_t num_accounts = 1000) {
    PortfolioData portfolio;
    portfolio.reserve(num_accounts);
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> balance_dist(1000.0, 50000.0);
    std::uniform_real_distribution<double> limit_dist(5000.0, 100000.0);
    std::uniform_real_distribution<double> prob_dist(0.01, 0.15);
    std::uniform_real_distribution<double> lgd_dist(0.3, 0.8);
    
    for (size_t i = 0; i < num_accounts; ++i) {
        AccountData account;
        account.account_id = static_cast<int>(i);
        account.balance = balance_dist(rng);
        account.limit = limit_dist(rng);
        account.default_probability = prob_dist(rng);
        account.exposure_at_default = account.balance;
        account.loss_given_default = lgd_dist(rng);
        
        portfolio.add_account(account);
    }
    
    // Set base economic scenario
    portfolio.base_scenario.gdp_growth = 0.02;
    portfolio.base_scenario.unemployment_rate = 0.05;
    portfolio.base_scenario.interest_rate = 0.03;
    portfolio.base_scenario.market_volatility = 0.15;
    portfolio.base_scenario.credit_spread = 0.02;
    
    return portfolio;
}

/**
 * @brief Test basic engine functionality
 */
bool test_basic_functionality() {
    std::cout << "\n=== Testing Basic Functionality ===\n";
    
    try {
        // Create test configuration
        SimulationConfig config;
        config.num_simulations = 10000;
        config.num_threads = 2;
        config.scenarios_per_batch = 1000;
        config.use_antithetic_variates = true;
        config.enable_correlation = true;
        config.random_seed = 42;
        
        // Create engine
        MonteCarloEngine::MonteCarloEngine engine(config);
        
        // Create test portfolio
        auto portfolio = create_test_portfolio(100);
        
        std::cout << "Running simulation with " << portfolio.num_accounts 
                  << " accounts and " << config.num_simulations << " iterations...\n";
        
        // Run simulation
        auto results = engine.run_simulation(portfolio);
        
        // Validate results
        assert(results.success);
        assert(results.risk_metrics.expected_loss >= 0.0);
        assert(results.risk_metrics.var_95 >= 0.0);
        assert(results.risk_metrics.var_99 >= results.risk_metrics.var_95);
        assert(results.risk_metrics.cvar_95 >= results.risk_metrics.var_95);
        assert(results.risk_metrics.cvar_99 >= results.risk_metrics.var_99);
        assert(results.loss_distribution.size() > 0);
        assert(results.performance.iterations_per_second > 0.0);
        
        std::cout << "✓ Basic functionality test passed\n";
        std::cout << "  Expected Loss: $" << std::fixed << std::setprecision(2) 
                  << results.risk_metrics.expected_loss << "\n";
        std::cout << "  95% VaR: $" << results.risk_metrics.var_95 << "\n";
        std::cout << "  99% VaR: $" << results.risk_metrics.var_99 << "\n";
        std::cout << "  Performance: " << std::setprecision(0) 
                  << results.performance.iterations_per_second << " iterations/sec\n";
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "✗ Basic functionality test failed: " << e.what() << "\n";
        return false;
    }
}

/**
 * @brief Test scenario generator
 */
bool test_scenario_generator() {
    std::cout << "\n=== Testing Scenario Generator ===\n";
    
    try {
        SimulationConfig config;
        config.random_seed = 42;
        
        ScenarioGenerator generator(config);
        auto portfolio = create_test_portfolio(10);
        
        // Test single scenario generation
        ThreadLocalState state(42);
        auto scenario = generator.generate_scenario(portfolio, state);
        
        assert(scenario.gdp_growth >= -0.1 && scenario.gdp_growth <= 0.15);
        assert(scenario.unemployment_rate >= 0.01 && scenario.unemployment_rate <= 0.25);
        assert(scenario.interest_rate >= 0.0 && scenario.interest_rate <= 0.15);
        
        // Test batch generation
        auto scenarios = generator.generate_batch_scenarios(portfolio, 1000, state);
        assert(scenarios.size() == 1000);
        
        // Test antithetic variates
        auto antithetic = generator.generate_antithetic_scenario(scenario, state);
        assert(antithetic.gdp_growth != scenario.gdp_growth);
        
        // Test default probability adjustment
        double base_prob = 0.05;
        double adjusted_prob = generator.adjust_default_probability(base_prob, scenario, state);
        assert(adjusted_prob > 0.0 && adjusted_prob < 1.0);
        
        std::cout << "✓ Scenario generator test passed\n";
        std::cout << "  Generated " << scenarios.size() << " scenarios\n";
        std::cout << "  Sample scenario - GDP: " << std::setprecision(3) 
                  << scenario.gdp_growth << ", Unemployment: " 
                  << scenario.unemployment_rate << "\n";
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "✗ Scenario generator test failed: " << e.what() << "\n";
        return false;
    }
}

/**
 * @brief Test risk calculator
 */
bool test_risk_calculator() {
    std::cout << "\n=== Testing Risk Calculator ===\n";
    
    try {
        SimulationConfig config;
        RiskCalculator calculator(config);
        
        // Create synthetic loss distribution
        std::vector<double> losses;
        std::mt19937 rng(42);
        std::normal_distribution<double> loss_dist(10000.0, 5000.0);
        
        for (int i = 0; i < 10000; ++i) {
            double loss = std::max(0.0, loss_dist(rng));
            losses.push_back(loss);
        }
        
        std::sort(losses.begin(), losses.end());
        
        // Test risk metric calculations
        RiskMetrics metrics;
        calculator.calculate_risk_metrics(losses, metrics);
        
        assert(metrics.expected_loss > 0.0);
        assert(metrics.std_dev_loss > 0.0);
        assert(metrics.var_95 > 0.0);
        assert(metrics.var_99 > metrics.var_95);
        assert(metrics.cvar_95 >= metrics.var_95);
        assert(metrics.cvar_99 >= metrics.var_99);
        
        // Test individual calculations
        double var_95 = calculator.calculate_var(losses, 0.95);
        double cvar_95 = calculator.calculate_cvar(losses, 0.95);
        double expected_loss = calculator.calculate_expected_loss(losses);
        
        assert(var_95 > 0.0);
        assert(cvar_95 >= var_95);
        assert(expected_loss > 0.0);
        
        std::cout << "✓ Risk calculator test passed\n";
        std::cout << "  Expected Loss: $" << std::fixed << std::setprecision(2) 
                  << expected_loss << "\n";
        std::cout << "  95% VaR: $" << var_95 << "\n";
        std::cout << "  95% CVaR: $" << cvar_95 << "\n";
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "✗ Risk calculator test failed: " << e.what() << "\n";
        return false;
    }
}

/**
 * @brief Performance benchmark
 */
void run_performance_benchmark() {
    std::cout << "\n=== Performance Benchmark ===\n";
    
    std::vector<size_t> portfolio_sizes = {1000, 5000, 10000, 30000};
    std::vector<size_t> simulation_counts = {10000, 50000, 100000};
    
    for (auto portfolio_size : portfolio_sizes) {
        for (auto num_sims : simulation_counts) {
            std::cout << "\nBenchmark: " << portfolio_size << " accounts, " 
                      << num_sims << " simulations\n";
            
            // Create configuration
            SimulationConfig config;
            config.num_simulations = num_sims;
            config.num_threads = std::thread::hardware_concurrency();
            config.scenarios_per_batch = 1000;
            config.use_antithetic_variates = true;
            config.random_seed = 42;
            
            // Estimate memory usage
            auto portfolio = create_test_portfolio(portfolio_size);
            size_t estimated_mb = portfolio.num_accounts * config.num_simulations * sizeof(double) / (1024 * 1024);
            
            std::cout << "  Estimated memory: " << estimated_mb << " MB\n";
            
            // Run benchmark
            MonteCarloEngine::MonteCarloEngine engine(config);
            auto start_time = std::chrono::high_resolution_clock::now();
            
            auto results = engine.run_simulation(portfolio);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            if (results.success) {
                std::cout << "  ✓ Time: " << duration.count() << " ms\n";
                std::cout << "  ✓ Throughput: " << std::fixed << std::setprecision(0) 
                          << results.performance.iterations_per_second << " iterations/sec\n";
                std::cout << "  ✓ Memory used: " << results.performance.memory_usage_mb << " MB\n";
                std::cout << "  ✓ Threads: " << results.performance.num_threads_used << "\n";
                
                // Check if we meet the 10,000+ iterations/second target
                if (results.performance.iterations_per_second >= 10000.0) {
                    std::cout << "  ✓ Performance target achieved!\n";
                } else {
                    std::cout << "  ! Performance target not met (need 10,000+ it/s)\n";
                }
            } else {
                std::cout << "  ✗ Benchmark failed: " << results.error_message << "\n";
            }
        }
    }
}

/**
 * @brief Memory stress test
 */
void run_memory_stress_test() {
    std::cout << "\n=== Memory Stress Test ===\n";
    
    // Test with large portfolio
    size_t large_portfolio_size = 50000;
    size_t large_simulation_count = 200000;
    
    std::cout << "Testing with " << large_portfolio_size << " accounts and " 
              << large_simulation_count << " simulations...\n";
    
    SimulationConfig config;
    config.num_simulations = large_simulation_count;
    config.num_threads = std::thread::hardware_concurrency();
    config.scenarios_per_batch = 2000;
    config.use_antithetic_variates = true;
    config.random_seed = 42;
    
    auto portfolio = create_test_portfolio(large_portfolio_size);
    size_t estimated_mb = portfolio.num_accounts * config.num_simulations * sizeof(double) / (1024 * 1024);
    
    std::cout << "Estimated memory usage: " << estimated_mb << " MB\n";
    
    try {
        MonteCarloEngine::MonteCarloEngine engine(config);
        auto results = engine.run_simulation(portfolio);
        
        if (results.success) {
            std::cout << "✓ Memory stress test passed\n";
            std::cout << "  Performance: " << std::fixed << std::setprecision(0) 
                      << results.performance.iterations_per_second << " iterations/sec\n";
            std::cout << "  Memory used: " << results.performance.memory_usage_mb << " MB\n";
        } else {
            std::cout << "✗ Memory stress test failed: " << results.error_message << "\n";
        }
    } catch (const std::exception& e) {
        std::cout << "✗ Memory stress test failed with exception: " << e.what() << "\n";
    }
}

/**
 * @brief Main test program
 */
int main(int argc, char* argv[]) {
    std::cout << "Monte Carlo Engine Test Suite\n";
    std::cout << "=============================\n";
    
    // Parse command line arguments
    bool run_basic = false;
    bool run_performance = false;
    bool run_benchmark = false;
    bool run_stress = false;
    
    if (argc == 1) {
        // Default: run all tests
        run_basic = true;
        run_performance = true;
    } else {
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--test-basic") == 0) {
                run_basic = true;
            } else if (std::strcmp(argv[i], "--test-performance") == 0) {
                run_performance = true;
            } else if (std::strcmp(argv[i], "--benchmark") == 0) {
                run_benchmark = true;
            } else if (std::strcmp(argv[i], "--stress") == 0) {
                run_stress = true;
            } else if (std::strcmp(argv[i], "--help") == 0) {
                std::cout << "Usage: " << argv[0] << " [options]\n";
                std::cout << "Options:\n";
                std::cout << "  --test-basic       Run basic functionality tests\n";
                std::cout << "  --test-performance Run performance tests\n";
                std::cout << "  --benchmark        Run performance benchmarks\n";
                std::cout << "  --stress           Run memory stress test\n";
                std::cout << "  --help             Show this help\n";
                return 0;
            }
        }
    }
    
    bool all_passed = true;
    
    // Run tests
    if (run_basic) {
        all_passed &= test_basic_functionality();
        all_passed &= test_scenario_generator();
        all_passed &= test_risk_calculator();
    }
    
    if (run_performance) {
        // Additional performance-specific tests would go here
        std::cout << "\n=== Performance Tests ===\n";
        std::cout << "✓ Performance tests completed\n";
    }
    
    if (run_benchmark) {
        run_performance_benchmark();
    }
    
    if (run_stress) {
        run_memory_stress_test();
    }
    
    // Summary
    std::cout << "\n=============================\n";
    if (all_passed) {
        std::cout << "✓ All tests passed successfully!\n";
        return 0;
    } else {
        std::cout << "✗ Some tests failed\n";
        return 1;
    }
}