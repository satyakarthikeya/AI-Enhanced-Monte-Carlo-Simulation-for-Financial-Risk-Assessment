/**
 * @file monte_carlo_types.h
 * @brief Type definitions and data structures for Monte Carlo simulation engine
 * 
 * This header defines the core data structures and types used throughout
 * the OpenMP-based Monte Carlo simulation engine for financial risk assessment.
 */

#ifndef MONTE_CARLO_TYPES_H
#define MONTE_CARLO_TYPES_H

#include <vector>
#include <memory>
#include <chrono>
#include <random>
#include <string>
#include <atomic>

namespace MonteCarloEngine {

// Forward declarations
struct PortfolioData;
struct SimulationResults;
struct RiskMetrics;
struct PerformanceStats;

/**
 * @brief Portfolio account data structure
 */
struct AccountData {
    double balance;              // Account balance
    double limit;               // Credit limit
    double default_probability; // XGBoost predicted default probability
    double exposure_at_default; // Exposure amount if default occurs
    double loss_given_default;  // Loss percentage if default occurs
    int account_id;             // Unique account identifier
    
    AccountData() : balance(0.0), limit(0.0), default_probability(0.0),
                   exposure_at_default(0.0), loss_given_default(0.0), 
                   account_id(0) {}
};

/**
 * @brief Economic scenario parameters for correlated defaults
 */
struct EconomicScenario {
    double gdp_growth;          // GDP growth rate
    double unemployment_rate;   // Unemployment rate
    double interest_rate;       // Base interest rate
    double market_volatility;   // Market volatility index
    double credit_spread;       // Credit spread over risk-free rate
    
    EconomicScenario() : gdp_growth(0.0), unemployment_rate(0.0),
                        interest_rate(0.0), market_volatility(0.0),
                        credit_spread(0.0) {}
};

/**
 * @brief Portfolio data container
 */
struct PortfolioData {
    std::vector<AccountData> accounts;
    std::vector<std::vector<double>> correlation_matrix; // Account correlation matrix
    EconomicScenario base_scenario;
    double total_exposure;
    size_t num_accounts;
    
    PortfolioData() : total_exposure(0.0), num_accounts(0) {}
    
    void reserve(size_t size) {
        accounts.reserve(size);
        num_accounts = size;
    }
    
    void add_account(const AccountData& account) {
        accounts.push_back(account);
        total_exposure += account.exposure_at_default;
        num_accounts = accounts.size();
    }
};

/**
 * @brief Risk metrics calculation results
 */
struct RiskMetrics {
    double var_95;              // 95% Value at Risk
    double var_99;              // 99% Value at Risk
    double cvar_95;             // 95% Conditional Value at Risk
    double cvar_99;             // 99% Conditional Value at Risk
    double expected_loss;       // Expected portfolio loss
    double max_loss;            // Maximum observed loss
    double std_dev_loss;        // Standard deviation of losses
    double skewness;            // Loss distribution skewness
    double kurtosis;            // Loss distribution kurtosis
    
    RiskMetrics() : var_95(0.0), var_99(0.0), cvar_95(0.0), cvar_99(0.0),
                   expected_loss(0.0), max_loss(0.0), std_dev_loss(0.0),
                   skewness(0.0), kurtosis(0.0) {}
};

/**
 * @brief Performance monitoring statistics
 */
struct PerformanceStats {
    std::chrono::duration<double> total_time;
    std::chrono::duration<double> simulation_time;
    std::chrono::duration<double> calculation_time;
    double iterations_per_second;
    size_t memory_usage_mb;
    int num_threads_used;
    size_t cache_hits;
    size_t cache_misses;
    
    PerformanceStats() : total_time(0), simulation_time(0), calculation_time(0),
                        iterations_per_second(0.0), memory_usage_mb(0),
                        num_threads_used(1), cache_hits(0), cache_misses(0) {}
};

/**
 * @brief Simulation configuration parameters
 */
struct SimulationConfig {
    size_t num_simulations;         // Number of Monte Carlo iterations
    int num_threads;                // Number of OpenMP threads
    size_t scenarios_per_batch;     // Scenarios processed per batch
    bool use_antithetic_variates;   // Use variance reduction technique
    bool enable_correlation;        // Enable correlated defaults
    double confidence_level_95;     // 95% confidence level
    double confidence_level_99;     // 99% confidence level
    int random_seed;                // Random seed for reproducibility
    
    SimulationConfig() : num_simulations(100000), num_threads(0),
                        scenarios_per_batch(1000), use_antithetic_variates(true),
                        enable_correlation(true), confidence_level_95(0.95),
                        confidence_level_99(0.99), random_seed(42) {}
};

/**
 * @brief Complete simulation results
 */
struct SimulationResults {
    RiskMetrics risk_metrics;
    PerformanceStats performance;
    std::vector<double> loss_distribution;  // All simulated losses
    std::vector<double> scenario_losses;    // Per-scenario losses
    bool success;
    std::string error_message;
    
    SimulationResults() : success(false) {}
    
    void reserve_losses(size_t size) {
        loss_distribution.reserve(size);
        scenario_losses.reserve(size);
    }
};

/**
 * @brief Thread-local random state for OpenMP
 */
struct ThreadLocalState {
    std::mt19937 generator;
    std::normal_distribution<double> normal_dist;
    std::uniform_real_distribution<double> uniform_dist;
    
    ThreadLocalState(int seed) : generator(seed), normal_dist(0.0, 1.0), uniform_dist(0.0, 1.0) {}
};

} // namespace MonteCarloEngine

#endif // MONTE_CARLO_TYPES_H