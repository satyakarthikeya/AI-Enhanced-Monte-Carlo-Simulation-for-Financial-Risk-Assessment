/**
 * @file monte_carlo_engine.h
 * @brief Main Monte Carlo simulation engine header
 * 
 * This header declares the main MonteCarloEngine class that orchestrates
 * the parallel Monte Carlo simulation using OpenMP.
 */

#ifndef MONTE_CARLO_ENGINE_H
#define MONTE_CARLO_ENGINE_H

#include "monte_carlo_types.h"
#include <omp.h>
#include <memory>
#include <vector>
#include <random>
#include <chrono>
#include <thread>

namespace MonteCarloEngine {

// Forward declarations
class ScenarioGenerator;
class RiskCalculator;

/**
 * @brief Main Monte Carlo simulation engine class
 * 
 * This class provides the main interface for running high-performance
 * Monte Carlo simulations with OpenMP parallelization.
 */
class MonteCarloEngine {
private:
    std::unique_ptr<ScenarioGenerator> scenario_generator_;
    std::unique_ptr<RiskCalculator> risk_calculator_;
    SimulationConfig config_;
    std::vector<ThreadLocalState> thread_states_;
    
    // Performance monitoring
    mutable std::chrono::high_resolution_clock::time_point start_time_;
    mutable std::chrono::high_resolution_clock::time_point end_time_;
    
    // Memory management
    std::vector<std::vector<double>> thread_local_losses_;
    std::vector<double> combined_losses_;
    
public:
    /**
     * @brief Constructor
     * @param config Simulation configuration parameters
     */
    explicit MonteCarloEngine(const SimulationConfig& config);
    
    /**
     * @brief Destructor
     */
    ~MonteCarloEngine();
    
    /**
     * @brief Run Monte Carlo simulation
     * @param portfolio Portfolio data with account information
     * @return Complete simulation results including risk metrics and performance
     */
    SimulationResults run_simulation(const PortfolioData& portfolio);
    
    /**
     * @brief Update simulation configuration
     * @param config New configuration parameters
     */
    void update_config(const SimulationConfig& config);
    
    /**
     * @brief Get current configuration
     * @return Current simulation configuration
     */
    const SimulationConfig& get_config() const { return config_; }
    
    /**
     * @brief Validate portfolio data
     * @param portfolio Portfolio to validate
     * @return True if portfolio is valid
     */
    bool validate_portfolio(const PortfolioData& portfolio) const;
    
    /**
     * @brief Get optimal number of threads for current system
     * @return Recommended number of threads
     */
    static int get_optimal_thread_count();
    
    /**
     * @brief Estimate memory requirements
     * @param portfolio Portfolio data
     * @param config Simulation configuration
     * @return Estimated memory usage in MB
     */
    static size_t estimate_memory_usage(const PortfolioData& portfolio, 
                                       const SimulationConfig& config);

private:
    /**
     * @brief Initialize OpenMP thread states
     */
    void initialize_thread_states();
    
    /**
     * @brief Allocate memory for parallel computation
     * @param portfolio Portfolio data for sizing
     */
    void allocate_simulation_memory(const PortfolioData& portfolio);
    
    /**
     * @brief Run parallel simulation batches
     * @param portfolio Portfolio data
     * @param results Results container
     */
    void run_parallel_batches(const PortfolioData& portfolio, 
                             SimulationResults& results);
    
    /**
     * @brief Aggregate results from all threads
     * @param results Results container to populate
     */
    void aggregate_thread_results(SimulationResults& results);
    
    /**
     * @brief Calculate performance metrics
     * @param results Results container to update
     */
    void calculate_performance_metrics(SimulationResults& results) const;
    
    /**
     * @brief Cleanup simulation resources
     */
    void cleanup_resources();
    
    /**
     * @brief Calculate scenario loss for a specific economic scenario
     * @param portfolio Portfolio data
     * @param scenario Economic scenario
     * @param thread_id Current thread ID
     * @return Portfolio loss for the scenario
     */
    double calculate_scenario_loss(const PortfolioData& portfolio,
                                  const EconomicScenario& scenario,
                                  int thread_id);
};

} // namespace MonteCarloEngine

#endif // MONTE_CARLO_ENGINE_H