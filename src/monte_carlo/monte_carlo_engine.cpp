/**
 * @file monte_carlo_engine.cpp
 * @brief Implementation of the main Monte Carlo simulation engine
 * 
 * This file implements the high-performance OpenMP-based Monte Carlo simulation
 * engine for financial risk assessment. It provides parallel execution,
 * efficient memory management, and comprehensive performance monitoring.
 */

#include "include/monte_carlo_engine.h"
#include "include/scenario_generator.h"
#include "include/risk_calculator.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <atomic>

namespace MonteCarloEngine {

MonteCarloEngine::MonteCarloEngine(const SimulationConfig& config) 
    : config_(config) {
    
    // Validate configuration
    if (config_.num_simulations == 0) {
        throw std::invalid_argument("Number of simulations must be greater than 0");
    }
    
    if (config_.num_threads <= 0) {
        config_.num_threads = get_optimal_thread_count();
    }
    
    // Set OpenMP configuration
    omp_set_num_threads(config_.num_threads);
    omp_set_dynamic(0); // Disable dynamic thread adjustment
    
    // Initialize components
    scenario_generator_ = std::make_unique<ScenarioGenerator>(config_);
    risk_calculator_ = std::make_unique<RiskCalculator>(config_);
    
    // Initialize thread-local states
    initialize_thread_states();
    
    std::cout << "Monte Carlo Engine initialized with " << config_.num_threads 
              << " threads for " << config_.num_simulations << " simulations\n";
}

MonteCarloEngine::~MonteCarloEngine() {
    cleanup_resources();
}

SimulationResults MonteCarloEngine::run_simulation(const PortfolioData& portfolio) {
    SimulationResults results;
    
    try {
        // Start timing
        start_time_ = std::chrono::high_resolution_clock::now();
        
        // Validate input
        if (!validate_portfolio(portfolio)) {
            results.error_message = "Invalid portfolio data";
            return results;
        }
        
        // Allocate memory for parallel computation
        allocate_simulation_memory(portfolio);
        
        // Initialize results container
        results.reserve_losses(config_.num_simulations);
        
        std::cout << "Starting Monte Carlo simulation...\n";
        std::cout << "Portfolio: " << portfolio.num_accounts << " accounts, "
                  << "Total exposure: $" << std::fixed << std::setprecision(2) 
                  << portfolio.total_exposure << "\n";
        
        // Run parallel simulation
        auto sim_start = std::chrono::high_resolution_clock::now();
        run_parallel_batches(portfolio, results);
        auto sim_end = std::chrono::high_resolution_clock::now();
        
        // Calculate risk metrics
        auto calc_start = std::chrono::high_resolution_clock::now();
        aggregate_thread_results(results);
        risk_calculator_->calculate_risk_metrics(results.loss_distribution, results.risk_metrics);
        auto calc_end = std::chrono::high_resolution_clock::now();
        
        // Calculate performance metrics
        end_time_ = std::chrono::high_resolution_clock::now();
        results.performance.simulation_time = sim_end - sim_start;
        results.performance.calculation_time = calc_end - calc_start;
        calculate_performance_metrics(results);
        
        results.success = true;
        
        // Print summary
        std::cout << "Simulation completed successfully!\n";
        std::cout << "Performance: " << std::fixed << std::setprecision(0) 
                  << results.performance.iterations_per_second << " iterations/second\n";
        std::cout << "Expected Loss: $" << std::fixed << std::setprecision(2) 
                  << results.risk_metrics.expected_loss << "\n";
        std::cout << "95% VaR: $" << results.risk_metrics.var_95 << "\n";
        std::cout << "99% VaR: $" << results.risk_metrics.var_99 << "\n";
        
    } catch (const std::exception& e) {
        results.success = false;
        results.error_message = std::string("Simulation error: ") + e.what();
        std::cerr << "Error: " << results.error_message << std::endl;
    }
    
    return results;
}

void MonteCarloEngine::run_parallel_batches(const PortfolioData& portfolio, 
                                           SimulationResults& results) {
    
    const size_t total_sims = config_.num_simulations;
    const size_t batch_size = config_.scenarios_per_batch;
    const size_t num_batches = (total_sims + batch_size - 1) / batch_size;
    
    // Progress tracking
    std::atomic<size_t> completed_batches(0);
    std::atomic<size_t> total_completed_sims(0);
    
    // OpenMP parallel execution
    #pragma omp parallel
    {
        const int thread_id = omp_get_thread_num();
        const int num_threads = omp_get_num_threads();
        
        // Thread-local loss accumulator
        std::vector<double> local_losses;
        local_losses.reserve(batch_size * 2); // Extra space for antithetic variates
        
        // Distribute batches across threads
        #pragma omp for schedule(dynamic, 1) nowait
        for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            const size_t start_sim = batch_idx * batch_size;
            const size_t end_sim = std::min(start_sim + batch_size, total_sims);
            const size_t sims_in_batch = end_sim - start_sim;
            
            // Generate scenarios for this batch
            auto scenarios = scenario_generator_->generate_batch_scenarios(
                portfolio, sims_in_batch, thread_states_[thread_id]);
            
            // Calculate losses for each scenario
            for (size_t i = 0; i < scenarios.size(); ++i) {
                double loss = calculate_scenario_loss(portfolio, scenarios[i], thread_id);
                local_losses.push_back(loss);
                
                // Antithetic variate for variance reduction
                if (config_.use_antithetic_variates && i < sims_in_batch) {
                    auto antithetic_scenario = scenario_generator_->generate_antithetic_scenario(
                        scenarios[i], thread_states_[thread_id]);
                    double antithetic_loss = calculate_scenario_loss(
                        portfolio, antithetic_scenario, thread_id);
                    local_losses.push_back(antithetic_loss);
                }
            }
            
            // Update progress
            completed_batches.fetch_add(1);
            total_completed_sims.fetch_add(sims_in_batch);
            
            // Progress reporting (only from thread 0)
            if (thread_id == 0 && completed_batches % 100 == 0) {
                double progress = (double)total_completed_sims / total_sims * 100.0;
                std::cout << "\rProgress: " << std::fixed << std::setprecision(1) 
                          << progress << "% (" << total_completed_sims 
                          << "/" << total_sims << " simulations)" << std::flush;
            }
        }
        
        // Store thread-local results
        #pragma omp critical
        {
            thread_local_losses_[thread_id] = std::move(local_losses);
        }
    }
    
    std::cout << "\nParallel simulation completed.\n";
}

double MonteCarloEngine::calculate_scenario_loss(const PortfolioData& portfolio,
                                                const EconomicScenario& scenario,
                                                int thread_id) {
    double total_loss = 0.0;
    ThreadLocalState& state = thread_states_[thread_id];
    
    // Calculate correlated defaults and losses
    for (size_t i = 0; i < portfolio.accounts.size(); ++i) {
        const auto& account = portfolio.accounts[i];
        
        // Adjust default probability based on economic scenario
        double adjusted_prob = scenario_generator_->adjust_default_probability(
            account.default_probability, scenario, state);
        
        // Generate random default event
        double random_val = state.uniform_dist(state.generator);
        
        if (random_val < adjusted_prob) {
            // Default occurred - calculate loss
            double loss_amount = account.exposure_at_default * account.loss_given_default;
            total_loss += loss_amount;
        }
    }
    
    return total_loss;
}

void MonteCarloEngine::aggregate_thread_results(SimulationResults& results) {
    // Combine all thread-local losses
    combined_losses_.clear();
    
    size_t total_losses = 0;
    for (const auto& thread_losses : thread_local_losses_) {
        total_losses += thread_losses.size();
    }
    
    combined_losses_.reserve(total_losses);
    
    for (const auto& thread_losses : thread_local_losses_) {
        combined_losses_.insert(combined_losses_.end(), 
                               thread_losses.begin(), 
                               thread_losses.end());
    }
    
    // Sort losses for quantile calculations
    std::sort(combined_losses_.begin(), combined_losses_.end());
    
    // Store in results
    results.loss_distribution = combined_losses_;
    
    std::cout << "Aggregated " << combined_losses_.size() 
              << " loss samples from " << config_.num_threads << " threads\n";
}

void MonteCarloEngine::calculate_performance_metrics(SimulationResults& results) const {
    results.performance.total_time = end_time_ - start_time_;
    
    double total_seconds = results.performance.total_time.count();
    results.performance.iterations_per_second = 
        static_cast<double>(results.loss_distribution.size()) / total_seconds;
    
    results.performance.num_threads_used = config_.num_threads;
    results.performance.memory_usage_mb = estimate_memory_usage(PortfolioData{}, config_);
}

bool MonteCarloEngine::validate_portfolio(const PortfolioData& portfolio) const {
    if (portfolio.accounts.empty()) {
        std::cerr << "Error: Portfolio contains no accounts\n";
        return false;
    }
    
    if (portfolio.total_exposure <= 0.0) {
        std::cerr << "Error: Total portfolio exposure must be positive\n";
        return false;
    }
    
    // Validate individual accounts
    for (const auto& account : portfolio.accounts) {
        if (account.default_probability < 0.0 || account.default_probability > 1.0) {
            std::cerr << "Error: Invalid default probability for account " 
                      << account.account_id << "\n";
            return false;
        }
        
        if (account.exposure_at_default < 0.0) {
            std::cerr << "Error: Negative exposure for account " 
                      << account.account_id << "\n";
            return false;
        }
        
        if (account.loss_given_default < 0.0 || account.loss_given_default > 1.0) {
            std::cerr << "Error: Invalid loss given default for account " 
                      << account.account_id << "\n";
            return false;
        }
    }
    
    return true;
}

void MonteCarloEngine::initialize_thread_states() {
    thread_states_.clear();
    thread_states_.reserve(config_.num_threads);
    
    for (int i = 0; i < config_.num_threads; ++i) {
        thread_states_.emplace_back(config_.random_seed + i);
    }
    
    thread_local_losses_.resize(config_.num_threads);
}

void MonteCarloEngine::allocate_simulation_memory(const PortfolioData& portfolio) {
    // Estimate memory needs
    size_t estimated_mb = estimate_memory_usage(portfolio, config_);
    
    std::cout << "Estimated memory usage: " << estimated_mb << " MB\n";
    
    // Pre-allocate thread-local storage
    const size_t losses_per_thread = (config_.num_simulations + config_.num_threads - 1) 
                                    / config_.num_threads;
    
    for (auto& thread_losses : thread_local_losses_) {
        thread_losses.clear();
        thread_losses.reserve(losses_per_thread * 2); // Account for antithetic variates
    }
    
    // Pre-allocate combined losses vector
    size_t total_capacity = config_.num_simulations;
    if (config_.use_antithetic_variates) {
        total_capacity *= 2;
    }
    combined_losses_.clear();
    combined_losses_.reserve(total_capacity);
}

void MonteCarloEngine::cleanup_resources() {
    thread_states_.clear();
    thread_local_losses_.clear();
    combined_losses_.clear();
    combined_losses_.shrink_to_fit();
}

void MonteCarloEngine::update_config(const SimulationConfig& config) {
    config_ = config;
    if (config_.num_threads <= 0) {
        config_.num_threads = get_optimal_thread_count();
    }
    
    omp_set_num_threads(config_.num_threads);
    initialize_thread_states();
    
    // Update components
    scenario_generator_->update_config(config_);
    risk_calculator_->update_config(config_);
}

int MonteCarloEngine::get_optimal_thread_count() {
    int max_threads = omp_get_max_threads();
    int logical_cores = std::thread::hardware_concurrency();
    
    // Use all available threads, but cap at a reasonable limit
    return std::min(max_threads, std::max(1, logical_cores));
}

size_t MonteCarloEngine::estimate_memory_usage(const PortfolioData& portfolio, 
                                              const SimulationConfig& config) {
    // Memory for loss storage
    size_t loss_storage = config.num_simulations * sizeof(double);
    if (config.use_antithetic_variates) {
        loss_storage *= 2;
    }
    
    // Memory for thread-local storage
    size_t thread_storage = config.num_threads * 
                           (config.num_simulations / config.num_threads) * 
                           sizeof(double);
    
    // Memory for portfolio data
    size_t portfolio_storage = portfolio.num_accounts * sizeof(AccountData);
    
    // Additional overhead
    size_t overhead = 100 * 1024 * 1024; // 100 MB overhead
    
    return (loss_storage + thread_storage + portfolio_storage + overhead) / (1024 * 1024);
}

} // namespace MonteCarloEngine