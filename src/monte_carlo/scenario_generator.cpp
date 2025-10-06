/**
 * @file scenario_generator.cpp
 * @brief Implementation of economic scenario generator
 * 
 * This file implements the ScenarioGenerator class that creates correlated
 * economic scenarios using factor models and provides thread-safe random
 * number generation for Monte Carlo simulation.
 */

#include "include/scenario_generator.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace MonteCarloEngine {

ScenarioGenerator::ScenarioGenerator(const SimulationConfig& config) 
    : config_(config) {
    initialize_default_parameters();
}

void ScenarioGenerator::update_config(const SimulationConfig& config) {
    config_ = config;
}

void ScenarioGenerator::initialize_default_parameters() {
    // Initialize factor model with empirically-derived parameters
    // Based on typical credit risk factor models used in financial institutions
    
    factor_model_.gdp_sensitivity = -2.5;          // Higher GDP reduces default risk
    factor_model_.unemployment_sensitivity = 1.8;  // Higher unemployment increases default risk
    factor_model_.interest_rate_sensitivity = 0.7; // Higher rates increase default risk
    factor_model_.market_vol_sensitivity = 1.2;    // Higher volatility increases default risk
    factor_model_.credit_spread_sensitivity = 2.0; // Higher spreads increase default risk
    
    // Initialize factor means (neutral economic conditions)
    factor_means_ = {
        0.02,   // GDP growth (2% baseline)
        0.05,   // Unemployment rate (5% baseline)
        0.03,   // Interest rate (3% baseline)
        0.15,   // Market volatility (15% baseline)
        0.02    // Credit spread (200 bps baseline)
    };
    
    // Initialize factor volatilities
    factor_volatilities_ = {
        0.01,   // GDP volatility (1%)
        0.015,  // Unemployment volatility (1.5%)
        0.008,  // Interest rate volatility (0.8%)
        0.05,   // Market volatility volatility (5%)
        0.01    // Credit spread volatility (1%)
    };
    
    // Initialize factor correlation matrix (5x5)
    factor_correlation_ = {
        {1.0,  -0.7,  0.3,   0.4,   0.6},  // GDP correlations
        {-0.7,  1.0, -0.2,  -0.3,  -0.5},  // Unemployment correlations
        {0.3,  -0.2,  1.0,   0.2,   0.4},  // Interest rate correlations
        {0.4,  -0.3,  0.2,   1.0,   0.5},  // Market volatility correlations
        {0.6,  -0.5,  0.4,   0.5,   1.0}   // Credit spread correlations
    };
    
    std::cout << "Scenario generator initialized with default factor model parameters\n";
}

std::vector<EconomicScenario> ScenarioGenerator::generate_batch_scenarios(
    const PortfolioData& portfolio,
    size_t num_scenarios,
    ThreadLocalState& thread_state) const {
    
    std::vector<EconomicScenario> scenarios;
    scenarios.reserve(num_scenarios);
    
    const EconomicScenario& base_scenario = portfolio.base_scenario;
    
    // Generate scenarios in batch for efficiency
    for (size_t i = 0; i < num_scenarios; ++i) {
        // Generate correlated factor values
        auto factors = generate_correlated_factors(thread_state);
        
        // Convert factors to economic scenario
        EconomicScenario scenario = factors_to_scenario(factors, base_scenario);
        scenarios.push_back(scenario);
    }
    
    return scenarios;
}

EconomicScenario ScenarioGenerator::generate_scenario(
    const PortfolioData& portfolio,
    ThreadLocalState& thread_state) const {
    
    auto factors = generate_correlated_factors(thread_state);
    return factors_to_scenario(factors, portfolio.base_scenario);
}

EconomicScenario ScenarioGenerator::generate_antithetic_scenario(
    const EconomicScenario& original_scenario,
    ThreadLocalState& thread_state) const {
    
    // For antithetic variates, we generate scenarios with opposite shocks
    // This reduces variance in the Monte Carlo estimation
    
    EconomicScenario antithetic;
    
    // Use opposite deviations from the mean
    antithetic.gdp_growth = 2 * factor_means_[0] - original_scenario.gdp_growth;
    antithetic.unemployment_rate = 2 * factor_means_[1] - original_scenario.unemployment_rate;
    antithetic.interest_rate = 2 * factor_means_[2] - original_scenario.interest_rate;
    antithetic.market_volatility = 2 * factor_means_[3] - original_scenario.market_volatility;
    antithetic.credit_spread = 2 * factor_means_[4] - original_scenario.credit_spread;
    
    // Ensure values remain in reasonable bounds
    antithetic.gdp_growth = std::max(-0.1, std::min(0.15, antithetic.gdp_growth));
    antithetic.unemployment_rate = std::max(0.01, std::min(0.25, antithetic.unemployment_rate));
    antithetic.interest_rate = std::max(0.0, std::min(0.15, antithetic.interest_rate));
    antithetic.market_volatility = std::max(0.05, std::min(0.8, antithetic.market_volatility));
    antithetic.credit_spread = std::max(0.0, std::min(0.1, antithetic.credit_spread));
    
    return antithetic;
}

double ScenarioGenerator::adjust_default_probability(
    double base_probability,
    const EconomicScenario& scenario,
    ThreadLocalState& thread_state) const {
    
    // Calculate systematic risk factor adjustment
    double systematic_factor = 0.0;
    
    // GDP impact (negative coefficient - higher GDP reduces default risk)
    systematic_factor += factor_model_.gdp_sensitivity * 
                        (scenario.gdp_growth - factor_means_[0]);
    
    // Unemployment impact (positive coefficient - higher unemployment increases risk)
    systematic_factor += factor_model_.unemployment_sensitivity * 
                        (scenario.unemployment_rate - factor_means_[1]);
    
    // Interest rate impact (positive coefficient - higher rates increase risk)
    systematic_factor += factor_model_.interest_rate_sensitivity * 
                        (scenario.interest_rate - factor_means_[2]);
    
    // Market volatility impact (positive coefficient - higher volatility increases risk)
    systematic_factor += factor_model_.market_vol_sensitivity * 
                        (scenario.market_volatility - factor_means_[3]);
    
    // Credit spread impact (positive coefficient - higher spreads increase risk)
    systematic_factor += factor_model_.credit_spread_sensitivity * 
                        (scenario.credit_spread - factor_means_[4]);
    
    // Apply logistic transformation to keep probability in [0,1]
    double log_odds = std::log(base_probability / (1.0 - base_probability));
    double adjusted_log_odds = log_odds + systematic_factor;
    double adjusted_probability = logistic_transform(adjusted_log_odds);
    
    // Ensure probability remains in valid range
    return std::max(1e-6, std::min(0.999, adjusted_probability));
}

std::vector<double> ScenarioGenerator::generate_correlated_factors(
    ThreadLocalState& thread_state) const {
    
    // Generate independent standard normal random variables
    std::vector<double> independent_values(factor_means_.size());
    for (size_t i = 0; i < independent_values.size(); ++i) {
        independent_values[i] = thread_state.normal_dist(thread_state.generator);
    }
    
    // Apply correlation structure using Cholesky decomposition
    auto correlated_values = apply_correlation_structure(independent_values);
    
    // Transform to economic factors
    std::vector<double> factors(factor_means_.size());
    for (size_t i = 0; i < factors.size(); ++i) {
        factors[i] = factor_means_[i] + 
                    factor_volatilities_[i] * correlated_values[i];
    }
    
    return factors;
}

std::vector<double> ScenarioGenerator::apply_correlation_structure(
    const std::vector<double>& random_values) const {
    
    // Compute Cholesky decomposition if not cached
    static thread_local std::vector<std::vector<double>> cholesky_matrix;
    static thread_local bool cholesky_computed = false;
    
    if (!cholesky_computed) {
        cholesky_matrix = cholesky_decomposition(factor_correlation_);
        cholesky_computed = true;
    }
    
    // Apply Cholesky transformation: correlated = L * independent
    std::vector<double> correlated(random_values.size(), 0.0);
    
    for (size_t i = 0; i < correlated.size(); ++i) {
        for (size_t j = 0; j <= i; ++j) {
            correlated[i] += cholesky_matrix[i][j] * random_values[j];
        }
    }
    
    return correlated;
}

EconomicScenario ScenarioGenerator::factors_to_scenario(
    const std::vector<double>& factors,
    const EconomicScenario& base_scenario) const {
    
    EconomicScenario scenario;
    
    // Apply bounds to ensure realistic economic values
    scenario.gdp_growth = std::max(-0.1, std::min(0.15, factors[0]));
    scenario.unemployment_rate = std::max(0.01, std::min(0.25, factors[1]));
    scenario.interest_rate = std::max(0.0, std::min(0.15, factors[2]));
    scenario.market_volatility = std::max(0.05, std::min(0.8, factors[3]));
    scenario.credit_spread = std::max(0.0, std::min(0.1, factors[4]));
    
    return scenario;
}

void ScenarioGenerator::calibrate_factor_model(const PortfolioData& portfolio) {
    // This method would typically use historical data to calibrate the factor model
    // For this implementation, we use reasonable defaults based on financial literature
    
    std::cout << "Using default factor model calibration\n";
    std::cout << "GDP sensitivity: " << factor_model_.gdp_sensitivity << "\n";
    std::cout << "Unemployment sensitivity: " << factor_model_.unemployment_sensitivity << "\n";
    std::cout << "Interest rate sensitivity: " << factor_model_.interest_rate_sensitivity << "\n";
    std::cout << "Market volatility sensitivity: " << factor_model_.market_vol_sensitivity << "\n";
    std::cout << "Credit spread sensitivity: " << factor_model_.credit_spread_sensitivity << "\n";
}

void ScenarioGenerator::set_factor_loadings(
    const std::vector<std::vector<double>>& loadings) {
    factor_loadings_ = loadings;
}

double ScenarioGenerator::logistic_transform(double x) const {
    // Prevent numerical overflow/underflow
    x = std::max(-500.0, std::min(500.0, x));
    return 1.0 / (1.0 + std::exp(-x));
}

std::vector<std::vector<double>> ScenarioGenerator::cholesky_decomposition(
    const std::vector<std::vector<double>>& correlation) const {
    
    size_t n = correlation.size();
    std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            if (i == j) {
                // Diagonal elements
                double sum = 0.0;
                for (size_t k = 0; k < j; ++k) {
                    sum += L[j][k] * L[j][k];
                }
                double val = correlation[j][j] - sum;
                if (val <= 0.0) {
                    throw std::runtime_error("Correlation matrix is not positive definite");
                }
                L[j][j] = std::sqrt(val);
            } else {
                // Off-diagonal elements
                double sum = 0.0;
                for (size_t k = 0; k < j; ++k) {
                    sum += L[i][k] * L[j][k];
                }
                L[i][j] = (correlation[i][j] - sum) / L[j][j];
            }
        }
    }
    
    return L;
}

} // namespace MonteCarloEngine