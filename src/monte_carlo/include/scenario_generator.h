/**
 * @file scenario_generator.h
 * @brief Economic scenario generator header for Monte Carlo simulation
 * 
 * This header declares the ScenarioGenerator class responsible for generating
 * correlated economic scenarios and adjusting default probabilities.
 */

#ifndef SCENARIO_GENERATOR_H
#define SCENARIO_GENERATOR_H

#include "monte_carlo_types.h"
#include <random>
#include <vector>
#include <memory>

namespace MonteCarloEngine {

/**
 * @brief Economic scenario generator for correlated defaults
 * 
 * This class generates correlated economic scenarios using factor models
 * and adjusts individual account default probabilities based on systematic
 * economic factors.
 */
class ScenarioGenerator {
private:
    SimulationConfig config_;
    
    // Factor model parameters
    std::vector<std::vector<double>> factor_loadings_;  // Factor loadings matrix
    std::vector<double> factor_means_;                  // Factor mean values
    std::vector<double> factor_volatilities_;          // Factor volatilities
    std::vector<std::vector<double>> factor_correlation_; // Factor correlation matrix
    
    // Economic scenario parameters
    struct FactorModel {
        double gdp_sensitivity;         // Sensitivity to GDP changes
        double unemployment_sensitivity; // Sensitivity to unemployment
        double interest_rate_sensitivity; // Sensitivity to interest rates
        double market_vol_sensitivity;   // Sensitivity to market volatility
        double credit_spread_sensitivity; // Sensitivity to credit spreads
    } factor_model_;
    
    // Variance reduction state
    mutable std::vector<std::vector<double>> antithetic_factors_;

public:
    /**
     * @brief Constructor
     * @param config Simulation configuration
     */
    explicit ScenarioGenerator(const SimulationConfig& config);
    
    /**
     * @brief Update configuration
     * @param config New simulation configuration
     */
    void update_config(const SimulationConfig& config);
    
    /**
     * @brief Generate a batch of economic scenarios
     * @param portfolio Portfolio data for calibration
     * @param num_scenarios Number of scenarios to generate
     * @param thread_state Thread-local random state
     * @return Vector of economic scenarios
     */
    std::vector<EconomicScenario> generate_batch_scenarios(
        const PortfolioData& portfolio,
        size_t num_scenarios,
        ThreadLocalState& thread_state) const;
    
    /**
     * @brief Generate single economic scenario
     * @param portfolio Portfolio data for calibration
     * @param thread_state Thread-local random state
     * @return Single economic scenario
     */
    EconomicScenario generate_scenario(
        const PortfolioData& portfolio,
        ThreadLocalState& thread_state) const;
    
    /**
     * @brief Generate antithetic scenario for variance reduction
     * @param original_scenario Original scenario
     * @param thread_state Thread-local random state
     * @return Antithetic scenario
     */
    EconomicScenario generate_antithetic_scenario(
        const EconomicScenario& original_scenario,
        ThreadLocalState& thread_state) const;
    
    /**
     * @brief Adjust default probability based on economic scenario
     * @param base_probability Base default probability from XGBoost
     * @param scenario Economic scenario
     * @param thread_state Thread-local random state
     * @return Adjusted default probability
     */
    double adjust_default_probability(
        double base_probability,
        const EconomicScenario& scenario,
        ThreadLocalState& thread_state) const;
    
    /**
     * @brief Calibrate factor model to historical data
     * @param portfolio Portfolio data for calibration
     */
    void calibrate_factor_model(const PortfolioData& portfolio);
    
    /**
     * @brief Set custom factor loadings
     * @param loadings Factor loadings matrix
     */
    void set_factor_loadings(const std::vector<std::vector<double>>& loadings);
    
    /**
     * @brief Get current factor model parameters
     * @return Factor model structure
     */
    const FactorModel& get_factor_model() const { return factor_model_; }

private:
    /**
     * @brief Initialize default factor model parameters
     */
    void initialize_default_parameters();
    
    /**
     * @brief Generate correlated factor values
     * @param thread_state Thread-local random state
     * @return Vector of factor values
     */
    std::vector<double> generate_correlated_factors(ThreadLocalState& thread_state) const;
    
    /**
     * @brief Apply Cholesky decomposition for correlation
     * @param random_values Independent random values
     * @return Correlated random values
     */
    std::vector<double> apply_correlation_structure(
        const std::vector<double>& random_values) const;
    
    /**
     * @brief Convert factors to economic scenario
     * @param factors Factor values
     * @param base_scenario Base economic scenario
     * @return Economic scenario
     */
    EconomicScenario factors_to_scenario(
        const std::vector<double>& factors,
        const EconomicScenario& base_scenario) const;
    
    /**
     * @brief Calculate logistic transformation
     * @param x Input value
     * @return Transformed value between 0 and 1
     */
    double logistic_transform(double x) const;
    
    /**
     * @brief Compute Cholesky decomposition of correlation matrix
     * @param correlation Correlation matrix
     * @return Lower triangular Cholesky matrix
     */
    std::vector<std::vector<double>> cholesky_decomposition(
        const std::vector<std::vector<double>>& correlation) const;
};

} // namespace MonteCarloEngine

#endif // SCENARIO_GENERATOR_H