/**
 * @file risk_calculator.h
 * @brief Risk metrics calculation header for Monte Carlo simulation
 * 
 * This header declares the RiskCalculator class responsible for computing
 * portfolio-level risk metrics including VaR, CVaR, and Expected Loss.
 */

#ifndef RISK_CALCULATOR_H
#define RISK_CALCULATOR_H

#include "monte_carlo_types.h"
#include <vector>
#include <algorithm>

namespace MonteCarloEngine {

/**
 * @brief Portfolio risk metrics calculator
 * 
 * This class provides high-performance calculation of portfolio risk metrics
 * using OpenMP parallelization for large-scale Monte Carlo simulations.
 */
class RiskCalculator {
private:
    SimulationConfig config_;
    
    // Cached quantile positions for efficiency
    mutable std::vector<size_t> quantile_positions_;
    mutable bool positions_cached_;

public:
    /**
     * @brief Constructor
     * @param config Simulation configuration
     */
    explicit RiskCalculator(const SimulationConfig& config);
    
    /**
     * @brief Update configuration
     * @param config New simulation configuration
     */
    void update_config(const SimulationConfig& config);
    
    /**
     * @brief Calculate comprehensive risk metrics from loss distribution
     * @param loss_distribution Sorted vector of portfolio losses
     * @param metrics Risk metrics structure to populate
     */
    void calculate_risk_metrics(const std::vector<double>& loss_distribution,
                               RiskMetrics& metrics) const;
    
    /**
     * @brief Calculate Value at Risk (VaR) at specified confidence level
     * @param loss_distribution Sorted vector of portfolio losses
     * @param confidence_level Confidence level (e.g., 0.95 for 95% VaR)
     * @return VaR value
     */
    double calculate_var(const std::vector<double>& loss_distribution,
                        double confidence_level) const;
    
    /**
     * @brief Calculate Conditional Value at Risk (CVaR) at specified confidence level
     * @param loss_distribution Sorted vector of portfolio losses
     * @param confidence_level Confidence level (e.g., 0.95 for 95% CVaR)
     * @return CVaR value
     */
    double calculate_cvar(const std::vector<double>& loss_distribution,
                         double confidence_level) const;
    
    /**
     * @brief Calculate Expected Loss (mean of loss distribution)
     * @param loss_distribution Vector of portfolio losses
     * @return Expected loss value
     */
    double calculate_expected_loss(const std::vector<double>& loss_distribution) const;
    
    /**
     * @brief Calculate loss distribution statistics
     * @param loss_distribution Vector of portfolio losses
     * @param mean Output parameter for mean
     * @param std_dev Output parameter for standard deviation
     * @param skewness Output parameter for skewness
     * @param kurtosis Output parameter for kurtosis
     */
    void calculate_distribution_stats(const std::vector<double>& loss_distribution,
                                     double& mean, double& std_dev,
                                     double& skewness, double& kurtosis) const;
    
    /**
     * @brief Calculate stress test metrics under adverse scenarios
     * @param loss_distribution Vector of portfolio losses
     * @param stress_percentile Percentile for stress testing (e.g., 99.9%)
     * @return Stress test loss value
     */
    double calculate_stress_loss(const std::vector<double>& loss_distribution,
                                double stress_percentile) const;
    
    /**
     * @brief Calculate portfolio concentration risk metrics
     * @param portfolio Portfolio data
     * @param loss_distribution Vector of portfolio losses
     * @return Concentration risk measure
     */
    double calculate_concentration_risk(const PortfolioData& portfolio,
                                       const std::vector<double>& loss_distribution) const;
    
    /**
     * @brief Validate loss distribution for calculation
     * @param loss_distribution Vector of portfolio losses
     * @return True if distribution is valid for calculations
     */
    bool validate_loss_distribution(const std::vector<double>& loss_distribution) const;

private:
    /**
     * @brief Calculate quantile from sorted distribution
     * @param sorted_losses Sorted vector of losses
     * @param quantile Quantile to calculate (0.0 to 1.0)
     * @return Quantile value
     */
    double calculate_quantile(const std::vector<double>& sorted_losses,
                             double quantile) const;
    
    /**
     * @brief Calculate mean of vector using parallel reduction
     * @param values Vector of values
     * @return Mean value
     */
    double parallel_mean(const std::vector<double>& values) const;
    
    /**
     * @brief Calculate variance using parallel computation
     * @param values Vector of values
     * @param mean Pre-calculated mean
     * @return Variance value
     */
    double parallel_variance(const std::vector<double>& values, double mean) const;
    
    /**
     * @brief Calculate higher-order moments (skewness, kurtosis)
     * @param values Vector of values
     * @param mean Pre-calculated mean
     * @param std_dev Pre-calculated standard deviation
     * @param skewness Output parameter for skewness
     * @param kurtosis Output parameter for kurtosis
     */
    void calculate_higher_moments(const std::vector<double>& values,
                                 double mean, double std_dev,
                                 double& skewness, double& kurtosis) const;
    
    /**
     * @brief Cache quantile positions for repeated calculations
     * @param distribution_size Size of the loss distribution
     */
    void cache_quantile_positions(size_t distribution_size) const;
    
    /**
     * @brief Linear interpolation between two values
     * @param x0 Lower bound
     * @param x1 Upper bound
     * @param y0 Value at lower bound
     * @param y1 Value at upper bound
     * @param x Target value
     * @return Interpolated value
     */
    double linear_interpolate(double x0, double x1, double y0, double y1, double x) const;
};

} // namespace MonteCarloEngine

#endif // RISK_CALCULATOR_H