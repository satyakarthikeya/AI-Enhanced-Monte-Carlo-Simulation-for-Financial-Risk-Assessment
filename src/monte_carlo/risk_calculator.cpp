/**
 * @file risk_calculator.cpp
 * @brief Implementation of portfolio risk metrics calculator
 * 
 * This file implements the RiskCalculator class that provides high-performance
 * calculation of VaR, CVaR, Expected Loss, and other risk metrics using
 * OpenMP parallelization for large-scale Monte Carlo simulations.
 */

#include "include/risk_calculator.h"
#include <omp.h>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <iostream>
#include <iomanip>

namespace MonteCarloEngine {

RiskCalculator::RiskCalculator(const SimulationConfig& config) 
    : config_(config), positions_cached_(false) {
}

void RiskCalculator::update_config(const SimulationConfig& config) {
    config_ = config;
    positions_cached_ = false; // Force recalculation of cached positions
}

void RiskCalculator::calculate_risk_metrics(const std::vector<double>& loss_distribution,
                                           RiskMetrics& metrics) const {
    
    if (!validate_loss_distribution(loss_distribution)) {
        throw std::invalid_argument("Invalid loss distribution for risk calculation");
    }
    
    std::cout << "Calculating risk metrics for " << loss_distribution.size() 
              << " loss samples...\n";
    
    // Calculate basic statistics
    double mean, std_dev, skewness, kurtosis;
    calculate_distribution_stats(loss_distribution, mean, std_dev, skewness, kurtosis);
    
    // Store basic metrics
    metrics.expected_loss = mean;
    metrics.std_dev_loss = std_dev;
    metrics.skewness = skewness;
    metrics.kurtosis = kurtosis;
    metrics.max_loss = *std::max_element(loss_distribution.begin(), loss_distribution.end());
    
    // Calculate VaR at different confidence levels
    metrics.var_95 = calculate_var(loss_distribution, config_.confidence_level_95);
    metrics.var_99 = calculate_var(loss_distribution, config_.confidence_level_99);
    
    // Calculate CVaR at different confidence levels
    metrics.cvar_95 = calculate_cvar(loss_distribution, config_.confidence_level_95);
    metrics.cvar_99 = calculate_cvar(loss_distribution, config_.confidence_level_99);
    
    // Print summary
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Risk Metrics Summary:\n";
    std::cout << "  Expected Loss: $" << metrics.expected_loss << "\n";
    std::cout << "  Loss Std Dev:  $" << metrics.std_dev_loss << "\n";
    std::cout << "  95% VaR:       $" << metrics.var_95 << "\n";
    std::cout << "  99% VaR:       $" << metrics.var_99 << "\n";
    std::cout << "  95% CVaR:      $" << metrics.cvar_95 << "\n";
    std::cout << "  99% CVaR:      $" << metrics.cvar_99 << "\n";
    std::cout << "  Maximum Loss:  $" << metrics.max_loss << "\n";
    std::cout << "  Skewness:      " << std::setprecision(3) << metrics.skewness << "\n";
    std::cout << "  Kurtosis:      " << metrics.kurtosis << "\n";
}

double RiskCalculator::calculate_var(const std::vector<double>& loss_distribution,
                                    double confidence_level) const {
    
    if (loss_distribution.empty()) {
        return 0.0;
    }
    
    // VaR is the quantile of the loss distribution
    return calculate_quantile(loss_distribution, confidence_level);
}

double RiskCalculator::calculate_cvar(const std::vector<double>& loss_distribution,
                                     double confidence_level) const {
    
    if (loss_distribution.empty()) {
        return 0.0;
    }
    
    // CVaR is the expected value of losses beyond VaR
    double var_level = calculate_var(loss_distribution, confidence_level);
    
    // Find all losses greater than or equal to VaR
    std::vector<double> tail_losses;
    tail_losses.reserve(loss_distribution.size() * (1.0 - confidence_level));
    
    for (double loss : loss_distribution) {
        if (loss >= var_level) {
            tail_losses.push_back(loss);
        }
    }
    
    if (tail_losses.empty()) {
        return var_level; // Return VaR if no tail losses
    }
    
    // Calculate mean of tail losses using parallel reduction
    return parallel_mean(tail_losses);
}

double RiskCalculator::calculate_expected_loss(const std::vector<double>& loss_distribution) const {
    return parallel_mean(loss_distribution);
}

void RiskCalculator::calculate_distribution_stats(const std::vector<double>& loss_distribution,
                                                 double& mean, double& std_dev,
                                                 double& skewness, double& kurtosis) const {
    
    if (loss_distribution.empty()) {
        mean = std_dev = skewness = kurtosis = 0.0;
        return;
    }
    
    // Calculate mean using parallel reduction
    mean = parallel_mean(loss_distribution);
    
    // Calculate variance using parallel computation
    double variance = parallel_variance(loss_distribution, mean);
    std_dev = std::sqrt(variance);
    
    // Calculate higher-order moments
    calculate_higher_moments(loss_distribution, mean, std_dev, skewness, kurtosis);
}

double RiskCalculator::calculate_stress_loss(const std::vector<double>& loss_distribution,
                                            double stress_percentile) const {
    return calculate_quantile(loss_distribution, stress_percentile);
}

double RiskCalculator::calculate_concentration_risk(const PortfolioData& portfolio,
                                                  const std::vector<double>& loss_distribution) const {
    
    // Calculate Herfindahl-Hirschman Index for concentration
    double hhi = 0.0;
    
    if (portfolio.total_exposure > 0.0) {
        for (const auto& account : portfolio.accounts) {
            double weight = account.exposure_at_default / portfolio.total_exposure;
            hhi += weight * weight;
        }
    }
    
    // Concentration risk multiplier based on HHI
    double concentration_multiplier = 1.0 + hhi;
    
    return concentration_multiplier;
}

bool RiskCalculator::validate_loss_distribution(const std::vector<double>& loss_distribution) const {
    if (loss_distribution.empty()) {
        std::cerr << "Error: Empty loss distribution\n";
        return false;
    }
    
    // Check for invalid values
    for (double loss : loss_distribution) {
        if (std::isnan(loss) || std::isinf(loss)) {
            std::cerr << "Error: Invalid loss value detected (NaN or Inf)\n";
            return false;
        }
        if (loss < 0.0) {
            std::cerr << "Error: Negative loss value detected\n";
            return false;
        }
    }
    
    return true;
}

double RiskCalculator::calculate_quantile(const std::vector<double>& sorted_losses,
                                         double quantile) const {
    
    if (sorted_losses.empty()) {
        return 0.0;
    }
    
    if (quantile <= 0.0) {
        return sorted_losses.front();
    }
    
    if (quantile >= 1.0) {
        return sorted_losses.back();
    }
    
    // Calculate position in sorted array
    double position = quantile * (sorted_losses.size() - 1);
    size_t lower_index = static_cast<size_t>(std::floor(position));
    size_t upper_index = static_cast<size_t>(std::ceil(position));
    
    if (lower_index == upper_index) {
        return sorted_losses[lower_index];
    }
    
    // Linear interpolation between adjacent values
    double weight = position - lower_index;
    return (1.0 - weight) * sorted_losses[lower_index] + 
           weight * sorted_losses[upper_index];
}

double RiskCalculator::parallel_mean(const std::vector<double>& values) const {
    if (values.empty()) {
        return 0.0;
    }
    
    double sum = 0.0;
    
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < values.size(); ++i) {
        sum += values[i];
    }
    
    return sum / static_cast<double>(values.size());
}

double RiskCalculator::parallel_variance(const std::vector<double>& values, double mean) const {
    if (values.size() <= 1) {
        return 0.0;
    }
    
    double sum_squared_diff = 0.0;
    
    #pragma omp parallel for reduction(+:sum_squared_diff)
    for (size_t i = 0; i < values.size(); ++i) {
        double diff = values[i] - mean;
        sum_squared_diff += diff * diff;
    }
    
    return sum_squared_diff / static_cast<double>(values.size() - 1);
}

void RiskCalculator::calculate_higher_moments(const std::vector<double>& values,
                                             double mean, double std_dev,
                                             double& skewness, double& kurtosis) const {
    
    if (values.size() <= 2 || std_dev == 0.0) {
        skewness = kurtosis = 0.0;
        return;
    }
    
    double sum_cubed = 0.0;
    double sum_fourth = 0.0;
    const double n = static_cast<double>(values.size());
    
    #pragma omp parallel for reduction(+:sum_cubed,sum_fourth)
    for (size_t i = 0; i < values.size(); ++i) {
        double standardized = (values[i] - mean) / std_dev;
        double cubed = standardized * standardized * standardized;
        double fourth = cubed * standardized;
        
        sum_cubed += cubed;
        sum_fourth += fourth;
    }
    
    // Sample skewness
    skewness = (n / ((n - 1) * (n - 2))) * sum_cubed / n;
    
    // Sample excess kurtosis (subtract 3 for excess over normal distribution)
    kurtosis = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * (sum_fourth / n) - 
               3.0 * (n - 1) * (n - 1) / ((n - 2) * (n - 3));
}

void RiskCalculator::cache_quantile_positions(size_t distribution_size) const {
    if (positions_cached_ && !quantile_positions_.empty()) {
        return;
    }
    
    quantile_positions_.clear();
    
    // Cache common quantile positions for efficiency
    std::vector<double> quantiles = {
        0.90, 0.95, 0.975, 0.99, 0.995, 0.999, 0.9999
    };
    
    for (double q : quantiles) {
        double position = q * (distribution_size - 1);
        quantile_positions_.push_back(static_cast<size_t>(position));
    }
    
    positions_cached_ = true;
}

double RiskCalculator::linear_interpolate(double x0, double x1, double y0, double y1, double x) const {
    if (x1 == x0) {
        return y0;
    }
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
}

} // namespace MonteCarloEngine