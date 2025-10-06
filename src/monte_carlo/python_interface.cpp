/**
 * @file python_interface.cpp
 * @brief Python C++ interface using pybind11 for Monte Carlo simulation engine
 * 
 * This file provides seamless integration between Python and the high-performance
 * C++ Monte Carlo simulation engine using pybind11 for data marshaling and
 * XGBoost model integration.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "include/monte_carlo_engine.h"
#include "include/scenario_generator.h"
#include "include/risk_calculator.h"
#include <iostream>
#include <stdexcept>
#include <thread>

namespace py = pybind11;
using namespace MonteCarloEngine;

/**
 * @brief Convert Python numpy array to portfolio data
 */
PortfolioData numpy_to_portfolio(py::array_t<double> balances,
                                py::array_t<double> limits,
                                py::array_t<double> default_probs,
                                py::array_t<double> exposures,
                                py::array_t<double> lgds,
                                py::array_t<int> account_ids) {
    
    auto bal_buf = balances.request();
    auto lim_buf = limits.request();
    auto prob_buf = default_probs.request();
    auto exp_buf = exposures.request();
    auto lgd_buf = lgds.request();
    auto id_buf = account_ids.request();
    
    if (bal_buf.size != lim_buf.size || bal_buf.size != prob_buf.size ||
        bal_buf.size != exp_buf.size || bal_buf.size != lgd_buf.size ||
        bal_buf.size != id_buf.size) {
        throw std::runtime_error("All input arrays must have the same size");
    }
    
    PortfolioData portfolio;
    portfolio.reserve(bal_buf.size);
    
    auto bal_ptr = static_cast<double*>(bal_buf.ptr);
    auto lim_ptr = static_cast<double*>(lim_buf.ptr);
    auto prob_ptr = static_cast<double*>(prob_buf.ptr);
    auto exp_ptr = static_cast<double*>(exp_buf.ptr);
    auto lgd_ptr = static_cast<double*>(lgd_buf.ptr);
    auto id_ptr = static_cast<int*>(id_buf.ptr);
    
    for (py::ssize_t i = 0; i < bal_buf.size; ++i) {
        AccountData account;
        account.balance = bal_ptr[i];
        account.limit = lim_ptr[i];
        account.default_probability = prob_ptr[i];
        account.exposure_at_default = exp_ptr[i];
        account.loss_given_default = lgd_ptr[i];
        account.account_id = id_ptr[i];
        
        portfolio.add_account(account);
    }
    
    return portfolio;
}

/**
 * @brief Convert simulation results to Python dictionary
 */
py::dict results_to_dict(const SimulationResults& results) {
    py::dict py_results;
    
    // Risk metrics
    py::dict risk_metrics;
    risk_metrics["var_95"] = results.risk_metrics.var_95;
    risk_metrics["var_99"] = results.risk_metrics.var_99;
    risk_metrics["cvar_95"] = results.risk_metrics.cvar_95;
    risk_metrics["cvar_99"] = results.risk_metrics.cvar_99;
    risk_metrics["expected_loss"] = results.risk_metrics.expected_loss;
    risk_metrics["max_loss"] = results.risk_metrics.max_loss;
    risk_metrics["std_dev_loss"] = results.risk_metrics.std_dev_loss;
    risk_metrics["skewness"] = results.risk_metrics.skewness;
    risk_metrics["kurtosis"] = results.risk_metrics.kurtosis;
    
    // Performance metrics
    py::dict performance;
    performance["total_time_seconds"] = results.performance.total_time.count();
    performance["simulation_time_seconds"] = results.performance.simulation_time.count();
    performance["calculation_time_seconds"] = results.performance.calculation_time.count();
    performance["iterations_per_second"] = results.performance.iterations_per_second;
    performance["memory_usage_mb"] = results.performance.memory_usage_mb;
    performance["num_threads_used"] = results.performance.num_threads_used;
    
    py_results["risk_metrics"] = risk_metrics;
    py_results["performance"] = performance;
    py_results["success"] = results.success;
    py_results["error_message"] = results.error_message;
    
    // Convert loss distribution to numpy array
    py_results["loss_distribution"] = py::cast(results.loss_distribution);
    
    return py_results;
}

/**
 * @brief Python wrapper class for Monte Carlo Engine
 */
class PyMonteCarloEngine {
private:
    std::unique_ptr<MonteCarloEngine::MonteCarloEngine> engine_;
    SimulationConfig config_;

public:
    PyMonteCarloEngine(py::dict config_dict = py::dict()) {
        // Parse configuration from Python dictionary
        config_.num_simulations = config_dict.contains("num_simulations") ? 
                                  config_dict["num_simulations"].cast<size_t>() : 100000;
        config_.num_threads = config_dict.contains("num_threads") ? 
                             config_dict["num_threads"].cast<int>() : 0;
        config_.scenarios_per_batch = config_dict.contains("scenarios_per_batch") ? 
                                     config_dict["scenarios_per_batch"].cast<size_t>() : 1000;
        config_.use_antithetic_variates = config_dict.contains("use_antithetic_variates") ? 
                                         config_dict["use_antithetic_variates"].cast<bool>() : true;
        config_.enable_correlation = config_dict.contains("enable_correlation") ? 
                                    config_dict["enable_correlation"].cast<bool>() : true;
        config_.random_seed = config_dict.contains("random_seed") ? 
                             config_dict["random_seed"].cast<int>() : 42;
        
        engine_ = std::make_unique<MonteCarloEngine::MonteCarloEngine>(config_);
    }
    
    py::dict run_simulation(py::array_t<double> balances,
                           py::array_t<double> limits,
                           py::array_t<double> default_probs,
                           py::array_t<double> exposures,
                           py::array_t<double> lgds,
                           py::array_t<int> account_ids,
                           py::dict economic_scenario = py::dict()) {
        
        try {
            // Convert Python data to C++ portfolio
            auto portfolio = numpy_to_portfolio(balances, limits, default_probs,
                                               exposures, lgds, account_ids);
            
            // Set economic scenario if provided
            if (!economic_scenario.empty()) {
                portfolio.base_scenario.gdp_growth = economic_scenario.contains("gdp_growth") ?
                                                     economic_scenario["gdp_growth"].cast<double>() : 0.02;
                portfolio.base_scenario.unemployment_rate = economic_scenario.contains("unemployment_rate") ?
                                                           economic_scenario["unemployment_rate"].cast<double>() : 0.05;
                portfolio.base_scenario.interest_rate = economic_scenario.contains("interest_rate") ?
                                                        economic_scenario["interest_rate"].cast<double>() : 0.03;
                portfolio.base_scenario.market_volatility = economic_scenario.contains("market_volatility") ?
                                                            economic_scenario["market_volatility"].cast<double>() : 0.15;
                portfolio.base_scenario.credit_spread = economic_scenario.contains("credit_spread") ?
                                                        economic_scenario["credit_spread"].cast<double>() : 0.02;
            }
            
            // Run simulation
            auto results = engine_->run_simulation(portfolio);
            
            // Convert results to Python dictionary
            return results_to_dict(results);
            
        } catch (const std::exception& e) {
            py::dict error_result;
            error_result["success"] = false;
            error_result["error_message"] = std::string("C++ Error: ") + e.what();
            return error_result;
        }
    }
    
    void update_config(py::dict config_dict) {
        if (config_dict.contains("num_simulations")) {
            config_.num_simulations = config_dict["num_simulations"].cast<size_t>();
        }
        if (config_dict.contains("num_threads")) {
            config_.num_threads = config_dict["num_threads"].cast<int>();
        }
        if (config_dict.contains("scenarios_per_batch")) {
            config_.scenarios_per_batch = config_dict["scenarios_per_batch"].cast<size_t>();
        }
        if (config_dict.contains("use_antithetic_variates")) {
            config_.use_antithetic_variates = config_dict["use_antithetic_variates"].cast<bool>();
        }
        if (config_dict.contains("enable_correlation")) {
            config_.enable_correlation = config_dict["enable_correlation"].cast<bool>();
        }
        if (config_dict.contains("random_seed")) {
            config_.random_seed = config_dict["random_seed"].cast<int>();
        }
        
        engine_->update_config(config_);
    }
    
    py::dict get_config() const {
        py::dict config;
        config["num_simulations"] = config_.num_simulations;
        config["num_threads"] = config_.num_threads;
        config["scenarios_per_batch"] = config_.scenarios_per_batch;
        config["use_antithetic_variates"] = config_.use_antithetic_variates;
        config["enable_correlation"] = config_.enable_correlation;
        config["random_seed"] = config_.random_seed;
        return config;
    }
    
    static int get_optimal_thread_count() {
        return std::thread::hardware_concurrency();
    }
    
    static size_t estimate_memory_usage(size_t num_accounts, size_t num_simulations) {
        PortfolioData dummy_portfolio;
        dummy_portfolio.num_accounts = num_accounts;
        
        SimulationConfig dummy_config;
        dummy_config.num_simulations = num_simulations;
        
        return num_accounts * num_simulations * sizeof(double) / (1024 * 1024);
    }
};

/**
 * @brief Utility functions for XGBoost integration
 */
py::dict validate_xgboost_predictions(py::array_t<double> predictions) {
    auto buf = predictions.request();
    auto ptr = static_cast<double*>(buf.ptr);
    
    py::dict validation_result;
    validation_result["valid"] = true;
    validation_result["message"] = "All predictions valid";
    
    size_t invalid_count = 0;
    for (py::ssize_t i = 0; i < buf.size; ++i) {
        if (ptr[i] < 0.0 || ptr[i] > 1.0 || std::isnan(ptr[i]) || std::isinf(ptr[i])) {
            invalid_count++;
        }
    }
    
    if (invalid_count > 0) {
        validation_result["valid"] = false;
        validation_result["message"] = "Found " + std::to_string(invalid_count) + 
                                      " invalid predictions out of " + std::to_string(buf.size);
        validation_result["invalid_count"] = invalid_count;
    }
    
    return validation_result;
}

py::array_t<double> adjust_predictions_for_stress(py::array_t<double> predictions, 
                                                  double stress_factor) {
    auto buf = predictions.request();
    auto result = py::array_t<double>(buf.size);
    auto result_buf = result.request();
    
    auto input_ptr = static_cast<double*>(buf.ptr);
    auto output_ptr = static_cast<double*>(result_buf.ptr);
    
    for (py::ssize_t i = 0; i < buf.size; ++i) {
        // Apply stress factor using logistic transformation
        double prob = input_ptr[i];
        double log_odds = std::log(prob / (1.0 - prob + 1e-10));
        double stressed_log_odds = log_odds + stress_factor;
        double stressed_prob = 1.0 / (1.0 + std::exp(-stressed_log_odds));
        
        output_ptr[i] = std::max(1e-6, std::min(0.999, stressed_prob));
    }
    
    return result;
}

// pybind11 module definition
PYBIND11_MODULE(monte_carlo_engine, m) {
    m.doc() = "High-Performance Monte Carlo Simulation Engine for Financial Risk Assessment";
    
    // Main Monte Carlo Engine class
    py::class_<PyMonteCarloEngine>(m, "MonteCarloEngine")
        .def(py::init<py::dict>(), 
             "Initialize Monte Carlo Engine",
             py::arg("config") = py::dict())
        .def("run_simulation", &PyMonteCarloEngine::run_simulation,
             "Run Monte Carlo simulation on portfolio data",
             py::arg("balances"), py::arg("limits"), py::arg("default_probs"),
             py::arg("exposures"), py::arg("lgds"), py::arg("account_ids"),
             py::arg("economic_scenario") = py::dict())
        .def("update_config", &PyMonteCarloEngine::update_config,
             "Update simulation configuration",
             py::arg("config"))
        .def("get_config", &PyMonteCarloEngine::get_config,
             "Get current simulation configuration")
        .def_static("get_optimal_thread_count", &PyMonteCarloEngine::get_optimal_thread_count,
                   "Get optimal number of threads for current system")
        .def_static("estimate_memory_usage", &PyMonteCarloEngine::estimate_memory_usage,
                   "Estimate memory usage for given portfolio size and simulations",
                   py::arg("num_accounts"), py::arg("num_simulations"));
    
    // Utility functions
    m.def("validate_xgboost_predictions", &validate_xgboost_predictions,
          "Validate XGBoost prediction values",
          py::arg("predictions"));
    
    m.def("adjust_predictions_for_stress", &adjust_predictions_for_stress,
          "Apply stress factor to XGBoost predictions",
          py::arg("predictions"), py::arg("stress_factor"));
    
    // Version information
    m.attr("__version__") = "1.0.0";
    m.attr("__author__") = "AI-Enhanced Monte Carlo Simulation Engine";
    
    // Configuration constants
    py::dict default_config;
    default_config["num_simulations"] = 100000;
    default_config["num_threads"] = 0;  // Auto-detect
    default_config["scenarios_per_batch"] = 1000;
    default_config["use_antithetic_variates"] = true;
    default_config["enable_correlation"] = true;
    default_config["random_seed"] = 42;
    m.attr("DEFAULT_CONFIG") = default_config;
}