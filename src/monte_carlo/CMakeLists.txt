cmake_minimum_required(VERSION 3.15)
project(MonteCarloEngine VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build type to Release by default for performance
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -ffast-math -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -pedantic")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -ffast-math -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -pedantic")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /Wall")
endif()

# Find required packages
find_package(OpenMP REQUIRED)
find_package(pybind11 REQUIRED)

# Python components
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Source files
set(SOURCES
    monte_carlo_engine.cpp
    scenario_generator.cpp
    risk_calculator.cpp
)

# Header files
set(HEADERS
    include/monte_carlo_types.h
    include/monte_carlo_engine.h
    include/scenario_generator.h
    include/risk_calculator.h
)

# Create static library for the core engine
add_library(monte_carlo_core STATIC ${SOURCES} ${HEADERS})
target_link_libraries(monte_carlo_core OpenMP::OpenMP_CXX)
target_include_directories(monte_carlo_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Enable position-independent code for static library (needed for shared module)
set_property(TARGET monte_carlo_core PROPERTY POSITION_INDEPENDENT_CODE ON)

# Create Python module using pybind11
pybind11_add_module(monte_carlo_engine python_interface.cpp)
target_link_libraries(monte_carlo_engine PRIVATE monte_carlo_core OpenMP::OpenMP_CXX)
target_compile_definitions(monte_carlo_engine PRIVATE VERSION_INFO=${PROJECT_VERSION})

# Optimization flags for the Python module
target_compile_options(monte_carlo_engine PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native -mtune=native>
    $<$<CONFIG:Debug>:-g -O0>
)

# Link-time optimization for Release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set_property(TARGET monte_carlo_engine PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    set_property(TARGET monte_carlo_core PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# Create standalone executable for testing (optional)
add_executable(monte_carlo_test test_main.cpp)
target_link_libraries(monte_carlo_test monte_carlo_core OpenMP::OpenMP_CXX)

# Installation - copy to local directory instead of system directory
install(TARGETS monte_carlo_engine 
        DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../../)

# Print configuration information
message(STATUS "Monte Carlo Engine Configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  OpenMP Found: ${OpenMP_FOUND}")
if(OpenMP_FOUND)
    message(STATUS "  OpenMP Version: ${OpenMP_CXX_VERSION}")
    message(STATUS "  OpenMP Flags: ${OpenMP_CXX_FLAGS}")
endif()
message(STATUS "  pybind11 Version: ${pybind11_VERSION}")
message(STATUS "  Python Executable: ${Python_EXECUTABLE}")
message(STATUS "  Python Version: ${Python_VERSION}")

# Custom targets for common operations
add_custom_target(install_python
    COMMAND ${CMAKE_COMMAND} --build . --target install
    COMMENT "Installing Python module"
)

add_custom_target(benchmark
    COMMAND ./monte_carlo_test --benchmark
    DEPENDS monte_carlo_test
    COMMENT "Running performance benchmarks"
)

# Testing configuration
enable_testing()
add_test(NAME basic_functionality 
         COMMAND monte_carlo_test --test-basic)
add_test(NAME performance_test 
         COMMAND monte_carlo_test --test-performance)

# Code coverage (for Debug builds)
if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(monte_carlo_core PRIVATE --coverage)
    target_link_libraries(monte_carlo_core --coverage)
endif()

# Memory sanitizer option
option(ENABLE_SANITIZER "Enable AddressSanitizer" OFF)
if(ENABLE_SANITIZER AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(monte_carlo_core PRIVATE -fsanitize=address -fno-omit-frame-pointer)
    target_link_libraries(monte_carlo_core -fsanitize=address)
endif()

# Profiling option
option(ENABLE_PROFILING "Enable profiling with gprof" OFF)
if(ENABLE_PROFILING)
    target_compile_options(monte_carlo_core PRIVATE -pg)
    target_link_libraries(monte_carlo_core -pg)
endif()

# Documentation with Doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)
endif()

# Package configuration
set(CPACK_PACKAGE_NAME "MonteCarloEngine")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "High-Performance Monte Carlo Simulation Engine")
set(CPACK_PACKAGE_VENDOR "AI-Enhanced Financial Risk Assessment")
include(CPack)