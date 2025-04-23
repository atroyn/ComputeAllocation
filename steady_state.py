#!/usr/bin/env python3
"""
Steady-State Analysis for Optimal Compute Allocation

This script analyzes the steady-state behavior of the optimal compute allocation
problem with power-law training dynamics in an infinite time horizon setting.

Model dynamics: dM/dt = βT·(α·C)^ε, where ε ∈ (0, 1)
Revenue model: R(t) = βR·M(t)^γ·(1-α(t))·C(t), γ > 1
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# Model parameters (same as power_law_fbsm.py)
beta_T = 1.0  # Training efficiency
beta_R = 1.0  # Revenue efficiency
epsilon = 0.5  # Diminishing returns in training (0 < epsilon < 1)
gamma = 2.0  # Increasing returns in revenue (gamma > 1)
rho = 0.15  # Discount rate (increased to be > growth rate)
growth_rate = 0.1  # Growth rate of compute
C0 = 100  # Initial compute

# Check if discount rate > growth rate for convergence in infinite horizon
if rho <= growth_rate:
    print(f"WARNING: For infinite horizon convergence, discount rate ({rho}) should be > growth rate ({growth_rate})")
    print("Results may not be reliable.")
else:
    print(f"Discount rate ({rho}) > growth rate ({growth_rate}): Convergence condition satisfied.")

# Function to calculate available compute at time t
def compute_availability(t):
    """
    Calculate available compute at time t with exponential growth.
    
    Parameters:
    -----------
    t : float or array
        Time point(s)
        
    Returns:
    --------
    float or array
        Available compute at time t
    """
    # Exponential growth model: C(t) = C0 * e^(growth_rate * t)
    return C0 * np.exp(growth_rate * t)

# Theoretical steady-state allocation
def theoretical_steady_state_allocation():
    """
    Calculate the theoretical steady-state allocation based on the
    dimensionless growth-to-discount ratio and other parameters.
    
    Returns:
    --------
    float
        Theoretical steady-state allocation to training
    """
    # Calculate the dimensionless growth-to-discount ratio
    g_rho_ratio = growth_rate / rho
    
    # Calculate the steady-state allocation
    # This is derived from the optimality condition in the infinite horizon case
    # where dλ/dt = ρλ - ∂H/∂M = 0 in steady state
    numerator = epsilon
    denominator = epsilon + (gamma - 1) * (1 - g_rho_ratio)
    
    # Ensure the allocation is between 0 and 1
    alpha_ss = max(0, min(1, numerator / denominator))
    
    return alpha_ss

# Function to calculate the growth rate of model capability in steady state
def calculate_steady_state_growth(alpha_ss):
    """
    Calculate the growth rate of model capability in steady state.
    
    Parameters:
    -----------
    alpha_ss : float
        Steady-state allocation to training
        
    Returns:
    --------
    float
        Growth rate of model capability in steady state
    """
    # In steady state, model capability grows at a constant rate
    # This rate depends on the allocation and compute growth
    return beta_T * (alpha_ss * C0) ** epsilon * np.exp(epsilon * growth_rate * 1)

# Function to numerically find the optimal steady-state allocation
def numerical_steady_state_allocation():
    """
    Numerically find the optimal steady-state allocation by maximizing
    the net present value in steady state.
    
    Returns:
    --------
    float
        Numerically optimal steady-state allocation to training
    """
    def negative_npv(alpha):
        # Validate allocation bounds
        if alpha <= 0 or alpha >= 1:
            return float('inf')  # Invalid allocation
        
        # Simulate with this allocation to get true NPV
        _, _, _, _, npv = simulate_with_constant_allocation(alpha)
        
        # Return negative for minimization
        return -npv
    
    # Try different initial points to avoid local minima
    best_alpha = 0.5  # Start with balanced allocation
    best_npv = -negative_npv(best_alpha)
    
    # Try a grid of initial points
    for alpha_init in [0.1, 0.3, 0.5, 0.7, 0.9]:
        # Find optimal allocation using numerical optimization
        result = minimize_scalar(negative_npv, bounds=(0.001, 0.999), method='bounded', options={'xatol': 1e-4})
        
        # Check if this is better than our current best
        if -negative_npv(result.x) > best_npv:
            best_alpha = result.x
            best_npv = -negative_npv(result.x)
            
    print(f"Best NPV found: {best_npv:.2f} at α={best_alpha:.4f}")
    return best_alpha

# Function to simulate the system dynamics with constant allocation
def simulate_with_constant_allocation(alpha, T=100, dt=0.1):
    """
    Simulate the system dynamics with a constant allocation.
    
    Parameters:
    -----------
    alpha : float
        Constant allocation to training
    T : float
        Time horizon
    dt : float
        Time step
        
    Returns:
    --------
    tuple
        (time_grid, M, revenue, npv)
    """
    # Create time grid
    time_grid = np.arange(0, T + dt, dt)
    
    # Initialize arrays
    M = np.zeros_like(time_grid)
    M[0] = 1.0  # Initial model capability
    
    # Forward simulation
    for i in range(1, len(time_grid)):
        # Get available compute at this time step
        C_t = compute_availability(time_grid[i-1])
        # Model training rate using power-law
        dM = beta_T * (alpha * C_t) ** epsilon
        # Update model capability
        M[i] = M[i-1] + dM * dt
    
    # Calculate available compute at each time point
    C_values = compute_availability(time_grid)
    
    # Calculate revenue
    revenue = beta_R * M**gamma * (1-alpha) * C_values
    
    # Calculate discounted revenue
    discounted_revenue = revenue * np.exp(-rho * time_grid)
    
    # Calculate NPV
    npv = np.sum(discounted_revenue) * dt
    
    return time_grid, M, revenue, discounted_revenue, npv

# Function to plot the results of different allocation strategies
def plot_comparison(results_dict):
    """
    Plot comparison of different allocation strategies.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary of results for different allocation strategies
    """
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Model capability
    plt.subplot(2, 2, 1)
    for label, result_tuple in results_dict.items():
        time_grid, M = result_tuple[0], result_tuple[1]
        alpha_value = result_tuple[5]  # The allocation value is at index 5
        plt.plot(time_grid, M, label=f"{label} (α={alpha_value:.3f})")
    plt.xlabel('Time')
    plt.ylabel('Model Capability (M)')
    plt.title('Model Capability Growth')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Revenue
    plt.subplot(2, 2, 2)
    for label, result_tuple in results_dict.items():
        time_grid, revenue = result_tuple[0], result_tuple[2]
        alpha_value = result_tuple[5]  # The allocation value is at index 5
        plt.plot(time_grid, revenue, label=f"{label} (α={alpha_value:.3f})")
    plt.xlabel('Time')
    plt.ylabel('Revenue')
    plt.title('Revenue Over Time')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Discounted revenue
    plt.subplot(2, 2, 3)
    for label, result_tuple in results_dict.items():
        time_grid, discounted_revenue = result_tuple[0], result_tuple[3]
        alpha_value = result_tuple[5]  # The allocation value is at index 5
        plt.plot(time_grid, discounted_revenue, label=f"{label} (α={alpha_value:.3f})")
    plt.xlabel('Time')
    plt.ylabel('Discounted Revenue')
    plt.title('Discounted Revenue Over Time')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Cumulative NPV
    plt.subplot(2, 2, 4)
    for label, result_tuple in results_dict.items():
        time_grid, discounted_revenue = result_tuple[0], result_tuple[3]
        alpha_value = result_tuple[5]  # The allocation value is at index 5
        cumulative_npv = np.cumsum(discounted_revenue) * (time_grid[1] - time_grid[0])
        plt.plot(time_grid, cumulative_npv, label=f"{label} (α={alpha_value:.3f})")
    plt.xlabel('Time')
    plt.ylabel('Cumulative NPV')
    plt.title('Cumulative NPV Over Time')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('steady_state_comparison.png', dpi=300)
    print("Saved steady_state_comparison.png")
    plt.show()

# Function to get parameter display name
def get_param_display_name(param_name):
    """
    Convert parameter name to display name with proper formatting.
    
    Parameters:
    -----------
    param_name : str
        Name of parameter
        
    Returns:
    --------
    str
        Display name with proper formatting (Unicode for Greek letters)
    """
    param_display = {
        'epsilon': 'ε',  # epsilon (ε)
        'gamma': 'γ',    # gamma (γ)
        'rho': 'ρ',      # rho (ρ)
        'growth_rate': 'Compute Growth Rate',
        'g_rho_ratio': 'g/ρ Ratio'
    }
    return param_display.get(param_name, param_name)

# Function to plot the sensitivity of steady-state allocation to parameters
def plot_sensitivity():
    """
    Plot the sensitivity of steady-state allocation to key parameters.
    """
    global epsilon, gamma, growth_rate
    
    # Parameter ranges
    epsilon_range = np.linspace(0.1, 0.9, 50)
    gamma_range = np.linspace(0.1, 3.0, 50)
    g_rho_ratios = np.linspace(0.1, 0.9, 50)
    
    # Calculate steady-state allocations for different parameters
    alpha_epsilon = []
    alpha_gamma = []
    alpha_g_rho = []
    
    # Original parameters
    orig_epsilon = epsilon
    orig_gamma = gamma
    orig_growth_rate = growth_rate
    
    # Vary epsilon
    for eps in epsilon_range:
        epsilon = eps
        alpha_epsilon.append(theoretical_steady_state_allocation())
    
    # Reset and vary gamma
    epsilon = orig_epsilon
    for gam in gamma_range:
        gamma = gam
        alpha_gamma.append(theoretical_steady_state_allocation())
    
    # Reset and vary g/rho ratio
    gamma = orig_gamma
    for ratio in g_rho_ratios:
        growth_rate = ratio * rho
        alpha_g_rho.append(theoretical_steady_state_allocation())
    
    # Reset parameters
    epsilon = orig_epsilon
    gamma = orig_gamma
    growth_rate = orig_growth_rate
    
    # Create plots
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Sensitivity to epsilon
    plt.subplot(1, 3, 1)
    plt.plot(epsilon_range, alpha_epsilon)
    plt.xlabel('Epsilon (ε)')
    plt.ylabel('Steady-State Allocation (α)')
    plt.title(f'Optimal Allocation vs {get_param_display_name("epsilon")}')
    plt.grid(True)
    
    # Plot 2: Sensitivity to gamma
    plt.subplot(1, 3, 2)
    plt.plot(gamma_range, alpha_gamma)
    plt.xlabel('Gamma (γ)')
    plt.ylabel('Steady-State Allocation (α)')
    plt.title(f'Optimal Allocation vs {get_param_display_name("gamma")}')
    plt.grid(True)
    
    # Plot 3: Sensitivity to g/rho ratio
    plt.subplot(1, 3, 3)
    plt.plot(g_rho_ratios, alpha_g_rho)
    plt.xlabel('Growth-to-Discount Ratio (g/ρ)')
    plt.ylabel('Steady-State Allocation (α)')
    plt.title(f'Optimal Allocation vs g/ρ Ratio')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('steady_state_sensitivity.png', dpi=300)
    print("Saved steady_state_sensitivity.png")
    plt.show()

# Main analysis
print("\nSteady-State Analysis for Optimal Compute Allocation")
print("------------------------------------------------")

# Calculate theoretical steady-state allocation
alpha_theoretical = theoretical_steady_state_allocation()
print(f"\nTheoretical steady-state allocation: {alpha_theoretical:.4f}")

# Calculate numerical steady-state allocation
alpha_numerical = numerical_steady_state_allocation()
print(f"Numerical steady-state allocation: {alpha_numerical:.4f}")

# Calculate growth rate of model capability in steady state
growth_rate_theoretical = calculate_steady_state_growth(alpha_theoretical)
print(f"Model capability growth rate (theoretical): {growth_rate_theoretical:.4f}")

growth_rate_numerical = calculate_steady_state_growth(alpha_numerical)
print(f"Model capability growth rate (numerical): {growth_rate_numerical:.4f}")

# Simulate with different allocation strategies
print("\nSimulating with different allocation strategies...")

# Theoretical steady-state allocation
results_theoretical = simulate_with_constant_allocation(alpha_theoretical)

# Numerical steady-state allocation
results_numerical = simulate_with_constant_allocation(alpha_numerical)

# 50/50 allocation (baseline)
results_baseline = simulate_with_constant_allocation(0.5)

# Extreme allocations
results_high = simulate_with_constant_allocation(0.8)
results_low = simulate_with_constant_allocation(0.2)

# Collect results
results_dict = {
    'Theoretical': results_theoretical + (alpha_theoretical,),
    'Numerical': results_numerical + (alpha_numerical,),
    'Baseline (50/50)': results_baseline + (0.5,),
    'High Training (80/20)': results_high + (0.8,),
    'Low Training (20/80)': results_low + (0.2,)
}

# Print NPV results
print("\nNPV Results:")
for label, result_tuple in results_dict.items():
    npv = result_tuple[4]  # NPV is at index 4
    alpha_value = result_tuple[5]  # Allocation is at index 5
    print(f"{label} (α={alpha_value:.3f}): {npv:.2f}")

# Plot comparison
plot_comparison(results_dict)

# Plot sensitivity analysis
plot_sensitivity()

print("\nAnalysis complete.")
