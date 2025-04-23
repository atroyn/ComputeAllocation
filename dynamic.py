#!/usr/bin/env python3
"""
Optimal Compute Allocation using Forward-Backward Sweep Method
with Power-Law Model for Training

Implementation of the Forward-Backward Sweep Method (FBSM) for the
optimal compute allocation problem with power-law training dynamics.

Model dynamics: dM/dt = βT·(α·C)^ε, where ε ∈ (0, 1)

And revenue model:
R(t) = βR·M(t)^γ·(1-α(t))·C(t), γ > 1"""

import numpy as np
import matplotlib.pyplot as plt

# Model parameters
beta_T = 1.0  # Training efficiency
beta_R = 1.0  # Revenue efficiency
epsilon = 0.5  # Diminishing returns in training (0 < epsilon < 1)
gamma = 2.0  # Increasing returns in revenue (gamma > 1)
rho = 0.05  # Discount rate
growth_rate = 0.1  # Growth rate of compute
C0 = 100  # Initial compute
M0 = 1.0  # Initial model capability
T = 10  # Time horizon (finite)
dt = 0.1      # Time step for simulation
time_grid = np.arange(0, T + dt, dt)

# Ensure discount rate > growth rate for convergence in infinite horizon
if rho <= growth_rate:
    print(f"WARNING: For infinite horizon convergence, discount rate ({rho}) should be > growth rate ({growth_rate})")
    print("Results may not be reliable for very long time horizons.")

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

# Initial conditions
M0 = 1.0       # Initial model capability
lambda_T = 0.0  # Terminal costate value (transversality condition)

# Initialize arrays
M = np.zeros_like(time_grid)
Lambda = np.zeros_like(time_grid)
alpha = np.ones_like(time_grid) * 0.5  # Initial guess for control

# Forward sweep function
def forward_sweep(M0, alpha):
    """
    Solve the state equation forward in time using the model dynamics.
    
    The model dynamics follow a power-law relationship:
    dM/dt = βT·(α·C)^ε, where ε ∈ (0, 1)
    
    Parameters:
    -----------
    M0 : float
        Initial model capability
    alpha : array
        Control trajectory (allocation to training)
        
    Returns:
    --------
    array
        Model capability trajectory over time
    """
    M = np.zeros_like(time_grid)
    M[0] = M0  # Set initial condition
    
    # Integrate the ODE forward in time using Euler's method
    for i in range(1, len(time_grid)):
        # Get available compute at this time step
        C_t = compute_availability(time_grid[i-1])
        # Model training rate using power-law relationship with diminishing returns
        dM = beta_T * (alpha[i-1] * C_t) ** epsilon
        # Update model capability
        M[i] = M[i-1] + dM * dt
        
    return M

# Backward sweep function
def backward_sweep(M, alpha):
    """
    Solve the costate equation backward in time.
    
    Parameters:
    -----------
    M : array
        Model capability trajectory
    alpha : array
        Control trajectory
        
    Returns:
    --------
    array
        Costate trajectory
    """
    Lambda = np.zeros_like(time_grid)
    Lambda[-1] = lambda_T  # Terminal condition
    
    for i in reversed(range(len(time_grid) - 1)):
        # Get available compute at this time step
        C_t = compute_availability(time_grid[i+1])
        # Costate equation from Pontryagin's principle
        dLambda = (rho * Lambda[i+1] - 
                  np.exp(-rho * time_grid[i+1]) * 
                  beta_R * gamma * 
                  M[i+1]**(gamma-1) * (1-alpha[i+1]) * C_t)
        Lambda[i] = Lambda[i+1] - dLambda * dt
        
    return Lambda

# Update control function
def update_control(M, Lambda):
    """
    Update the control based on the optimality condition.
    
    Parameters:
    -----------
    M : array
        Model capability trajectory
    Lambda : array
        Costate trajectory
        
    Returns:
    --------
    array
        Updated control trajectory
    """
    new_alpha = np.zeros_like(time_grid)
    
    for i in range(len(time_grid)):
        # Get available compute at this time step
        C_t = compute_availability(time_grid[i])
        # Optimality condition: ∂H/∂α = 0
        term1 = np.exp(-rho * time_grid[i]) * beta_R * M[i]**gamma * C_t
        term2 = Lambda[i] * beta_T * epsilon * C_t**epsilon
        
        if term1 <= 0 or term2 <= 0:
            new_alpha[i] = 0
        else:
            # We need to be careful with the negative exponent
            exponent = 1/(epsilon-1)  # This is negative since epsilon < 1
            base = term1 / term2
            
            # For negative exponents, we need positive base
            if base > 0:
                alpha_opt = base**exponent
                new_alpha[i] = np.clip(alpha_opt, 0, 1)
            else:
                new_alpha[i] = 0
    
    return new_alpha

# Main FBSM algorithm
def run_fbsm(max_iterations=100, tolerance=1e-4):
    """
    Run the forward-backward sweep method.
    
    Parameters:
    -----------
    max_iterations : int
        Maximum number of iterations
    tolerance : float
        Convergence tolerance
        
    Returns:
    --------
    tuple
        Results of the optimization
    """
    global M, Lambda, alpha
    
    # Initialize
    M[0] = M0
    Lambda[-1] = lambda_T
    
    # Iteratively solve
    for iteration in range(max_iterations):
        # Store previous control for convergence check
        prev_alpha = alpha.copy()
        
        # Forward sweep
        M = forward_sweep(M0, alpha)
        
        # Backward sweep
        Lambda = backward_sweep(M, alpha)
        
        # Update control
        new_alpha = update_control(M, Lambda)
        
        # Apply relaxation for better convergence
        relaxation = 0.3
        alpha = (1-relaxation) * alpha + relaxation * new_alpha
        
        # Check convergence
        change = np.max(np.abs(alpha - prev_alpha))
        print(f"Iteration {iteration+1}: max change = {change:.6f}")
        
        if change < tolerance:
            print(f"Converged after {iteration+1} iterations")
            break
            
    # Calculate available compute at each time point
    C_values = compute_availability(time_grid)
    
    # Calculate revenue and NPV
    revenue = beta_R * M**gamma * (1-alpha) * C_values
    discounted_revenue = revenue * np.exp(-rho * time_grid)
    npv = np.trapezoid(discounted_revenue, time_grid)
    
    print(f"\nResults:\n")
    print(f"Total NPV: {npv:.2f}")
    print(f"Final model capability: {M[-1]:.2f}")
    print(f"Average allocation to training: {np.mean(alpha):.2f}")
    
    return M, alpha, revenue, npv

# Run the FBSM algorithm
print("Running Forward-Backward Sweep Method optimization...")
M, alpha, revenue, npv = run_fbsm()

# Calculate available compute at each time point
C_values = compute_availability(time_grid)

# Calculate inference compute (portion allocated to serving)
inference_compute = (1 - alpha) * C_values

# Calculate discounted revenue for plotting
discounted_revenue = revenue * np.exp(-rho * time_grid)

# Function to create comprehensive plots of the results
def create_plots(time_grid, alpha, M, revenue, discounted_revenue, C_values):
    """
    Create comprehensive plots of the optimization results.
    
    Parameters:
    -----------
    time_grid : array
        Time points
    alpha : array
        Control trajectory (allocation to training)
    M : array
        Model capability trajectory
    revenue : array
        Revenue trajectory
    discounted_revenue : array
        Discounted revenue trajectory
    C_values : array
        Available compute trajectory
    """
    # Calculate derived quantities
    inference_compute = (1 - alpha) * C_values
    training_compute = alpha * C_values
    cumulative_revenue = np.cumsum(discounted_revenue) * dt
    
    # Create a comprehensive dashboard of results
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Optimal allocation strategy
    plt.subplot(3, 2, 1)
    plt.plot(time_grid, alpha, 'b-', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Training Allocation (α)')
    plt.title('Training Allocation Over Time')
    plt.grid(True)
    plt.ylim(0, 1)
    
    # Plot 2: Compute allocation
    plt.subplot(3, 2, 2)
    plt.plot(time_grid, training_compute, 'r-', label='Training')
    plt.plot(time_grid, inference_compute, 'g-', label='Inference')
    plt.xlabel('Time')
    plt.ylabel('Compute')
    plt.title('Total Compute Allocation Over Time')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Model capability
    plt.subplot(3, 2, 3)
    plt.plot(time_grid, M, 'r-', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Model Capability (M)')
    plt.title('Model Capability Over Time')
    plt.grid(True)
    
    # Plot 4: Available compute
    plt.subplot(3, 2, 4)
    plt.plot(time_grid, C_values, 'k-', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Available Compute')
    plt.title('Available Compute Over Time')
    plt.grid(True)
    
    # Plot 5: Revenue
    plt.subplot(3, 2, 5)
    plt.plot(time_grid, revenue, 'g-', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Revenue')
    plt.title('Revenue Over Time')
    plt.grid(True)
    
    # Plot 6: Cumulative NPV
    plt.subplot(3, 2, 6)
    plt.plot(time_grid, cumulative_revenue, 'b-', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Cumulative NPV')
    plt.title('Cumulative NPV Over Time')
    plt.grid(True)
    
    # Save and display the plot
    plt.tight_layout()
    plt.savefig('power_law_fbsm_results.png', dpi=300)
    print("Saved power_law_fbsm_results.png")
    plt.show()  # Display the plot

# Generate comprehensive plots
create_plots(time_grid, alpha, M, revenue, revenue * np.exp(-rho * time_grid), compute_availability(time_grid))


# Sensitivity analysis
def run_sensitivity(param_name, param_values):
    """
    Run sensitivity analysis on a parameter.
    
    Parameters:
    -----------
    param_name : str
        Name of parameter to vary
    param_values : array
        Values to try for the parameter
    """
    global beta_T, beta_R, epsilon, gamma, rho, growth_rate
    
    results = []
    
    # Store original parameter value
    original_values = {
        'beta_T': beta_T,
        'beta_R': beta_R,
        'epsilon': epsilon,
        'gamma': gamma,
        'rho': rho,
        'growth_rate': growth_rate
    }
    
    for value in param_values:
        # Set parameter value
        if param_name == 'beta_T':
            beta_T = value
        elif param_name == 'beta_R':
            beta_R = value
        elif param_name == 'epsilon':
            epsilon = value
        elif param_name == 'gamma':
            gamma = value
        elif param_name == 'rho':
            rho = value
        elif param_name == 'growth_rate':
            growth_rate = value
        
        print(f"\nRunning with {param_name} = {value}")
        
        # Reset arrays
        global M, Lambda, alpha
        M = np.zeros_like(time_grid)
        Lambda = np.zeros_like(time_grid)
        alpha = np.ones_like(time_grid) * 0.5
        
        # Run FBSM
        M_result, alpha_result, revenue_result, npv_result = run_fbsm()
        
        # Store results
        results.append({
            'param_value': value,
            'npv': npv_result,
            'final_M': M_result[-1],
            'avg_alpha': np.mean(alpha_result)
        })
    
    # Restore original parameter values
    beta_T = original_values['beta_T']
    beta_R = original_values['beta_R']
    epsilon = original_values['epsilon']
    gamma = original_values['gamma']
    rho = original_values['rho']
    growth_rate = original_values['growth_rate']
    
    return results

# Sensitivity Analysis Functions

# Helper function to get display name for parameters
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
        'growth_rate': 'Compute Growth Rate'
    }
    return param_display.get(param_name, param_name)

# Helper function to normalize values for comparison
def normalize(values):
    """
    Normalize values to range [0,1] for comparison.
    
    Parameters:
    -----------
    values : list or array
        Values to normalize
        
    Returns:
    --------
    list
        Normalized values
    """
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return [0.5 for _ in values]  # Handle constant case
    return [(v - min_val) / (max_val - min_val) for v in values]

# Function to plot individual parameter sensitivity
def plot_sensitivity_results(param_name, param_values, results):
    """
    Create plots for sensitivity analysis results for a single parameter.
    
    Parameters:
    -----------
    param_name : str
        Name of parameter that was varied
    param_values : array
        Values used for the parameter
    results : list
        List of dictionaries containing results for each parameter value
    """
    # Extract results into arrays for plotting
    npv_values = [result['npv'] for result in results]
    final_M_values = [result['final_M'] for result in results]
    avg_alpha_values = [result['avg_alpha'] for result in results]
    
    # Get display name for parameter
    display_name = get_param_display_name(param_name)
    
    # Create a figure with 3 subplots
    plt.figure(figsize=(15, 5))
    
    # Create subplots for different metrics
    metrics = [
        {'data': npv_values, 'color': 'b', 'ylabel': 'Net Present Value', 'title': f'NPV vs {display_name}'},
        {'data': final_M_values, 'color': 'r', 'ylabel': 'Final Model Capability', 'title': f'Final Model Capability vs {display_name}'},
        {'data': avg_alpha_values, 'color': 'g', 'ylabel': 'Average Training Allocation', 'title': f'Avg Training Allocation vs {display_name}'}
    ]
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 3, i)
        plt.plot(param_values, metric['data'], f"{metric['color']}o-", linewidth=2)
        plt.xlabel(display_name)
        plt.ylabel(metric['ylabel'])
        plt.title(metric['title'])
        plt.grid(True)
        if i == 3:  # For allocation plot
            plt.ylim(0, 1)
    
    # Save and display the plot
    plt.tight_layout()
    plt.savefig(f'sensitivity_{param_name}.png', dpi=300)
    print(f"Saved sensitivity_{param_name}.png")
    plt.show()

# Function to plot combined sensitivity analysis
def plot_combined_sensitivity(param_configs):
    """
    Create a combined plot showing sensitivity analyses for multiple parameters.
    
    Parameters:
    -----------
    param_configs : list of dict
        List of parameter configurations, each containing:
        - name: parameter name
        - values: array of parameter values
        - results: list of results for each value
        - color: color to use for plotting
    """
    plt.figure(figsize=(15, 10))
    
    # Define the metrics to plot
    plot_configs = [
        {'metric': 'npv', 'title': 'Sensitivity of NPV to Parameters', 'ylabel': 'Normalized NPV', 'normalize': True, 'pos': 1},
        {'metric': 'final_M', 'title': 'Sensitivity of Final Model Capability to Parameters', 'ylabel': 'Normalized Final Model Capability', 'normalize': True, 'pos': 2},
        {'metric': 'avg_alpha', 'title': 'Sensitivity of Allocation Strategy to Parameters', 'ylabel': 'Average Training Allocation', 'normalize': False, 'pos': 3}
    ]
    
    # Create each subplot
    for config in plot_configs:
        plt.subplot(2, 2, config['pos'])
        
        for param in param_configs:
            # Extract data for this metric
            data = [r[config['metric']] for r in param['results']]
            
            # Normalize if needed
            if config['normalize']:
                data = normalize(data)
                
            # Plot the data
            plt.plot(param['values'], data, f"{param['color']}o-", label=get_param_display_name(param['name']))
            
        plt.xlabel('Parameter Value')
        plt.ylabel(config['ylabel'])
        plt.title(config['title'])
        plt.legend()
        plt.grid(True)
        
        # Set y-limits for allocation plot
        if config['metric'] == 'avg_alpha':
            plt.ylim(0, 1)
    
    # Add a textual summary
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    # Generate summary text
    summary_lines = ["Sensitivity Analysis Summary:\n\n"]
    
    for param in param_configs:
        name = param['name']
        display_name = get_param_display_name(name)
        values = param['values']
        results = param['results']
        
        # Add description for parameter
        if name == 'epsilon':
            description = "diminishing returns in training"
        elif name == 'gamma':
            description = "increasing returns in revenue"
        elif name == 'rho':
            description = "discount rate"
        elif name == 'growth_rate':
            description = "compute availability"
        else:
            description = name
        
        # Add parameter summary
        if name != 'growth_rate':
            summary_lines.append(f"{display_name} ({description}):\n")
        else:
            summary_lines.append(f"{display_name}:\n")
            
        summary_lines.append(f"  Range: {min(values):.2f} to {max(values):.2f}\n")
        summary_lines.append(f"  NPV range: {min([r['npv'] for r in results]):.2f} to {max([r['npv'] for r in results]):.2f}\n\n")
    
    # Join all summary lines
    summary_text = "".join(summary_lines)
    plt.text(0, 0.5, summary_text, fontsize=10, verticalalignment='center')
    
    # Save and display the combined plot
    plt.tight_layout()
    plt.savefig('combined_sensitivity_analysis.png', dpi=300)
    print("Saved combined_sensitivity_analysis.png")
    plt.show()

# Run sensitivity analyses
def run_all_sensitivity_analyses(num_samples=10):
    """
    Run sensitivity analyses for all parameters.
    
    Parameters:
    -----------
    num_samples : int
        Number of samples to use for each parameter
    
    Returns:
    --------
    list
        List of parameter configurations with results
    """
    # Define parameter ranges and colors
    param_configs = [
        {'name': 'epsilon', 'range': (0.1, 0.9), 'color': 'b'},
        {'name': 'gamma', 'range': (0.1, 3.0), 'color': 'r'},
        {'name': 'rho', 'range': (0.01, 0.2), 'color': 'g'},
        {'name': 'growth_rate', 'range': (0.05, 0.5), 'color': 'm'}
    ]
    
    # Run sensitivity analysis for each parameter
    for param in param_configs:
        # Generate parameter values
        param['values'] = np.linspace(param['range'][0], param['range'][1], num_samples)
        
        # Run sensitivity analysis
        print(f"Running sensitivity analysis for {param['name']}...")
        param['results'] = run_sensitivity(param['name'], param['values'])
        
        # Plot individual results
        plot_sensitivity_results(param['name'], param['values'], param['results'])
    
    # Create combined plot
    plot_combined_sensitivity(param_configs)
    
    return param_configs

# Run all sensitivity analyses
sensitivity_results = run_all_sensitivity_analyses(num_samples=10)
