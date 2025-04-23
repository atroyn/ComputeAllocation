# Optimal Compute Allocation Simulation

This project simulates and solves the optimal control problem for allocating compute resources between training new AI models and serving existing models to maximize net present value.

## Research Paper

This implementation is based on the research paper:

- **[Economically Optimal Compute Allocation](Economically_Optimal_Compute_Allocation.pdf)**

If you use this code or the results in your research, please cite the paper using the following BibTeX:

```bibtex
@misc{troynikov2025optimal,
  title={Economically Optimal Compute Allocation},
  author={Troynikov, Anton},
  year={2025},
  howpublished={\url{https://github.com/antontroynikov/ComputeAllocation}},
}
```

## Problem Formulation

The problem is formulated as an optimal control problem:

- **Objective**: Maximize the net present value of all revenue
  ```
  ∫(0 to T) e^(-ρt) · βR·M(t)^γ·(1-α(t))·C(t) dt
  ```

- **Subject to**: Model capability growth dynamics
  ```
  dM/dt = βT·(α(t)·C(t))^ε, ε ∈ (0, 1)
  ```

- **Variables**:
  - M(t): Model capability at time t
  - α(t) ∈ [0,1]: Fraction of compute allocated to training
  - C(t): Total available compute at time t
  - R(M): Revenue as a function of model capability

## Key Files

This project provides two main analysis approaches:

- **`steady_state.py`**: Analyzes the steady-state behavior in an infinite time horizon
- **`dynamic.py`**: Implements dynamic optimization using the Forward-Backward Sweep Method
- **`requirements.txt`**: Python dependencies

## Steady-State Analysis (`steady_state.py`)

This script analyzes the optimal allocation in a steady-state, infinite time horizon setting.

### Features

- Theoretical derivation of the steady-state allocation based on model parameters
- Numerical optimization to validate theoretical predictions
- Simulation of different allocation strategies (theoretical, numerical, baseline 50/50, high/low training)
- Sensitivity analysis to key parameters (ε, γ, growth-to-discount ratio)
- Comprehensive visualizations of results

### Usage

```
python steady_state.py
```

### Key Functions

- `theoretical_steady_state_allocation()`: Calculates the theoretical optimal allocation
- `numerical_steady_state_allocation()`: Finds the optimal allocation through direct NPV optimization
- `simulate_with_constant_allocation(alpha)`: Simulates system dynamics with a fixed allocation
- `plot_sensitivity()`: Analyzes how optimal allocation changes with different parameters

## Dynamic Optimization (`dynamic.py`)

This script implements the Forward-Backward Sweep Method (FBSM) to solve the dynamic optimal control problem over a finite time horizon.

### Features

- Full implementation of the FBSM algorithm for optimal control
- Forward simulation of model dynamics and backward solving of costate equations
- Optimality condition-based control updates
- Comprehensive visualizations of optimal trajectories
- Sensitivity analysis for all key parameters

### Usage

```
python dynamic.py
```

### Key Functions

- `forward_sweep(M0, alpha)`: Solves the state equation forward in time
- `backward_sweep(M, alpha)`: Solves the costate equation backward in time
- `update_control(M, Lambda)`: Updates the control based on optimality conditions
- `run_fbsm()`: Executes the complete FBSM algorithm
- `run_all_sensitivity_analyses()`: Performs sensitivity analysis on all parameters

## Results

Recent analysis with parameters ρ = 0.15, g = 0.1, ε = 0.5, γ = 2.0 shows:

- Theoretical steady-state allocation: 60%
- Numerical steady-state allocation: 50%
- Balanced allocation strategies (around 50%) perform best in NPV terms
- Both high training (80%) and low training (20%) performed similarly

The theoretical formula provides a good approximation but slightly overestimates the optimal training allocation compared to direct NPV optimization.

## Customization

You can modify the following parameters in either script:

- `beta_T`: Training efficiency scaling constant
- `beta_R`: Revenue efficiency scaling constant
- `epsilon`: Diminishing returns exponent in training (0 < ε < 1)
- `gamma`: Increasing returns exponent in revenue (γ > 1)
- `rho`: Discount rate
- `growth_rate`: Rate of compute growth
- `T`: Time horizon (for dynamic.py)
- `M0`: Initial model capability

You can also modify the `compute_availability` function to model different scenarios of compute growth over time.
