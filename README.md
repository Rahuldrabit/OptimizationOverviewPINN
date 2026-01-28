# PINN Benchmarks + HPO (Reorganized)

Complete Physics-Informed Neural Network (PINN) benchmark suite with hyperparameter optimization using Genetic Algorithm (GA), Particle Swarm Optimization (PSO), and Ant Colony Optimization (ACO).

## 🔥  All PDE Benchmarks

### Implemented Benchmarks
- **ODE**: Exponential decay (analytic solution available)
- **Burgers 1D**: Viscous Burgers equation with shock formation
- **Heat Equation**: 1D diffusion with analytic solution
- **Allen-Cahn**: Phase-field equation with interface dynamics
- **Reaction-Diffusion**: Gray-Scott system (pattern formation)
- **2D Navier-Stokes**: Lid-driven cavity flow
- **Wave Equation**: 1D hyperbolic PDE with analytic solution
- **Helmholtz**: Elliptic PDE with manufactured solution

### Hyperparameters Optimized
- **Learning rate** (log-scale: 1e-4 to 5e-2)
- **Optimizer**: Adam, AdamW, L-BFGS
- **Network depth**: 1-6 hidden layers
- **Network width**: 8-256 neurons per layer
- **Activation**: tanh, sine, swish (SiLU)
- **Physics vs BC loss weights**
- **Collocation point count**: 64-1024

## 📁 Project Structure

```
src/
  benchmarks/          # All PDE benchmark implementations
    ode/              # Exponential decay ODE
    burgers/          # 1D Burgers equation  
    heat/             # Heat/diffusion equation
    allen_cahn/       # Allen-Cahn phase field
    reaction_diffusion/# Gray-Scott system
    navier_stokes/    # 2D incompressible flow
    wave/             # Wave & Helmholtz equations
  models/             # Neural network architectures
  training/           # PINN training logic + benchmark factory
  hpo/               # Hyperparameter optimization methods
    ga.py            # Genetic Algorithm (PyGAD)
    pso.py           # Particle Swarm Optimization (PySwarm)
    aco.py           # Ant Colony Optimization (custom ACOR)
    search_space.py  # Shared hyperparameter encoding
  utils.py           # Utilities (seeding, file I/O)
tests/               # Comprehensive test suite
scripts/             # Command-line runners
outputs/             # Results organized by method and benchmark
```

## 🚀 Setup (Windows PowerShell)

```powershell
cd e:\PINN
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Install PyTorch (choose one):
# CPU version:
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
# GPU version (CUDA 12.1):
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 🧪 Usage Examples

### Run Single Benchmark (Baseline)
```powershell
# ODE benchmark (default)
python scripts\run_baseline.py

# Specific benchmark
python scripts\run_baseline.py burgers
python scripts\run_baseline.py heat
python scripts\run_baseline.py allen_cahn
```

### Run All Benchmarks (Comparison)
```powershell
python scripts\run_all_benchmarks.py
```

### Hyperparameter Optimization

```powershell
# Genetic Algorithm
python scripts\run_ga.py ode
python scripts\run_ga.py heat

# Particle Swarm Optimization  
python scripts\run_pso.py ode
python scripts\run_pso.py burgers

# Ant Colony Optimization
python scripts\run_aco.py ode
python scripts\run_aco.py wave
```

### Run Tests
```powershell
python scripts\run_tests.py
```

## 📊 Results Structure

Results are organized as:
```
outputs/
  baseline/
    ode/metrics.json
    burgers/metrics.json
    heat/metrics.json
    ...
  ga/
    ode/ga_best_metrics.json
    heat/ga_best_metrics.json 
    ...
  pso/
    ode/pso_best_metrics.json
    ...
  aco/
    ode/aco_best_metrics.json
    ...
  comparison/
    baseline_comparison.json
```

## 🔬 Benchmark Details

### ODE: Exponential Decay
- **Equation**: `y'(t) = -y(t)`, `y(0) = 1`
- **Domain**: `t ∈ [0, 5]`
- **Analytic**: `y(t) = exp(-t)`
- **Status**: ✅ Fully implemented with training

### Burgers 1D
- **Equation**: `u_t + u*u_x = ν*u_xx`
- **IC**: `u(x,0) = sin(πx)`
- **BC**: `u(±1,t) = 0`
- **Status**: 📐 Benchmark defined, placeholder training

### Heat Equation
- **Equation**: `u_t = α*u_xx`
- **IC**: `u(x,0) = sin(πx/L)`
- **BC**: `u(0,t) = u(L,t) = 0`
- **Analytic**: `u(x,t) = sin(πx/L)*exp(-α*π²t/L²)`
- **Status**: 📐 Benchmark defined, placeholder training

### Allen-Cahn
- **Equation**: `u_t = D*u_xx + u - u³`
- **IC**: `u(x,0) = tanh((x-x₀)/√(2D))`
- **BC**: Neumann `u_x = 0`
- **Status**: 📐 Benchmark defined, placeholder training

### Reaction-Diffusion (Gray-Scott)
- **Equations**: 
  - `u_t = D_u*∇²u - uv² + f(1-u)`
  - `v_t = D_v*∇²v + uv² - (f+k)v`
- **Domain**: 2D with periodic BC
- **Status**: 📐 Benchmark defined, placeholder training

### 2D Navier-Stokes
- **Equations**: 
  - `u_t + u·∇u = -∇p + ν∇²u`
  - `∇·u = 0`
- **Setup**: Lid-driven cavity
- **Status**: 📐 Benchmark defined, placeholder training

### Wave Equation
- **Equation**: `u_tt = c²*u_xx`
- **IC**: `u(x,0) = sin(πx/L)`, `u_t(x,0) = 0`
- **Analytic**: `u(x,t) = sin(πx/L)*cos(cπt/L)`
- **Status**: 📐 Benchmark defined, placeholder training

### Helmholtz
- **Equation**: `∇²u + k²u = f(x,y)`
- **Source**: `f = 2π²sin(πx)sin(πy)`
- **Analytic**: `u = sin(πx)sin(πy)` when `k² = 2π²`
- **Status**: 📐 Benchmark defined, placeholder training

## ⚙️ HPO Method Details

### Genetic Algorithm (PyGAD)
- **Population**: 10 individuals
- **Generations**: 8  
- **Selection**: Steady-state selection
- **Crossover**: Single-point
- **Mutation**: 20% gene mutation rate

### Particle Swarm Optimization (PySwarm)
- **Swarm size**: 12 particles
- **Iterations**: 6-8
- **Inertia/acceleration**: PySwarm defaults

### Ant Colony Optimization (Custom ACOR)
- **Ants**: 10
- **Iterations**: 8
- **Archive size**: 10
- **Gaussian sampling**: σ based on solution diversity

## 🎯 Next Steps

1. **Install PyTorch** and run baseline: `python scripts\run_baseline.py`
2. **Run tests**: `python scripts\run_tests.py`
3. **Compare all benchmarks**: `python scripts\run_all_benchmarks.py`
4. **Try HPO**: `python scripts\run_ga.py ode`

