# PINN Benchmarks + Hyperparameter Optimization (HPO) Suite

Complete Physics-Informed Neural Network (PINN) benchmark and Hyperparameter Optimization framework comparing **Genetic Algorithms (GA)**, **Particle Swarm Optimization (PSO)**, **Gravitational Search Algorithm (GSA)**, **Ant Colony Optimization (ACO)**, **Fuzzy-Adaptive Search**, and **Hybrid Metaheuristics**.

## 🏛️ System Architecture

```mermaid
graph TD
    subgraph "Standalone Optimizers"
        GA["GA (Genetic Algorithm)"]
        PSO["PSO (Particle Swarm)"]
        ACO["ACO (Ant Colony / ACOR)"]
        GSA["GSA (Gravitational Search)"]
    end

    subgraph "Hybrid Algorithms"
        GAPSO["GA-PSO Hybrid"]
        PSOGSA["PSO-GSA Hybrid"]
        ACOGA["ACO-GA Hybrid"]
    end

    subgraph "Fuzzy-Enhanced"
        FLC["Fuzzy Logic Controller (Mamdani)"]
        FPSO["Fuzzy-PSO"]
        FGA["Fuzzy-GA"]
        FACO["Fuzzy-ACO"]
    end

    subgraph "Comparison & Reporting"
        CMP["Comparison Engine (Config Search)"]
        RPT["Report & Plot Generator"]
    end

    GA --> GAPSO
    PSO --> GAPSO
    PSO --> PSOGSA
    GSA --> PSOGSA
    ACO --> ACOGA
    GA --> ACOGA

    FLC --> FPSO
    FLC --> FGA
    FLC --> FACO
    PSO --> FPSO
    GA --> FGA
    ACO --> FACO

    GA --> CMP
    PSO --> CMP
    ACO --> CMP
    GSA --> CMP
    GAPSO --> CMP
    PSOGSA --> CMP
    ACOGA --> CMP
    FPSO --> CMP
    FGA --> CMP
    FACO --> CMP

    CMP --> RPT
```

---

## 🔥 Evaluated PDE Benchmarks

- **ODE**: Exponential decay $\frac{dy}{dt} = -y$, $y(0) = 1$ (analytic solution: $y = e^{-t}$)
- **Burgers 1D**: Viscous Burgers equation $u_t + u u_x = \nu u_{xx}$ with shock dynamics
- **Heat Equation**: 1D diffusion $u_t = \alpha u_{xx}$ with Dirichlet boundaries
- **Allen-Cahn**: Phase-field equation $u_t = D u_{xx} + u - u^3$
- **Reaction-Diffusion**: Gray-Scott system (2D pattern formation)
- **2D Navier-Stokes**: Lid-driven cavity incompressible flow
- **Wave Equation**: 1D hyperbolic PDE $u_{tt} = c^2 u_{xx}$
- **Helmholtz**: Elliptic PDE $\nabla^2 u + k^2 u = f(x,y)$

---

## 🎯 8-Dimensional Hyperparameter Search Space

| Hyperparameter | Range / Options | Encoding |
| :--- | :--- | :--- |
| **Network Depth** | 1 to 6 hidden layers | Integer |
| **Network Width** | 8 to 256 neurons/layer | Integer |
| **Activation Function** | `tanh`, `sine` (Siren), `swish` (SiLU) | Categorical |
| **Optimizer** | `Adam`, `AdamW`, `L-BFGS` | Categorical |
| **Learning Rate** | $10^{-4}$ to $5 \times 10^{-2}$ | Continuous ($\log_{10}$) |
| **Physics Loss Weight** ($w_{phys}$) | 0.1 to 10.0 | Continuous |
| **Initial/Boundary Weight** ($w_{ic}$) | 0.1 to 50.0 | Continuous |
| **Collocation Points** | 64 to 1024 points | Integer |

---

## 🔬 Algorithm Categories

### 1. Standalone Metaheuristics
- **GA (Genetic Algorithm)**: Tournament selection, uniform crossover, random mutation, elitism.
- **PSO (Particle Swarm Optimization)**: Directional particle velocity updates with cognitive ($c_1$) and social ($c_2$) memory.
- **ACO (Ant Colony Optimization)**: Continuous ACOR with Gaussian kernel archive sampling.
- **GSA (Gravitational Search Algorithm)**: Agents attract via Newtonian gravity proportional to fitness masses; gravitational constant $G(t)$ decays over time.

### 2. Fuzzy-Adaptive Search (Fuzzy Logic Controller)
A Mamdani Fuzzy Inference System dynamically estimates population diversity, improvement rate, and search progress:
- **Fuzzy-PSO**: Dynamically adapts inertia weight $w(t) \in [0.3, 0.9]$ and social factor $c_2(t) \in [1.0, 2.5]$.
- **Fuzzy-GA**: Dynamically adjusts mutation rate $p_m(t) \in [0.05, 0.45]$ and crossover rate $p_c(t) \in [0.50, 0.95]$.
- **Fuzzy-ACO**: Dynamically adjusts dispersion $\zeta(t) \in [0.35, 1.20]$ and Gaussian sharpness $q(t) \in [0.15, 0.80]$.

### 3. Hybrid Metaheuristics
- **GA-PSO Hybrid**: Alternates between GA evolutionary exploration (crossover & mutation) and PSO particle velocity exploitation.
- **PSO-GSA Hybrid**: Unified velocity equation $V(t+1) = w V + c_1' r_1 a_{GSA} + c_2' r_2 (g_{best} - X)$ combining gravitational exploration with swarm exploitation (Mirjalili & Hashim).
- **ACO-GA Hybrid**: Uses ACO continuous archive for global exploration, then injects elite candidates to initialize GA for rapid schema recombination.

---

## 📁 Project Structure

```
d:/OptimizationOverviewPINN/
├── src/
│   ├── benchmarks/          # 8 PDE benchmark implementations
│   ├── models/              # Neural network architectures (MLP, Siren, activations)
│   ├── training/            # PINN trainer & benchmark factory
│   ├── hpo/                 # Hyperparameter Optimization methods
│   │   ├── search_space.py      # 8-dim search space & decoding logic
│   │   ├── ga.py                # Genetic Algorithm
│   │   ├── pso.py               # Particle Swarm Optimization
│   │   ├── aco.py               # Ant Colony Optimization (ACOR)
│   │   ├── gsa.py               # Gravitational Search Algorithm
│   │   ├── fuzzy_controller.py  # Mamdani Fuzzy Logic Controller
│   │   ├── fuzzy_pso.py         # Fuzzy-Adaptive PSO
│   │   ├── fuzzy_ga.py          # Fuzzy-Adaptive GA
│   │   ├── fuzzy_aco.py         # Fuzzy-Adaptive ACO
│   │   ├── hybrid_ga_pso.py     # GA-PSO Hybrid
│   │   ├── hybrid_pso_gsa.py    # PSO-GSA Hybrid
│   │   ├── hybrid_aco_ga.py     # ACO-GA Hybrid
│   │   ├── comparison.py        # Config-style multi-seed search engine
│   │   └── report_generator.py  # Automated plots & Markdown report builder
│   └── utils.py             # File I/O and reproducibility utilities
├── scripts/
│   ├── run_baseline.py          # Baseline PINN runner
│   ├── run_all_benchmarks.py    # Run baselines on all 8 PDEs
│   ├── run_ga.py                # Standalone GA runner
│   ├── run_pso.py               # Standalone PSO runner
│   ├── run_aco.py               # Standalone ACO runner
│   ├── run_gsa.py               # Standalone GSA runner
│   ├── run_fuzzy.py             # Fuzzy optimizers runner (--method pso/ga/aco/all)
│   ├── run_hybrids.py           # Hybrid optimizers runner (--method ga_pso/pso_gsa/aco_ga/all)
│   ├── run_full_comparison.py   # Master benchmark suite & report generator
│   └── run_tests.py             # Unit test suite
├── outputs/
│   ├── comparison/              # Master comparison data, report, and plots
│   │   ├── plots/               # Convergence, boxplot, radar, heatmap figures
│   │   ├── hpo_comparison_results.json
│   │   └── HPO_INVESTIGATION_REPORT.md
│   └── ...                      # Individual optimizer results
└── tests/                       # Comprehensive unit & integration tests
```

---

## 🚀 Execution Guide (PowerShell / Command Line)

### 1. Run Complete Investigation Suite & Generate Plots + Report
```powershell
# Full comparison across 4 benchmarks (ODE, Heat, Burgers, Wave) with 3 seeds
python scripts\run_full_comparison.py

# Rapid smoke test mode
python scripts\run_full_comparison.py --quick
```

### 2. Run Individual Optimizer Categories

```powershell
# Standalone optimizers
python scripts\run_ga.py ode
python scripts\run_pso.py ode
python scripts\run_aco.py ode
python scripts\run_gsa.py ode

# Fuzzy-adaptive optimizers
python scripts\run_fuzzy.py ode --method all
python scripts\run_fuzzy.py heat --method pso

# Hybrid metaheuristics
python scripts\run_hybrids.py ode --method all
python scripts\run_hybrids.py burgers --method pso_gsa
```

### 3. Run Test Suite
```powershell
python scripts\run_tests.py
```
