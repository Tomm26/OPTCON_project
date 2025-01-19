# Optimal Control of a Flexible Robotic Arm

Implementation of optimal control strategies for a flexible robotic arm, modeled as a planar two-link robot with torque applied to the first joint.

## Project Structure
```
OPTCON_project/
├── src/
│   ├── __init__.py
│   ├── parameters.py          # System parameters and constants
│   ├── dynamics.py           # System dynamics implementation
│   ├── derivatives.py        # Symbolic derivatives for optimal control
│   ├── equilibrium.py        # Equilibrium computation and trajectory generation
│   ├── controllers/
│   │   ├── __init__.py
│   │   ├── cost.py          # Cost functions and derivatives
│   │   ├── lqr.py           # LQR solver implementation
│   │   └── armijo.py        # Line search implementation
│   ├── tasks/
│   │   ├── __init__.py
│   │   └── task1.py         # Task 1 implementation
│   └── visualization/
│       ├── __init__.py
│       └── animate.py       # Robot animation utilities
└── README.md
```

## Requirements
- Python 3.8+
- NumPy
- SciPy
- SymPy
- Matplotlib
- tqdm

## Installation
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Task Implementation Status

### Task 0 - Problem Setup ✓
- [x] Dynamics discretization
- [x] State-space equations
- [x] Dynamics function implementation
- [x] Basic visualization

### Task 1 - Trajectory Generation (I) ✓
- [x] System equilibria computation
- [x] Reference curve definition
- [x] Newton-like algorithm implementation
- [x] Smooth trajectory generation
- [x] Numerical stabilization

### Task 2 - Trajectory Generation (II)
- [ ] Smooth state-input curve generation
- [ ] Trajectory generation implementation
- [ ] Quasi-static trajectory computation

### Task 3 - LQR Tracking
- [ ] Robot dynamics linearization
- [ ] LQR algorithm implementation
- [ ] Perturbed initial conditions testing

### Task 4 - MPC Tracking
- [ ] MPC algorithm implementation
- [ ] Tracking performance testing
- [ ] LQR comparison

### Task 5 - Animation ✓
- [x] Basic animation implementation
- [x] Trajectory visualization
- [ ] Phase space visualization

## Usage
To run Task 1 (equilibrium trajectory):
```bash
python -m src.tasks.task1
```

## File Description

### Core Files
- `parameters.py`: System constants and parameters
- `dynamics.py`: Robot dynamics implementation
- `derivatives.py`: Symbolic computation of derivatives
- `equilibrium.py`: Equilibrium finding and trajectory generation

### Controllers
- `cost.py`: Cost functions and their derivatives
- `lqr.py`: LTV-LQR solver with numerical stabilization
- `armijo.py`: Line search implementation for optimization

### Tasks
- `task1.py`: Implementation of equilibrium finding and trajectory generation

### Visualization
- `animate.py`: Robot animation and trajectory visualization