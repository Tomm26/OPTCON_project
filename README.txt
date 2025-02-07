# README

## Project Overview
This project implements a flexible robotic arm simulation, control, and optimization framework. It includes modules for dynamics modeling, trajectory generation, cost evaluation, optimization using Newton's method, and visualization through animation.

## Project Structure
The project is organized as follows:

- **`animate.py`**: Handles the animation of the robotic arm using `matplotlib`.
- **`cost.py`**: Defines the cost function for trajectory optimization.
- **`dynamics.py`**: Implements the dynamics of the flexible robotic arm.
- **`newton.py`**: Contains the Newton-based optimization algorithm.
- **`parameters.py`**: Defines the physical and control parameters of the robotic system.
- **`task1.py`**: Executes a predefined trajectory optimization task.
- **`task2.py`**: Implements an extended trajectory optimization task with data saving capabilities.
- **`task3_4.py`**: Implements controllers using Linear Quadratic Regulator (LQR) and Model Predictive Control (MPC).
- **`traj.py`**: Provides functions for trajectory generation and interpolation.
- **`utils.py`**: Utility functions, including gravity compensation function to generate reference for the input.

## Requirements
To run this project, you need the following dependencies:
- Python 3.x
- `numpy>=2.1.3`
- `scipy>=1.14.0`
- `sympy>=1.12.0`
- `matplotlib>=2.1.0`
- `pandas>=2.2.3`
- `cvxopt>=1.3.2`
- `control>=0.10.1`

Install the required packages using:
```sh
pip install -r requirements.txt
```

## How to Run

### Task 1
To run the first task with Newton optimization:
```sh
python task1.py
```

### Task 2
To execute the second task and save optimal trajectories:
```sh
python task2.py
```

### Task 3 & 4 (LQR and MPC Controllers)
To compare the performance of LQR and MPC controllers:
```sh
python task3_4.py
```

## Outputs
- The optimized trajectories are visualized using `matplotlib`.
- The animation of the robotic arm can be saved as an animation file.
- `task2.py` saves the optimal trajectory in CSV format.