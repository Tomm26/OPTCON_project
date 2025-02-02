"""
Task 4: MPC Control Implementation for Flexible Robotic Arm System
This implementation focuses on Model Predictive Control (MPC) for trajectory tracking
"""
import os
import numpy as np
from numpy import diag, eye, array, hstack, zeros, ones
import pandas as pd
from datetime import datetime
import cvxopt
from cvxopt import matrix, solvers
from enum import Enum

from parameters import dt, ns, ni  # System parameters
from dynamics import FlexibleRoboticArm  # System dynamics

# Configure CVXOPT solver parameters for better numerical stability
solvers.options['show_progress'] = False  # Disable solver output
solvers.options['abstol'] = 1e-10        # Absolute tolerance
solvers.options['reltol'] = 1e-10        # Relative tolerance
solvers.options['feastol'] = 1e-10       # Feasibility tolerance

class RegulatorType(Enum):
    """
    Enumeration of controller types
    Note: This is used for labeling purposes only - the actual implementation
    uses MPC regardless of this setting
    """
    MPC = "MPC"  # Model Predictive Control
    #used to use other type of regulators

class GetData:
    """
    Class to store and manage trajectory tracking control data
    This class handles both reference and actual trajectories along with performance metrics
    """
    def __init__(self, 
                 xx_ref,                    # Reference state trajectory
                 uu_ref,                    # Reference input trajectory
                 tracking_results,          # List of (state, input) tuples for each tracking run
                 initial_conditions,        # List of initial states used
                 measurement_noise,         # Boolean flag for measurement noise
                 regulator_type,            # Type of controller used
                 start_time,                # Start time of computation
                 end_time):                 # End time of computation
        self.xx_ref = xx_ref
        self.uu_ref = uu_ref
        self.tracking_results = tracking_results
        self.initial_conditions = initial_conditions
        self.measurement_noise = measurement_noise
        self.regulator_type = regulator_type
        self.start_time = start_time
        self.end_time = end_time
        self.computation_time = (end_time - start_time).total_seconds()

    def get_tracking_error(self, run_index):
        """
        Calculate tracking error for a specific run
        Uses the L2 norm of the difference between actual and reference trajectories
        """
        xx_track = self.tracking_results[run_index][0]
        # Use the shorter length between reference and tracking
        T = min(xx_track.shape[1], self.xx_ref.shape[1])
        return np.linalg.norm(xx_track[:,:T] - self.xx_ref[:,:T], axis=0)

    def get_control_effort(self, run_index):
        """
        Calculate control effort for a specific run
        Uses the L2 norm of the difference between actual and reference inputs
        """
        uu_track = self.tracking_results[run_index][1]
        # Use the shorter length between reference and tracking
        T = min(uu_track.shape[1], self.uu_ref.shape[1])
        return np.linalg.norm(uu_track[:,:T] - self.uu_ref[:,:T], axis=0)

def task4(xx_traj, uu_traj, lazyExecution=False):
    """
    Task 4: MPC tracking control after linearization around trajectory
    
    Args:
        xx_traj: Reference state trajectory (4xT matrix)
        uu_traj: Reference input trajectory (1xT matrix)
        lazyExecution: Flag for lazy execution (not implemented)
    
    Returns:
        GetData object containing tracking results and performance metrics
    """
    # Initialize Flexible Robotic Arm system with parameters from Set 3
    fra_system = FlexibleRoboticArm(
        m1=1.5,  # mass of first link [kg]
        m2=1.5,  # mass of second link [kg]
        l1=2.0,  # length of first link [m]
        l2=2.0,  # length of second link [m]
        r1=1.0,  # distance to center of mass of first link [m]
        r2=1.0,  # distance to center of mass of second link [m]
        I1=2.0,  # inertia of first link [kg*m^2]
        I2=2.0,  # inertia of second link [kg*m^2]
        g=9.81,  # gravitational acceleration [m/s^2]
        f1=0.1,  # viscous friction coefficient first joint
        f2=0.1,  # viscous friction coefficient second joint
        dt=dt,   # discretization step from parameters.py
        ns=ns,   # number of states (4: [θ1, θ2, θ̇1, θ̇2])
        ni=ni,   # number of inputs (1: torque)
        method='rk'  # integration method (Runge-Kutta)
    )
    
    # MPC prediction horizon (reduced from 400 to 15 for computational efficiency)
    N = 15
    
    # Cost matrices for MPC optimization
    Q = diag([16.0, 16.0, 6.0, 6.0])  # State cost matrix (higher weights on positions)
    R = 0.0001 * eye(ni)              # Input cost matrix (small weight to minimize control effort)
    
    # Get trajectory dimensions
    T = xx_traj.shape[1]  # Total time steps
    
    # Extend trajectories for prediction horizon
    xx_ext = extend_trajectory(xx_traj, N)  # Extended state trajectory
    uu_ext = extend_trajectory(uu_traj, N)  # Extended input trajectory
    
    # Initialize linearization matrices for each time step
    AA = np.zeros((ns, ns, T+N))  # State transition matrices
    BB = np.zeros((ns, ni, T+N))  # Input matrices
    
    # Compute linearization along trajectory
    print("Computing linearization along trajectory...")
    for t in range(T+N):
        # Extract current state and input
        x_t = xx_ext[:, t]
        u_t = uu_ext[:, t]
        
        try:
            # Calculate linearization matrices at current point
            A_t, B_t = fra_system.get_gradients(x_t, u_t)
            
            # Convert to numpy arrays and store
            AA[:,:,t] = np.array(A_t, dtype=float)
            BB[:,:,t] = np.array(B_t, dtype=float)
            
            # Debug printing for first few time steps
            if t < 3:
                print(f"Time {t}:")
                print("State:", x_t)
                print("Input:", u_t)
                print("A shape:", AA[:,:,t].shape)
                print("B shape:", BB[:,:,t].shape)
                
        except Exception as e:
            print(f"Error at time step {t}:")
            print(f"State: {x_t}")
            print(f"Input: {u_t}")
            raise e
    
    # Define disturbance levels for robustness testing
    disturbance_levels = [0.0, 0.1, 0.2]  # 0%, 10%, and 20% perturbations
    measurement_noise = False  # Flag for measurement noise (not implemented)
    
    # Run MPC tracking for each disturbance level
    start_time = datetime.now()
    initial_conditions = []
    tracking_results = []
    
    for level in disturbance_levels:
        print(f"Running MPC tracking with {level*100}% disturbance...")
        # Generate perturbed initial condition
        x0 = generate_initial_condition(xx_traj[:, 0], level)
        initial_conditions.append(x0)
        
        # Run MPC controller
        xx_cl, uu_cl = run_mpc_controller(
            fra_system,
            x0,
            xx_ext,
            uu_ext,
            AA,
            BB,
            Q,
            R,
            N,
            measurement_noise
        )
        
        # Store results (excluding prediction horizon extension)
        tracking_results.append((
            xx_cl[:, :-N],
            uu_cl[:, :-N]
        ))
    
    # Create results object with all tracking data
    results = GetData(
        xx_traj,
        uu_traj,
        tracking_results,
        initial_conditions,
        measurement_noise,
        RegulatorType.MPC,
        start_time,
        datetime.now()
    )
    
    return results

def run_mpc_controller(system, x0, xx_ref, uu_ref, AA, BB, Q, R, N, use_noise):
    """
    Run MPC controller for the complete trajectory
    
    Args:
        system: FlexibleRoboticArm instance
        x0: Initial state
        xx_ref: Reference state trajectory
        uu_ref: Reference input trajectory
        AA: Sequence of linearized state matrices
        BB: Sequence of linearized input matrices
        Q: State cost matrix
        R: Input cost matrix
        N: Prediction horizon
        use_noise: Flag for measurement noise
    
    Returns:
        xx: State trajectory
        uu: Input trajectory
    """
    T = xx_ref.shape[1] - N  # Total time steps (adjusted for prediction horizon)
    ns = system.ns           # Number of states
    ni = system.ni           # Number of inputs
    
    # Initialize state and input trajectories
    xx = zeros((ns, T))     # State trajectory storage
    uu = zeros((ni, T-1))   # Input trajectory storage
    xx[:, 0] = x0           # Set initial state
    
    # Input constraints for the system
    u_max = 10.0            # Maximum allowable input
    u_min = -10.0           # Minimum allowable input
    
    # Run MPC loop for each time step
    for t in range(T-1):
        # Compute current state error
        x_error = xx[:, t] - xx_ref[:, t]
        
        # Solve MPC optimization problem
        u_mpc = solve_mpc_problem(
            x_error,
            xx_ref[:, t:t+N+1],  # Reference trajectory segment
            uu_ref[:, t:t+N],    # Reference input segment
            AA[:, :, t:t+N],     # Linearized A matrices
            BB[:, :, t:t+N],     # Linearized B matrices
            Q,
            R,
            u_min,
            u_max,
            N
        )
        
        # Apply first input (receding horizon principle)
        uu[:, t] = u_mpc[0] + uu_ref[:, t]
        
        # Simulate system forward one step
        xx[:, t+1] = system.discrete_dynamics(xx[:, t], uu[:, t])
        
        # Add measurement noise if enabled
        if use_noise:
            xx[:, t+1] += np.random.normal(0, 0.01, ns)
    
    return xx, uu

def solve_mpc_problem(x0, xx_ref, uu_ref, AA, BB, Q, R, u_min, u_max, N):
    """
    Solve the MPC optimization problem using CVXOPT quadratic programming
    
    Args:
        x0: Current state error
        xx_ref: Reference state trajectory segment
        uu_ref: Reference input trajectory segment
        AA: Sequence of linearized A matrices
        BB: Sequence of linearized B matrices
        Q: State cost matrix
        R: Input cost matrix
        u_min: Minimum input constraint
        u_max: Maximum input constraint
        N: Prediction horizon
    
    Returns:
        Optimal input sequence
    """
    ns = x0.shape[0]    # Number of states
    ni = BB.shape[1]    # Number of inputs
    
    # Formulate quadratic programming matrices
    P, q = formulate_cost_matrices(AA, BB, Q, R, N, x0)
    G, h = formulate_constraint_matrices(N, u_min, u_max)
    
    # Solve quadratic program using CVXOPT
    sol = solvers.qp(
        matrix(P),       # Quadratic cost matrix
        matrix(q),       # Linear cost vector
        matrix(G),       # Constraint matrix
        matrix(h)        # Constraint vector
    )
    
    # Check solution status
    if sol['status'] != 'optimal':
        raise RuntimeError("QP solver did not converge to optimal solution")
    
    # Extract and reshape solution
    return np.array(sol['x']).reshape(-1, ni)

def formulate_cost_matrices(AA, BB, Q, R, N, x0):
    """
    Generate cost matrices for quadratic programming problem
    
    Args:
        AA: Sequence of linearized A matrices
        BB: Sequence of linearized B matrices
        Q: State cost matrix
        R: Input cost matrix
        N: Prediction horizon
        x0: Initial state error
    
    Returns:
        P: Quadratic cost matrix
        q: Linear cost vector
    """
    ns = AA.shape[0]    # Number of states
    ni = BB.shape[1]    # Number of inputs
    
    # Build block diagonal input cost matrix
    P = zeros((N*ni, N*ni))
    for i in range(N):
        P[i*ni:(i+1)*ni, i*ni:(i+1)*ni] = R
    
    # Build linear cost vector
    q = zeros((N*ni, 1))
    x_pred = x0
    for i in range(N):
        # Compute cost based on predicted state evolution
        q[i*ni:(i+1)*ni] = 2 * BB[:,:,i].T @ Q @ x_pred
        # Predict next state
        x_pred = AA[:,:,i] @ x_pred + BB[:,:,i] @ zeros(ni)
    
    return P, q

def formulate_constraint_matrices(N, u_min, u_max):
    """
    Generate constraint matrices for input bounds in the QP problem
    
    Args:
        N: Prediction horizon
        u_min: Minimum input constraint
        u_max: Maximum input constraint
    
    Returns:
        G: Constraint matrix
        h: Constraint vector
    """
    ni = 1  # Single input system
    
    # Input constraints: u_min ≤ u ≤ u_max
    G = zeros((2*N*ni, N*ni))  # Matrix for both upper and lower bounds
    h = zeros(2*N*ni)          # Vector for bound values
    
    # Build constraint matrices for each step in horizon
    for i in range(N):
        # Upper bound constraints
        G[i*ni:(i+1)*ni, i*ni:(i+1)*ni] = eye(ni)
        # Lower bound constraints
        G[N*ni+i*ni:N*ni+(i+1)*ni, i*ni:(i+1)*ni] = -eye(ni)
        # Constraint values
        h[i*ni:(i+1)*ni] = u_max                    # Upper bounds
        h[N*ni+i*ni:N*ni+(i+1)*ni] = -u_min         # Lower bounds
    
    return G, h

def plot_mpc_results(results):
    """
    Plot MPC tracking results and performance metrics
    
    Args:
        results: GetData object containing tracking results
    """
    import matplotlib.pyplot as plt
    
    # Find common length for all trajectories
    T_ref = results.xx_ref.shape[1]
    T_track = min(track[0].shape[1] for track in results.tracking_results)
    T = min(T_ref, T_track)
    
    # Create time vector
    time = np.arange(0, T*dt, dt)
    
    # Create main figure for state and input trajectories
    fig, axs = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
    fig.suptitle('MPC Tracking Results', fontsize=16)
    
    # Labels for states and disturbance levels
    state_labels = [r'$\theta_1$', r'$\theta_2$', r'$\dot{\theta}_1$', r'$\dot{\theta}_2$']
    disturbance_labels = ['0%', '10%', '20%']
    
    # Plot states
    for i in range(4):
        ax = axs[i]
        # Plot reference trajectory
        ax.plot(time, results.xx_ref[i, :T], 'k--', label='Reference', alpha=0.7)
        
        # Plot actual trajectories for each disturbance level
        for j, (xx, _) in enumerate(results.tracking_results):
            ax.plot(time, xx[i, :T], 
                   label=f'Disturbance {disturbance_labels[j]}',
                   alpha=0.8)
        
        ax.set_ylabel(state_labels[i])
        ax.grid(True)
        ax.legend()
    
    # Plot control input
    ax = axs[4]
    ax.plot(time, results.uu_ref[0, :T], 'k--', label='Reference', alpha=0.7)
    
    for j, (_, uu) in enumerate(results.tracking_results):
        ax.plot(time[:-1], uu[0, :(T-1)], 
               label=f'Disturbance {disturbance_labels[j]}',
               alpha=0.8)
    
    ax.set_ylabel('Control Input u')
    ax.set_xlabel('Time [s]')
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    
    # Create second figure for performance metrics
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig2.suptitle('MPC Performance Metrics', fontsize=16)
    
    # Plot tracking errors
    ax1.set_title('State Tracking Error')
    for j in range(len(results.tracking_results)):
        error = results.get_tracking_error(j)
        ax1.plot(time, error[:T], 
                label=f'Disturbance {disturbance_labels[j]}')
    ax1.set_ylabel('Error Norm')
    ax1.grid(True)
    ax1.legend()
    
    # Plot control effort
    ax2.set_title('Control Effort')
    for j in range(len(results.tracking_results)):
        effort = results.get_control_effort(j)
        ax2.plot(time[:-1], effort[:(T-1)], 
                label=f'Disturbance {disturbance_labels[j]}')
    ax2.set_ylabel('Input Difference Norm')
    ax2.set_xlabel('Time [s]')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save plots to file
    try:
      
        fig.savefig('plots/mpc_tracking.png')
        fig2.savefig('plots/mpc_performance.png')
        print("Plots saved")
    except Exception as e:
        print(f"Error saving plots: {e}")
    
    plt.show()

def extend_trajectory(traj, N):
    """
    Extend trajectory by N steps by repeating last value
    
    Args:
        traj: Original trajectory
        N: Number of steps to extend
    
    Returns:
        Extended trajectory
    """
    last_value = traj[:, -1:]  # Keep 2D shape
    extension = np.tile(last_value, (1, N))
    return np.hstack((traj, extension))

def generate_initial_condition(x_ref, disturbance_level):
    """
    Generate perturbed initial condition
    
    Args:
        x_ref: Reference initial state
        disturbance_level: Magnitude of perturbation (as fraction)
    
    Returns:
        Perturbed initial state
    """
    return x_ref + disturbance_level * np.random.randn(*x_ref.shape)

if __name__ == "__main__":

    try:
        # Load trajectory data from CSV
        df = pd.read_csv('traj/optimal_trajectories.csv')
        print("Available columns in CSV:", df.columns)
        
        # Extract state and input trajectories
        xx_traj = np.array([
            df['x1'].values,  # θ1: angle of first link
            df['x2'].values,  # θ2: angle of second link
            df['x3'].values,  # θ̇1: angular velocity of first link
            df['x4'].values,  # θ̇2: angular velocity of second link
        ])
        
        uu_traj = np.array([df['u'].values])  # Control input
        
        # Reshape trajectories to correct dimensions
        xx_traj = xx_traj.reshape(4, -1)  # 4 states x T timesteps
        uu_traj = uu_traj.reshape(1, -1)  # 1 input x T timesteps
        
        print("Shapes:")
        print("xx_traj shape:", xx_traj.shape)
        print("uu_traj shape:", uu_traj.shape)
        
        # Run MPC tracking
        results = task4(xx_traj, uu_traj)
        print("Task 4 completed successfully!")
        
        # Generate and save plots
        plot_mpc_results(results)
        
    except FileNotFoundError:
        print("Error: Could not find trajectory file")
    except KeyError as e:
        print(f"Error: Column not found in CSV file: {e}")
        print("Available columns:", df.columns.tolist() if 'df' in locals() else "CSV not loaded")
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
