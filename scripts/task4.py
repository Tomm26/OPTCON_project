"""
Task 4: MPC Control Implementation for Flexible Robotic Arm System
"""
import os
import numpy as np
from numpy import diag, eye, array, hstack, zeros, ones
import pandas as pd
from datetime import datetime
import cvxopt
from cvxopt import matrix, solvers
from enum import Enum

from parameters import dt, ns, ni
from dynamics import FlexibleRoboticArm

# Configure CVXOPT solver
solvers.options['show_progress'] = False
solvers.options['abstol'] = 1e-10
solvers.options['reltol'] = 1e-10
solvers.options['feastol'] = 1e-10

class RegulatorType(Enum):
    """Enumeration of controller types"""
    MPC = "MPC"
    LQR = "LQR"
    PID = "PID"

class GetData:
    """Class to store trajectory tracking control data"""
    def __init__(self, 
                 xx_ref, 
                 uu_ref,
                 tracking_results,
                 initial_conditions,
                 measurement_noise,
                 regulator_type,
                 start_time,
                 end_time):
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
        """Calculate tracking error for a specific run"""
        xx_track = self.tracking_results[run_index][0]
        # Use the shorter length between reference and tracking
        T = min(xx_track.shape[1], self.xx_ref.shape[1])
        return np.linalg.norm(xx_track[:,:T] - self.xx_ref[:,:T], axis=0)

    def get_control_effort(self, run_index):
        """Calculate control effort for a specific run"""
        uu_track = self.tracking_results[run_index][1]
        # Use the shorter length between reference and tracking
        T = min(uu_track.shape[1], self.uu_ref.shape[1])
        return np.linalg.norm(uu_track[:,:T] - self.uu_ref[:,:T], axis=0)

def task4(xx_traj, uu_traj, lazyExecution=False):
    """
    Task 4: MPC tracking control after linearization around trajectory
    """
    # Initialize FRA system
    fra_system = FlexibleRoboticArm(
        m1=1.5,  # mass of first link
        m2=1.5,  # mass of second link
        l1=2.0,  # length of first link
        l2=2.0,  # length of second link
        r1=1.0,  # distance to center of mass of first link
        r2=1.0,  # distance to center of mass of second link
        I1=2.0,  # inertia of first link
        I2=2.0,  # inertia of second link
        g=9.81,  # gravitational acceleration
        f1=0.1,  # viscous friction coefficient first joint
        f2=0.1,  # viscous friction coefficient second joint
        dt=dt,   # discretization step from parameters.py
        ns=ns,   # number of states
        ni=ni,   # number of inputs
        method='rk'  # integration method
    )
    
    # MPC parameters
    N = 15 # Prediction horizon
    
    # Cost matrices
    Q = diag([16.0, 16.0, 6.0, 6.0])  # State cost
    R = 0.0001 * eye(ni)  # Input cost
    
    # Get trajectory dimensions
    T = xx_traj.shape[1]
    
    # Extend trajectories for prediction horizon
    xx_ext = extend_trajectory(xx_traj, N)
    uu_ext = extend_trajectory(uu_traj, N)
    
    # Initialize linearization matrices
    AA = np.zeros((ns, ns, T+N))
    BB = np.zeros((ns, ni, T+N))
    
    # Compute linearization along trajectory
    print("Computing linearization along trajectory...")
    for t in range(T+N):
        # Estrai lo stato e l'input correnti
        x_t = xx_ext[:, t]
        u_t = uu_ext[:, t]
        
        try:
            # Calcola le matrici di linearizzazione per il punto corrente
            A_t, B_t = fra_system.get_gradients(x_t, u_t)
            
            # Converti in array numpy se necessario
            AA[:,:,t] = np.array(A_t, dtype=float)
            BB[:,:,t] = np.array(B_t, dtype=float)
            
            # Stampa di debug per i primi punti
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
    
    # Initial conditions with different perturbation levels
    disturbance_levels = [0.0, 0.1, 0.2]
    measurement_noise = False
    
    # Run MPC tracking for each disturbance level
    start_time = datetime.now()
    initial_conditions = []
    tracking_results = []
    
    for level in disturbance_levels:
        print(f"Running MPC tracking with {level*100}% disturbance...")
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
    
    # Create results object
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
    """Run MPC controller for the complete trajectory"""
    T = xx_ref.shape[1] - N  # Adjust for prediction horizon
    ns = system.ns
    ni = system.ni
    
    xx = zeros((ns, T))
    uu = zeros((ni, T-1))
    xx[:, 0] = x0
    
    # Input constraints
    u_max = 10.0
    u_min = -10.0
    
    # Run MPC loop
    for t in range(T-1):
        # Current state error
        x_error = xx[:, t] - xx_ref[:, t]
        
        # Solve MPC optimization problem
        u_mpc = solve_mpc_problem(
            x_error,
            xx_ref[:, t:t+N+1],
            uu_ref[:, t:t+N],
            AA[:, :, t:t+N],
            BB[:, :, t:t+N],
            Q,
            R,
            u_min,
            u_max,
            N
        )
        
        # Apply first input
        uu[:, t] = u_mpc[0] + uu_ref[:, t]
        
        # Simulate system forward
        xx[:, t+1] = system.discrete_dynamics(xx[:, t], uu[:, t])
        
        # Add measurement noise if enabled
        if use_noise:
            xx[:, t+1] += np.random.normal(0, 0.01, ns)
    
    return xx, uu

def solve_mpc_problem(x0, xx_ref, uu_ref, AA, BB, Q, R, u_min, u_max, N):
    """Solve the MPC optimization problem using CVXOPT"""
    ns = x0.shape[0]
    ni = BB.shape[1]
    
    # Formulate QP matrices
    P, q = formulate_cost_matrices(AA, BB, Q, R, N, x0)
    G, h = formulate_constraint_matrices(N, u_min, u_max)
    
    # Solve QP
    sol = solvers.qp(
        matrix(P),
        matrix(q),
        matrix(G),
        matrix(h)
    )
    
    if sol['status'] != 'optimal':
        raise RuntimeError("QP solver did not converge to optimal solution")
    
    return np.array(sol['x']).reshape(-1, ni)

def formulate_cost_matrices(AA, BB, Q, R, N, x0):
    """Generate cost matrices for QP problem"""
    ns = AA.shape[0]
    ni = BB.shape[1]
    
    # Build block diagonal Q matrix
    P = zeros((N*ni, N*ni))
    for i in range(N):
        P[i*ni:(i+1)*ni, i*ni:(i+1)*ni] = R
    
    # Build q vector
    q = zeros((N*ni, 1))
    x_pred = x0
    for i in range(N):
        q[i*ni:(i+1)*ni] = 2 * BB[:,:,i].T @ Q @ x_pred
        x_pred = AA[:,:,i] @ x_pred + BB[:,:,i] @ zeros(ni)
    
    return P, q

def formulate_constraint_matrices(N, u_min, u_max):
    """Generate constraint matrices for QP problem"""
    ni = 1  # Single input system
    
    # Input constraints: u_min ≤ u ≤ u_max
    G = zeros((2*N*ni, N*ni))
    h = zeros(2*N*ni)
    
    for i in range(N):
        G[i*ni:(i+1)*ni, i*ni:(i+1)*ni] = eye(ni)  # Upper bound
        G[N*ni+i*ni:N*ni+(i+1)*ni, i*ni:(i+1)*ni] = -eye(ni)  # Lower bound
        h[i*ni:(i+1)*ni] = u_max
        h[N*ni+i*ni:N*ni+(i+1)*ni] = -u_min
    
    return G, h

def plot_mpc_results(results):
    """
    Plot MPC tracking results
    """
    import matplotlib.pyplot as plt
    
    # Get the shortest length among all trajectories to ensure consistency
    T_ref = results.xx_ref.shape[1]
    T_track = min(track[0].shape[1] for track in results.tracking_results)
    T = min(T_ref, T_track)
    
    # Create time vector with correct length
    time = np.arange(0, T*dt, dt)
    
    # Create subplots
    fig, axs = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
    fig.suptitle('MPC Tracking Results', fontsize=16)
    
    # State labels
    state_labels = [r'$\theta_1$', r'$\theta_2$', r'$\dot{\theta}_1$', r'$\dot{\theta}_2$']
    
    # Disturbance labels
    disturbance_labels = ['0%', '10%', '20%']
    
    # Plot states
    for i in range(4):
        ax = axs[i]
        # Reference trajectory
        ax.plot(time, results.xx_ref[i, :T], 'k--', label='Reference', alpha=0.7)
        
        # Actual trajectories for each disturbance level
        for j, (xx, _) in enumerate(results.tracking_results):
            ax.plot(time, xx[i, :T], 
                   label=f'Disturbance {disturbance_labels[j]}',
                   alpha=0.8)
        
        ax.set_ylabel(state_labels[i])
        ax.grid(True)
        ax.legend()
    
    # Plot control input
    ax = axs[4]
    # Reference input
    ax.plot(time, results.uu_ref[0, :T], 'k--', label='Reference', alpha=0.7)
    
    # Actual inputs for each disturbance level
    for j, (_, uu) in enumerate(results.tracking_results):
        ax.plot(time[:-1], uu[0, :(T-1)], 
               label=f'Disturbance {disturbance_labels[j]}',
               alpha=0.8)
    
    ax.set_ylabel('Control Input u')
    ax.set_xlabel('Time [s]')
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    
    # Plot tracking errors
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig2.suptitle('MPC Performance Metrics', fontsize=16)
    
    # Tracking errors
    ax1.set_title('State Tracking Error')
    for j in range(len(results.tracking_results)):
        error = results.get_tracking_error(j)
        ax1.plot(time, error[:T], 
                label=f'Disturbance {disturbance_labels[j]}')
    ax1.set_ylabel('Error Norm')
    ax1.grid(True)
    ax1.legend()
    
    # Control effort
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
    
    # Save plots
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(current_dir)
        plots_dir = os.path.join(project_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        fig.savefig(os.path.join(plots_dir, 'mpc_tracking.png'))
        fig2.savefig(os.path.join(plots_dir, 'mpc_performance.png'))
        print(f"Plots saved in {plots_dir}")
    except Exception as e:
        print(f"Error saving plots: {e}")
    
    plt.show()


def extend_trajectory(traj, N):
    """Extend trajectory by N steps by repeating last value"""
    last_value = traj[:, -1:]  # Mantiene la dimensione 2D
    extension = np.tile(last_value, (1, N))
    return np.hstack((traj, extension))

def generate_initial_condition(x_ref, disturbance_level):
    """Generate perturbed initial condition"""
    return x_ref + disturbance_level * np.random.randn(*x_ref.shape)

if __name__ == "__main__":
    # Get the correct path to the CSV file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    csv_path = os.path.join(project_dir, 'traj', 'optimal_trajectories.csv')

    try:
        # Load data using pandas
        df = pd.read_csv(csv_path)
        
        # Print column names to debug
        print("Available columns in CSV:", df.columns.tolist())
        
        # Extract state and input trajectories
        xx_traj = np.array([
            df['x1'].values,  # invece di theta1
            df['x2'].values,  # invece di theta2
            df['x3'].values,  # invece di dtheta1
            df['x4'].values,  # invece di dtheta2
        ])
        
        uu_traj = np.array([df['u'].values])
        
        # Ensure correct shapes
        xx_traj = xx_traj.reshape(4, -1)  # 4 states x T timesteps
        uu_traj = uu_traj.reshape(1, -1)  # 1 input x T timesteps
        
        print("Shapes:")
        print("xx_traj shape:", xx_traj.shape)
        print("uu_traj shape:", uu_traj.shape)
        
        # Run task4
        results = task4(xx_traj, uu_traj)
        print("Task 4 completed successfully!")
        
        # Plot results
        plot_mpc_results(results)
        
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
        
    except FileNotFoundError:
        print(f"Error: Could not find trajectory file at {csv_path}")
    except KeyError as e:
        print(f"Error: Column not found in CSV file: {e}")
        print("Available columns:", df.columns.tolist() if 'df' in locals() else "CSV not loaded")
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
