"""
Task 4: MPC Control Implementation for Flexible Robotic Arm System
This implementation focuses on Model Predictive Control (MPC) for trajectory tracking.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from cvxopt import matrix, solvers
from enum import Enum
import matplotlib.pyplot as plt

from parameters import (
    m1, m2, l1, l2, r1, r2, I1, I2, g, f1, f2, dt, ns, ni
)
from dynamics import FlexibleRoboticArm  # System dynamics

# Configure CVXOPT solver parameters
solvers.options['show_progress'] = False
solvers.options['abstol'] = 1e-10
solvers.options['reltol'] = 1e-10
solvers.options['feastol'] = 1e-10


class RegulatorType(Enum):
    MPC = "MPC"


class GetData:
    """
    Class to store and manage trajectory tracking control data.
    """
    def __init__(self, xx_ref, uu_ref, tracking_results, initial_conditions,
                 regulator_type, start_time, end_time):
        self.xx_ref = xx_ref
        self.uu_ref = uu_ref
        self.tracking_results = tracking_results
        self.initial_conditions = initial_conditions
        self.regulator_type = regulator_type
        self.start_time = start_time
        self.end_time = end_time
        self.computation_time = (end_time - start_time).total_seconds()

    def get_tracking_error(self, run_index):
        """
        Calculate the L2-norm tracking error between the actual and reference state trajectories.
        """
        xx_track = self.tracking_results[run_index][0]
        T = min(xx_track.shape[1], self.xx_ref.shape[1])
        return np.linalg.norm(xx_track[:, :T] - self.xx_ref[:, :T], axis=0)

    def get_control_effort(self, run_index):
        """
        Calculate the L2-norm control effort error between the actual and reference inputs.
        """
        uu_track = self.tracking_results[run_index][1]
        T = min(uu_track.shape[1], self.uu_ref.shape[1])
        return np.linalg.norm(uu_track[:, :T] - self.uu_ref[:, :T], axis=0)


def extend_trajectory(traj, N):
    """
    Extend trajectory by N steps by repeating the last value.
    """
    last_value = traj[:, -1:]
    extension = np.tile(last_value, (1, N))
    return np.hstack((traj, extension))


def generate_initial_condition(x_ref, disturbance_level):
    """
    Generate a perturbed initial condition.
    """
    return x_ref + disturbance_level * np.random.randn(*x_ref.shape)


def formulate_cost_matrices(AA, BB, Q, R, N, x0):
    """
    Formulate the quadratic cost matrices for the QP problem.
    """
    ns = AA.shape[0]
    ni = BB.shape[1]
    P = np.zeros((N * ni, N * ni))
    for i in range(N):
        P[i * ni:(i + 1) * ni, i * ni:(i + 1) * ni] = R

    q = np.zeros((N * ni, 1))
    x_pred = x0
    for i in range(N):
        q[i * ni:(i + 1) * ni] = 2 * BB[:, :, i].T @ Q @ x_pred
        # No control deviation is assumed in the prediction (zero input deviation)
        x_pred = AA[:, :, i] @ x_pred
    return P, q


def formulate_constraint_matrices(N, u_min, u_max):
    """
    Generate constraint matrices for input bounds in the QP problem.
    """
    ni = 1  # single input system
    G = np.zeros((2 * N * ni, N * ni))
    h = np.zeros(2 * N * ni)
    for i in range(N):
        # Upper bound: u <= u_max
        G[i * ni:(i + 1) * ni, i * ni:(i + 1) * ni] = np.eye(ni)
        h[i * ni:(i + 1) * ni] = u_max
        # Lower bound: -u <= -u_min  (i.e. u >= u_min)
        G[N * ni + i * ni:N * ni + (i + 1) * ni, i * ni:(i + 1) * ni] = -np.eye(ni)
        h[N * ni + i * ni:N * ni + (i + 1) * ni] = -u_min
    return G, h


def solve_mpc_problem(x0, xx_ref, uu_ref, AA, BB, Q, R, u_min, u_max, N):
    """
    Solve the MPC QP problem using CVXOPT.
    """
    ns = x0.shape[0]
    ni = BB.shape[1]
    P, q = formulate_cost_matrices(AA, BB, Q, R, N, x0)
    G, h = formulate_constraint_matrices(N, u_min, u_max)

    sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))
    if sol['status'] != 'optimal':
        raise RuntimeError("QP solver did not converge to an optimal solution")
    return np.array(sol['x']).reshape(-1, ni)


def run_mpc_controller(system, x0, xx_ref, uu_ref, AA, BB, Q, R, N):
    """
    Run the MPC controller over the trajectory.
    """
    T = xx_ref.shape[1] - N  # Adjust time steps for prediction horizon
    ns = system.ns
    ni = system.ni

    xx = np.zeros((ns, T))
    uu = np.zeros((ni, T - 1))
    xx[:, 0] = x0

    u_max = 10.0
    u_min = -10.0

    for t in range(T - 1):
        x_error = xx[:, t] - xx_ref[:, t]
        u_mpc = solve_mpc_problem(
            x_error,
            xx_ref[:, t:t + N + 1],
            uu_ref[:, t:t + N],
            AA[:, :, t:t + N],
            BB[:, :, t:t + N],
            Q,
            R,
            u_min,
            u_max,
            N
        )
        # Apply the first control action (receding horizon)
        uu[:, t] = u_mpc[0] + uu_ref[:, t]
        xx[:, t + 1] = system.discrete_dynamics(xx[:, t], uu[:, t])

    return xx, uu


def task4(arm, xx_traj, uu_traj, N, Q, R):
    """
    Run MPC tracking control after linearization around the trajectory.
    """
    T = xx_traj.shape[1]
    xx_ext = extend_trajectory(xx_traj, N)
    uu_ext = extend_trajectory(uu_traj, N)

    # Precompute the linearization matrices along the extended trajectory
    AA = np.zeros((ns, ns, T + N))
    BB = np.zeros((ns, ni, T + N))
    for t in range(T + N):
        x_t = xx_ext[:, t]
        u_t = uu_ext[:, t]
        A_t, B_t = arm.get_gradients(x_t, u_t)
        AA[:, :, t] = np.array(A_t, dtype=float)
        BB[:, :, t] = np.array(B_t, dtype=float)

    # Run MPC tracking for different disturbance levels
    disturbance_levels = [0.0, 0.1, 0.2]
    start_time = datetime.now()
    initial_conditions = []
    tracking_results = []

    for level in disturbance_levels:
        x0 = generate_initial_condition(xx_traj[:, 0], level)
        initial_conditions.append(x0)
        xx_cl, uu_cl = run_mpc_controller(
            arm,
            x0,
            xx_ext,
            uu_ext,
            AA,
            BB,
            Q,
            R,
            N
        )
        # Exclude the extended horizon when storing results
        tracking_results.append((xx_cl[:, :-N], uu_cl[:, :-N]))

    results = GetData(
        xx_traj,
        uu_traj,
        tracking_results,
        initial_conditions,
        RegulatorType.MPC,
        start_time,
        datetime.now()
    )
    return results


def plot_mpc_results(results):
    """
    Plot the state trajectories, control inputs, and performance metrics.
    """
    T_ref = results.xx_ref.shape[1]
    T_track = min(track[0].shape[1] for track in results.tracking_results)
    T = min(T_ref, T_track)
    time = np.arange(0, T * dt, dt)

    # Plot state trajectories and control inputs
    fig, axs = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
    fig.suptitle('MPC Tracking Results', fontsize=16)
    state_labels = [r'$\theta_1$', r'$\theta_2$', r'$\dot{\theta}_1$', r'$\dot{\theta}_2$']
    disturbance_labels = ['0%', '10%', '20%']

    for i in range(4):
        ax = axs[i]
        ax.plot(time, results.xx_ref[i, :T], 'k--', label='Reference', alpha=0.7)
        for j, (xx, _) in enumerate(results.tracking_results):
            ax.plot(time, xx[i, :T], label=f'Disturbance {disturbance_labels[j]}', alpha=0.8)
        ax.set_ylabel(state_labels[i])
        ax.grid(True)
        ax.legend()

    ax = axs[4]
    ax.plot(time, results.uu_ref[0, :T], 'k--', label='Reference', alpha=0.7)
    for j, (_, uu) in enumerate(results.tracking_results):
        ax.plot(time[:-1], uu[0, :(T - 1)], label=f'Disturbance {disturbance_labels[j]}', alpha=0.8)
    ax.set_ylabel('Control Input u')
    ax.set_xlabel('Time [s]')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    # Plot performance metrics: tracking error and control effort
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig2.suptitle('MPC Performance Metrics', fontsize=16)

    ax1.set_title('State Tracking Error')
    for j in range(len(results.tracking_results)):
        error = results.get_tracking_error(j)
        ax1.plot(time, error[:T], label=f'Disturbance {disturbance_labels[j]}')
    ax1.set_ylabel('Error Norm')
    ax1.grid(True)
    ax1.legend()

    ax2.set_title('Control Effort')
    for j in range(len(results.tracking_results)):
        effort = results.get_control_effort(j)
        ax2.plot(time[:-1], effort[:(T - 1)], label=f'Disturbance {disturbance_labels[j]}')
    ax2.set_ylabel('Input Difference Norm')
    ax2.set_xlabel('Time [s]')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    try:
        fig.savefig('plots/mpc_tracking.png')
        fig2.savefig('plots/mpc_performance.png')
        print("Plots saved")
    except Exception as e:
        print(f"Error saving plots: {e}")
    plt.show()



if __name__ == "__main__":
    try:

        N = 15  # MPC prediction horizon
        Q = np.diag([16.0, 16.0, 6.0, 6.0])
        R = 0.0001 * np.eye(ni)

        # Initialize the flexible robotic arm system
        arm = FlexibleRoboticArm(m1, m2, l1, l2, r1, r2, I1, I2, g, f1, f2, dt, ns, ni, method='euler')

        df = pd.read_csv('traj/optimal_trajectories.csv')
        xx_traj = np.array([
            df['x1'].values,  # θ1
            df['x2'].values,  # θ2
            df['x3'].values,  # θ̇1
            df['x4'].values   # θ̇2
        ])
        uu_traj = np.array([df['u'].values])  # Control input

        # Ensure proper shaping: states (4 x T), input (1 x T)
        xx_traj = xx_traj.reshape(4, -1)
        uu_traj = uu_traj.reshape(1, -1)

        results = task4(arm, xx_traj, uu_traj, N, Q, R)
        print("Task 4 completed successfully!")
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
