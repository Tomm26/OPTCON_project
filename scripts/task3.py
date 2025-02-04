import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from dynamics import FlexibleRoboticArm
from newton import NewtonOptimizer
from parameters import (
    m1, m2, l1, l2, r1, r2, I1, I2, g, f1, f2, dt, ns, ni
)
from animate import FlexibleRobotAnimator


def load_trajectory(csv_path):
    """
    Load the optimal trajectory from a CSV file and return the state and input arrays.
    
    Parameters:
        csv_path (str): Path to the CSV file containing the optimal trajectory.
        
    Returns:
        x_optimal (np.ndarray): Optimal state trajectory, shape (4, T).
        u_optimal (np.ndarray): Optimal input trajectory, shape (1, T).
    """
    df = pd.read_csv(csv_path)
    x_optimal = np.array([
        df['x1'].values,
        df['x2'].values,
        df['x3'].values,
        df['x4'].values,
    ]).reshape(4, -1)
    u_optimal = np.array([df['u'].values]).reshape(1, -1)
    return x_optimal, u_optimal


def simulate_system(arm, x_optimal, u_optimal, KK, disturbances):
    """
    Simulate the closed-loop system with feedback control while applying any disturbances.
    
    Parameters:
        arm: The robotic arm dynamics object.
        x_optimal (np.ndarray): Optimal state trajectory, shape (ns, T).
        u_optimal (np.ndarray): Optimal input trajectory, shape (ni, T).
        KK (np.ndarray): Gain matrices, shape (ni, ns, T).
        disturbances (dict): Dictionary with flags and parameters for disturbances.
    
    Returns:
        xx (np.ndarray): Simulated state trajectory, shape (ns, T).
        uu (np.ndarray): Simulated input trajectory, shape (ni, T).
    """
    TT = x_optimal.shape[1]
    ns = x_optimal.shape[0]
    ni = u_optimal.shape[0]

    uu = np.zeros((ni, TT))
    xx = np.zeros((ns, TT))

    # Set the initial state (optionally perturbed)
    perturbed_state = disturbances.get("perturbed_state", False)
    x0_perturb = disturbances.get("x0_perturb", np.zeros(ns))
    xx[:, 0] = x_optimal[:, 0] + x0_perturb * int(perturbed_state)

    # Disturbance parameters
    gaussian_noise = disturbances.get("gaussian_noise", False)
    mean_noise = disturbances.get("gaussian_mean", 0)
    std_noise = disturbances.get("gaussian_std", 0.003)

    # LQR
    for t in range(TT - 1):
        
        # Calculate the feedback control input 
        uu[:, t] = u_optimal[:, t] + KK[:, :, t] @ (xx[:, t] - x_optimal[:, t])

        # Simulate the discrete dynamics and add Gaussian noise if activated
        xx[:, t+1] = arm.discrete_dynamics(xx[:, t], uu[:, t])
        if gaussian_noise:
            noise = np.random.normal(mean_noise, std_noise, size=ns)
            xx[:, t+1] += noise

    return xx, uu


def plot_results(x_optimal, xx, u_optimal, uu):
    """
    Plot the simulation results by comparing the optimal and simulated trajectories
    for all 4 states and the control input.
    
    Parameters:
        x_optimal (np.ndarray): Optimal state trajectory (4 x T).
        xx (np.ndarray): Simulated state trajectory (4 x T).
        u_optimal (np.ndarray): Optimal input trajectory (1 x T).
        uu (np.ndarray): Simulated input trajectory (1 x T).
    """
    TT = x_optimal.shape[1]
    # Create a time vector in seconds
    time = np.arange(TT) * dt

    # Create a figure for the states (4 subplots)
    state_labels = ['theta1', 'theta2', 'omega1', 'omega2']
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for i, label in enumerate(state_labels):
        axs[i].plot(time, x_optimal[i], 'r--', label=f'Optimal {label}')
        axs[i].plot(time, xx[i], 'g-', label=f'Approx {label}')
        axs[i].set_ylabel(label)
        axs[i].legend(loc="best")
        axs[i].grid(True)
    axs[-1].set_xlabel("Time [s]")
    fig.suptitle("Comparison of State Trajectories", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # Create a figure for the control input
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 4))
    ax2.plot(time, u_optimal.flatten(), 'r--', label='Optimal Input')
    ax2.plot(time, uu.flatten(), 'g-', label='Approx Input')
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Control Input")
    ax2.set_title("Comparison of Control Input Trajectory")
    ax2.legend(loc="best")
    ax2.grid(True)
    fig2.tight_layout()
    plt.show()


def main():
    # LQR regulator parameters
    QQreg = np.diag([15.0, 15.0, 1.0, 1.0])
    RRreg = 0.001 * np.eye(1)
    QQTreg = np.diag([20.0, 20.0, 1.0, 1.0])

    # Initialize the system
    arm = FlexibleRoboticArm(m1, m2, l1, l2, r1, r2, I1, I2, g, f1, f2, dt, ns, ni, method='euler')
    optimizer = NewtonOptimizer(arm)

    # Load the optimal trajectory
    x_optimal, u_optimal = load_trajectory('traj/optimal_trajectories.csv')

    # Linearize the dynamics along the optimal trajectory
    AA_traj, BB_traj = arm.linearize_over_traj(x_optimal, u_optimal)
    TT = x_optimal.shape[1]

    # Solve the LQ problem to obtain the gain matrices
    KK, *_ = optimizer.affine_lqp(
        AA_traj, BB_traj, QQreg, RRreg,
        np.zeros((ni, ns, TT)), QQTreg, TT,
        np.zeros((ns)), np.zeros((ns, TT)),
        np.zeros((ni, TT)), np.zeros((ns))
    )

    # Define disturbance parameters (modify flags to enable/disable)
    disturbances = {
        "gaussian_noise": True,        # Set to True to add Gaussian noise
        "perturbed_state": True,      # Set to True to perturb the initial state
        "perturbed_params": True,     # Set to True to add uncertanties to robot param
        # Parameters
        "x0_perturb": np.array([0.02, 0.1, 0.1, 0.1]),
        "gaussian_mean": 0,
        "gaussian_std": 0.002,
    }

    # If parameter perturbation is desired, reinitialize the arm with new parameters
    if disturbances.get("perturbed_params"):
        arm = FlexibleRoboticArm(m1 * 1.03, m2 * 1.05, l1 * 1.02, l2 * 1.04, r1, r2, I1, I2, g, f1, f2, dt, ns, ni)

    # Simulate the closed-loop system
    xx, uu = simulate_system(arm, x_optimal, u_optimal, KK, disturbances)

    # Plot the improved results
    plot_results(x_optimal, xx, u_optimal, uu)

    # Animate the simulation
    animator = FlexibleRobotAnimator(xx.T, dt=dt)
    animator.animate()


if __name__ == "__main__":
    main()
