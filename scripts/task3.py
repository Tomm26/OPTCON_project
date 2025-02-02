import numpy as np
import matplotlib.pyplot as plt
from dynamics import FlexibleRoboticArm
from cost import Cost
import pandas as pd
from newton import NewtonOptimizer
from utils import generate_initial_input_trajectory_2
from parameters import m1, m2, l1, l2, r1, r2, I1, I2, g, f1, f2, dt, ns, ni, fixed_stepsize, stepsize_0, cc, beta, armijo_maxiters
from animate import FlexibleRobotAnimator


if __name__ == "__main__":


    QQ = np.diag([15.0, 15.0, 1.0, 1.0])
    RR = 0.001*np.eye(1)
    QQT = np.diag([20.0, 20.0, 1.0, 1.0])
    
    #parameters for LQR regolator
    QQreg = np.diag([15.0, 15.0, 1.0, 1.0])
    RRreg = 0.001*np.eye(1)
    QQTreg = np.diag([20.0, 20.0, 1.0, 1.0])
    
    # Initialize system
    arm = FlexibleRoboticArm(m1, m2, l1, l2, r1, r2, I1, I2, g, f1, f2, dt, ns, ni)
    cost = Cost(QQ, RR, QQT)
    optimizer = NewtonOptimizer(arm, cost, dt, fixed_stepsize, stepsize_0, cc, beta, armijo_maxiters)

    # Load optimized trajectory
    # Load trajectory data from CSV
    df = pd.read_csv('OPTCON_project_giorgio/OPTCON_project/traj/optimal_trajectories.csv')
    print("Available columns in CSV:", df.columns)
    
    # Extract state and input trajectories
    x_optimal = np.array([
        df['x1'].values,  # θ1: angle of first link
        df['x2'].values,  # θ2: angle of second link
        df['x3'].values,  # θ̇1: angular velocity of first link
        df['x4'].values,  # θ̇2: angular velocity of second link
    ])
    
    u_optimal = np.array([df['u'].values])  # Control input
    
    # Reshape trajectories to correct dimensions
    x_optimal = x_optimal.reshape(4, -1)  # 4 states x T timesteps
    u_optimal = u_optimal.reshape(1, -1)  # 1 input x T timesteps
    x0 = x_optimal[:, 0]


    # Linearize the dynamics over the optimal trajectory
    AA_traj, BB_traj = arm.linearize_over_traj(x_optimal, u_optimal)

    TT = x_optimal.shape[-1]

    # Solve the LQ problem to obtain the K gain matrices
    KK, *_ = optimizer.solve_lqp(AA_traj, BB_traj, QQreg, RRreg, np.zeros((ni,ns, TT)), QQTreg, TT, 
                                   np.zeros((ns)), np.zeros((ns, TT)), np.zeros((ni, TT)), np.zeros((ns)))
    
    # Apply the feedback controller on the non linear dynamics
    uu = np.zeros((ni, TT))
    xx = np.zeros((ns, TT))
    
    
    # Define different kinds of disturbances
    impulse_torque = False
    gaussian_noise = False
    perturbed_state = False
    perturbed_params = False

    # Impulse torque parameters
    uu_noise = 0     
    delta = 1/dt * 1     # Duration (in s)
    starting_time = TT/3 # Starting time
    intensity = 1.3

    # Gaussian noise parameters
    mean = 0
    std = 0.003

    # Perturbed initial state x0
    xx[:, 0] = x0 + np.array([0.02, 0.1, 0.1, 0.1]) * int(perturbed_state)

    # Perturbed parameters
    if perturbed_params:
        arm = FlexibleRoboticArm(m1*1.10, m2*1.20, l1*1.05, l2*1.05, r1, r2, I1, I2, g, f1, f2, dt, ns, ni)

    for t in range(TT-1):

        if starting_time - delta <= t <= starting_time + delta and impulse_torque:
            print(t)
            # add impulse torque
            uu_noise = -uu[:, t-1] * intensity
        else:
            uu_noise = 0

        uu[:, t] = u_optimal[:, t] +  KK[:, :, t] @ (xx[:, t] - x_optimal[:, t]) + uu_noise
        xx[:, t+1] = arm.discrete_dynamics(xx[:, t], uu[:, t]) + np.random.normal(mean, std, size=ns)*int(gaussian_noise)

    
    # Plot results
    
    plt.plot(range(TT), x_optimal[0], 'r', label = 'optimal theta1')
    plt.plot(range(TT), xx[0], 'g', label = 'approx theta1')
    plt.show()

    plt.plot(range(TT), x_optimal[1], 'r', label = 'optimal theta1')
    plt.plot(range(TT), xx[1], 'g', label = 'approx theta1')
    plt.show()

    plt.plot(range(TT), u_optimal.T, 'r', label = 'optimal theta1')
    plt.plot(range(TT), uu.T, 'g', label = 'approx theta1')
    plt.show()

    animator = FlexibleRobotAnimator(xx.T, dt=dt)
    animator.animate()




