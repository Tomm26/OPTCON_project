import numpy as np
import matplotlib.pyplot as plt
from dynamics import FlexibleRoboticArm
from cost import Cost
from traj import TrajectoryGenerator, TrajectoryType
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
    
    plot = True

    # Initialize system
    arm = FlexibleRoboticArm(m1, m2, l1, l2, r1, r2, I1, I2, g, f1, f2, dt, ns, ni)

    cost = Cost(QQ, RR, QQT)
    
    # Create reference trajectory
    T = 13.0  # Total time
    waypoint_times = np.array([0, 2.5, 4.5, 7])
    x_waypoints = np.array([
        [0, 0, 0, 0], 
        [np.deg2rad(30), -np.deg2rad(30), 0, 0],
        [np.deg2rad(-45), -np.deg2rad(-45), 0, 0],
        [np.deg2rad(60), -np.deg2rad(60), 0, 0]
        ])
    
    print('Generating reference trajectory...')
    traj_gen = TrajectoryGenerator(x_waypoints, waypoint_times, T, dt)
    x_ref, t_array = traj_gen.generate_trajectory(TrajectoryType.STEP)
    u_ref = generate_initial_input_trajectory_2(arm, x_ref)
    print('Reference trajectory generated.')

    if plot:
        traj_gen.plot_trajectory(TrajectoryType.STEP)
        plt.plot(u_ref)
        plt.xlabel('Time (s)')
        plt.ylabel('Control Input (u)')
        plt.title('Reference Control Input')
        plt.grid()
        plt.show()

    # Save x0 and remove it from x_ref
    x0 = x_ref[0]
    x_ref = x_ref[1:]

    x_ref = x_ref.T
    u_ref = u_ref.T

    # Initialize optimizer
    optimizer = NewtonOptimizer(arm, cost, dt,fixed_stepsize, stepsize_0, cc, beta, armijo_maxiters)
    
    print('Optimizing...')

    # Run optimization
    x_optimal, u_optimal, costs = optimizer.newton_optimize(x_ref, u_ref, 
                                            max_iters=20, 
                                            threshold_grad=1e-3,
                                            use_armijo=True,
                                            show_plots_armijo=False)
    
    AA_traj, BB_traj = arm.linearize_over_traj(x_optimal[:, :, -1], u_optimal[:, :, -1])

    TT = x_optimal[:, :, -1].shape[-1]
    KK, *_ = optimizer.solve_lqp(AA_traj, BB_traj, QQreg, RRreg, np.zeros((ni,ns, TT)), QQTreg, TT, 
                                   np.zeros((ns)), np.zeros((ns, TT)), np.zeros((ni, TT)), np.zeros((ns)))
    
    # Apply the feedback controller on the non linear dynamics
    uu = np.zeros((ni, TT))
    xx = np.zeros((ns, TT))
    xx[:, 0] = x0 + np.array([0.2, 0.1, 0.1,0.1]) #x0 perturbed (da migliorare)

    for t in range(TT-1):
        uu[:, t] = u_optimal[:, t, -1] +  KK[:, :, t] @ (xx[:, t] - x_optimal[:, t, -1])
        xx[:, t+1] = arm.discrete_dynamics(xx[:, t], uu[:, t])

    animator = FlexibleRobotAnimator(xx.T, dt=dt)
    animator.animate()
    
    """# Plot results
    optimizer.plot_results(x_optimal, u_optimal, x_ref, u_ref)
    optimizer.plot_convergence(costs)"""




