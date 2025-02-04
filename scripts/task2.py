import numpy as np
import pandas as pd
from dynamics import FlexibleRoboticArm
from cost import Cost
from traj import TrajectoryGenerator, TrajectoryType
from newton import NewtonOptimizer
from parameters import m1, m2, l1, l2, r1, r2, I1, I2, g, f1, f2, dt, ns, ni, fixed_stepsize, stepsize_0, cc, beta, armijo_maxiters
from parameters import Q2, R2, QT2
from animate import FlexibleRobotAnimator

if __name__ == "__main__":

    T = 10.0  # Total time
    plot = False
    plot_armijo = False
    save_traj = True
    
    # Initialize system
    arm = FlexibleRoboticArm(m1, m2, l1, l2, r1, r2, I1, I2, g, f1, f2, dt, ns, ni, method='euler')
    cost = Cost(Q2, R2, QT2)
    
    # Create reference trajectory
    x_waypoints = np.array([
        [0, 0, 0, 0], 
        [np.deg2rad(45), -np.deg2rad(45), 0, 0],
        [-np.deg2rad(60), np.deg2rad(60), 0, 0],
        [np.deg2rad(80), -np.deg2rad(80), 0, 0]
        ])
    waypoint_times = np.linspace(0, T-0.7, len(x_waypoints))

    
    print('\nGenerating reference trajectory...\n')
    traj_gen = TrajectoryGenerator(x_waypoints, waypoint_times, T, dt)
    x_ref, t_array = traj_gen.generate_trajectory(TrajectoryType.CUBIC)
    print('Reference trajectory generated.\n')

    if plot:
        traj_gen.plot_trajectory(TrajectoryType.CUBIC)

    x_ref = x_ref.T
    u_ref = np.zeros((1, x_ref.shape[1]))

    # Initialize optimizer
    optimizer = NewtonOptimizer(arm, cost, dt, fixed_stepsize, stepsize_0, cc, beta, armijo_maxiters)
    
    print('Optimizing...\n')
    x_optimal, u_optimal, costs = optimizer.newton_optimize(x_ref, u_ref, 
                                            max_iters=25, 
                                            threshold_grad=1e-3,
                                            use_armijo=True,
                                            show_plots_armijo=plot_armijo)

    x_opt_final = x_optimal[:, :, -1].T
    u_opt_final = u_optimal[:, :, -1].T 

    if save_traj :
        df_x = pd.DataFrame(x_opt_final, columns=["x1", "x2", "x3", "x4"])
        df_u = pd.DataFrame(u_opt_final, columns=["u"])

        # Combine the DataFrames
        df_combined = pd.concat([df_x, df_u], axis=1)

        # Save the combined DataFrame to a CSV file
        df_combined.to_csv("traj/optimal_trajectories.csv", index=False)

        print("Optimal trajectories saved to 'optimal_trajectories.csv'.")

    # Plot results
    optimizer.plot_results(x_optimal, u_optimal, x_ref, u_ref)
    optimizer.plot_convergence(costs)

    animator = FlexibleRobotAnimator(x_opt_final, dt=dt)
    animator.animate()