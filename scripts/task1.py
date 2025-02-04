import numpy as np
from dynamics import FlexibleRoboticArm
from cost import Cost
from traj import TrajectoryGenerator, TrajectoryType
from newton import NewtonOptimizer
from parameters import m1, m2, l1, l2, r1, r2, I1, I2, g, f1, f2, dt, ns, ni, fixed_stepsize, stepsize_0, cc, beta, armijo_maxiters
from parameters import Q1, R1, QT1
from animate import FlexibleRobotAnimator

if __name__ == "__main__":

    T = 3.0  # Total time
    plot = False
    plot_armijo = False
    
    # Initialize system
    arm = FlexibleRoboticArm(m1, m2, l1, l2, r1, r2, I1, I2, g, f1, f2, dt, ns, ni, method='rk')
    cost = Cost(Q1, R1, QT1)
    
    # Create reference trajectory
    waypoint_times = np.array([0, T/2])
    x_waypoints = np.array([[0, 0, 0, 0], [np.pi, 0, 0, 0]])
    
    print('\nGenerating reference trajectory...\n')
    traj_gen = TrajectoryGenerator(x_waypoints, waypoint_times, T, dt)
    x_ref, t_array = traj_gen.generate_trajectory(TrajectoryType.STEP)
    print('Reference trajectory generated.\n')

    if plot:
        traj_gen.plot_trajectory(TrajectoryType.STEP)

    x_ref = x_ref.T
    u_ref = np.zeros((ni, x_ref.shape[1]))

    # Initialize optimizer
    optimizer = NewtonOptimizer(arm, cost, dt, fixed_stepsize, stepsize_0, cc, beta, armijo_maxiters)
    
    print('Optimizing...\n')
    # Run optimization
    x_optimal, u_optimal, costs = optimizer.newton_optimize(x_ref, u_ref, 
                                            max_iters=25, 
                                            threshold_grad=1e-3,
                                            use_armijo=True,
                                            show_plots_armijo=plot_armijo)
    
    # Plot results
    optimizer.plot_results(x_optimal, u_optimal, x_ref, u_ref)
    optimizer.plot_convergence(costs)

    animator = FlexibleRobotAnimator(x_optimal[:, :, -1].T, dt=dt)
    animator.animate()