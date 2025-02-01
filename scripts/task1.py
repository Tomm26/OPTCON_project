import numpy as np
import matplotlib.pyplot as plt
from dynamics import FlexibleRoboticArm
from cost import Cost
from traj import TrajectoryGenerator, TrajectoryType
from newton import NewtonOptimizer
from parameters import m1, m2, l1, l2, r1, r2, I1, I2, g, f1, f2, dt, ns, ni, fixed_stepsize, stepsize_0, cc, beta, armijo_maxiters
from animate import FlexibleRobotAnimator

if __name__ == "__main__":

    QQ = np.diag([10.0, 10.0, 0.1, 0.1])
    RR = 0.001*np.eye(1)
    QQT = np.diag([20.0, 20.0, 3.0, 3.0])

    plot = False
    plot_armijo = False
    
    # Initialize system
    arm = FlexibleRoboticArm(m1, m2, l1, l2, r1, r2, I1, I2, g, f1, f2, dt, ns, ni, method='rk')
    cost = Cost(QQ, RR, QQT)
    
    # Create reference trajectory
    T = 10.0  # Total time
    waypoint_times = np.array([0, T/2])
    x_waypoints = np.array([[0, 0, 0, 0], [np.pi, 0, 0, 0]])
    
    print('Generating reference trajectory...')
    traj_gen = TrajectoryGenerator(x_waypoints, waypoint_times, T, dt)
    x_ref, t_array = traj_gen.generate_trajectory(TrajectoryType.STEP)
    print('Reference trajectory generated.')

    if plot:
        traj_gen.plot_trajectory(TrajectoryType.STEP)

    # Save x0 and remove it from x_ref
    x0 = x_ref[0]
    x_ref = x_ref[1:]

    x_ref = x_ref.T
    u_ref = np.zeros((1, x_ref.shape[1]))

    # Initialize optimizer
    optimizer = NewtonOptimizer(arm, cost, dt, fixed_stepsize, stepsize_0, cc, beta, armijo_maxiters)
    
    print('Optimizing...')
    # Run optimization
    x_optimal, u_optimal, costs = optimizer.newton_optimize(x_ref, u_ref, 
                                            max_iters=14, 
                                            threshold_grad=1e-3,
                                            use_armijo=True,
                                            show_plots_armijo=plot_armijo)
    
    # Plot results
    optimizer.plot_results(x_optimal, u_optimal, x_ref, u_ref)
    optimizer.plot_convergence(costs)

    animator = FlexibleRobotAnimator(x_optimal[:, :, -1].T, dt=dt)
    animator.animate()