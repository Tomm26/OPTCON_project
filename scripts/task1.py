import numpy as np
import matplotlib.pyplot as plt
from dynamics import FlexibleRoboticArm
from cost import Cost
from traj import TrajectoryGenerator, TrajectoryType
from newton import NewtonOptimizer
from utils import generate_initial_input_trajectory
from parameters import dt, ns, ni

if __name__ == "__main__":
    
    # Initialize system
    arm = FlexibleRoboticArm()
    cost = Cost()
    
    # Create reference trajectory
    T = 16.0  # Total time
    waypoint_times = np.array([0, T/2])
    x_waypoints = np.array([
        [0.0, 0.0, 0.0, 0.0],  # Initial state
        [np.pi, 0.0, 0.0, 0.0],  # Vertical position
    ])
    
    traj_gen = TrajectoryGenerator(x_waypoints, waypoint_times, T, dt)
    x_ref, t_array = traj_gen.generate_trajectory(TrajectoryType.EXPONENTIAL)

    # Define input waypoints (initial guess)
    u_ref = generate_initial_input_trajectory(arm, x_ref)

    # # plot u_ref
    # plt.plot(u_ref)
    # plt.show()

    # Save x0 and remove it from x_ref
    x0 = x_ref[0]
    x_ref = x_ref[1:]

    x_ref = x_ref.T
    u_ref = u_ref.T

    # Initialize optimizer
    optimizer = NewtonOptimizer(arm, cost)
    
    # Run optimization
    x_optimal, u_optimal, costs = optimizer.newton_optimize(x_ref, u_ref, 
                                            max_iters=10, 
                                            threshold_grad=1e-3,
                                            use_armijo=True,
                                            show_plots_armijo=False)
    
    # Plot results
    optimizer.plot_results(x_optimal, u_optimal, x_ref, u_ref)
    optimizer.plot_convergence(costs)