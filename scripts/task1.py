import numpy as np
import matplotlib.pyplot as plt
from dynamics import FlexibleRoboticArm
from cost import Cost
from traj import TrajectoryInterpolation
from newton import NewtonOptimizer
from utils import generate_initial_input_trajectory
from parameters import dt, ns, ni

def main():
    # Initialize the robotic arm system
    arm = FlexibleRoboticArm()
    
    # Define time horizon and waypoints for swing-up
    T = 5.0  # Total time in seconds
    waypoint_times = np.array([0, T/2, T])
    
    # Define state waypoints for swing-up motion
    # State: [theta1, theta2, dtheta1, dtheta2]
    x_waypoints = np.array([
        [0.0, 0.0, 0.0, 0.0],  # Initial state
        [np.pi, 0.0, 0.0, 0.0],  # Vertical position
        [np.pi, 0.0, 0.0, 0.0]  # Vertical position
    ])
    # Define input waypoints (initial guess)
    u_waypoints = generate_initial_input_trajectory(arm, x_waypoints)
    
    # Create trajectory interpolation
    traj_interp = TrajectoryInterpolation(
        x_waypoints, u_waypoints, waypoint_times, 
        dt=dt, interpolation_type='step'
    )
    
    # Get interpolated trajectories
    t, x_ref, u_ref = traj_interp.get_trajectories()

    traj_interp.plot_trajectories()
    plt.show()
    
    # Initialize optimizer
    optimizer = NewtonOptimizer(arm)
    
    # Initial state (hanging down)
    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    Q = np.diag([12.0, 12.0, 12.0, 12.0])
    R = np.diag([0.1])
    
    # Solve trajectory optimization
    print("Optimizing trajectory...")
    x_opt, u_opt, info = optimizer.solve_trajectory(
        x_ref, u_ref, x0,
        Q, R,
        max_iters=100,
        tol=1e-6,
        do_plot=True
    )
    
    print(f"Optimization completed in {info['iterations']} iterations")
    print(f"Final gradient norm: {info['grad_norms'][-1]:.2e}")
    print(f"Final cost: {info['costs'][-1]:.2e}")
    

if __name__ == "__main__":
    main()