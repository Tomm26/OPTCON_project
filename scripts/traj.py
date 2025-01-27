import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from parameters import ns, ni

class TrajectoryInterpolation:
    """Class for interpolating state and input trajectories."""
    
    def __init__(self, x_waypoints, u_waypoints, waypoint_times, dt, interpolation_type='cubic'):
        """
        Initialize the trajectory interpolation.
        
        Args:
            x_waypoints (np.ndarray): State waypoints with shape (n_waypoints, ns)
            u_waypoints (np.ndarray): Input waypoints with shape (n_waypoints, ni)
            waypoint_times (np.ndarray): Time points for waypoints with shape (n_waypoints,)
            dt (float): Time step for interpolation
            interpolation_type (str): Type of interpolation ('step', 'cubic', or 'trapezoidal')
        """
        self.x_waypoints = np.array(x_waypoints)
        self.u_waypoints = np.array(u_waypoints)
        self.waypoint_times = np.array(waypoint_times)
        self.dt = dt
        self.interpolation_type = interpolation_type
        
        # Validate inputs
        self._validate_inputs()
        
        # Generate time vector
        self.t = np.arange(self.waypoint_times[0], 
                          self.waypoint_times[-1] + self.dt, 
                          self.dt)
        
        # Perform interpolation
        self.x_trajectory, self.u_trajectory = self._interpolate()
        
    def _validate_inputs(self):
        """Validate input dimensions and parameters."""
        # Check shapes
        assert self.x_waypoints.shape[1] == ns, f"State dimension must be {ns}"
        assert self.u_waypoints.shape[1] == ni, f"Input dimension must be {ni}"
        assert self.x_waypoints.shape[0] == self.waypoint_times.shape[0], \
            "Number of state waypoints must match number of time points"
        assert self.u_waypoints.shape[0] == self.waypoint_times.shape[0], \
            "Number of input waypoints must match number of time points"
            
        # Check time vector
        assert np.all(np.diff(self.waypoint_times) > 0), \
            "Time points must be strictly increasing"
            
        # Check interpolation type
        assert self.interpolation_type in ['step', 'cubic', 'trapezoidal'], \
            "Interpolation type must be 'step', 'cubic', or 'trapezoidal'"
            
        # Check dt
        assert self.dt > 0, "Time step must be positive"
        assert self.dt < self.waypoint_times[-1] - self.waypoint_times[0], \
            "Time step must be smaller than total time span"
            
    def _step_interpolation(self):
        """Perform step interpolation."""
        x_trajectory = np.zeros((len(self.t), ns))
        u_trajectory = np.zeros((len(self.t), ni))
        
        # For each time point, find the last waypoint
        for i, t_current in enumerate(self.t):
            idx = np.searchsorted(self.waypoint_times, t_current, side='right') - 1
            idx = max(0, idx)  # Ensure non-negative index
            
            x_trajectory[i] = self.x_waypoints[idx]
            u_trajectory[i] = self.u_waypoints[idx]
            
        return x_trajectory, u_trajectory
    
    def _cubic_interpolation(self):
        """Perform cubic spline interpolation for states and linear for inputs."""
        # Cubic interpolation for states
        x_interpolators = [CubicSpline(self.waypoint_times, self.x_waypoints[:, i]) 
                         for i in range(ns)]
        x_trajectory = np.column_stack([x_interp(self.t) for x_interp in x_interpolators])
        
        # Linear interpolation for inputs to avoid oscillations
        u_trajectory = np.zeros((len(self.t), ni))
        for i, t_current in enumerate(self.t):
            idx = np.searchsorted(self.waypoint_times, t_current, side='right') - 1
            idx = min(max(0, idx), len(self.waypoint_times) - 2)
            
            t0 = self.waypoint_times[idx]
            t1 = self.waypoint_times[idx + 1]
            alpha = (t_current - t0) / (t1 - t0)
            
            u_trajectory[i] = (1 - alpha) * self.u_waypoints[idx] + \
                            alpha * self.u_waypoints[idx + 1]
        
        return x_trajectory, u_trajectory

    def _trapezoidal_interpolation(self):
        """Perform trapezoidal interpolation."""
        x_trajectory = np.zeros((len(self.t), ns))
        u_trajectory = np.zeros((len(self.t), ni))
        
        for i, t_current in enumerate(self.t):
            # Find the surrounding waypoints with proper boundary handling
            idx = np.searchsorted(self.waypoint_times, t_current, side='right') - 1
            idx = min(max(0, idx), len(self.waypoint_times) - 2)
            
            # Get time interval
            t0 = self.waypoint_times[idx]
            t1 = self.waypoint_times[idx + 1]
            alpha = (t_current - t0) / (t1 - t0)
            
            # Linear interpolation for both states and inputs
            x_trajectory[i] = (1 - alpha) * self.x_waypoints[idx] + \
                            alpha * self.x_waypoints[idx + 1]
            u_trajectory[i] = (1 - alpha) * self.u_waypoints[idx] + \
                            alpha * self.u_waypoints[idx + 1]
            
        return x_trajectory, u_trajectory
    
    def _interpolate(self):
        """Perform interpolation based on specified type."""
        if self.interpolation_type == 'step':
            return self._step_interpolation()
        elif self.interpolation_type == 'trapezoidal':
            return self._trapezoidal_interpolation()
        elif self.interpolation_type == 'cubic':
            return self._cubic_interpolation()
        else:
            raise ValueError("Invalid interpolation type")
    
    def get_trajectories(self):
        """
        Return interpolated trajectories and time vector.
        
        Returns:
            tuple: (time vector, state trajectory, input trajectory)
        """
        return self.t, self.x_trajectory, self.u_trajectory
        
    def plot_trajectories(self, figsize=(15, 10)):
        """
        Plot the interpolated trajectories as points.
        
        Args:
            figsize (tuple): Figure size (width, height)
            
        Returns:
            tuple: (figure, axes) for further customization if needed
        """
        fig, axes = plt.subplots(ns + ni, 1, figsize=figsize)
        if ns + ni == 1:
            axes = [axes]
        fig.suptitle(f'Trajectory Points ({self.interpolation_type})')
        
        # Plot states
        for i in range(ns):
            ax = axes[i]
            # Plot interpolated trajectory points
            ax.plot(self.t, self.x_trajectory[:, i], '.', markersize=8,
                   label='Interpolated')
            
            ax.set_ylabel(f'State {i+1}')
            ax.grid(True)
            ax.legend()
        
        # Plot inputs
        for i in range(ni):
            ax = axes[ns + i]
            # Plot interpolated trajectory points
            ax.plot(self.t, self.u_trajectory[:, i], '.', markersize=8,
                   label='Interpolated')
            
            ax.set_ylabel(f'Input {i+1}')
            ax.grid(True)
            ax.legend()
        
        # Set xlabel for bottom subplot only
        axes[-1].set_xlabel('Time')
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        return fig, axes


if __name__ == "__main__":
    # Test the implementation
    
    # Create sample waypoints
    t_waypoints = np.array([0, 1, 2])
    x_waypoints = np.array([
        [0, 0, 0, 0],  # Initial state
        [1, 0.5, 0, 0],  # Intermediate state
        [2, 1, 0, 0]  # Final state
    ])
    u_waypoints = np.array([
        [0],  # Initial input
        [1],  # Intermediate input
        [0]   # Final input
    ])
    
    # Test all interpolation types
    interpolation_types = ['step', 'cubic', 'trapezoidal']
    
    for interp_type in interpolation_types:
        interpolator = TrajectoryInterpolation(
            x_waypoints, u_waypoints, t_waypoints, 
            dt=0.1, interpolation_type=interp_type
        )
        
        # Plot trajectories
        interpolator.plot_trajectories()
        # plt.savefig(f'{interp_type}_interpolation.png')
        
        # Get trajectories for printing
        t, x, u = interpolator.get_trajectories()
        
        # Print first few points
        print(f"\n{interp_type.capitalize()} Interpolation:")
        print("Time points:", t[:5], "...")
        print("State trajectory (first 5 points):", x[:5])
        print("Input trajectory (first 5 points):", u[:5])
    
    plt.show()