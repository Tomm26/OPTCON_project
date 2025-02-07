import numpy as np
from scipy.interpolate import CubicSpline
from enum import Enum
import matplotlib.pyplot as plt

class TrajectoryType(Enum):
    EXPONENTIAL  = "exponential"
    CUBIC        = "cubic"
    STEP         = "step"

class TrajectoryGenerator:
    """
    Class for generating and plotting multi-dimensional trajectories.
    
    It is assumed that the input xPoints has shape (p, n), i.e.,
    p points (rows) and n dimensions (columns).
    In the constructor, xPoints is transposed to (n, p).

    tPoints must have shape (p,):
       - times associated with the p points
    T  : total time
    dt : sampling time
    
    Available interpolation methods:
        - exponential
        - cubic
        - step
    """

    def __init__(self, xPoints, tPoints, T, dt):
        # Convert to a numpy array and TRANSPOSE xPoints 
        # (now the shape becomes (n, p) internally)
        self.xPoints = np.array(xPoints, dtype=float).T  
        self.tPoints = np.array(tPoints, dtype=float)
        self.T       = float(T)
        self.dt      = float(dt)

        # Basic checks
        if len(self.tPoints.shape) != 1:
            raise ValueError("tPoints must be a one-dimensional vector.")
        if len(self.xPoints.shape) != 2:
            raise ValueError("xPoints, after transposition, must be a 2D array.")
        
        p1 = self.xPoints.shape[1]
        p2 = self.tPoints.shape[0]
        if p1 != p2:
            raise ValueError("After transposition, the number of columns of xPoints must match the length of tPoints.")
        
        if self.tPoints[0] < 0 or self.tPoints[-1] > self.T:
            raise ValueError("The times in tPoints must be within [0, T].")
        if self.dt <= 0:
            raise ValueError("dt must be positive.")

        # Number of dimensions and number of known points
        self.n = self.xPoints.shape[0]  # dimensions (rows)
        self.p = self.xPoints.shape[1]  # known points (columns)

    def generate_trajectory(self, traj_type=TrajectoryType.EXPONENTIAL):
        """
        Generates the sampled trajectory (X_array, t_array) based on the requested type.
        
        Parameters:
        -----------
        traj_type : TrajectoryType
            One of the values in [EXPONENTIAL, CUBIC, STEP]
        
        Returns:
        --------
        (X_array, t_array) : (np.ndarray, np.ndarray)
            X_array: a matrix of shape (n, N) containing the values of x for each dimension
                     (n = number of dimensions, N = number of samples)
            t_array: a vector of the corresponding times (length N)
        """
        # Regular time vector
        t_array = np.arange(0, self.T, self.dt)
        N = len(t_array)

        # Matrix for x values for each dimension (n rows, N columns)
        X_array = np.zeros((self.n, N))

        # Interpolate on each interval [tPoints[j], tPoints[j+1]]
        for j in range(self.p - 1):
            t1 = self.tPoints[j]
            t2 = self.tPoints[j+1]

            idx_start = np.searchsorted(t_array, t1)
            idx_end   = np.searchsorted(t_array, t2)

            for i in range(self.n):
                v1 = self.xPoints[i, j]
                v2 = self.xPoints[i, j+1]

                # If the two points are the same, use a constant value
                if v1 == v2:
                    X_array[i, idx_start:idx_end] = v1
                else:
                    # Compute the sub-trajectory (x_loc, t_loc)
                    if traj_type == TrajectoryType.EXPONENTIAL:
                        x_loc, t_loc = self._exponential_segment(t1, t2, v1, v2)
                    elif traj_type == TrajectoryType.CUBIC:
                        x_loc, t_loc = self._cubic_segment(t1, t2, v1, v2)
                    elif traj_type == TrajectoryType.STEP:
                        x_loc, t_loc = self._step_segment(t1, t2, v1, v2)
                    else:
                        raise ValueError("Trajectory type not recognized.")

                    # Fill in the corresponding indices
                    seg_len = min(idx_end - idx_start, len(x_loc))
                    X_array[i, idx_start:idx_start + seg_len] = x_loc[:seg_len]

        # If the last defined time is less than T, continue with the last value
        last_idx = np.searchsorted(t_array, self.tPoints[-1])
        for i in range(self.n):
            last_val = self.xPoints[i, -1]
            X_array[i, last_idx:] = last_val

        return X_array.T, t_array

    def plot_trajectory(self, traj_type=TrajectoryType.EXPONENTIAL, show_points=True):
        """
        Generates the trajectory and plots it in separate subplots,
        one for each dimension (row) of xPoints.
        
        Parameters:
        -----------
        traj_type   : TrajectoryType
            Type of trajectory to generate (default: EXPONENTIAL)
        show_points : bool
            If True, show the known points for each dimension using scatter.
        """
        # Generate the trajectory
        X_array, t_array = self.generate_trajectory(traj_type)
        X_array = X_array.T

        fig, axs = plt.subplots(self.n, 1, figsize=(8, 2*self.n), sharex=True)
        if self.n == 1:
            axs = [axs]  # To handle the case of a single dimension

        for i in range(self.n):
            ax = axs[i]
            ax.plot(t_array, X_array[i, :], label=f"Dimension {i+1}", lw=2)

            if show_points:
                # The known points correspond to self.xPoints[i, :]
                ax.scatter(self.tPoints, self.xPoints[i, :],
                           facecolor='none', edgecolor='k', zorder=10, s=50, 
                           label=f"Known points dim {i+1}")

            ax.set_ylabel(f"x[{i}]")
            ax.grid(True)
            ax.legend()

        fig.suptitle(f"Trajectory type: {traj_type.value}", fontsize=14)
        axs[-1].set_xlabel("Time [s]")
        plt.tight_layout()
        plt.show()

    def _exponential_segment(self, t1, t2, v1, v2):
        dt_local = self.dt
        t_loc = np.arange(t1, t2, dt_local)
        T = (t2 - t1)
        if T < 1e-8:
            return (np.array([v1]), np.array([t1]))

        k = 3.0 / T
        denom = 1.0 - np.exp(-k * T)
        x_loc = v1 + (v2 - v1) * (1 - np.exp(-k * (t_loc - t1))) / denom
        return x_loc, t_loc

    def _cubic_segment(self, t1, t2, v1, v2):
        dt_local = self.dt
        t_loc = np.arange(t1, t2, dt_local)
        if (t2 - t1) < 1e-8:
            return (np.array([v1]), np.array([t1]))
        
        spline = CubicSpline([t1, t2], [v1, v2], bc_type=((1, 0.0), (1, 0.0)))
        x_loc = spline(t_loc)
        return x_loc, t_loc

    def _step_segment(self, t1, t2, v1, v2):
        t_loc = np.arange(t1, t2, self.dt)
        if (t2 - t1) < 1e-8:
            return (np.array([v1]), np.array([t1]))

        # Remain at v1 for the entire interval [t1, t2)
        x_loc = np.full_like(t_loc, v1)
        return x_loc, t_loc

