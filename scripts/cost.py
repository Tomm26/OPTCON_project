import numpy as np
from parameters import ns, ni

class Cost:
    def __init__(self, QQt=None, RRt=None):
        self.ns = ns
        self.ni = ni
        self.Q = np.diag([10.0, 5.0, 1.0, 1.0]) if QQt is None else QQt
        self.R = 0.1 * np.eye(ni) if RRt is None else RRt
        self._check_cost_matrices()
        
    def _check_cost_matrices(self):
        assert self.Q.shape == (self.ns, self.ns), f"Q must be {self.ns}x{self.ns}"
        assert self.R.shape == (self.ni, self.ni), f"R must be {self.ni}x{self.ni}"
        
    def _check_dimensions(self, x, u, x_ref, u_ref):
        assert x.shape == (self.ns,), f"State must be {self.ns}-dimensional"
        assert u.shape == (self.ni,), f"Input must be {self.ni}-dimensional"
        assert x_ref.shape == (self.ns,), f"Reference state must be {self.ns}-dimensional"
        assert u_ref.shape == (self.ni,), f"Reference input must be {self.ni}-dimensional"
        
    def _check_trajectory_dimensions(self, x_traj, u_traj, x_ref_traj, u_ref_traj):
        T = u_traj.shape[0]
        assert x_traj.shape == (T+1, self.ns), f"State trajectory must be {T+1}x{self.ns}"
        assert u_traj.shape == (T, self.ni), f"Input trajectory must be {T}x{self.ni}"
        assert x_ref_traj.shape == (T+1, self.ns), f"Reference state trajectory must be {T+1}x{self.ns}"
        assert u_ref_traj.shape == (T, self.ni), f"Reference input trajectory must be {T}x{self.ni}"
        
    def stage_cost(self, x, u, x_ref, u_ref):
        self._check_dimensions(x, u, x_ref, u_ref)
        x_dev = x - x_ref
        u_dev = u - u_ref

        l = 0.5 * x_dev.T @ self.Q @ x_dev + 0.5 * u_dev.T @ self.R @ u_dev

        grad_x = self.Q @ x_dev
        grad_u = self.R @ u_dev
        return l, grad_x, grad_u, self.Q, self.R
    
    def terminal_cost(self, x, x_ref):
        self._check_dimensions(x, np.zeros(self.ni), x_ref, np.zeros(self.ni))
        x_dev = x - x_ref

        lT = 0.5 * x_dev.T @ self.Q @ x_dev

        grad_x = self.Q @ x_dev
        return lT, grad_x, self.Q
        
    def complete_cost(self, x_traj, u_traj, x_ref_traj, u_ref_traj):
        self._check_trajectory_dimensions(x_traj, u_traj, x_ref_traj, u_ref_traj)
        T = u_traj.shape[0]
        l_traj = np.zeros(T)
        grad_x_traj = np.zeros((T+1, self.ns))
        grad_u_traj = np.zeros((T, self.ni))
        
        for t in range(T):
            l_traj[t], grad_x_traj[t], grad_u_traj[t], _, _ = self.stage_cost(
                x_traj[t], u_traj[t], x_ref_traj[t], u_ref_traj[t])
            
        lT, grad_x_traj[T], _ = self.terminal_cost(x_traj[T], x_ref_traj[T])

        J = np.sum(l_traj) + lT
        
        return J, l_traj, grad_x_traj, grad_u_traj


if __name__ == "__main__":

    # Example parameters
    QQt = None  # Use default Q matrix
    RRt = None  # Use default R matrix
    cost = Cost(QQt=QQt, RRt=RRt)

    # Define trajectory length
    T = 5

    # Generate sample trajectories
    x_traj = np.random.randn(T+1, cost.ns)
    u_traj = np.random.randn(T, cost.ni)
    x_ref_traj = np.zeros((T+1, cost.ns))
    u_ref_traj = np.zeros((T, cost.ni))

    # Compute the complete cost
    J, l_traj, grad_x_traj, grad_u_traj = cost.complete_cost(
        x_traj, u_traj, x_ref_traj, u_ref_traj
    )

    # Print the results
    print("Total Cost:", J)
    print("Stage Costs:", l_traj)
    print("Gradient w.r.t States:", grad_x_traj)
    print("Gradient w.r.t Inputs:", grad_u_traj)