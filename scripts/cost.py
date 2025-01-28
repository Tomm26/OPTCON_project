import numpy as np
from parameters import ns, ni

class Cost:
    def __init__(self, QQt=None, RRt=None, QQT=None):
        self.ns = ns
        self.ni = ni
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0]) if QQt is None else QQt
        self.R = 0.01 * np.eye(ni) if RRt is None else RRt
        self.QT = np.diag([1.0, 1.0, 1.0, 1.0]) if QQT is None else QQT
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

        ll = 0.5 * x_dev.T @ self.Q @ x_dev + 0.5 * u_dev.T @ self.R @ u_dev

        grad_x = self.Q @ x_dev
        grad_u = self.R @ u_dev
        return ll, grad_x, grad_u, self.Q, self.R
    
    def terminal_cost(self, x, x_ref):
        self._check_dimensions(x, np.zeros(self.ni), x_ref, np.zeros(self.ni))
        x_dev = x - x_ref

        lT = 0.5 * x_dev.T @ self.QT @ x_dev

        grad_x = self.QT @ x_dev
        return lT, grad_x, self.QT
        
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
    QQt = np.diag([1.0, 1.0, 1.0, 1.0])
    RRt = 0.01 * np.eye(ni)
    cost = Cost(QQt=QQt, RRt=RRt)

    # Define trajectory length
    T = 5

    # Generate sample trajectories
    x_traj = np.ones((T+1, ns))
    u_traj = np.ones((T, ni))
    x_ref_traj = np.zeros((T+1, ns))
    u_ref_traj = np.zeros((T, ni))

    # Compute the stage cost
    l, grad_x, grad_u, Q, R = cost.stage_cost(x_traj[0], u_traj[0], x_ref_traj[0], u_ref_traj[0])
    print("Stage Cost:", l)
    print("Gradient w.r.t States:", grad_x)
    print("Gradient w.r.t Inputs:", grad_u)

    # Compute the terminal cost
    lT, grad_xT, Q = cost.terminal_cost(x_traj[-1], x_ref_traj[-1])
    print("Terminal Cost:", lT)
    print("Gradient w.r.t States:", grad_xT)

    # # Compute the complete cost
    # J, l_traj, grad_x_traj, grad_u_traj = cost.complete_cost(
    #     x_traj, u_traj, x_ref_traj, u_ref_traj
    # )

    # # Print the results
    # print("Total Cost:", J)
    # print("Stage Costs:", l_traj)
    # print("Gradient w.r.t States:", grad_x_traj)
    # print("Gradient w.r.t Inputs:", grad_u_traj)