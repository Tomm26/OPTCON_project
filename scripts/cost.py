import numpy as np
from parameters import ns, ni

class Cost:
    def __init__(self, QQt=None, RRt=None, QQT=None):
        self.ns = ns
        self.ni = ni
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0]) if QQt is None else QQt
        self.R = 0.01 * np.eye(ni) if RRt is None else RRt
        self.QT = QQT
        self.S = np.zeros((ni, ns))
        
    def _check_cost_matrices(self):
        assert self.Q.shape == (self.ns, self.ns), f"Q must be {self.ns}x{self.ns}"
        assert self.R.shape == (self.ni, self.ni), f"R must be {self.ni}x{self.ni}"
        assert self.QT.shape == (self.ns, self.ns), f"QT must be {self.ns}x{self.ns}"
        
    def _check_dimensions(self, x, u, x_ref, u_ref):
        assert x.shape == (self.ns,), f"State must be {self.ns}-dimensional"
        assert u.shape == (self.ni,), f"Input must be {self.ni}-dimensional"
        assert x_ref.shape == (self.ns,), f"Reference state must be {self.ns}-dimensional"
        assert u_ref.shape == (self.ni,), f"Reference input must be {self.ni}-dimensional"
        
    def stage_cost(self, x, u, x_ref, u_ref):
        self._check_cost_matrices()
        self._check_dimensions(x, u, x_ref, u_ref)
        x_dev = x - x_ref
        u_dev = u - u_ref

        ll = 0.5 * x_dev.T @ self.Q @ x_dev + 0.5 * u_dev.T @ self.R @ u_dev

        grad_x = self.Q @ x_dev
        grad_u = self.R @ u_dev
        return ll, grad_x, grad_u, self.Q, self.S, self.R
    
    def terminal_cost(self, x, x_ref):
        self._check_cost_matrices()
        self._check_dimensions(x, np.zeros(self.ni), x_ref, np.zeros(self.ni))
        x_dev = x - x_ref

        lT = 0.5 * x_dev.T @ self.QT @ x_dev

        grad_x = self.QT @ x_dev
        return lT, grad_x, self.QT

    def get_QT(self):
        return self.QT

    def get_Q(self):
        return self.Q
    
    def get_R(self):
        return self.R
    
    def set_QT(self, new_QT):

        if new_QT.shape != (self.ns, self.ns):
            raise ValueError(f"Terminal cost matrix QT must be of shape ({self.ns}, {self.ns})")
        self.QT = new_QT

    def set_Q(self, new_Q):

        if new_Q.shape != (self.ns, self.ns):
            raise ValueError(f"Q must be of shape ({self.ns}, {self.ns})")
        self.Q = new_Q

    def set_R(self, new_R):

        if new_R.shape != (self.ni, self.ni):
            raise ValueError(f"R must be of shape ({self.ni}, {self.ni})")
        self.R = new_R

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