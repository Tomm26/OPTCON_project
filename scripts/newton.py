import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from dynamics import FlexibleRoboticArm
from cost import Cost
from parameters import ns, ni, dt

class NewtonOptimizer:
    def __init__(self, arm: FlexibleRoboticArm):
        """
        Initialize Newton optimizer for trajectory optimization.
        
        Args:
            arm: FlexibleRoboticArm instance
            cost_fn: Cost instance for optimization
        """
        self.arm = arm
        
        # Parametri di Armijo
        self.c = 0.5
        self.beta = 0.7
        self.max_armijo_iters = 20
        
    def solve_trajectory(self, 
                         x_ref, 
                         u_ref, 
                         x0,
                         Q, R,
                         max_iters=100, 
                         tol=1e-6, 
                         do_plot=True):
        """
        Solve trajectory optimization problem.
        
        Args:
            x_ref: Reference state trajectory (T+1, ns)
            u_ref: Reference input trajectory (T, ni)
            x0: Initial state
            Q: Stage state cost matrix (usata nella cost function)
            R: Stage input cost matrix (usata nella cost function)
            max_iters: Maximum number of iterations
            tol: Convergence tolerance on the gradient norm
            do_plot: Boolean per abilitare/disabilitare i plot a fine esecuzione
        
        Returns:
            x: optimized state trajectory (T+1, ns)
            u: optimized input trajectory (T, ni)
            info: dictionary with debugging information
        """

        self.cost_fn = Cost(Q, R)

        T = u_ref.shape[0]
        
        # Inizializza traiettorie
        x = np.zeros((T+1, ns))
        u = np.zeros((T, ni))
        x[0] = x0
        
        # Forward pass iniziale (open-loop con u=0)
        for t in range(T):
            x[t+1] = self.arm.discrete_dynamics(x[t], u[t])
        
        costs = []
        grad_norms = []
        
        # Iterazioni
        for iteration in trange(max_iters):
            # Backward pass: ottieni costate e derivate
            lambda_, As, Bs, qs, rs, Qs, Rs = self._backward_pass(x, u, x_ref, u_ref)
            
            # Calcolo costo totale
            cost = self._compute_cost(x, u, x_ref, u_ref)
            costs.append(cost)
            
            # Gradiente w.r.t. gli input
            grad = self._compute_gradient(lambda_, x, u, rs, Bs)
            grad_norm = np.linalg.norm(grad)
            grad_norms.append(grad_norm)
            
            # Check convergenza
            if grad_norm < tol:
                print(f"[INFO] Convergenza raggiunta a iterazione {iteration} (norm grad={grad_norm:.2e})")
                break
                
            # Sottoproblema LQR -> guadagni e feedforward
            K, sigma = self._solve_lqr_subproblem(As, Bs, Qs, Rs, qs, rs)
            
            # Ricerca Armijo
            step_size = self._armijo_search(x, u, x_ref, u_ref, K, sigma, cost, grad)
            
            if step_size <= 0:
                print("[WARNING] Armijo search fallita o direzione non di discesa. Interruzione.")
                break
                
            # Aggiorno traiettoria in anello chiuso
            x_new = np.zeros_like(x)
            u_new = np.zeros_like(u)
            x_new[0] = x[0].copy()
            
            for t in range(T):
                u_new[t] = u[t] + K[t] @ (x_new[t] - x[t]) + step_size * sigma[t]
                x_new[t+1] = self.arm.discrete_dynamics(x_new[t], u_new[t])
            
            x = x_new
            u = u_new
            
        # Costruisco dizionario di info
        info = {
            'costs': costs,
            'grad_norms': grad_norms,
            'iterations': len(costs),
            'converged': (grad_norms[-1] < tol) if grad_norms else False
        }
        
        # Eventuale plot
        if do_plot:
            self._plot_results(costs, grad_norms, x, x_ref)
        
        return x, u, info

    def _backward_pass(self, x, u, x_ref, u_ref):
        """Compute costate and derivatives via backward pass."""
        T = u.shape[0]
        
        lambda_ = np.zeros((T+1, ns))
        As = np.zeros((T, ns, ns))
        Bs = np.zeros((T, ns, ni))
        Qs = np.zeros((T+1, ns, ns))
        Rs = np.zeros((T, ni, ni))
        qs = np.zeros((T+1, ns))
        rs = np.zeros((T, ni))
        
        # Terminal cost
        lf, qf, Qf = self.cost_fn.terminal_cost(x[-1], x_ref[-1])
        lambda_[-1] = qf
        Qs[-1] = Qf
        qs[-1] = qf
        
        # Itero da T-1 a 0
        for t in range(T-1, -1, -1):
            # stage cost
            _, qt, rt, Qt, Rt = self.cost_fn.stage_cost(x[t], u[t], x_ref[t], u_ref[t])
            
            # dinamica linearizzata
            At, Bt = self.arm.get_gradients(x[t], u[t])[0:2]
            
            As[t] = At
            Bs[t] = Bt
            Qs[t] = Qt
            Rs[t] = Rt
            qs[t] = qt
            rs[t] = rt
            
            # costate
            lambda_[t] = qt + At.T @ lambda_[t+1]
        
        return lambda_, As, Bs, qs, rs, Qs, Rs
    
    def _solve_lqr_subproblem(self, As, Bs, Qs, Rs, qs, rs):
        """Solve the LQR subproblem for feedback gains and feedforward."""
        T = As.shape[0]
        
        K = np.zeros((T, ni, ns))
        sigma = np.zeros((T, ni))
        P = np.zeros((T+1, ns, ns))
        p = np.zeros((T+1, ns))
        
        # terminal
        P[-1] = Qs[-1]
        p[-1] = qs[-1]
        
        # backward pass di Riccati
        for t in range(T-1, -1, -1):
            M = Rs[t] + Bs[t].T @ P[t+1] @ Bs[t]
            
            K[t] = -np.linalg.solve(M, Bs[t].T @ P[t+1] @ As[t])
            sigma[t] = -np.linalg.solve(M, rs[t] + Bs[t].T @ p[t+1])
            
            P[t] = Qs[t] + As[t].T @ P[t+1] @ As[t] + K[t].T @ M @ K[t]
            p[t] = qs[t] + As[t].T @ p[t+1] + K[t].T @ M @ sigma[t]
            
        return K, sigma
    
    def _compute_cost(self, x, u, x_ref, u_ref):
        """Compute total trajectory cost."""
        cost, _, _, _ = self.cost_fn.complete_cost(x, u, x_ref, u_ref)
        return cost
    
    def _compute_gradient(self, lambda_, x, u, rs, Bs):
        """Compute gradient of cost with respect to inputs."""
        T = u.shape[0]
        grad = np.zeros_like(u)
        for t in range(T):
            grad[t] = rs[t] + Bs[t].T @ lambda_[t+1]
        return grad
    
    def _armijo_search(self, x, u, x_ref, u_ref, K, sigma, cost, grad):
        """
        Perform Armijo line search.
        Returns step_size or 0.0 if fails.
        """
        grad_dot_sigma = np.sum(grad * sigma)
        if grad_dot_sigma >= 0:
            print("[WARNING] sigma not a descent direction")
            return 0.0
            
        step_size = 1.0
        
        for _ in range(self.max_armijo_iters):
            x_new = np.zeros_like(x)
            u_new = np.zeros_like(u)
            x_new[0] = x[0].copy()
            
            try:
                for t in range(u.shape[0]):
                    u_new[t] = u[t] + K[t] @ (x_new[t] - x[t]) + step_size * sigma[t]
                    x_new[t+1] = self.arm.discrete_dynamics(x_new[t], u_new[t])
                
                cost_new = self._compute_cost(x_new, u_new, x_ref, u_ref)
                
                if cost_new <= cost + self.c * step_size * grad_dot_sigma:
                    return step_size
                    
            except (np.linalg.LinAlgError, RuntimeWarning):
                print("[WARNING] Simulation failed during Armijo search.")
                pass
                
            step_size *= self.beta
        
        print("[WARNING] Armijo search failed (no suitable step found).")
        return 0.0

    def _plot_results(self, costs, grad_norms, x, x_ref):
        """Simple plot of cost, gradient norm, and state vs reference."""
        iters = range(len(costs))
        
        plt.figure(figsize=(10,4))
        # Costo
        plt.subplot(1,2,1)
        plt.plot(iters, costs, 'b-o')
        plt.xlabel('Iterazione')
        plt.ylabel('Costo')
        plt.title('Evoluzione del costo')
        
        # Norma del gradiente
        plt.subplot(1,2,2)
        plt.plot(iters, grad_norms, 'r-o')
        plt.xlabel('Iterazione')
        plt.ylabel('||grad||')
        plt.title('Evoluzione della norma del gradiente')
        
        plt.tight_layout()
        plt.show()
        
        # Stato vs Riferimento
        plt.figure()
        for i in range(x.shape[1]):
            plt.plot(x[:, i], label=f'x[{i}]')
            plt.plot(x_ref[:, i], '--', label=f'x_ref[{i}]')
        plt.xlabel('Time step')
        plt.title('Confronto stati vs riferimento')
        plt.legend()
        plt.show()