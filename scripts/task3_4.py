import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from cvxopt import matrix, solvers
from enum import Enum
from typing import Dict, Tuple

from dynamics import FlexibleRoboticArm
from newton import NewtonOptimizer
from animate import FlexibleRobotAnimator
from parameters import (
    m1, m2, l1, l2, r1, r2, I1, I2, g, f1, f2, dt, ns, ni, Q3, R3, QT3, Q4, R4, QT4, N4
    )

# Configure CVXOPT solver
solvers.options.update({'show_progress': False, 'abstol': 1e-10, 'reltol': 1e-10, 'feastol': 1e-10})

class ControllerType(Enum):
    LQR = "LQR"
    MPC = "MPC"

class ControllerBase:
    """Base class for controllers"""
    def __init__(self, arm: FlexibleRoboticArm):
        self.arm = arm
        self.ns = arm.ns
        self.ni = arm.ni

    @staticmethod
    def load_trajectory(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load reference trajectory from CSV"""
        df = pd.read_csv(csv_path)
        x_ref = np.array([df[f'x{i+1}'].values for i in range(4)]).reshape(4, -1)
        u_ref = np.array([df['u'].values]).reshape(1, -1)
        return x_ref, u_ref

class LQRController(ControllerBase):
    """LQR Controller Implementation"""
    def __init__(self, arm: FlexibleRoboticArm, Q: np.ndarray, R: np.ndarray, QT: np.ndarray):
        super().__init__(arm)
        self.Q = Q
        self.R = R
        self.QT = QT
        self.optimizer = NewtonOptimizer(arm)

    def compute_gains(self, x_ref: np.ndarray, u_ref: np.ndarray) -> np.ndarray:
        """Compute LQR gain matrices"""
        AA_traj, BB_traj = self.arm.linearize_over_traj(x_ref, u_ref)
        TT = x_ref.shape[1]
        KK, *_ = self.optimizer.affine_lqp(
            AA_traj, BB_traj, self.Q, self.R,
            np.zeros((self.ni, self.ns, TT)), self.QT, TT,
            np.zeros((self.ns)), np.zeros((self.ns, TT)),
            np.zeros((self.ni, TT)), np.zeros((self.ns))
        )
        return KK

    def simulate(self, x_ref: np.ndarray, u_ref: np.ndarray, 
                disturbances: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate system with LQR control"""
        TT = x_ref.shape[1]
        KK = self.compute_gains(x_ref, u_ref)
        
        xx = np.zeros((self.ns, TT))
        uu = np.zeros((self.ni, TT))
        
        # Apply initial state perturbation if specified
        x0_perturb = disturbances.get("x0_perturb", np.zeros(self.ns))
        if disturbances.get("perturbed_state", False):
            xx[:, 0] = x_ref[:, 0] + x0_perturb
        else:
            xx[:, 0] = x_ref[:, 0]

        # Simulation loop
        for t in range(TT - 1):
            # Compute control input
            uu[:, t] = u_ref[:, t] + KK[:, :, t] @ (xx[:, t] - x_ref[:, t])
            
            # Simulate dynamics
            xx[:, t+1] = self.arm.discrete_dynamics(xx[:, t], uu[:, t])
            
            # Add noise if specified
            if disturbances.get("gaussian_noise", False):
                noise = np.random.normal(
                    disturbances.get("gaussian_mean", 0),
                    disturbances.get("gaussian_std", 0.002),
                    size=self.ns
                )
                xx[:, t+1] += noise

        return xx, uu

class MPCController(ControllerBase):
    """MPC Controller Implementation – versione migliorata con costo sullo stato esplicito."""
    def __init__(self, arm: FlexibleRoboticArm, Q: np.ndarray, R: np.ndarray, N: int,
                 Qf: np.ndarray = None, u_min: float = -60.0, u_max: float = 60.0):
        """
        :param Q: Matrice di costo sullo stato (stadio)
        :param R: Matrice di costo sull'input
        :param N: Lunghezza dell'orizzonte predittivo
        :param Qf: Matrice di costo terminale (se None, si usa Q)
        """
        super().__init__(arm)
        self.Q = Q
        self.R = R
        self.N = N
        self.Qf = Qf if Qf is not None else Q  # costo terminale
        self.u_min = u_min
        self.u_max = u_max

    def formulate_qp_matrices(self, AA: np.ndarray, BB: np.ndarray, x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Costruisce le matrici Hessiana H e il vettore lineare f per il QP
        considerando il costo:
            J = sum_{i=0}^{N-1} [ x_{i+1}^T Q x_{i+1} + u_i^T R u_i ] + x_{N+1}^T Qf x_{N+1}
        con la dinamica predittiva:
            x_{i+1} = A_i x_i + B_i u_i.
        Eliminando le variabili di stato, si ricava una formulazione in funzione di U = [u0, ..., u_{N-1}].
        """
        m = self.ni
        n = self.ns
        N = self.N

        # Calcola la traiettoria nominale propagando x0 con u=0.
        x_nom = np.zeros((n, N+1))
        x_nom[:, 0] = x0
        for i in range(N):
            x_nom[:, i+1] = AA[:, :, i] @ x_nom[:, i]

        # Costruzione delle matrici di sensibilità S_i tali che:
        #   x_i = x_nom[:, i] + S_i * U,
        # dove U è il vettore impilato degli input.
        S = []
        for i in range(N+1):
            S_i = np.zeros((n, N * m))
            # Per ogni controllo u_j che influenza x_i (con j = 0,..., i-1)
            for j in range(i):
                prod = np.eye(n)
                # Prodotto dei Jacobiani A per passare da u_j a x_i:
                for k in range(j, i):
                    prod = AA[:, :, k] @ prod
                # Inserisce il contributo di u_j nelle colonne corrispondenti
                S_i[:, j*m:(j+1)*m] = prod @ BB[:, :, j]
            S.append(S_i)

        # Inizializza Hessiana H e vettore f (nota: il costo è espresso come  U^T H U + 2 f^T U)
        H = np.zeros((N * m, N * m))
        f = np.zeros((N * m, 1))

        # Accumula il costo stage per i passi 1,...,N-1
        for i in range(1, N):
            H += S[i].T @ self.Q @ S[i]
            f += S[i].T @ (2 * self.Q @ x_nom[:, i]).reshape(-1, 1)
        # Aggiunge il costo terminale al passo N
        H += S[N].T @ self.Qf @ S[N]
        f += S[N].T @ (2 * self.Qf @ x_nom[:, N]).reshape(-1, 1)

        # Aggiunge il costo sull'input per ogni istante
        for i in range(N):
            idx = slice(i * m, (i+1) * m)
            H[idx, idx] += self.R

        return H, f

    def formulate_constraints(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Formulazione dei vincoli sugli input.
        Si costruiscono le matrici G e il vettore h tali che:
            G * U <= h,
        per imporre u_min <= u_i <= u_max per i = 0,...,N-1.
        """
        m = self.ni
        N = self.N
        G = np.zeros((2 * N * m, N * m))
        h = np.zeros(2 * N * m)
        
        for i in range(N):
            # Vincolo superiore: u_i <= u_max
            G[i * m:(i+1) * m, i * m:(i+1) * m] = np.eye(m)
            h[i * m:(i+1) * m] = self.u_max
            
            # Vincolo inferiore: -u_i <= -u_min  <=> u_i >= u_min
            G[N * m + i * m:N * m + (i+1) * m, i * m:(i+1) * m] = -np.eye(m)
            h[N * m + i * m:N * m + (i+1) * m] = -self.u_min
            
        return G, h

    def solve_mpc(self, x0: np.ndarray, AA: np.ndarray, BB: np.ndarray) -> np.ndarray:
        """
        Risolve il QP a orizzonte finito.
        :param x0: stato iniziale (errore rispetto al riferimento)
        :param AA: array di matrici A lungo l'orizzonte (dimensione: ns x ns x N)
        :param BB: array di matrici B lungo l'orizzonte (dimensione: ns x ni x N)
        :return: Sequenza ottimale degli input (impilati)
        """
        H, f = self.formulate_qp_matrices(AA, BB, x0)
        G, h = self.formulate_constraints()

        # CVXOPT richiede le matrici in formato "matrix"
        sol = solvers.qp(matrix(H), matrix(f), matrix(G), matrix(h))
        
        if sol['status'] != 'optimal':
            raise RuntimeError("MPC QP solver did not converge to optimal solution")
            
        # Restituisce U con dimensione (N, ni)
        U_opt = np.array(sol['x']).reshape(-1, self.ni)
        return U_opt

    def simulate(self, x_ref: np.ndarray, u_ref: np.ndarray, 
                 disturbances: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Simula il sistema con controllo MPC"""
        TT = x_ref.shape[1] - self.N  # si garantisce la disponibilità della linearizzazione per l'intero orizzonte

        # Inizializza le traiettorie
        xx = np.zeros((self.ns, TT))
        uu = np.zeros((self.ni, TT-1))
        
        # Condizione iniziale (con eventuale perturbazione)
        x0_perturb = disturbances.get("x0_perturb", np.zeros(self.ns))
        if disturbances.get("perturbed_state", False):
            xx[:, 0] = x_ref[:, 0] + x0_perturb
        else:
            xx[:, 0] = x_ref[:, 0]

        # Pre-calcola le matrici di linearizzazione per l'intera traiettoria richiesta
        AA = np.zeros((self.ns, self.ns, TT + self.N))
        BB = np.zeros((self.ns, self.ni, TT + self.N))
        for t in range(TT + self.N):
            A_t, B_t = self.arm.get_gradients(x_ref[:, t], u_ref[:, t])
            AA[:, :, t] = np.array(A_t, dtype=float)
            BB[:, :, t] = np.array(B_t, dtype=float)

        # Loop di simulazione MPC
        for t in range(TT-1):
            # Calcola l'errore di stato (si lavora in variabili erroriali rispetto al riferimento)
            x_error = xx[:, t] - x_ref[:, t]
            
            # Risolve il QP a orizzonte finito
            U_opt = self.solve_mpc(x_error, AA[:, :, t:t+self.N], BB[:, :, t:t+self.N])
            
            # Applica il primo comando ottimale, aggiungendolo al riferimento
            uu[:, t] = U_opt[0] + u_ref[:, t]
            
            # Evolvi la dinamica del sistema
            xx[:, t+1] = self.arm.discrete_dynamics(xx[:, t], uu[:, t])
            
            # Aggiungi rumore se specificato
            if disturbances.get("gaussian_noise", False):
                noise = np.random.normal(
                    disturbances.get("gaussian_mean", 0),
                    disturbances.get("gaussian_std", 0.002),
                    size=self.ns
                )
                xx[:, t+1] += noise

        return xx, uu


class SimulationManager:
    """Manages simulation execution and results visualization"""
    def __init__(self, arm: FlexibleRoboticArm):
        self.arm = arm
        
    def run_comparison(self, x_ref: np.ndarray, u_ref: np.ndarray, 
                      disturbances: Dict) -> Dict:
        """Run both LQR and MPC controllers and compare results"""
        # Initialize controllers
        lqr = LQRController(
            self.arm,
            Q=Q3,
            R=R3,
            QT=QT3
        )
        
        mpc = MPCController(
            self.arm,
            Q=Q4,
            R=R4,
            N=N4,
            Qf=QT4
        )
        
        # Run simulations
        xx_lqr, uu_lqr = lqr.simulate(x_ref, u_ref, disturbances)
        xx_mpc, uu_mpc = mpc.simulate(x_ref, u_ref, disturbances)
        
        return {
            'lqr': (xx_lqr, uu_lqr),
            'mpc': (xx_mpc, uu_mpc)
        }
    
    def run_lqr(self, x_ref: np.ndarray, u_ref: np.ndarray, 
                disturbances: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Run only the LQR controller"""
        lqr = LQRController(
            self.arm,
            Q=Q3,
            R=R3,
            QT=QT3
        )
        return lqr.simulate(x_ref, u_ref, disturbances)
    
    def run_mpc(self, x_ref: np.ndarray, u_ref: np.ndarray, 
                disturbances: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Run only the MPC controller"""
        mpc = MPCController(
            self.arm,
            Q=Q4,
            R=R4,
            N=N4
        )
        return mpc.simulate(x_ref, u_ref, disturbances)
    
    def plot_comparison(self, x_ref: np.ndarray, u_ref: np.ndarray, 
                    results: Dict, save_plots: bool = True):
        """Plot comparison of LQR and MPC results"""
        # Find the minimum length among all trajectories
        T_ref = x_ref.shape[1]
        T_lqr = results['lqr'][0].shape[1]
        T_mpc = results['mpc'][0].shape[1]
        T = min(T_ref, T_lqr, T_mpc)
        
        # Create time vector for the shortened horizon
        time = np.arange(T) * dt
        
        # Create figure for states
        fig, axs = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
        fig.suptitle('Controller Comparison: LQR vs MPC', fontsize=16)
        
        state_labels = [r'$\theta_1$', r'$\theta_2$', 
                    r'$\dot{\theta}_1$', r'$\dot{\theta}_2$']
        
        # Plot states
        for i in range(4):
            ax = axs[i]
            ax.plot(time, x_ref[i, :T], 'k--', label='Reference', alpha=0.7)
            ax.plot(time, results['lqr'][0][i, :T], 'b-', label='LQR', alpha=0.8)
            ax.plot(time, results['mpc'][0][i, :T], 'r-', label='MPC', alpha=0.8)
            ax.set_ylabel(state_labels[i])
            ax.grid(True)
            ax.legend()

        # Plot control inputs
        ax = axs[4]
        ax.plot(time, u_ref[0, :T], 'k--', label='Reference', alpha=0.7)
        ax.plot(time[:-1], results['lqr'][1][0, :(T-1)], 'b-', label='LQR', alpha=0.8)
        ax.plot(time[:-1], results['mpc'][1][0, :(T-1)], 'r-', label='MPC', alpha=0.8)
        ax.set_ylabel('Control Input u')
        ax.set_xlabel('Time [s]')
        ax.grid(True)
        ax.legend()

        plt.tight_layout()
        
        # Create performance metrics plot
        fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig2.suptitle('Performance Comparison: LQR vs MPC', fontsize=16)
        
        # Plot tracking error
        for controller, (xx, _) in results.items():
            error = np.linalg.norm(xx[:, :T] - x_ref[:, :T], axis=0)
            ax1.plot(time, error, label=controller.upper())
        ax1.set_ylabel('State Error Norm')
        ax1.grid(True)
        ax1.legend()
        
        # Plot control effort
        for controller, (_, uu) in results.items():
            effort = np.linalg.norm(uu[:, :(T-1)] - u_ref[:, :(T-1)], axis=0)
            ax2.plot(time[:-1], effort, label=controller.upper())
        ax2.set_ylabel('Control Effort')
        ax2.set_xlabel('Time [s]')
        ax2.grid(True)
        ax2.legend()
        
        if save_plots:
            try:
                fig.savefig('plots/controller_comparison.png')
                fig2.savefig('plots/performance_comparison.png')
                print("Plots saved successfully")
            except Exception as e:
                print(f"Error saving plots: {e}")
        
        plt.show()
    
    def plot_single(self, x_ref: np.ndarray, u_ref: np.ndarray, 
                    result: Tuple[np.ndarray, np.ndarray], controller_name: str,
                    save_plot: bool = True):
        """Plot simulation results for a single controller"""
        xx, uu = result
        T_ref = x_ref.shape[1]
        T_sim = xx.shape[1]
        T = min(T_ref, T_sim)
        time = np.arange(T) * dt
        
        fig, axs = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
        fig.suptitle(f'Simulation with {controller_name}', fontsize=16)
        
        state_labels = [r'$\theta_1$', r'$\theta_2$', 
                        r'$\dot{\theta}_1$', r'$\dot{\theta}_2$']
        for i in range(4):
            ax = axs[i]
            ax.plot(time, x_ref[i, :T], 'k--', label='Reference', alpha=0.7)
            ax.plot(time, xx[i, :T], 'b-', label=controller_name, alpha=0.8)
            ax.set_ylabel(state_labels[i])
            ax.grid(True)
            ax.legend()
            
        ax = axs[4]
        ax.plot(time, u_ref[0, :T], 'k--', label='Reference', alpha=0.7)
        # Note: The control input trajectory length is T-1
        ax.plot(time[:-1], uu[0, :(T-1)], 'b-', label=controller_name, alpha=0.8)
        ax.set_ylabel('Control Input u')
        ax.set_xlabel('Time [s]')
        ax.grid(True)
        ax.legend()
        
        plt.tight_layout()
        if save_plot:
            try:
                fig.savefig(f'plots/{controller_name.lower()}_simulation.png')
                print(f"Plot for {controller_name} saved successfully")
            except Exception as e:
                print(f"Error saving plot: {e}")
        plt.show()

def main():
    """Main execution function"""
    try:
        # Initialize system
        arm = FlexibleRoboticArm(m1, m2, l1, l2, r1, r2, I1, I2, g, f1, f2, dt, ns, ni)
        
        # Load reference trajectory
        x_ref, u_ref = ControllerBase.load_trajectory('traj/optimal_trajectories.csv')
        
        # Define disturbances
        disturbances = {
            "gaussian_noise": True,
            "perturbed_state": True,
            "perturbed_params": True,
            "x0_perturb": np.array([0.02, 0.1, 0.1, 0.1]),
            "gaussian_mean": 0,
            "gaussian_std": 0.002,
        }
        
        # If you want to also perturb the system parameters
        if disturbances.get("perturbed_params"):
            arm = FlexibleRoboticArm(m1 * 1.03, m2 * 1.05, l1 * 1.02, l2 * 1.04, r1, r2, I1, I2, g, f1, f2, dt, ns, ni)
        
        sim_manager = SimulationManager(arm)
        
        # Choose simulation mode
        print("Select simulation mode:")
        print("1 - LQR only")
        print("2 - MPC only")
        print("3 - Comparison (LQR and MPC)")
        mode = input("Enter 1, 2 or 3: ").strip()
        
        if mode == "1":
            print("Running simulation with LQR...")
            result = sim_manager.run_lqr(x_ref, u_ref, disturbances)
            sim_manager.plot_single(x_ref, u_ref, result, "LQR")
            
        elif mode == "2":
            print("Running simulation with MPC...")
            result = sim_manager.run_mpc(x_ref, u_ref, disturbances)
            sim_manager.plot_single(x_ref, u_ref, result, "MPC")
            
        elif mode == "3":
            print("Running simulation for LQR vs MPC comparison...")
            results = sim_manager.run_comparison(x_ref, u_ref, disturbances)
            sim_manager.plot_comparison(x_ref, u_ref, results)
            
            # Example: animate the MPC result (optional)
            animate = input("Do you want to animate the MPC result? (y/n): ").strip().lower() == 'y'
            if animate:
                animator = FlexibleRobotAnimator(results['mpc'][0].T, dt=dt)
                animator.animate()
        else:
            print("Invalid mode. Terminating the program.")
        
    except FileNotFoundError:
        print("Error: Could not find trajectory file")
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

