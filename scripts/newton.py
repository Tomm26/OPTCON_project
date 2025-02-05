import numpy as np
from matplotlib import pyplot as plt
from dynamics import FlexibleRoboticArm
from cost import Cost
from parameters import ns, ni, dt
from control import dare

class NewtonOptimizer:
    def __init__(self, arm:FlexibleRoboticArm, cost:Cost = None, dt=dt, fixed_stepsize=0.7, stepsize_0=1, cc=0.5, beta=0.7, armijo_maxiters=15):
        """
        Initialize the Newton Optimizer.
        
        Args:
            arm: FlexibleRoboticArm instance for system dynamics
            cost: Cost instance for cost calculations
            dt: Time step for discretization
            fixed_stepsize: Fixed step size when not using Armijo rule
            stepsize_0: Initial step size for Armijo rule
            cc: Armijo rule parameter
            beta: Step size reduction factor for Armijo rule
            armijo_maxiters: Maximum Armijo iterations
        """
        self.arm = arm
        self.cost = cost

        self.dt = dt
        self.stepsize_0 = stepsize_0
        self.cc = cc
        self.beta = beta
        self.armijo_maxiters = armijo_maxiters
        self.fixed_stepsize = fixed_stepsize
    
    def check_qt(self, xx_ref, uu_ref):
        
        if self.cost.get_QT() is None:
            
            A, B = self.arm.get_gradients(xx_ref[:, -1], uu_ref[:, -1])
            Q = self.cost.get_Q()
            R = self.cost.get_R()
            self.cost.set_QT(dare(A, B, Q, R)[0])
            print('QT computed :\n', self.cost.get_QT())

    def stagecost(self, x, u, x_ref, u_ref):
        """Wrapper for stage cost that matches optimizer interface."""
        ll, grad_x, grad_u, Q, S, R = self.cost.stage_cost(x, u, x_ref, u_ref)
        return ll, grad_x, grad_u, Q, S, R

    def termcost(self, x, x_ref):
        """Wrapper for terminal cost that matches optimizer interface."""
        lT, grad_x, QT = self.cost.terminal_cost(x, x_ref)
        return lT, grad_x, QT

    def select_stepsize(self, xx_ref, uu_ref, x0, xx, uu, JJ, descent_arm, KK, sigma, TT, plot=False):
        """
        Computes the stepsize using Armijo's rule.
        """
        stepsizes = []
        costs_armijo = []
        stepsize = self.stepsize_0
        
        # Main Armijo loop
        for i in range(self.armijo_maxiters):
            print(f"Armijo iteration {i+1}")
            # Temporary solution update
            xx_temp = np.zeros((ns, TT))
            uu_temp = np.zeros((ni, TT))
            xx_temp[:, 0] = x0

            for tt in range(TT - 1):
                
                uu_temp[:, tt] = uu[:, tt] + KK[:, :, tt] @ (xx_temp[:, tt] - xx[:, tt]) + stepsize * sigma[:, tt]
                xx_temp[:, tt + 1] = self.arm.discrete_dynamics(xx_temp[:, tt], uu_temp[:, tt])

            # Temporary cost calculation
            JJ_temp = 0
            for tt in range(TT - 1):
                JJ_temp += self.stagecost(xx_temp[:, tt], uu_temp[:, tt], xx_ref[:, tt], uu_ref[:, tt])[0]

            JJ_temp += self.termcost(xx_temp[:, -1], xx_ref[:, -1])[0]

            stepsizes.append(stepsize)
            costs_armijo.append(JJ_temp)

            if JJ_temp >= JJ + self.cc * stepsize * descent_arm:
                stepsize *= self.beta
            else:
                print(f'Armijo stepsize = {stepsize:.3e}, computed in {i+1} iterations')
                break

            if i == self.armijo_maxiters - 1:
                print("WARNING: no stepsize was found with armijo rule!")
                stepsize = self.fixed_stepsize
                break

        if plot:
            self._plot_armijo_descent(stepsize, x0, xx, uu, KK, sigma,
                                    xx_ref, uu_ref, JJ, descent_arm,
                                    stepsizes, costs_armijo, ns, ni, TT)

        return stepsize

    def _plot_armijo_descent(self, stepsize_0, x0, xx, uu, KK, sigma,
                            xx_ref, uu_ref, JJ, descent_arm,
                            stepsizes, costs_armijo, ns, ni, TT):
        """
        Plots the cost as a function of the stepsize in the Armijo rule.
        Optimized version using vectorized operations.
        """
        # Generate uniform stepsize grid (ridotto il numero di punti)
        steps = np.linspace(0, stepsize_0, 10)  # Ridotto da 20 a 10 punti
        costs = np.zeros(len(steps))

        # Pre-allocate arrays for all steps at once
        xx_temp_all = np.zeros((len(steps), ns, TT))
        uu_temp_all = np.zeros((len(steps), ni, TT))
        xx_temp_all[:, :, 0] = x0

        # Compute all trajectories in parallel
        for tt in range(TT - 1):
            # Broadcast operations for all stepsizes at once
            state_diff = xx_temp_all[:, :, tt] - xx[:, tt]
            feedback_term = np.einsum('ijk,lj->li', KK[:, :, tt:tt+1], state_diff)
            stepsize_term = np.outer(steps, sigma[:, tt])
            
            uu_temp_all[:, :, tt] = (uu[:, tt] + 
                                    feedback_term + 
                                    stepsize_term)
            
            # Compute next states for all trajectories
            for i in range(len(steps)):
                xx_temp_all[i, :, tt + 1] = self.arm.discrete_dynamics(xx_temp_all[i, :, tt], uu_temp_all[i, :, tt])

        # Compute costs efficiently
        for i in range(len(steps)):
            JJ_temp = 0
            xx_temp = xx_temp_all[i]
            uu_temp = uu_temp_all[i]
            
            # Vectorize stage cost computation
            stage_costs = np.array([
                self.stagecost(xx_temp[:, t], uu_temp[:, t],
                            xx_ref[:, t], uu_ref[:, t])[0]
                for t in range(TT - 1)
            ])
            JJ_temp = np.sum(stage_costs)
            
            # Add terminal cost
            JJ_temp += self.termcost(xx_temp[:, -1], xx_ref[:, -1])[0]
            costs[i] = JJ_temp

        # Plotting (ora più efficiente)
        plt.figure(1)
        plt.clf()
        
        # Plot tutto in un'unica figura
        plt.plot(steps, costs, 'y-', 
                steps, JJ + descent_arm * steps, 'r-',
                steps, JJ + self.cc * descent_arm * steps, 'g--')
        
        plt.scatter(stepsizes[-1], costs_armijo[-1], marker='*', c='b', zorder=5)
        
        # Aggiungi le labels dopo
        plt.gca().lines[0].set_label('$J(\\mathbf{u}^k - \\text{stepsize}\\cdot d^k)$')
        plt.gca().lines[1].set_label('$J(\\mathbf{u}^k) + \\text{stepsize} \\cdot \\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
        plt.gca().lines[2].set_label('$J(\\mathbf{u}^k) + c\\cdot\\text{stepsize} \\cdot \\nabla J(\\mathbf{u}^k)^{\\top} d^k$')

        plt.grid(True)
        plt.xlabel('stepsize')
        plt.legend()
        plt.show()

    def affine_lqp(self, AAin, BBin, QQin, RRin, SSin, QQfin, TT, x0, qqin, rrin, qqfin):
        """
        Affine Linear Quadratic Optimal Control Problem Solver
        """
        # Handle 2D input matrices
        try:
            ns, lA = AAin.shape[1:]
        except ValueError:
            AAin = AAin[:, :, None]
            ns, lA = AAin.shape[1:]

        try:
            ni, lB = BBin.shape[1:]
        except ValueError:
            BBin = BBin[:, :, None]
            ni, lB = BBin.shape[1:]

        try:
            nQ, lQ = QQin.shape[1:]
        except ValueError:
            QQin = QQin[:, :, None]
            nQ, lQ = QQin.shape[1:]

        try:
            nR, lR = RRin.shape[1:]
        except ValueError:
            RRin = RRin[:, :, None]
            nR, lR = RRin.shape[1:]

        try:
            nSi, nSs, lS = SSin.shape
        except ValueError:
            SSin = SSin[:, :, None]
            nSi, nSs, lS = SSin.shape

        # Dimension checks
        if nQ != ns:
            raise ValueError("Matrix Q not compatible with number of states.")
        if nR != ni:
            raise ValueError("Matrix R not compatible with number of inputs.")
        if (nSs != ns) or (nSi != ni):
            raise ValueError("Matrix S not compatible with state/input dimensions.")

        # Expand if necessary
        if lA < TT: AAin = AAin.repeat(TT, axis=2)
        if lB < TT: BBin = BBin.repeat(TT, axis=2)
        if lQ < TT: QQin = QQin.repeat(TT, axis=2)
        if lR < TT: RRin = RRin.repeat(TT, axis=2)
        if lS < TT: SSin = SSin.repeat(TT, axis=2)

        # Initialize arrays
        KK = np.zeros((ni, ns, TT))
        sigma = np.zeros((ni, TT))
        PP = np.zeros((ns, ns, TT))
        pp = np.zeros((ns, TT))
        xx = np.zeros((ns, TT))
        uu = np.zeros((ni, TT))
        xx[:, 0] = x0

        # Terminal conditions
        PP[:, :, -1] = QQfin
        pp[:, -1] = qqfin

        # Backward pass
        for t in reversed(range(TT - 1)):
            AAt = AAin[:, :, t]
            BBt = BBin[:, :, t]
            QQt = QQin[:, :, t]
            RRt = RRin[:, :, t]
            SSt = SSin[:, :, t]
            qqt = qqin[:, t][:, None]
            rrt = rrin[:, t][:, None]
            PP_next = PP[:, :, t + 1]
            pp_next = pp[:, t + 1][:, None]

            M_inv = np.linalg.inv(RRt + BBt.T @ PP_next @ BBt)
            m_t = rrt + BBt.T @ pp_next

            PP[:, :, t] = (
                AAt.T @ PP_next @ AAt
                - (BBt.T @ PP_next @ AAt + SSt).T @ M_inv @ (BBt.T @ PP_next @ AAt + SSt)
                + QQt
            )
            pp[:, t] = (
                AAt.T @ pp_next
                - (BBt.T @ PP_next @ AAt + SSt).T @ M_inv @ m_t
                + qqt
            ).ravel()

        # Calculate KK and sigma
        for t in range(TT - 1):
            AAt = AAin[:, :, t]
            BBt = BBin[:, :, t]
            RRt = RRin[:, :, t]
            SSt = SSin[:, :, t]
            PP_next = PP[:, :, t + 1]
            pp_next = pp[:, t + 1][:, None]

            M_inv = np.linalg.inv(RRt + BBt.T @ PP_next @ BBt)
            m_t = rrin[:, t][:, None] + BBt.T @ pp_next

            KK[:, :, t] = -M_inv @ (BBt.T @ PP_next @ AAt + SSt)
            sigma[:, t] = (-M_inv @ m_t).ravel()

        # Forward pass
        for t in range(TT - 1):
            uu[:, t] = KK[:, :, t] @ xx[:, t] + sigma[:, t]
            xx[:, t + 1] = AAin[:, :, t] @ xx[:, t] + BBin[:, :, t] @ uu[:, t]

        return KK, sigma, PP, xx, uu

    def newton_optimize(self, xx_ref, uu_ref, max_iters=15, threshold_grad=1e-4, use_armijo=True, show_plots_armijo=False):
        """
        Regularized Newton's method in closed-loop.
        
        Args:
            xx_ref: Reference state trajectory (ns x TT)
            uu_ref: Reference input trajectory (ni x TT)
            max_iters: Maximum optimization iterations
            threshold_grad: Gradient norm threshold for convergence
            use_armijo: Whether to use Armijo line search
            show_plots_armijo: Whether to plot Armijo line search
            
        Returns:
            xx: Optimized state trajectories
            uu: Optimized input trajectories
        """

        self.check_qt(xx_ref, uu_ref)

        TT = xx_ref.shape[1]
        
        # Initialize sequences
        xx = np.zeros((ns, TT, max_iters))
        uu = np.zeros((ni, TT-1, max_iters))

        # Set initial conditions
        x0 = xx_ref[:, 0]
        xx[:, 0, 0] = x0
        

        # Initialize first trajectory
        for tt in range(TT - 1):
            xx[:, tt+1, 0] = self.arm.discrete_dynamics(xx[:, tt, 0], uu[:, tt, 0])

        JJ = np.zeros(max_iters-1)

        for kk in range(max_iters-1):

            # Initialize matrices
            QQt = np.zeros((ns, ns, TT))
            RRt = np.zeros((ni, ni, TT))
            SSt = np.zeros((ni, ns, TT))
            AA = np.zeros((ns, ns, TT))
            BB = np.zeros((ns, ni, TT))
            qqt = np.zeros((ns, TT))
            rrt = np.zeros((ni, TT))
            lmbda = np.zeros_like(xx)
            grdJdu = np.zeros_like(uu)

            # Terminal cost and lambda initialization
            JJ[kk], qqT, QQt[:, :, -1] = self.termcost(xx[:, -1, kk], xx_ref[:, -1])
            lmbda[:, TT-1, kk] = qqT

            # Backward pass
            for tt in reversed(range(TT-1)):
                # Stage cost
                JJtemp, qqtemp, rrtemp, QQtemp, SStemp, RRtemp = self.stagecost(
                    xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt]
                )
                JJ[kk] += JJtemp

                # Store matrices
                QQt[:, :, tt] = QQtemp
                SSt[:, :, tt] = SStemp
                RRt[:, :, tt] = RRtemp
                qqt[:, tt] = qqtemp
                rrt[:, tt] = rrtemp

                # Dynamics linearization
                AA[:, :, tt], BB[:, :, tt] = self.arm.get_gradients(xx[:, tt, kk], uu[:, tt, kk])

                # Costate equation
                lmbda[:,tt, kk] = qqt[:,tt] + AA[:,:,tt].T @ lmbda[:,tt+1, kk]
                grdJdu[:,tt, kk] = rrt[:,tt] + BB[:,:,tt].T @ lmbda[:,tt+1, kk]

            # Check convergence
            normGrad = np.linalg.norm(grdJdu[:, :, kk])
            print(f"\nIteration {kk+1}: ")
            print("Current cost: ", JJ[kk])
            print("Current gradient norm: ", normGrad)

            if normGrad < threshold_grad:
                print("Optimization finished: gradient norm is below threshold")
                return xx[:, :, :kk], uu[:, :, :kk], JJ[:kk]

            # LQP solution
            KK, sigma, _, _, deltau = self.affine_lqp(AA, BB, QQt, RRt, SSt, self.cost.QT, TT, np.zeros_like(x0), qqt, rrt, qqT)

            # Step size selection
            if use_armijo:
                descent_arm = 0
                for tt in range(TT-1):
                    descent_arm += grdJdu[:, tt, kk].T @ deltau[:,tt]

                stepsize = self.select_stepsize(
                    xx_ref, uu_ref, x0, xx[:, :, kk], uu[:, :, kk], 
                    JJ[kk], descent_arm, 
                    KK, sigma, TT, show_plots_armijo
                )
            else:
                stepsize = self.fixed_stepsize

            # Forward pass: update input and state in closed-loop
            xx[:, 0, kk+1] = x0
            for tt in range(TT-1):
                uu[:, tt, kk+1] = (uu[:, tt, kk] + KK[:, :, tt] @ (xx[:, tt, kk+1] - xx[:, tt, kk]) + stepsize * sigma[:, tt])
                xx[:, tt+1, kk+1] = self.arm.discrete_dynamics(xx[:, tt, kk+1], uu[:, tt, kk+1])


        print("\nOptimization finished: maximum iterations reached")
        return xx, uu, JJ

    def plot_results(self, xx, uu, xx_ref, uu_ref):
        """
        Plot optimization results showing state and input trajectories.
        
        Args:
            xx: Optimized state trajectories (ns x TT x iterations)
            uu: Optimized input trajectories (ni x TT x iterations)
            xx_ref: Reference state trajectory (ns x TT)
            uu_ref: Reference input trajectory (ni x TT)
        """
        TT = xx_ref.shape[1]
        iterations = xx.shape[2]
        
        # Create figure with subplots
        fig, axs = plt.subplots(5, 1, figsize=(12, 15))
        fig.suptitle('Optimization Results', fontsize=15)
        
        # Plot state trajectories
        state_labels = ['θ₁', 'θ₂', 'ω₁', 'ω₂']
        for i in range(4):
            ax = axs[i]
            # Plot reference trajectory
            ax.plot(range(TT), xx_ref[i], 'r--', label='Reference', linewidth=2)
            
            # Plot optimization iterations with increasing opacity
            for k in range(iterations):
                alpha = 0.2 + 0.8 * (k / (iterations - 1))
                ax.plot(range(TT), xx[i, :, k], 'b-', alpha=alpha, linewidth=1)
            
            # Plot final trajectory
            ax.plot(range(TT), xx[i, :, -1], 'g-', label='Optimized', linewidth=2)
            
            ax.set_xlabel('Time step')
            ax.set_ylabel(f'{state_labels[i]}')
            ax.grid(True)
            ax.legend()
        
        # Plot input trajectory
        ax = axs[4]
        ax.plot(range(TT-1), uu_ref[0], 'r--', label='Reference', linewidth=2)
        
        # Plot optimization iterations with increasing opacity
        for k in range(iterations):
            alpha = 0.2 + 0.8 * (k / (iterations - 1))
            ax.plot(range(TT-1), uu[0, :, k], 'b-', alpha=alpha, linewidth=1)
            
        ax.plot(range(TT-1), uu[0, :, -1], 'g-', label='Optimized', linewidth=2)
        ax.set_xlabel('Time step')
        ax.set_ylabel('Input torque')
        ax.grid(True)
        ax.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_convergence(self, costs):
        """
        Plot the convergence of the cost function over iterations.
        
        Args:
            costs: Array of costs for each iteration
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(costs)), costs, 'b-o')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('Cost Convergence')
        plt.grid(True)
        plt.yscale('log')
        plt.show()
