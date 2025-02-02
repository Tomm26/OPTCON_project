import numpy as np
from scipy.integrate import solve_ivp
import sympy as sy
from typing import Tuple
import numpy.typing as npt
from parameters import m1, m2, l1, l2, r1, r2, I1, I2, g, f1, f2, dt, ns, ni

class FlexibleRoboticArm:
    def __init__(self, m1: float, m2: float, l1: float, l2: float, r1: float, r2: float, 
                 I1: float, I2: float, g: float, f1: float, f2: float, dt: float, 
                 ns: int = 4, ni: int = 1, method: str = 'euler'):
        """
        Initialize the Flexible Robotic Arm system with given parameters.
        
        Args:
            m1, m2: Masses of links 1 and 2
            l1, l2: Lengths of links 1 and 2
            r1, r2: Distances to center of mass for links 1 and 2
            I1, I2: Moments of inertia for links 1 and 2
            g: Gravitational constant
            f1, f2: Friction coefficients
            dt: Time step
            ns: Number of states (default: 4)
            ni: Number of inputs (default: 1)
            method: Method of discretization ('euler' or 'rk'), rk use RK23
        """
        self.params = {
            'm1': m1, 'm2': m2, 'l1': l1, 'l2': l2, 
            'r1': r1, 'r2': r2, 'I1': I1, 'I2': I2,
            'g': g, 'f1': f1, 'f2': f2, 'dt': dt
        }
        self.ns = ns
        self.ni = ni
        self.method = method
        
        # Pre-compute constant system parameters
        self._compute_system_constants()
        
        # Initialize symbolic variables for gradients
        self._init_symbolic_vars()

    def _compute_system_constants(self) -> None:
        """Pre-compute constant parameters used in dynamics calculations."""
        self._a = (self.params['I1'] + self.params['I2'] + 
                  self.params['m1'] * self.params['r1']**2 + 
                  self.params['m2'] * (self.params['l1']**2 + self.params['r2']**2))
        self._b = 2 * self.params['m2'] * self.params['l1'] * self.params['r2']
        self._c = self.params['I2'] + self.params['m2'] * self.params['r2']**2
        self._d = self.params['m2'] * self.params['l1'] * self.params['r2']
        self._z = self.params['g'] * (self.params['m1'] * self.params['r1'] + 
                                    self.params['m2'] * self.params['l1'])
        self._e = self.params['g'] * self.params['m2'] * self.params['r2']
        self._friction_matrix = np.diag([self.params['f1'], self.params['f2']])

    def mass_matrix(self, theta2: float) -> npt.NDArray[np.float64]:
        """
        Compute mass matrix M(theta) with caching for repeated calculations.
        
        Args:
            theta2: Second joint angle
            
        Returns:
            2x2 mass matrix
        """
        cos_theta2 = np.cos(theta2)
        M = np.zeros((2, 2))
        M[0, 0] = self._a + self._b * cos_theta2
        M[0, 1] = M[1, 0] = self._c + self._d * cos_theta2
        M[1, 1] = self._c
        return M

    def coriolis_vector(self, theta2: float, dtheta1: float, dtheta2: float) -> npt.NDArray[np.float64]:
        """
        Compute Coriolis vector C(theta,dtheta).
        
        Args:
            theta2: Second joint angle
            dtheta1, dtheta2: Joint angular velocities
            
        Returns:
            2D Coriolis vector
        """
        sin_theta2 = np.sin(theta2)
        return np.array([
            -self._d * sin_theta2 * dtheta2 * (dtheta2 + 2 * dtheta1),
            self._d * sin_theta2 * dtheta1**2
        ])

    def gravity_vector(self, theta1: float, theta2: float) -> npt.NDArray[np.float64]:
        """
        Compute gravity vector G(theta).
        
        Args:
            theta1, theta2: Joint angles
            
        Returns:
            2D gravity vector
        """
        sin_theta1 = np.sin(theta1)
        sin_sum = np.sin(theta1 + theta2)
        return np.array([
            self._z * sin_theta1 + self._e * sin_sum,
            self._e * sin_sum
        ])

    def continuous_dynamics(self, x: npt.NDArray[np.float64], 
                          u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Compute continuous time dynamics efficiently.
        
        Args:
            x: State vector [theta1, theta2, dtheta1, dtheta2]
            u: Control input [u1]
            
        Returns:
            State derivative vector
        """
        theta1, theta2, dtheta1, dtheta2 = x
        
        # Get dynamics components
        M = self.mass_matrix(theta2)
        C = self.coriolis_vector(theta2, dtheta1, dtheta2)
        G = self.gravity_vector(theta1, theta2)
        
        # Compute acceleration
        ddtheta = np.linalg.solve(
            M,
            np.array([u[0], 0]) - C - self._friction_matrix @ np.array([dtheta1, dtheta2]) - G
        )
        
        return np.array([dtheta1, dtheta2, ddtheta[0], ddtheta[1]])

    def discrete_dynamics(self, x: npt.NDArray[np.float64], 
                        u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Compute discrete time dynamics using specified integration method.
        
        Args:
            x: State vector
            u: Control input
            method: Integration method ('euler' or 'rk')
            
        Returns:
            Next state vector
        """
        if self.method.lower() == 'euler':
            return x + self.params['dt'] * self.continuous_dynamics(x, u)
        elif self.method.lower() == 'rk':
            sol = solve_ivp(
                lambda t, x: self.continuous_dynamics(x, u),
                t_span=[0, self.params['dt']],
                y0=x,
                method='RK23',
                rtol=1e-6,
                atol=1e-6
            )
            return sol.y[:, -1]
        else:
            raise ValueError("Unsupported integration method. Choose 'euler' or 'rk'.")

    def linearize_over_traj(self, x_traj, u_traj):
        """
        Linearize the dynamics along the feasible x_traj, u_traj over a finite horizon
        x_traj: (ns x TT)
        u_traj: (ni x TT)
        """
        T = x_traj.shape[-1]

        AA_traj = np.zeros((self.ns, self.ns, T)) # tensor storing the A matrices along t
        BB_traj = np.zeros((self.ns, self.ni, T)) # tensor storing the B matrices along t
        
        for t in range(T):
            AA_traj[:,:, t], BB_traj[:,:, t] = self.get_gradients(x_traj[:, t], u_traj[:, t])

        return AA_traj, BB_traj

    def _augmented_ode(self, t, aug, in_u, sys_params):
        ns = self.ns
        x = aug[:ns]
        X = aug[ns:ns + ns * ns].reshape(ns, ns)
        Y = aug[ns + ns * ns:].reshape(ns, self.ni)
        
        dxdt = self.continuous_dynamics(x, in_u)
        A = np.array(self._lambda_grad(*x, *in_u, *sys_params))
        B = np.array(self._lambda_grad_u(*x, *in_u, *sys_params))
        
        dXdt = A @ X
        dYdt = A @ Y + B
        
        return np.concatenate([dxdt, dXdt.flatten(), dYdt.flatten()])

    def get_gradients(self, 
                    state: npt.NDArray[np.float64], 
                    in_u: npt.NDArray[np.float64]
                    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Compute gradients of the dynamics with respect to state and input
        using either Euler or RK23 integration.
        
        Args:
            state: Current state vector.
            in_u: Control input vector.
        
        Returns:
            Tuple (gradx, gradu) where:
            gradx: Discretized gradient ∂x₊/∂x (ns x ns)
            gradu: Discretized gradient ∂x₊/∂u (ns x ni)
        """
        sys_params = (self._a, self._b, self._c, self._d, 
                    self.params['f1'], self.params['f2'], 
                    self._z, self._e)
        dt = self.params['dt']
        ns = self.ns
        ni = self.ni

        if self.method.lower() == 'euler':
            # Euler discretization for the gradients:
            # gradx ≈ I + dt * ∂f/∂x
            # gradu ≈ dt * ∂f/∂u
            gradx = np.eye(ns) + dt * np.array(
                self._lambda_grad(*state, *in_u, *sys_params)
            )
            gradu = dt * np.array(
                self._lambda_grad_u(*state, *in_u, *sys_params)
            )
            return gradx, gradu

        elif self.method.lower() == 'rk':
            # Use RK23 integration on the augmented system.
            # The augmented state includes:
            #   - the state x (length ns),
            #   - X = ∂x/∂x₀ (ns x ns, flattened),
            #   - Y = ∂x/∂u (ns x ni, flattened).
            x0 = state
            X0 = np.eye(ns)
            Y0 = np.zeros((ns, ni))
            aug0 = np.concatenate([x0, X0.flatten(), Y0.flatten()])

            # Integrate the augmented system over [0, dt] using RK23.
            sol = solve_ivp(
                lambda t, aug: self._augmented_ode(t, aug, in_u, sys_params),
                t_span=[0, dt],
                y0=aug0,
                method='RK23',
                t_eval=[dt],
                rtol=1e-6,
                atol=1e-6
            )
            aug_final = sol.y[:, -1]
            gradx = aug_final[ns:ns + ns * ns].reshape(ns, ns)
            gradu = aug_final[ns + ns * ns:].reshape(ns, ni)
            return gradx, gradu

        else:
            raise ValueError("Unsupported integration method. Use 'euler' or 'rk'.")

    def _init_symbolic_vars(self) -> None:
        """Initialize symbolic variables for gradient computation."""
        # Create symbolic variables
        self._sym_vars = sy.symbols('x1 x2 x3 x4 u a b c d f1 f2 z e')
        x1, x2, x3, x4, u, a, b, c, d, f1, f2, z, e = self._sym_vars
        
        # Create state vector
        x = sy.Matrix([[x1], [x2], [x3], [x4]])
        
        # Define symbolic matrices
        M = sy.Matrix([[a + b*sy.cos(x2), c + d*sy.cos(x2)], 
                      [c + d*sy.cos(x2), c]])
        C = sy.Matrix([[-d*x4*sy.sin(x2)*(x4 + 2*x3)], 
                      [d*sy.sin(x2)*x3*x3]])
        G = sy.Matrix([[z*sy.sin(x1) + e*sy.sin(x1 + x2)],
                      [e*sy.sin(x1 + x2)]])
        F = sy.Matrix([[f1, 0], [0, f2]])
        U = sy.Matrix([[u], [0]])
        
        # Compute dynamics
        f34 = M.inv()*(U - C - F*sy.Matrix([[x3], [x4]]) - G)
        f12 = sy.Matrix([[x3], [x4]])
        self._sym_f = sy.Matrix.vstack(f12, f34)
        
        # Pre-compute gradients
        self._sym_grad = self._sym_f.jacobian(x)
        self._sym_grad_u = self._sym_f.diff(u)
        
        # Create lambdified functions
        self._create_lambda_functions()

    def _create_lambda_functions(self) -> None:
        """Create lambda functions for gradients."""
        pars = sy.Matrix([[*self._sym_vars]])
        
        self._lambda_grad = sy.lambdify(pars, self._sym_grad, modules='numpy')
        self._lambda_grad_u = sy.lambdify(pars, self._sym_grad_u, modules='numpy')


if __name__ == "__main__":
    # Create an instance of the FlexibleRoboticArm
    arm = FlexibleRoboticArm(m1, m2, l1, l2, r1, r2, I1, I2, g, f1, f2, dt, ns, ni)
    
    # Initial state: [theta1, theta2, dtheta1, dtheta2]
    x0 = [np.pi/4, np.pi/4, 0.0, 0.0]
    u = [1.0]  # Input torque
    
    # Test 1: Basic dynamics
    x_next = arm.discrete_dynamics(x0, u)
    print("Test 1 - Next state (x0 = [0, 0, 0, 0], u = 1):\n", x_next)
    
    # # Test 2: Different initial state
    # x0 = [0.1, -0.1, 0.5, -0.5]
    # x_next = arm.discrete_dynamics(x0, u)
    # print("Test 2 - Next state (x0 = [0.1, -0.1, 0.5, -0.5], u = 1):", x_next)
    
    # # Test 3: Zero input torque
    # x0 = [0.0, 0.0, 0.0, 0.0]
    # u = [0.0]
    # x_next = arm.discrete_dynamics(x0, u)
    # print("Test 3 - Next state (x0 = [0, 0, 0, 0], u = 0):", x_next)
    
    # # Test 4: Steady-state response
    # x0 = [np.pi/4, np.pi/4, 0.0, 0.0]
    # u = [5.0]
    # x_next = arm.discrete_dynamics(x0, u)
    # print("Test 4 - Next state (x0 = [pi/4, pi/4, 0, 0], u = 5):", x_next)
    
    # Test 5: Gradients at a specific state
    # x0 = [0.0, 0.0, 0.0, 0.0]
    # u = [1.0]
    grad_x, grad_u = arm.get_gradients(x0, u)
    # print("Test 5 - Gradients at x0 =\n", x0, "\nand u =\n", u)
    print("    Gradient wrt state (grad_x):\n", grad_x)
    print("    Gradient wrt input (grad_u):\n", grad_u)