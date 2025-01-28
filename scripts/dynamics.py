import numpy as np
from scipy.integrate import solve_ivp
import sympy as sy
from typing import Tuple
import numpy.typing as npt
from parameters import m1, m2, l1, l2, r1, r2, I1, I2, g, f1, f2, dt, ns, ni

class FlexibleRoboticArm:
    def __init__(self, m1: float, m2: float, l1: float, l2: float, r1: float, r2: float, 
                 I1: float, I2: float, g: float, f1: float, f2: float, dt: float, 
                 ns: int = 4, ni: int = 1):
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
        """
        self.params = {
            'm1': m1, 'm2': m2, 'l1': l1, 'l2': l2, 
            'r1': r1, 'r2': r2, 'I1': I1, 'I2': I2,
            'g': g, 'f1': f1, 'f2': f2, 'dt': dt
        }
        self.ns = ns
        self.ni = ni
        
        # Pre-compute constant system parameters
        self._compute_system_constants()
        
        # Initialize symbolic variables for gradients and hessians
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
                        u: npt.NDArray[np.float64], 
                        method: str = 'rk45') -> npt.NDArray[np.float64]:
        """
        Compute discrete time dynamics using specified integration method.
        
        Args:
            x: State vector
            u: Control input
            method: Integration method ('euler' or 'rk45')
            
        Returns:
            Next state vector
        """
        if method == 'euler':
            return x + self.params['dt'] * self.continuous_dynamics(x, u)
        elif method == 'rk45':
            sol = solve_ivp(
                lambda t, x: self.continuous_dynamics(x, u),
                t_span=[0, self.params['dt']],
                y0=x,
                method='RK45',
                rtol=1e-6,
                atol=1e-6
            )
            return sol.y[:, -1]
        else:
            raise ValueError("Unsupported integration method. Choose 'euler' or 'rk45'.")

    def _init_symbolic_vars(self) -> None:
        """Initialize symbolic variables for gradient and Hessian computation."""
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
        
        # Pre-compute Hessians
        self._sym_hess_xx = [self._sym_grad[:, i].jacobian(x).T for i in range(self.ns)]
        self._sym_hess_xu = sy.Matrix([self._sym_grad[:,i].diff(u) for i in range(self.ns)]).reshape(4,4).T
        self._sym_hess_ux = self._sym_grad_u.jacobian(x).T
        self._sym_hess_uu = self._sym_grad_u.diff(u).T
        
        # Create lambdified functions
        self._create_lambda_functions()

    def _create_lambda_functions(self) -> None:
        """Create lambda functions for gradients and Hessians."""
        pars = sy.Matrix([[*self._sym_vars]])
        
        self._lambda_grad = sy.lambdify(pars, self._sym_grad)
        self._lambda_grad_u = sy.lambdify(pars, self._sym_grad_u)
        self._lambda_hess_xx = [sy.lambdify(pars, h) for h in self._sym_hess_xx]
        self._lambda_hess_xu = sy.lambdify(pars, self._sym_hess_xu)
        self._lambda_hess_ux = sy.lambdify(pars, self._sym_hess_ux)
        self._lambda_hess_uu = sy.lambdify(pars, self._sym_hess_uu)

    def get_gradients(self, state: npt.NDArray[np.float64], 
                     in_u: npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.float64], 
                                                           npt.NDArray[np.float64]]:
        """
        Compute gradients of the dynamics with respect to state and input.
        
        Args:
            state: State vector
            in_u: Control input
            
        Returns:
            Tuple of gradients with respect to state and input
        """
        sys_params = (self._a, self._b, self._c, self._d, 
                     self.params['f1'], self.params['f2'], 
                     self._z, self._e)
        
        gradx = np.identity(self.ns) + self.params['dt'] * np.array(
            self._lambda_grad(*state, *in_u, *sys_params)
        )
        gradu = self.params['dt'] * np.array(
            self._lambda_grad_u(*state, *in_u, *sys_params)
        )
        return gradx, gradu

    def get_hessians(self, state: npt.NDArray[np.float64], 
                    in_u: npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.float64], ...]:
        """
        Compute Hessians of the dynamics with respect to state and input.
        
        Args:
            state: State vector
            in_u: Control input
            
        Returns:
            Tuple of Hessians (xx, xu, ux, uu)
        """
        sys_params = (self._a, self._b, self._c, self._d, 
                     self.params['f1'], self.params['f2'], 
                     self._z, self._e)
        
        return (
            np.array([self._lambda_hess_xx[i](*state, *in_u, *sys_params) 
                     for i in range(self.ns)]),
            np.array(self._lambda_hess_xu(*state, *in_u, *sys_params)),
            np.array(self._lambda_hess_ux(*state, *in_u, *sys_params)),
            np.array(self._lambda_hess_uu(*state, *in_u, *sys_params))
        )

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
    
    # Test 6: Hessians at a specific state
    hess_x, hess_u, hess_xu, hess_ux = arm.get_hessians(x0, u)
    # print("Test 6 - Hessians at x0 =\n", x0, "\nand u =\n", u)
    print("    Hessian wrt state (hess_x):\n", hess_x)
    print("    Hessian wrt input (hess_u):\n", hess_u)
    print("    Mixed Hessian (hess_xu):\n", hess_xu)
    print("    Mixed Hessian transpose (hess_ux):\n", hess_ux)
