import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp
from parameters import m1, m2, l1, l2, r1, r2, I1, I2, g, f1, f2, dt, ns, ni

class FlexibleRoboticArm:
    def __init__(self):
        # System parameters
        self.m1 = m1
        self.m2 = m2 
        self.l1 = l1
        self.l2 = l2
        self.r1 = r1
        self.r2 = r2
        self.I1 = I1
        self.I2 = I2
        self.g = g
        self.f1 = f1
        self.f2 = f2
        self.dt = dt
        self.ns = ns
        self.ni = ni
        
        # Create symbolic variables for states and inputs
        self.setup_symbolic_variables()
        
        # Compute symbolic expressions
        self.setup_symbolic_expressions()
        
    def setup_symbolic_variables(self):
        """Create symbolic variables for states and inputs."""
        # States: theta1, theta2, dtheta1, dtheta2
        self.theta1_sym = sp.Symbol('theta1')
        self.theta2_sym = sp.Symbol('theta2')  
        self.dtheta1_sym = sp.Symbol('dtheta1')
        self.dtheta2_sym = sp.Symbol('dtheta2')
        
        # Input torque
        self.u_sym = sp.Symbol('u')
        
        # State vector
        self.x_sym = sp.Matrix([self.theta1_sym, self.theta2_sym, 
                              self.dtheta1_sym, self.dtheta2_sym])
        
    def setup_symbolic_expressions(self):
        """Compute symbolic expressions for dynamics and derivatives."""
        # Mass matrix
        M = sp.Matrix([
            [self.I1 + self.I2 + self.m1*self.r1**2 + 
             self.m2*(self.l1**2 + self.r2**2) + 
             2*self.m2*self.l1*self.r2*sp.cos(self.theta2_sym),
             self.I2 + self.m2*self.r2**2 + 
             self.m2*self.l1*self.r2*sp.cos(self.theta2_sym)],
            [self.I2 + self.m2*self.r2**2 + 
             self.m2*self.l1*self.r2*sp.cos(self.theta2_sym),
             self.I2 + self.m2*self.r2**2]
        ])
        
        # Coriolis vector
        C = sp.Matrix([
            -self.m2*self.l1*self.r2*self.dtheta2_sym*sp.sin(self.theta2_sym)*
            (self.dtheta2_sym + 2*self.dtheta1_sym),
            self.m2*self.l1*self.r2*sp.sin(self.theta2_sym)*self.dtheta1_sym**2
        ])
        
        # Gravity vector
        G = sp.Matrix([
            self.g*(self.m1*self.r1 + self.m2*self.l1)*sp.sin(self.theta1_sym) + 
            self.g*self.m2*self.r2*sp.sin(self.theta1_sym + self.theta2_sym),
            self.g*self.m2*self.r2*sp.sin(self.theta1_sym + self.theta2_sym)
        ])
        
        # Friction matrix
        F = sp.Matrix([[self.f1, 0], [0, self.f2]])
        
        # Dynamics equation
        ddtheta = M.inv() * (sp.Matrix([self.u_sym, 0]) - C - 
                            F*sp.Matrix([self.dtheta1_sym, self.dtheta2_sym]) - G)
        
        # Full dynamics
        self.f_sym = sp.Matrix([
            self.dtheta1_sym,
            self.dtheta2_sym,
            ddtheta[0],
            ddtheta[1]
        ])
        
        # Compute gradients
        self.compute_symbolic_gradients()
        
    def compute_symbolic_gradients(self):
        """Compute symbolic gradients and Hessians."""
        # Gradient with respect to state
        self.grad_x_sym = self.f_sym.jacobian(self.x_sym)
        
        # Gradient with respect to input
        self.grad_u_sym = self.f_sym.diff(self.u_sym)
        
        # Hessian with respect to state
        self.hess_x_sym = sp.Matrix([[self.grad_x_sym.diff(x) for x in self.x_sym]])
        
        # Hessian with respect to input 
        self.hess_u_sym = self.grad_u_sym.diff(self.u_sym)
        
        # Mixed Hessian
        self.hess_xu_sym = sp.Matrix([[self.grad_u_sym.diff(x) for x in self.x_sym]])
        
        # Create lambda functions for fast numerical evaluation
        self.create_lambda_functions()
        
    def create_lambda_functions(self):
        """Create lambda functions for numerical evaluation."""
        # Convert symbolic expressions to lambda functions
        self.grad_x_fn = sp.lambdify(
            (self.theta1_sym, self.theta2_sym, self.dtheta1_sym, 
             self.dtheta2_sym, self.u_sym),
            self.grad_x_sym
        )
        
        self.grad_u_fn = sp.lambdify(
            (self.theta1_sym, self.theta2_sym, self.dtheta1_sym, 
             self.dtheta2_sym, self.u_sym),
            self.grad_u_sym
        )
        
        self.hess_x_fn = sp.lambdify(
            (self.theta1_sym, self.theta2_sym, self.dtheta1_sym, 
             self.dtheta2_sym, self.u_sym),
            self.hess_x_sym
        )
        
        self.hess_u_fn = sp.lambdify(
            (self.theta1_sym, self.theta2_sym, self.dtheta1_sym, 
             self.dtheta2_sym, self.u_sym),
            self.hess_u_sym
        )
        
        self.hess_xu_fn = sp.lambdify(
            (self.theta1_sym, self.theta2_sym, self.dtheta1_sym, 
             self.dtheta2_sym, self.u_sym),
            self.hess_xu_sym
        )
        
    def get_gradients(self, x, u):
        """
        Compute gradients at given state and input.
        
        Args:
            x: State vector [theta1, theta2, dtheta1, dtheta2]
            u: Input torque (scalar)
            
        Returns:
            grad_x: Gradient with respect to state
            grad_u: Gradient with respect to input
        """
        grad_x = np.array(self.grad_x_fn(x[0], x[1], x[2], x[3], u[0]))
        grad_u = np.array(self.grad_u_fn(x[0], x[1], x[2], x[3], u[0]))
        return grad_x, grad_u
    
    def get_hessians(self, x, u):
        """
        Compute Hessians at given state and input.
        
        Args:
            x: State vector [theta1, theta2, dtheta1, dtheta2]
            u: Input torque (scalar)
            
        Returns:
            hess_x: Hessian with respect to state
            hess_u: Hessian with respect to input
            hess_xu: Mixed Hessian
            hess_ux: Mixed Hessian transpose
        """
        hess_x = np.array(self.hess_x_fn(x[0], x[1], x[2], x[3], u[0]))
        hess_u = np.array(self.hess_u_fn(x[0], x[1], x[2], x[3], u[0]))
        hess_xu = np.array(self.hess_xu_fn(x[0], x[1], x[2], x[3], u[0]))
        hess_ux = hess_xu.T
        return hess_x, hess_u, hess_xu, hess_ux

    def mass_matrix(self, theta2):
        """Compute mass matrix M(theta)."""
        M = np.zeros((2, 2))
        M[0,0] = (self.I1 + self.I2 + self.m1*self.r1**2 +
                  self.m2*(self.l1**2 + self.r2**2) +
                  2*self.m2*self.l1*self.r2*np.cos(theta2))
        M[0,1] = self.I2 + self.m2*self.r2**2 + self.m2*self.l1*self.r2*np.cos(theta2)
        M[1,0] = M[0,1]
        M[1,1] = self.I2 + self.m2*self.r2**2
        return M

    def coriolis_vector(self, theta2, dtheta1, dtheta2):
        """Compute Coriolis vector C(theta,dtheta)."""
        C = np.zeros(2)
        C[0] = -self.m2*self.l1*self.r2*dtheta2*np.sin(theta2)*(dtheta2 + 2*dtheta1)
        C[1] = self.m2*self.l1*self.r2*np.sin(theta2)*dtheta1**2
        return C

    def gravity_vector(self, theta1, theta2):
        """Compute gravity vector G(theta)."""
        G = np.zeros(2)
        G[0] = (self.g*(self.m1*self.r1 + self.m2*self.l1)*np.sin(theta1) +
                self.g*self.m2*self.r2*np.sin(theta1 + theta2))
        G[1] = self.g*self.m2*self.r2*np.sin(theta1 + theta2)
        return G

    def friction_matrix(self):
        """Compute friction matrix F."""
        return np.diag([self.f1, self.f2])

    def continuous_dynamics(self, x, u):
        """Compute continuous time dynamics."""
        theta1, theta2, dtheta1, dtheta2 = x
        M = self.mass_matrix(theta2)
        C = self.coriolis_vector(theta2, dtheta1, dtheta2)
        G = self.gravity_vector(theta1, theta2)
        F = self.friction_matrix()
        
        ddtheta = np.linalg.solve(
            M,
            np.array([u[0], 0]) - C - F@np.array([dtheta1, dtheta2]) - G
        )
        return np.array([dtheta1, dtheta2, ddtheta[0], ddtheta[1]])

    def discrete_dynamics(self, x, u):
        """Compute discrete time dynamics using RK45 integration."""
        def system(t, x):
            return self.continuous_dynamics(x, u)
            
        sol = solve_ivp(
            system,
            t_span=[0, self.dt],
            y0=x,
            method='RK45'
        )
        return sol.y[:,-1]
    

if __name__ == "__main__":
    # Create an instance of the FlexibleRoboticArm
    arm = FlexibleRoboticArm()
    
    # Initial state: [theta1, theta2, dtheta1, dtheta2]
    x0 = [0.0, 0.0, 0.0, 0.0]
    u = [1.0]  # Input torque
    
    # Test 1: Basic dynamics
    x_next = arm.discrete_dynamics(x0, u)
    print("Test 1 - Next state (x0 = [0, 0, 0, 0], u = 1):", x_next)
    
    # Test 2: Different initial state
    x0 = [0.1, -0.1, 0.5, -0.5]
    x_next = arm.discrete_dynamics(x0, u)
    print("Test 2 - Next state (x0 = [0.1, -0.1, 0.5, -0.5], u = 1):", x_next)
    
    # Test 3: Zero input torque
    x0 = [0.0, 0.0, 0.0, 0.0]
    u = [0.0]
    x_next = arm.discrete_dynamics(x0, u)
    print("Test 3 - Next state (x0 = [0, 0, 0, 0], u = 0):", x_next)
    
    # Test 4: Steady-state response
    x0 = [np.pi/4, np.pi/4, 0.0, 0.0]
    u = [5.0]
    x_next = arm.discrete_dynamics(x0, u)
    print("Test 4 - Next state (x0 = [pi/4, pi/4, 0, 0], u = 5):", x_next)
    
    # Test 5: Gradients at a specific state
    x0 = [0.2, 0.1, -0.1, 0.2]
    u = [2.0]
    grad_x, grad_u = arm.get_gradients(x0, u)
    print("Test 5 - Gradients at x0 =", x0, "and u =", u)
    print("    Gradient wrt state (grad_x):", grad_x)
    print("    Gradient wrt input (grad_u):", grad_u)
    
    # Test 6: Hessians at a specific state
    hess_x, hess_u, hess_xu, hess_ux = arm.get_hessians(x0, u)
    print("Test 6 - Hessians at x0 =", x0, "and u =", u)
    print("    Hessian wrt state (hess_x):", hess_x)
    print("    Hessian wrt input (hess_u):", hess_u)
    print("    Mixed Hessian (hess_xu):", hess_xu)
    print("    Mixed Hessian transpose (hess_ux):", hess_ux)
