import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp
from parameters import m1, m2, l1, l2, r1, r2, I1, I2, g, f1, f2, dt, ns, ni

# Define symbolic variables and matrices for gradient computation
x1, x2, x3, x4, u, a, b, c, d, f1_sym, f2_sym, z, e = sp.symbols('x1 x2 x3 x4 u a b c d f1 f2 z e')
x = sp.Matrix([[x1], [x2], [x3], [x4]])

# Define symbolic matrices
M = sp.Matrix([[a + b*sp.cos(x2), c + d*sp.cos(x2)],
               [c + d*sp.cos(x2), c]])
C = sp.Matrix([[-d*x4*sp.sin(x2)*(x4 + 2*x3)],
               [d*sp.sin(x2)*x3*x3]])
G = sp.Matrix([[z*sp.sin(x1) + e*sp.sin(x1 + x2)],
               [e*sp.sin(x1 + x2)]])
F = sp.Matrix([[f1_sym, 0],
               [0, f2_sym]])
U = sp.Matrix([[u], [0]])

# Define dynamics
f34 = M.inv()*(U - C - F*sp.Matrix([[x3], [x4]]) - G)
f12 = sp.Matrix([[x3], [x4]])
f = sp.Matrix.vstack(f12, f34)

# Compute gradients
grad = f.jacobian(x)
grad_u = f.diff(u)

# Compute Hessians
hess_xx = [grad[:, i].jacobian(x).T for i in range(ns)]
hess_xu = sp.Matrix([grad[:,i].diff(u) for i in range(ns)]).reshape(4,4).T
hess_ux = grad_u.jacobian(x).T
hess_uu = grad_u.diff(u).T

# Lambdify expressions
pars = sp.Matrix([[x1], [x2], [x3], [x4], [u], [a], [b], [c], [d], [f1_sym], [f2_sym], [z], [e]])
lgrad = sp.lambdify(pars, grad)
lgrad_u = sp.lambdify(pars, grad_u)
lhess_xx = [sp.lambdify(pars, hess_xx[i]) for i in range(ns)]
lhess_xu = sp.lambdify(pars, hess_xu)
lhess_ux = sp.lambdify(pars, hess_ux)
lhess_uu = sp.lambdify(pars, hess_uu)

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

    def discrete_dynamics(self, x, u, method='euler'):
        """Compute discrete time dynamics using the specified integration method.
        
        Parameters:
        x (array-like): State vector [theta1, theta2, dtheta1, dtheta2]
        u (array-like): Control input [u1]
        method (str): Integration method, 'euler' or 'rk45'. Default is 'euler'.
        
        Returns:
        np.ndarray: Next state vector
        """
        if method == 'euler':
            dx = self.continuous_dynamics(x, u)
            return np.array(x) + self.dt * dx
        
        elif method == 'rk45':
            def system(t, x):
                return self.continuous_dynamics(x, u)
                
            sol = solve_ivp(
                system,
                t_span=[0, self.dt],
                y0=x,
                method='RK45'
            )
            return sol.y[:, -1]
        else:
            raise ValueError("Unsupported integration method. Choose 'euler' or 'rk45'.")

    def get_gradients(self, x, u):
        """
        Compute the gradient of the discrete dynamics with respect to state and input.
        
        Parameters:
        x (np.ndarray): State vector [theta1, theta2, dtheta1, dtheta2]
        u (np.ndarray): Control input [u1]
        
        Returns:
        tuple: (gradx, gradu) where:
            gradx is the gradient with respect to state (4x4 matrix)
            gradu is the gradient with respect to input (4x1 matrix)
        """
        # System parameters for symbolic computation
        a = self.I1 + self.I2 + self.m1*self.r1**2 + self.m2*(self.l1**2 + self.r2**2)
        b = 2*self.m2*self.l1*self.r2
        c = self.I2 + self.m2*self.r2**2
        d = self.m2*self.l1*self.r2
        z = self.g*(self.m1*self.r1 + self.m2*self.l1)
        e = self.g*self.m2*self.r2
        
        # Stack parameters for lambdified functions
        params = [*x, *u, a, b, c, d, self.f1, self.f2, z, e]
        
        # Compute gradients using lambdified functions
        gradx = np.identity(self.ns) + self.dt * np.array(lgrad(*params))
        gradu = self.dt * np.array(lgrad_u(*params))
        
        return gradx, gradu

    def get_hessians(self, x, u):
        """
        Compute the Hessian matrices of the discrete dynamics.
        
        Parameters:
        x (np.ndarray): State vector [theta1, theta2, dtheta1, dtheta2]
        u (np.ndarray): Control input [u1]
        
        Returns:
        tuple: (hess_xx, hess_xu, hess_ux, hess_uu) where:
            hess_xx is the Hessian with respect to state (4x4x4 tensor)
            hess_xu is the mixed Hessian (4x4 matrix)
            hess_ux is the mixed Hessian (4x4 matrix)
            hess_uu is the Hessian with respect to input (4x4 matrix)
        """
        # System parameters for symbolic computation
        a = self.I1 + self.I2 + self.m1*self.r1**2 + self.m2*(self.l1**2 + self.r2**2)
        b = 2*self.m2*self.l1*self.r2
        c = self.I2 + self.m2*self.r2**2
        d = self.m2*self.l1*self.r2
        z = self.g*(self.m1*self.r1 + self.m2*self.l1)
        e = self.g*self.m2*self.r2
        
        # Stack parameters for lambdified functions
        params = [*x, *u, a, b, c, d, self.f1, self.f2, z, e]
        
        # Compute Hessians using lambdified functions
        hess_xx = np.array([lhess_xx[i](*params) for i in range(self.ns)])
        hess_xu = np.array(lhess_xu(*params))
        hess_ux = np.array(lhess_ux(*params))
        hess_uu = np.array(lhess_uu(*params))
        
        # # Scale by dt for discrete dynamics
        # hess_xx = self.dt * hess_xx
        # hess_xu = self.dt * hess_xu
        # hess_ux = self.dt * hess_ux
        # hess_uu = self.dt * hess_uu
        
        return hess_xx, hess_xu, hess_ux, hess_uu

if __name__ == "__main__":
    # Create an instance of the FlexibleRoboticArm
    arm = FlexibleRoboticArm()
    
    # Initial state: [theta1, theta2, dtheta1, dtheta2]
    x0 = [0.0, 0.0, 0.0, 0.0]
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
    x0 = [0.0, 0.0, 0.0, 0.0]
    u = [2.0]
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
