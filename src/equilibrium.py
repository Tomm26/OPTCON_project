import numpy as np
from scipy.optimize import fsolve
from .dynamics import dynamics as dyn

def find_equilibrium(theta1_desired, theta2_desired):
    """
    Finds the equilibrium point for given desired angles.
    Uses fsolve to find the equilibrium where accelerations are zero.
    
    Args:
        theta1_desired: Desired angle for first joint
        theta2_desired: Desired angle for second joint
        
    Returns:
        x_eq: Equilibrium state [theta1, theta2, 0, 0]
        u_eq: Required input torque at equilibrium
    """
    def equilibrium_equations(vars):
        theta1, theta2, u = vars
        # State at equilibrium (velocities are zero)
        x = np.array([theta1, theta2, 0, 0])
        
        # Get accelerations from dynamics
        x_next = dyn(x, np.array([u]))[0]
        
        # At equilibrium, velocities should be constant (accelerations zero)
        # and positions should match desired values
        return [
            x_next[2],                 # Zero acceleration 1
            x_next[3],                 # Zero acceleration 2
            theta1 - theta1_desired,   # Position error 1
        ]
    
    # Initial guess - use desired angles and zero input
    x0 = [theta1_desired, theta2_desired, 0]
    
    # Solve equilibrium equations
    sol = fsolve(equilibrium_equations, x0, full_output=True)
    
    if sol[2] != 1:
        raise ValueError("Could not find equilibrium - solver did not converge")
        
    theta1_eq, theta2_eq, u_eq = sol[0]
    
    x_eq = np.array([theta1_eq, theta2_eq, 0, 0])
    
    return x_eq, u_eq

def generate_reference_trajectory(x_eq_start, x_eq_end, u_eq_start, u_eq_end, T, dt):
    """
    Generates a smooth reference trajectory between two equilibrium points.
    Uses a minimum jerk trajectory for smooth transitions.
    
    Args:
        x_eq_start: Starting equilibrium state
        x_eq_end: Ending equilibrium state
        u_eq_start: Starting equilibrium input
        u_eq_end: Ending equilibrium input
        T: Total time
        dt: Time step
        
    Returns:
        x_ref: Reference state trajectory
        u_ref: Reference input trajectory
    """
    steps = int(T/dt)
    t = np.linspace(0, T, steps)
    
    # Minimum jerk trajectory coefficients
    def min_jerk_coeffs(t):
        tau = t/T
        return 10*tau**3 - 15*tau**4 + 6*tau**5
    
    # Generate trajectories
    s = min_jerk_coeffs(t)
    
    x_ref = np.zeros((4, steps))
    u_ref = np.zeros((1, steps))
    
    # Position trajectories
    x_ref[0,:] = x_eq_start[0] + (x_eq_end[0] - x_eq_start[0])*s
    x_ref[1,:] = x_eq_start[1] + (x_eq_end[1] - x_eq_start[1])*s
    
    # Velocity trajectories (derivative of position)
    ds_dt = np.gradient(s, dt)
    x_ref[2,:] = (x_eq_end[0] - x_eq_start[0])*ds_dt
    x_ref[3,:] = (x_eq_end[1] - x_eq_start[1])*ds_dt
    
    # Input trajectory
    u_ref[0,:] = u_eq_start + (u_eq_end - u_eq_start)*s
    
    return x_ref, u_ref