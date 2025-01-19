import numpy as np
from scipy.optimize import fsolve
from .dynamics import dynamics as dyn

def find_equilibrium(theta1_desired, theta2_desired):
    def equilibrium_equations(vars):
        theta1, theta2, u = vars
        x = np.array([theta1, theta2, 0, 0])
        x_next = dyn(x, np.array([u]))[0]
        
        # Normalizza gli angoli
        theta1_error = np.arctan2(np.sin(theta1 - theta1_desired), 
                                np.cos(theta1 - theta1_desired))
        theta2_error = np.arctan2(np.sin(theta2 - theta2_desired), 
                                np.cos(theta2 - theta2_desired))
        
        return [
            x_next[2],  # acc1 = 0
            x_next[3],  # acc2 = 0 
            theta1_error  # angoli desiderati
        ]
    
    # Usa più punti iniziali
    x0_attempts = [
        [theta1_desired, theta2_desired, 0],
        [theta1_desired + 0.1, theta2_desired + 0.1, 1],
        [theta1_desired - 0.1, theta2_desired - 0.1, -1]
    ]
    
    for x0 in x0_attempts:
        sol = fsolve(equilibrium_equations, x0, full_output=True)
        if sol[2] == 1:  # Convergenza
            return np.array([sol[0][0], sol[0][1], 0, 0]), sol[0][2]
            
    raise ValueError("Could not find equilibrium")

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