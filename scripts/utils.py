import numpy as np
from dynamics import FlexibleRoboticArm

def generate_initial_input_trajectory(arm:FlexibleRoboticArm, x_traj, max_iter=100, step_size=0.1, tol=1e-4):
    """
    Generate initial input trajectory using gradient descent based on robot dynamics.
    
    Args:
        arm: FlexibleRoboticArm instance
        x_traj: State trajectory array of shape (T+1, 4) where T is time horizon
        max_iter: Maximum iterations for gradient descent
        step_size: Step size for gradient updates
        tol: Tolerance for convergence
        
    Returns:
        u_traj: Input trajectory array of shape (T, 1)
    """
    T = x_traj.shape[0] - 1
    u_traj = np.zeros((T, 1))  # Initialize input trajectory
    
    # For each timestep, find input that best achieves next state
    for t in range(T):
        x_current = x_traj[t]
        x_next_desired = x_traj[t+1]
        u_current = np.array([0.0])  # Initialize input at zero
        
        for i in range(max_iter):
            # Get next state using current input
            x_next = arm.discrete_dynamics(x_current, u_current)
            
            # Compute state error
            state_error = x_next - x_next_desired
            
            # Get gradient of dynamics with respect to input
            _, grad_u = arm.get_gradients(x_current, u_current)
            
            # Update input using gradient
            u_update = -step_size * grad_u.T @ state_error
            u_current += u_update
            
            # Check convergence
            if np.abs(u_update) < tol:
                break
                
        u_traj[t] = u_current
        
    return u_traj


def generate_initial_input_trajectory_2(arm:FlexibleRoboticArm, x_traj, max_iter=100, step_size=0.1, tol=1e-4):
    """
    Generate initial input trajectory using gradient descent based on robot dynamics.
    
    Args:
        arm: FlexibleRoboticArm instance
        x_traj: State trajectory array of shape (T+1, 4) where T is time horizon
        max_iter: Maximum iterations for gradient descent
        step_size: Step size for gradient updates
        tol: Tolerance for convergence
        
    Returns:
        u_traj: Input trajectory array of shape (T, 1)
    """
    T = x_traj.shape[0] - 1
    u_traj = np.zeros((T, 1))  # Initialize input trajectory
    
    # For each timestep, find input that best achieves next state
    for t in range(T):
        x_current = [x_traj[t][0], -x_traj[t][0], 0,0]

        diff_u = lambda uu: (arm.coriolis_vector(x_current[1], x_current[2], x_current[3]).reshape(2,1) + 
                             arm.gravity_vector(x_current[0], x_current[1]).reshape(2,1) - np.concatenate((uu, np.array([0]))).reshape(2,1))
        
        uu = np.zeros((1, max_iter))
        uu[:, 0] = 0

        for k in range(max_iter-1):

            uu[:, k+1] = uu[:, k] + (step_size * diff_u(uu[:, k]))[0]
        
            if np.abs(uu[:,k+1] - uu[:, k]) < tol:
                
                u_traj[t] = uu[:, k+1]
                break
                
        u_traj[t] = uu[:, k+1]
        
    return u_traj