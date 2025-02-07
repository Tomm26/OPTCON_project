import numpy as np
from dynamics import FlexibleRoboticArm


def generate_u_ref(arm:FlexibleRoboticArm, x_traj):
    """
    Gravity Compensation
    """
    T = x_traj.shape[0] - 1
    u_traj = np.zeros((T, 1))
    
    for t in range(T):
                
        u_traj[t] =  (arm.gravity_vector(x_traj[t][0], x_traj[t][1]))[0]
        
    return u_traj