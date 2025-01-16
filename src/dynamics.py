import numpy as np
from parameters import dt, ns, m1, m2, l1, l2, r1, r2, I1, I2, g, f1, f2

def dynamics(xx, uu):
    # Ensure inputs are numpy arrays
    xx = np.array(xx)
    uu = np.array([uu]) if np.isscalar(uu) else np.array(uu)
    
    M = np.array([[I1 + I2 + m1*r1*r1 + m2*(l1*l1 + r2*r2) + 2*m2*l1*r2*np.cos(xx[1]), I2 + m2*r2*r2 + m2*l1*r2*np.cos(xx[1])], 
                  [I2 + m2*r2*r2 + m2*l1*r2*np.cos(xx[1]), I2 + m2*r2*r2]])
    
    C = np.array([[-m2*l1*r2*xx[3]*np.sin(xx[1])*(xx[3] + 2*xx[2])], 
                  [m2*l1*r2*np.sin(xx[1])*xx[2]*xx[2]]])
    
    G = np.array([[g*(m1*r1 + m2*l1)*np.sin(xx[0]) + g*m2*r2*np.sin(xx[0] + xx[1])],
                  [g*m2*r2*np.sin(xx[0] + xx[1])]])
    
    F = np.array([[f1, 0], 
                  [0, f2]])
    
    xx_plus = np.zeros((ns,))
    
    xx_plus[0] = xx[0] + xx[2] * dt
    xx_plus[1] = xx[1] + xx[3] * dt
    
    res = np.linalg.inv(M) @ (np.array([uu[0], 0]).reshape(2,1) - C - F @ np.array([xx[2], xx[3]]).reshape(2,1) - G)
    
    res = res.flatten()
    xx_plus[2] = xx[2] + res[0] * dt
    xx_plus[3] = xx[3] + res[1] * dt
    
    return xx_plus  # Return only the next state