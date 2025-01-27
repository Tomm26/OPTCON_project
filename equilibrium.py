#this will compute equilibrium
import dynamics as dyn
import numpy as np
from parameters import ni
def findEq(xeq):
    """
    Finds equilibrium given the xeq input for the dynamic 
    """

    xx_in = [xeq, -xeq, 0,0] #this must be the case for an equilibrium
    maxiters = int(1e3)
    stepsize = 0.5
    diffThreshold = 0.01

    diff_u = lambda uu: dyn.getCoriolisAt(xx_in) + dyn.getGravityAt(xx_in) - np.concatenate([uu, np.array([0])]).reshape(2,1)
    uu = np.zeros((ni, maxiters))
    uu[:, 0] = 0

    for k in range(maxiters-1):
        
        uu[:, k+1] = uu[:, k] + (stepsize * diff_u(uu[:, k]))[0]
        
        if np.abs(uu[:,k+1] - uu[:, k]) < diffThreshold:
            break

    return uu[:, k+1]
        
