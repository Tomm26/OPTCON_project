import numpy as np
import derivatives as dv
from parameters import *

#this is to prevent multiple computation of the symbolic
gradient= dv.gradient
hessian = dv.hessian

def dynamics(xx, uu):

    #xx[0], xx[1], xx[2], xx[3] = phi1, phi2, phi1', phi2'

    M = np.array([[I1 + I2 + m1*r1*r1 + m2*(l1*l1 + r2*r2) + 2*m2*l1*r2*np.cos(xx[1]), I2 + m2*r2*r2 + m2*l1*r2*np.cos(xx[1])], 
                  [I2 + m2*r2*r2 + m2*l1*r2*np.cos(xx[1]), I2 + m2*r2*r2]])
    
    C = np.array([[-m2*l1*r2*xx[3]*np.sin(xx[1])*(xx[3] + 2*xx[2])], 
                  [m2*l1*r2*np.sin(xx[1])*xx[2]*xx[2]]])
    
    G = np.array([[g*(m1*r1 + m2*l1)*np.sin(xx[0]) + g*m2*r2*np.sin(xx[0] + xx[1])],
                  [g*m2*r2*np.sin(xx[0] + xx[1])]])
    
    F = np.array([[f1, 0], 
                  [0, f2]])
    
    #update x
    xx_plus = np.zeros((ns, ))

    xx_plus[0] = xx[0] + xx[2] * dt
    xx_plus[1] = xx[1] + xx[3] * dt

    res = np.linalg.inv(M) @ (np.array([uu[0], 0]).reshape(2,1) - C - F @ np.array([xx[2], xx[3]]).reshape(2,1) - G)

    res = res.flatten()
    xx_plus[2] = xx[2] + res[0] * dt
    xx_plus[3] = xx[3] + res[1] * dt
    
    return xx_plus, *gradient(xx, uu), *hessian(xx,uu)


def getCoriolisAt(xx):
    return np.array([[-m2*l1*r2*xx[3]*np.sin(xx[1])*(xx[3] + 2*xx[2])], 
                  [m2*l1*r2*np.sin(xx[1])*xx[2]*xx[2]]])

def getGravityAt(xx):
    return np.array([[g*(m1*r1 + m2*l1)*np.sin(xx[0]) + g*m2*r2*np.sin(xx[0] + xx[1])],
                  [g*m2*r2*np.sin(xx[0] + xx[1])]])