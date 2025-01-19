import numpy as np
from ..parameters import dt, ns, ni 
from ..dynamics import dynamics

# Increased weights for positions and reduced for velocities
QQt = np.diag([100.0, 100.0, 1.0, 1.0])
RRt = 0.1*np.eye(1)

def stagecost(xx, uu, xx_ref, uu_ref):
    """Stage cost function and its derivatives"""
    ll = 0.5*(xx - xx_ref).T @ QQt @ (xx - xx_ref) + 0.5*(uu - uu_ref).T @ RRt @ (uu - uu_ref)

    grad_x_l = QQt@(xx - xx_ref)
    grad_u_l = RRt@(uu - uu_ref)

    hess_xx = QQt
    hess_uu = RRt
    
    return ll, grad_x_l, grad_u_l, hess_xx, hess_uu

def termcost(xT, xT_ref, QQT=None):
    """Terminal cost function and its derivatives"""
    if QQT is None:
        QQT = 5*QQt  # Aumentato il peso terminale
        
    llT = 0.5*(xT - xT_ref).T @ QQT @ (xT - xT_ref)

    grad_x_l = QQT@(xT - xT_ref)
    hess_l = QQT
    
    return llT, grad_x_l, hess_l