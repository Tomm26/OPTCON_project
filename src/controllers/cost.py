import numpy as np
from ..parameters import dt, ns, ni 
from ..dynamics import dynamics as dyn

# Increased position weights for better tracking
QQt = np.diag([100.0, 100.0, 10.0, 10.0])  # More weight on positions
RRt = 1.0*np.eye(1)  # Slightly increased input weight

def stagecost(xx, uu, xx_ref, uu_ref):
    """
    Stage cost function and its derivatives
    
    Args:
        xx: Current state
        uu: Current input
        xx_ref: Reference state
        uu_ref: Reference input
        
    Returns:
        ll: Cost value
        grad_x_l: Cost gradient w.r.t. state
        grad_u_l: Cost gradient w.r.t. input
        hess_xx: Cost hessian w.r.t. state
        hess_uu: Cost hessian w.r.t. input
    """
    ll = 0.5*(xx - xx_ref).T @ QQt @ (xx - xx_ref) + 0.5*(uu - uu_ref).T @ RRt @ (uu - uu_ref)

    grad_x_l = QQt@(xx - xx_ref)
    grad_u_l = RRt@(uu - uu_ref)

    hess_xx = QQt
    hess_uu = RRt
    
    return ll, grad_x_l, grad_u_l, hess_xx, hess_uu

def termcost(xT, xT_ref, QQT=None):
    """
    Terminal cost function and its derivatives
    
    Args:
        xT: Terminal state
        xT_ref: Reference terminal state
        QQT: Terminal cost weight matrix (optional)
        
    Returns:
        llT: Terminal cost value
        grad_x_l: Cost gradient w.r.t. state
        hess_l: Cost hessian w.r.t. state
    """
    if QQT is None:
        QQT = 2*QQt  # Increased terminal weight for better convergence
        
    llT = 0.5*(xT - xT_ref).T @ QQT @ (xT - xT_ref)

    grad_x_l = QQT@(xT - xT_ref)
    hess_l = QQT
    
    return llT, grad_x_l, hess_l