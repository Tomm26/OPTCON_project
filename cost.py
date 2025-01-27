import numpy as np
import parameters as dyn

dt = dyn.dt
ns = dyn.ns
ni = dyn.ni

QQt = 80*np.diag([1.0, 1.0, 1.0, 1.0])
RRt = 0.01*np.eye(1)

def stagecost(xx,uu, xx_ref, uu_ref):

    ll = 0.5*(xx - xx_ref).T @ QQt @ (xx - xx_ref) + 0.5*(uu - uu_ref).T @ RRt @ (uu - uu_ref)

    #gradient wrt x, u
    grad_x_l = QQt@(xx - xx_ref)
    grad_u_l = RRt@(uu - uu_ref)

    #hessian wrt x, u
    hess_xx = QQt
    hess_xu = np.zeros((1,ns))
    hess_uu = RRt
    return ll, grad_x_l, grad_u_l, hess_xx, hess_xu, hess_uu

def termcost(xT, xT_ref, QQT = QQt):
    
    llT = 0.5*(xT - xT_ref).T @ QQT @ (xT - xT_ref)

    #gradient
    grad_x_l = QQt@(xT - xT_ref)

    #hessian 
    hess_l = QQt
    return llT, grad_x_l, hess_l
