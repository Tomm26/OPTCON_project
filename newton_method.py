import numpy as np
import armijo 
import lqr

def compute_descent_direction(AA, BB, QQt, RRt, SSt, qqt, rrt, qqT, x0, TT):
    """Calcola la direzione di discesa tramite LQR lineare tempo-variante"""
    KK, sigma, PP, pp, deltau = lqr.ltv_LQR(
        AA, BB, QQt, RRt, SSt, QQt[:,:,-1], TT, np.zeros_like(x0), qqt, rrt, qqT)
    return KK, sigma, deltau

def backward_pass(xx, uu, xx_ref, uu_ref, dynamics, stagecost, termcost, ns, ni, TT):
    """Esegue il backward pass per calcolare gradienti e linearizzazioni"""
    QQt = np.zeros((ns, ns, TT))
    RRt = np.zeros((ni, ni, TT))
    SSt = np.zeros((ni, ns, TT))
    AA = np.zeros((ns, ns, TT))
    BB = np.zeros((ns, ni, TT))
    qqt = np.zeros((ns, TT))
    rrt = np.zeros((ni, TT))
    lmbda = np.zeros((ns, TT))
    grdJdu = np.zeros((ni, TT))

    # Terminal cost
    JJ, qqT, QQt[:, :, -1] = termcost(xx[:, -1], xx_ref[:, -1])
    lmbda[:, -1] = qqT

    # Backward recursion
    for tt in reversed(range(TT-1)):
        JJ_temp, qq, rr, QQ, SS, RR = stagecost(xx[:, tt], uu[:, tt], xx_ref[:, tt], uu_ref[:, tt])
        JJ += JJ_temp
        
        QQt[:, :, tt] = QQ
        RRt[:, :, tt] = RR
        SSt[:, :, tt] = SS
        qqt[:, tt] = qq
        rrt[:, tt] = rr

        _, AA[:, :, tt], BB[:, :, tt], *_ = dynamics(xx[:, tt], uu[:, tt])

        lmbda[:, tt] = qqt[:, tt] + AA[:, :, tt].T @ lmbda[:, tt+1]
        grdJdu[:, tt] = rrt[:, tt] + BB[:, :, tt].T @ lmbda[:, tt+1]

    return JJ, QQt, RRt, SSt, AA, BB, qqt, rrt, lmbda, grdJdu

def select_stepsize(stepsize_0, armijo_maxiters, cc, beta, deltau, xx_ref, uu_ref, x0, xx_curr, uu_curr, JJ_current, descent_dir, KK, sigma, TT):
    """Armijo line search per la selezione dello stepsize"""
    return armijo.select_stepsize(
        stepsize_0, armijo_maxiters, cc, beta, deltau, xx_ref, uu_ref, x0, 
        xx_curr, uu_curr, JJ_current, descent_dir, KK, sigma, TT)