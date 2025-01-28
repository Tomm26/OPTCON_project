import numpy as np
from parameters import *
import cost
import dynamics as dyn
import lqr
import armijo 

def newton_optimizer(xx, uu, xx_ref, uu_ref, x0, TT, max_iters, thresholdGrad, Armijo = True, showPlots = False):
    """
    Regularized Newton's method in closed-loop    
    
    xx:            state (ns x TT x max_iters)
    uu:            input (ni x TT x max_iters)
    xx_ref:        referiment state (ns x TT)
    uu_ref:        referiment input (ni x TT)
    x0:            starting input (ns)
    TT: time       istants (num of sec * dt)
    max_iters:     max num of iteration before stop
    thresholdGrad: threshold wrt the norm of the gradient descent direction
    Armijo:        if the opt should use Armijo or fixed stepsize
    showPlots:     plots armijo steps

    returns
    xx:            state for each iter (ns x TT x max_iters)
    uu:            input for each iter (ni x TT x max_iters)
    """
    #define local functions to improve performaces
    stagecost = cost.stagecost 
    termcost = cost.termcost
    dynamics = dyn.dynamics
    ltv_LQR = lqr.ltv_LQR
    select_stepsize = armijo.select_stepsize
    JJ = np.zeros(max_iters) # costs 

    for kk in range(max_iters-1):

        # initialize matrices
        QQt = np.zeros((ns, ns, TT))
        RRt = np.zeros((1,1,TT))
        SSt = np.zeros((1,ns,TT))
        AA = np.zeros((ns,ns,TT))
        BB = np.zeros((ns,1,TT))
        qqt = np.zeros((ns, TT))
        rrt = np.zeros((ni, TT))

        lmbda = np.zeros_like(xx)
        grdJdu = np.zeros_like(uu)

        # initialize lambda and cost and compute backward the costate equation
        JJ[kk], qqT, QQt[:, :, -1] = termcost(xx[:, -1, kk], xx_ref[:, -1])
        lmbda[:, TT-1, kk] = qqT

        for tt in reversed(range(TT-1)):
            JJtemp, qqtemp, rrtemp, QQtemp, SStemp, RRtemp = stagecost(xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt])
            JJ[kk] += JJtemp

            QQt[:, :, tt] = QQtemp
            SSt[:, :, tt] = SStemp
            RRt[:, :, tt] = RRtemp
            qqt[:, tt] = qqtemp
            rrt[:, tt] = rrtemp

            _, AA[:, :, tt], BB[:, :, tt], *_ = dynamics(xx[:, tt, kk], uu[:, tt, kk])

            lmbda[:,tt, kk] = qqt[:,tt] + AA[:,:,tt].T @ lmbda[:,tt+1, kk]
            grdJdu[:,tt, kk] = rrt[:,tt] + BB[:,:,tt].T @ lmbda[:,tt+1, kk]

        normGrad = np.linalg.norm(grdJdu[:, :, kk])

        print(f"\n{kk} iteration: ")
        print("current cost: ", JJ[kk])
        print("current gradient of J (norm): ", normGrad)

        if normGrad < thresholdGrad:
            print("finished as the gradient is sufficiently small")
            break

        # solve the lqr with approx hessians 
        KK, sigma, *_, deltau = ltv_LQR(AA, BB, QQt, RRt, SSt,QQt[:,:,-1], TT,np.zeros_like(x0),qqt, rrt, qqT)

        if Armijo:

            stepsize = select_stepsize(
                stepsize_0, armijo_maxiters, cc, beta, deltau, xx_ref, uu_ref, x0, 
                xx[:, :, kk], uu[:, :, kk], JJ[kk], (deltau @ grdJdu[:, :, kk].T).squeeze(), KK, sigma, TT, plot=showPlots)
        else:
            stepsize = fixed_stepsize


        # update input and state in closed-loop
        xx[:,0,kk+1] = x0
        for tt in range(TT-1):
            uu[:, tt, kk+1] = uu[:, tt, kk] + KK[:, :, tt] @ (xx[:, tt, kk+1] - xx[:, tt, kk]) + stepsize * sigma[:,tt]
            xx[:,tt+1,kk+1] = dynamics(xx[:,tt, kk+1], uu[:, tt, kk+1])[0]

    return xx, uu


