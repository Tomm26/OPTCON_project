import numpy as np
from matplotlib import pyplot as plt
from parameters import *
import cost
import dynamics as dyn
import lqr 
from tqdm import tqdm
import equilibrium as eq
import trajectory as traj
import armijo 

ltv_LQR = lqr.ltv_LQR
dynamics = dyn.dynamics
stagecost = cost.stagecost
termcost = cost.termcost
select_stepsize = armijo.select_stepsize

sec = 1/dt #should be a second? yes
tf = 16 #time in seconds
TT = int(tf*sec) 
max_iters =35

fixed_stepsize = 0.5

# ARMIJO PARAMETERS
Armijo = False
stepsize_0 = 1
cc =0.5
beta = 0.7
armijo_maxiters = 20 # number of Armijo iterations

#define inputs and ref
xx_init = np.zeros((ns, TT))
uu_init = np.zeros((ni, TT))

xx_ref = traj.stepFun([np.pi/8, -np.pi/8, 0,0], [-np.pi/8, np.pi/8, 0,0], TT, 6)
uu_ref = np.array([eq.findEq(xx_ref[0, tt]) for tt in range(TT)]).reshape((ni, TT))

xx = np.zeros((ns, TT, max_iters))   # state seq.
uu = np.zeros((ni, TT, max_iters))   # input seq.

#Starting at the same position as the xref (ie equilibrium)
uu[:,:, 0] = np.repeat(uu_ref[:,0].reshape(-1,1), TT, axis=1)
x0 = xx_ref[:,0]
xx[:, 0,0] = x0

for tt in range(1, TT-1):
    xx[:, tt+1, 0] = dynamics(xx[:, tt, 0], uu[:, tt, 0])[0]


JJ = np.zeros(max_iters)
deltau = np.zeros((ni,TT, max_iters)) #descent direction
descent_arm = np.zeros(max_iters) 
lmbd = np.zeros((ns, TT, max_iters)) # lambdas - costate seq.

for kk in tqdm(range(max_iters-1)):

    # compute the cost and the direction descent
    QQt = np.zeros((ns, ns, TT))
    RRt = np.zeros((1,1,TT))
    SSt = np.zeros((1,ns,TT))
    AA = np.zeros((ns,ns,TT))
    BB = np.zeros((ns,1,TT))
    qqt = np.zeros((ns, TT))
    rrt = np.zeros((ni, TT))

    lmbda = np.zeros_like(xx)
    grdJdu = np.zeros_like(uu)

    
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

    print("current cost: ", JJ[kk])
  
    KK, sigma, *_, deltau = ltv_LQR(AA, BB, QQt, RRt, SSt,QQt[:,:,-1], TT,np.zeros_like(x0),qqt, rrt, qqT)
    
    if Armijo:
        
        stepsize = select_stepsize(
            stepsize_0, armijo_maxiters, cc, beta, deltau, xx_ref, uu_ref, x0, 
            xx[:, :, kk], uu[:, :, kk], JJ[kk], (deltau @ grdJdu[:, :, kk].T).squeeze(), KK, sigma, TT)
    else:
        stepsize = fixed_stepsize

    #update input and state
    
    xx[:,0,kk+1] = x0
    for tt in range(TT-1):
        uu[:, tt, kk+1] = uu[:, tt, kk] + KK[:, :, tt] @ (xx[:, tt, kk+1] - xx[:, tt, kk]) + stepsize * sigma[:,tt]
        xx[:,tt+1,kk+1] = dynamics(xx[:,tt, kk+1], uu[:, tt, kk+1])[0]


# Plotting the results

# Adding labels and title

plt.plot(list(range(TT)), xx_ref[0], color='r', label='ref function (theta 1 state)')
plt.plot(list(range(TT)), xx[0, :, max_iters-1], color='g', label='opt func (theta 1 state)')

plt.plot(list(range(TT)), xx_ref[1], color='r', label='ref function (theta 1 state)')
plt.plot(list(range(TT)), xx[1, :, max_iters-1], color='g', label='opt func (theta 1 state)')
# Show the plot
plt.xlabel('time')
plt.ylabel('theta val')
plt.grid(True)
plt.legend()
plt.show()

plt.plot(list(range(TT)), xx_ref[2], color='r', label='ref function (theta 1 state)')
plt.plot(list(range(TT)), xx[2, :, max_iters-1], color='g', label='opt func (theta 1 state)')
plt.show()