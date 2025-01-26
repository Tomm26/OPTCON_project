import numpy as np
from matplotlib import pyplot as plt
from parameters import *
import cost
import dynamics as dyn
import lqr 
from tqdm import tqdm
import armijo

sec = 1/dt #should be a second? yes
tf = 1 #time in seconds
TT = int(tf*sec) 
max_iters =500

fixed_stepsize = 1e-3

# ARMIJO PARAMETERS
Armijo = True
stepsize_0 = 1
cc =0.8
beta = 0.2
armijo_maxiters = 20 # number of Armijo iterations

#define inputs and ref
xx_init = np.zeros((ns, TT))
uu_init = np.zeros((ni, TT))

xx_ref = []
uu_ref = []


for tt in range(TT):
    if tt<TT/2:
        xx_ref.append(np.array([np.pi/4, -np.pi/4, 0,0]))
        uu_ref.append(np.array(dyn.g*(dyn.m1*dyn.r1 + dyn.m2*dyn.l1)*np.sin(np.pi/4)))
    else:
        xx_ref.append(np.array([-np.pi/4, np.pi/4, 0,0]))
        uu_ref.append(np.array(dyn.g*(dyn.m1*dyn.r1 + dyn.m2*dyn.l1)*np.sin(-np.pi/4)))

xx_ref = np.array(xx_ref).T
uu_ref = np.array(uu_ref).reshape(ni, TT)

#Starting at the same position as the xref (ie equilibrium)
xx_init = np.repeat(xx_ref[:,0].reshape(-1,1), TT, axis=1)
uu_init = np.repeat(uu_ref[:,0].reshape(-1,1), TT, axis=1)

xx = np.zeros((ns, TT, max_iters))   # state seq.
uu = np.zeros((ni, TT, max_iters))   # input seq.

xx[:,:,0] = xx_init
uu[:,:,0] = uu_init
x0 = xx_ref[:,0]

JJ = np.zeros(max_iters)
deltau = np.zeros((ni,TT, max_iters)) #descent direction
descent_arm = np.zeros(max_iters) 
lmbd = np.zeros((ns, TT, max_iters)) # lambdas - costate seq.

for kk in tqdm(range(max_iters-1)):

    # compute the cost
    for tt in range(TT-1):
        JJ[kk]+= cost.stagecost(xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt])[0]

    JJ[kk]+= cost.termcost(xx[:, -1, kk], xx_ref[:, -1])[0]

    if kk%1 ==0:
        print("cost: ", JJ[kk])
    
    # Descent direction calculation

    QQt = np.zeros((ns, ns, TT))

    #for the first iter approx
    if kk < max_iters:
        RRt = np.zeros((1,1,TT-1))
        SSt = np.zeros((1,ns,TT-1))
        for tt in range(TT-1):
            QQt[:, :, tt] = cost.stagecost(xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt])[3]
            RRt[:,:, tt] = cost.stagecost(xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt])[4]
        
        QQt[:, :, TT-1] = cost.termcost(xx[:,TT-1,kk], xx_ref[:,TT-1])[2]
        
    else:
        lmbd[:, -1, kk] = cost.termcost(xx[:,-1,kk], xx_ref[:,-1])[1]
    
        #find the lambdas (solve backwards the costate equation)
        for tt in reversed(range(TT-1)):
            print(lmbd[:, tt+1, kk])
            lmbd[:, tt, kk] = dyn.dynamics(xx[:, tt, kk], uu[:, tt, kk])[1] @ lmbd[:, tt+1, kk] + cost.stagecost(xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt])[1]
        

        #integrate backward in time (for Qt, Rt, St)
        RRt = np.zeros((ns, ns, TT-1))
        SSt = np.zeros((ns, ns, TT-1))

        for tt in range(TT-1):
            *_, hess_l_11, hess_l_22 = cost.stagecost(xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt])
            *_, hess_f_11, hess_f_12, _, hess_f_22 = dyn.dynamics(xx[:, tt, kk], uu[:, tt, kk])

            QQt[:,:, tt] = hess_l_11 + hess_f_11 @ lmbd[:, tt+1, kk]
            RRt[:,:, tt] = hess_l_22 + hess_f_22 @ lmbd[:, tt+1, kk]
            SSt[:,:, tt] = hess_f_12 @ lmbd[:, tt+1, kk]

        QQt[:,:, TT-1] = cost.termcost(xx[:,TT-1,kk], xx_ref[:,TT-1])[2]

    grads_fx = np.array([dyn.dynamics(xx[:, tt, kk], uu[:, tt, kk])[1] for tt in range(TT-1)]).reshape((ns,ns,TT-1))
    grads_fu = np.array([dyn.dynamics(xx[:, tt, kk], uu[:, tt, kk])[2] for tt in range(TT-1)]).reshape((ns,1,TT-1))
    qqin = np.array([cost.stagecost(xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt])[1] for tt in range(TT-1)]).reshape((ns,TT-1))
    rrin = np.array([cost.stagecost(xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt])[2] for tt in range(TT-1)]).reshape((1,TT-1))
    qqfin = cost.termcost(xx[:,TT-1,kk], xx_ref[:,TT-1])[1]

    res = lqr.ltv_LQR(grads_fx, grads_fu, QQt, RRt, SSt,QQt[:,:,TT-1], TT,0,qqin, rrin, qqfin)
    
    #update input and state
    if Armijo:
        for tt in range(TT-1):
            descent_arm[kk]+= np.linalg.norm(res[-1][0,tt])**2

        #print(descent_arm[kk])
        stepsize = armijo.select_stepsize(stepsize_0, armijo_maxiters, cc, beta,
                                res[-1], xx_ref, uu_ref, 0, 
                                uu[:,:,kk], JJ[kk], descent_arm[kk], False)
    else:
        stepsize = fixed_stepsize

    print(stepsize)
    uu[:,:,kk+1] = uu[:,:, kk] + stepsize * res[-1]

    xx[:,0,kk+1] = x0
    for tt in range(TT-1):
        xx[:,tt+1,kk+1] = dyn.dynamics(xx[:,tt, kk+1], uu[:, tt, kk+1])[0]


# Plotting the results

# Adding labels and title

plt.plot(list(range(TT)), xx_ref[0], marker='o', linestyle='-', color='r', label='ref function (theta 1 state)')
plt.plot(list(range(TT)), xx[0, :, max_iters-1], marker='x', linestyle='--', color='b', label='opt func (theta 1 state)')
# Show the plot
plt.xlabel('time')
plt.ylabel('theta val')
plt.grid(True)
plt.legend()
plt.show()
    


