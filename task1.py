import numpy as np
from matplotlib import pyplot as plt
from parameters import *
import cost
import dynamics as dyn
from newton_method import compute_descent_direction, backward_pass, select_stepsize
import equilibrium as eq
import trajectory as traj


# Inizializzazione parametri e variabili (come nel codice originale)
sec = 1/dt
tf = 16 
TT = int(tf*sec)
max_iters = 14
thresholdGrad = 1e-5
fixed_stepsize = 1
Armijo = True
stepsize_0 = 1
cc = 0.5
beta = 0.7
armijo_maxiters = 15

# Inizializzazione stati e controlli
xx_init = np.zeros((ns, TT))
uu_init = np.zeros((ni, TT))
xx_ref = traj.stepFun([np.pi/8, -np.pi/8, 0,0], [-np.pi/8, np.pi/8, 0,0], TT, 6)
uu_ref = np.array([eq.findEq(xx_ref[0, tt]) for tt in range(TT)]).reshape((ni, TT))

xx = np.zeros((ns, TT, max_iters))
uu = np.zeros((ni, TT, max_iters))
uu[:,:, 0] = np.repeat(uu_ref[:,0].reshape(-1,1), TT, axis=1)
x0 = xx_ref[:,0]
xx[:, 0,0] = x0

# Inizializzazione variabili di ottimizzazione
JJ = np.zeros(max_iters)
deltau = np.zeros((ni,TT, max_iters))
descent_arm = np.zeros(max_iters)
lmbd = np.zeros((ns, TT, max_iters))

# Loop principale di ottimizzazione
for kk in range(max_iters-1):
    # Forward simulation
    for tt in range(TT-1):
        xx[:, tt+1, 0] = dyn.dynamics(xx[:, tt, 0], uu[:, tt, 0])[0]

    # Backward pass
    JJ[kk], QQt, RRt, SSt, AA, BB, qqt, rrt, lmbd[:, :, kk], grdJdu = backward_pass(
        xx[:, :, kk], uu[:, :, kk], xx_ref, uu_ref, dyn.dynamics, cost.stagecost, cost.termcost, ns, ni, TT)

    # Controllo convergenza
    normGrad = np.linalg.norm(grdJdu)
    print(f"\n{kk} iteration: Cost {JJ[kk]}, GradNorm {normGrad}")
    if normGrad < thresholdGrad:
        print("Convergenza raggiunta")
        break

    # Calcolo direzione di discesa
    KK, sigma, deltau = compute_descent_direction(AA, BB, QQt, RRt, SSt, qqt, rrt, lmbd[:, -1, kk], x0, TT)

    # Selezione stepsize
    if Armijo:
        stepsize = select_stepsize(
            stepsize_0, armijo_maxiters, cc, beta, deltau, xx_ref, uu_ref, x0,
            xx[:, :, kk], uu[:, :, kk], JJ[kk], np.sum(grdJdu*deltau), KK, sigma, TT)
    else:
        stepsize = fixed_stepsize

    # Aggiornamento controllo e stato
    xx[:,0,kk+1] = x0
    for tt in range(TT-1):
        uu[:, tt, kk+1] = uu[:, tt, kk] + KK[:, :, tt] @ (xx[:, tt, kk+1] - xx[:, tt, kk]) + stepsize * sigma[:,tt]
        xx[:,tt+1,kk+1] = dyn.dynamics(xx[:,tt, kk+1], uu[:, tt, kk+1])[0]

# Plotting (come nel codice originale)
plt.figure()
plt.plot(xx_ref[0], 'r--', label='ref')
plt.plot(xx[0, :, -1], 'g', label='opt')
plt.xlabel('time')
plt.ylabel('theta1')
plt.grid()
plt.legend()

plt.figure()
plt.plot(xx_ref[1], 'r--', label='ref')
plt.plot(xx[1, :, -1], 'g', label='opt')
plt.xlabel('time')
plt.ylabel('theta2')
plt.grid()
plt.legend()

plt.figure()
plt.plot(xx_ref[2], 'r--', label='ref')
plt.plot(xx[2, :, -1], 'g', label='opt')
plt.xlabel('time')
plt.ylabel('velocity1')
plt.grid()
plt.legend()

plt.show()