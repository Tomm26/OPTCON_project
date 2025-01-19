import numpy as np
from ..parameters import dt, ns, ni 
from ..dynamics import dynamics
from . import cost

def select_stepsize(stepsize_0, armijo_maxiters, cc, beta, deltau, xx_ref, uu_ref, x0, uu, JJ, descent_arm, plot=False):
    """
    Computes the stepsize using Armijo's rule.
    
    Args:
        stepsize_0: Initial stepsize guess
        armijo_maxiters: Maximum number of iterations for armijo rule
        cc: c parameter for sufficient decrease condition
        beta: Beta parameter for stepsize reduction
        deltau: Descent direction for the control action
        xx_ref: Reference trajectory state
        uu_ref: Reference trajectory input
        x0: Initial state
        uu: Input at current iteration
        JJ: Cost at current iteration
        descent_arm: Armijo descent direction at current iteration
        plot: Whether to plot descent direction
        
    Returns:
        stepsize: Selected stepsize
    """
    TT = uu.shape[1]
    stepsizes = []  # list of stepsizes
    costs_armijo = []

    stepsize = stepsize_0

    for ii in range(armijo_maxiters):
        # Temporary solution update
        xx_temp = np.zeros((ns,TT))
        uu_temp = np.zeros((ni,TT))
        xx_temp[:,0] = x0

        for tt in range(TT-1):
            uu_temp[:,tt] = uu[:,tt] + stepsize*deltau[:,tt]
            xx_temp[:,tt+1] = dynamics(xx_temp[:,tt], uu_temp[:,tt])[0]

        # Calculate cost with temporary solution
        JJ_temp = 0
        for tt in range(TT-1):
            temp_cost = cost.stagecost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt])[0]
            JJ_temp += temp_cost

        JJ_temp += cost.termcost(xx_temp[:,-1], xx_ref[:,-1])[0]

        stepsizes.append(stepsize)
        costs_armijo.append(np.min([JJ_temp, 100*JJ]))

        # Check Armijo condition
        if JJ_temp > JJ + cc*stepsize*descent_arm:
            stepsize = beta*stepsize
        else:
            if ii > 0:  # Solo se non è la prima iterazione
                print('Armijo stepsize = {:.3e}'.format(stepsize))
            break

        if ii == armijo_maxiters-1:
            print("WARNING: no stepsize found with Armijo rule!")

    if plot:
        import matplotlib.pyplot as plt
        steps = np.linspace(0, stepsize_0, int(2e1))
        costs = np.zeros(len(steps))

        for ii in range(len(steps)):
            step = steps[ii]
            xx_temp = np.zeros((ns,TT))
            uu_temp = np.zeros((ni,TT))
            xx_temp[:,0] = x0

            for tt in range(TT-1):
                uu_temp[:,tt] = uu[:,tt] + step*deltau[:,tt]
                xx_temp[:,tt+1] = dynamics(xx_temp[:,tt], uu_temp[:,tt])[0]

            JJ_temp = 0
            for tt in range(TT-1):
                temp_cost = cost.stagecost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt])[0]
                JJ_temp += temp_cost

            JJ_temp += cost.termcost(xx_temp[:,-1], xx_ref[:,-1])[0]
            costs[ii] = np.min([JJ_temp, 100*JJ])

        plt.figure()
        plt.clf()
        plt.plot(steps, costs, color='g', label='$J(\\mathbf{u}^k + stepsize*d^k)$')
        plt.plot(steps, JJ + descent_arm*steps, color='r', 
                label='$J(\\mathbf{u}^k) + stepsize*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
        plt.plot(steps, JJ + cc*descent_arm*steps, color='g', linestyle='dashed',
                label='$J(\\mathbf{u}^k) + stepsize*c*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
        plt.scatter(stepsizes, costs_armijo, marker='*')
        plt.grid()
        plt.xlabel('stepsize')
        plt.legend()
        plt.show()

    return stepsize