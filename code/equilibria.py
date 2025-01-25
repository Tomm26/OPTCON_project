
# Flexible Robotic Arm Equilibrium Points Searcher

from dynamics import discretizedDynamicFRA, dynamicG, dynamicC
from parameters import ns,ni
from numpy import zeros, eye, linalg, array, squeeze

def getFRAEquilibriumPoint(uu, xx0):
    """
    Search for a particular equilibrium point of the Flexible Robotic Arm, for input uu, starting from a certain provided initial guess.
    These points are computed using the Newton Method for zero root finding applied on the function r(x)=f(x)-x,
    where f(x) is the discretized dynamic function of the Flexible Robotic Arm itself.
    Arguments:
    - uu: 1x1 input value for which compute Flexible Robotic Arm equilibria
    - xx0: 4x1 column vector state initial guess for the equilibrium point (recall that x=[x0, x1, x2, x3]'=[ϑ1, ϑ2, dϑ1dt, dϑ2dt]')
    Returns:
    - xeq: 4x1 column vector equilibrium states
    """
    # Notice that the definition of the discretized function f=[f0, f1, f2, f3]' is the following:
    # f[0] = xx[0] + dt*xx[2]
    # f[1] = xx[1] + dt*xx[3]
    # f[2:4] = xx[2:4] + dt*invM@(U-G-F@array([xx[2], xx[3]])-C)

    uu = uu.squeeze()

    maximumIteration = int(5e5)
    stepsize = 5e-1
    tolerance = 1e-10
    xx = zeros((ns, maximumIteration))
    xx[:,0] = xx0
    solved = False

    for i in range(maximumIteration-1):
        xxp, dfdx, _ = discretizedDynamicFRA(xx[:,i], uu)
        xxp = xxp - xx[:,i]
        dfdx = dfdx - eye(ns)
        direction = -linalg.inv(dfdx)@xxp;
        xx[:,i+1] = xx[:,i] + (stepsize*direction).reshape(1, ns);
        if (abs(xx[:,i+1]-xx[:,i]) < tolerance).all():
            solved = True
            break
    if not solved:
        raise TimeoutError("No equilibrium point found in " + str(maximumIteration) + " maximum number of iterations")
    
    return xx[:,i+1]

def searchFRAInputGivenAnEquilibria(eqxx):
    """
    Search the input value that generates the one equilibrium point of the FRA characterized by:
    [x0, x1, x2, x3] = [eqxx, -eqxx, 0, 0],
    alias the equilibrium point in which the first link is related to angle eqxx and the second link is pointing downwards.
    Notice that the resulted input should be the same one obtained considering the equilibrium point in which the second link is pointing upwards.
    Arguments:
    - eqxx: scalar equilibrium value for x0 (alias -x1)
    Returns:
    - uu: scalar input value that generates the requested equilibrium point
    """
    
    maximumIteration = int(5e4)
    stepsize = 5e-1
    tolerance = 1e-10
    eqxx = array([eqxx, -eqxx, 0, 0])
    fToNullify = lambda uu: (dynamicC(eqxx) + dynamicG(eqxx) - array([uu, 0]))[0]
    uu = zeros((ni, maximumIteration))
    uu[:,0] = 0
    solved = False

    for i in range(maximumIteration-1):
        uu[:,i+1] = uu[:,i] + stepsize*fToNullify(squeeze(uu[:,i]))
        if (abs(uu[:,i+1]-uu[:,i]) < tolerance).all():
            solved = True
            break
    if not solved:
        raise TimeoutError("No suitable input value found in " + str(maximumIteration) + " maximum number of iterations")
    
    return uu[:,i+1]