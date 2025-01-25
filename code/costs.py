
# Definition of the cost functions for a Trajectory Generation Optimal Control Problem

from numpy import squeeze, zeros

def stageCostTrkTrj(ns, ni, xx, uu, xx_des, uu_des, QQt, RRt):
    """
    Stage cost function for a trajectory generation optimization problem.
    This cost function is typically defined as the sum of two quadratic function, one for the state, one for the input.
    Definition as follows: l(x,u) = 1/2(x-x_des)'Q(x-x_des) + 1/2(u-u_des)'R(u-u_des)
    Arguments:
    - ns: number of states
    - ni: number of inputs
    - xx: nsx1 actual column vector state at time t (recall: x=[x0, x1, x2, x3]'=[ϑ1, ϑ2, dϑ1dt, dϑ2dt]' in the FRA case)
    - xx_des: nsx1 desired column vector state at time t
    - uu: nix1 actual input value at time t
    - uu_des: nix1 desired input value at time t
    - QQt: nsxns state cost matrix at time t
    - RRt: nixni input cost matrix at time t
    Returns:
    - ll: scalar stage cost at xx,uu
    - dldx: nsx1 gradient of the cost wrt x at xx,uu (recall: the gradient is the transpose of the Jacobian and viceversa)
    - dldu: nix1 gradient of the cost wrt u at xx,uu (recall: the gradient is the transpose of the Jacobian and viceversa)
    - d2ldx2: nsxns hessian of the cost wrt x two times at xx,uu
    - d2ldxdu: nsxni hessian of the cost first wrt x and then wrt u at xx,uu
    - d2ldu2: nixni hessian of the cost wrt u two times at xx,uu
    """

    xx = xx.reshape(ns, 1)
    uu = uu.reshape(ni, 1)
    xx_des = xx_des.reshape(ns, 1)
    uu_des = uu_des.reshape(ni, 1)

    ll = 0.5*(xx-xx_des).T@QQt@(xx-xx_des) + 0.5*(uu-uu_des).T@RRt@(uu-uu_des)
    dldx = QQt@(xx-xx_des)
    dldu = RRt@(uu-uu_des)

    d2ldxdx = QQt
    d2ldxdu = zeros((ns, ni))
    d2ldudu = RRt

    return squeeze(ll), dldx, dldu, d2ldxdx, d2ldxdu, d2ldudu

def termCostTrkTrj(ns, xT, xT_des, QQT):
    """
    Terminal cost function for a trajectory generation optimization problem.
    This cost function is typically defined as a quadratic function of the state (in its terminal value).
    Definition as follows: l_T(x) = 1/2(x-x_des)'Q_T(x-x_des)
    Arguments:
    - xT: nsx1 actual column vector state at the terminal time T
    - xT_des: nsx1 desired column vector state at the terminal time T
    - QQT: nsxns terminal state cost matrix
    Returns:
    - ll: scalar terminal cost at xT
    - dldx: nsx1 gradient of the cost wrt x at xT (recall: the gradient is the transpose of the Jacobian and viceversa)
    - d2ldxdx: nsxns hessian of the cost wrt x two times at xT
    """

    xT = xT.reshape(ns, 1)
    xT_des = xT_des.reshape(ns, 1)
    ll = 0.5*(xT-xT_des).T@QQT@(xT-xT_des)
    dldx = QQT@(xT-xT_des)
    d2ldxdx = QQT

    return squeeze(ll), dldx, d2ldxdx

def totalCostFunction(xx, uu, xx_des, uu_des, TT, stageCostFunction, terminalCostFunction):
    """
    Generic implementation of a total cost function for an Optimal Control Problem,
    that depends on a certain input-state trajectory and a desired (eventually unfeasible) input-state curve for it.
    Arguments:
    - xx: column vector actual state trajectory (dimension: ns*TT) (coupled with uu; this is a state-input trajectory)
    - uu: column vector actual input trajectory (dimension: ni*TT) (coupled with xx; this is a state-input trajectory)
    - xx_des: column vector state desired curve (dimension: ns*TT)
    - uu_des: column vector input desired curve (dimension: ni*TT)
    - TT: number of time steps (each one of duration dt, enough for evolve from t=0 to t=T, where [0, T] is the considered horizon)
    - stageCostFunction: function of (xx_t, uu_t, xx_des_t, uu_des_t, t) that computes the stage cost requiring as arguments respectively:
                         trajectory state and input at time t, desired state and input at time t, the time t
    - terminalCostFunction: function of (xT, xT_des) that computes the terminal cost requiring as arguments respectively:
                            trajectory terminal state value and desired terminal state value
    Returns:
    - ll: scalar total cost of the trajectory xx, uu having xx_des, uu_des as desired curves
    """
    ll = 0
    for tt in range(TT-1):
        ll += stageCostFunction(xx[:,tt], uu[:,tt], xx_des[:,tt], uu_des[:,tt], tt)[0]
    return ll + terminalCostFunction(xx[:,-1], xx_des[:,-1])[0]
