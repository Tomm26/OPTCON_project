
# Flexible Robotic Arm Discretized Dynamics

from parameters import I1, I2, m1, m2, r1, r2, l1, f1, f2, g, ns, ni
from parameters import discretizationStep as dt
from numpy import array, cos, sin, linalg, zeros
from miscellaneous import correctStateInputCurvesShapes

def discretizedDynamicFRA(xx, uu, onlyZeroOrderDynamic = False):
    """
    Discretized Dynamics of the Flexible Robotic Arm.
    Arguments:
    - xx: 4x1 column vector state at time t (x=[x0, x1, x2, x3]'=[ϑ1, ϑ2, dϑ1dt, dϑ2dt]')
    - uu: 1x1 input value at time t
    - onlyZeroOrderDynamic: if True, only the zero order dynamic is retourned (i.e. the column vector state at time t+1, xpp)
    Returns:
    - xxp: 4x1 column vector state at time t+1
    - dfdx: 4x4 jacobian of the dynamic wrt x at xx,uu
    - dfdu: 4x1 jacobian of the dynamic wrt u at xx,uu
    """

    xx = xx.squeeze()
    uu = uu.squeeze()

    # Inertia matrix (2x2)
    M = array([
        [I1+I2+ m1*(r1)**2+m2*((l1)**2+(r2)**2)+2*m2*l1*r2*cos(xx[1]),  I2+m2*(r2)**2+m2*l1*r2*cos(xx[1]) ],
        [I2+m2*(r2)**2+m2*l1*r2*cos(xx[1]),                             I2+m2*(r2)**2                     ]
    ])
    # Coriolis and centrifugal forces matrix (2x1)
    C = dynamicC(xx)
    # Gravity forces matrix (2x1)
    G = dynamicG(xx)
    # Friction forces matrix (2x2)
    F = array([
        [f1, 0], 
        [0, f2]
    ])
    # Input matrix (2x1)
    U = array([uu, 0])

    # Derivative of the inertia matrix w.r.t. the second state x1 (2x2)
    dMdx1 = array([
        [-2*m2*l1*r2*sin(xx[1]), -m2*l1*r2*sin(xx[1]) ], 
        [-m2*l1*r2*sin(xx[1]),   0                    ]
    ])
    # Tensor derivative of the intertia matrix w.r.t. the state vector (2x2x4)
    # dMdx = [zeros((2,2)), dMdx1, zeros((2,2)), zeros((2,2))]
    # Inverse of the inertia matrix (2x2)
    invM = linalg.pinv(M)
    # Derivative of the inverse of the inertia matrix w.r.t. the second state x1 (2x2)
    dinvMdx1 = -invM@dMdx1@invM

    # Tensor derivative of the Coriolis and centrifugal forces matrix w.r.t. the state vector (2x1x4)
    dCdx0 = array([0, 0])
    dCdx1 = array([
        -m2*l1*r2*xx[3]*cos(xx[1])*(xx[3]+2*xx[2]),
        m2*l1*r2*cos(xx[1])*(xx[2])**2
    ])
    dCdx2 = array([
        -2*m2*l1*r2*xx[3]*sin(xx[1]), 
        2*m2*l1*r2*sin(xx[1])*(xx[2])
    ])
    dCdx3 = array([
        -m2*l1*r2*sin(xx[1])*(2*xx[3] + 2*xx[2]),
        0
    ])
    dCdx = array([dCdx0, dCdx1, dCdx2, dCdx3])

    # Tensor derivative of the gravity forces matrix w.r.t. the state vector (2x1x4)
    dGdx0 = array([
        g*(m1*r1+m2*l1)*cos(xx[0])+g*m2*r2*cos(xx[0]+xx[1]),
        g*m2*r2*cos(xx[0]+xx[1])
    ])
    dGdx1 = array([
        g*m2*r2*cos(xx[0]+xx[1]), 
        g*m2*r2*cos(xx[0]+xx[1])
    ])
    dGdx2 = array([0, 0])
    dGdx3 = array([0, 0])
    dGdx = array([dGdx0, dGdx1, dGdx2, dGdx3])

    # Tensor derivative of the friction forces matrix w.r.t. the state vector (2x2x4)
    # dFdx = [zeros((2,2)), zeros((2,2)), zeros((2,2)), zeros((2,2))]

    # Definition of the 4x1 column vector state at time t+1 according to the discretized dynamic.
    # (notice that this is the definition of the discretized dynamic function f=[f0, f1, f2, f3]')
    xxp = zeros((ns, 1))
    xxp[0] = xx[0] + dt*xx[2]
    xxp[1] = xx[1] + dt*xx[3]
    xxp[2:4] = (xx[2:4] + dt*invM@(U-G-F@array([xx[2], xx[3]])-C)).reshape(2,1)

    if onlyZeroOrderDynamic: return xxp.squeeze()

    # Jacobian of the dynamic function w.r.t. x at xx,uu (dfdx, 4x4) (notice f=[f0, f1, f2, f3]')
    # Notice that the minor 2x2 extracted from dfdx at last two rows and first two columns if the Jacobian of [f2, f3]' w.r.t. [x0, x1]'
    # Notice that the minor 2x2 extracted from dfdx at last two rows and last two columns if the Jacobian of [f2, f3]' w.r.t. [x2, x3]'
    # Notice that [f2, f3]'=invM(x1)*(U-G(x0,x1)-F*[x2, x3]'-C(x0,x1,x2,x3))
    dfdx = zeros((ns, ns))
    dfdx[0, :] = [1, 0, dt, 0]
    dfdx[1, :] = [0, 1, 0, dt]
    dfdx[2:4, 0] = dt*(invM@(-dGdx[0]-dCdx[0]))
    dfdx[2:4, 1] = dt*(invM@(-dGdx[1]-dCdx[1]) + dinvMdx1@(U-F@array([xx[2], xx[3]])-C-G))
    dfdx[2:4, 2] = array([1, 0]) + dt*(invM@(-dGdx[2]-F@array([1, 0])-dCdx[2]))
    dfdx[2:4, 3] = array([0, 1]) + dt*(invM@(-dGdx[3]-F@array([0, 1])-dCdx[3]))

    # Jacobian of the dynamic function w.r.t. u at xx,uu (dfdu, 4x1) (notice f=[f0, f1, f2, f3]')
    dfdu = zeros((ns, ni))
    dfdu[2:4, 0] = dt*invM@array([1, 0])

    return xxp.squeeze(), dfdx, dfdu

def runDynamicFunction(discretizedDynamicFuntion, uu, xx0):
    """
    Generic implementation of a forward-in-time evolution (in an open loop fashon) of the dynamic of a certain system.
    Arguments:
    - discretizedDynamicFuntion: functions of (xx_t, uu_t) that implements the discretized dynamics of the system that is being considered,
                                 requiring as arguments respectively the state and input values at time t,
                                 returning the state value at time t+1 AND all the jacobians and hessians of the dynamics wrt state and input in the following order:
                                 xxp, dfdx, dfdu, d2fdxdx, d2fdxdu, d2fdudx, d2fdudu
    - uu: nixTT column vector input curve
    - xx0: nsx1 column vector initial state
    Returns:
    - xx: column vector state trajectory obtained by running the dynamic of the system (from a python variable p.o.f., this is of shape (ns,TT))
    """

    # TT is the number of time steps (each one of duration dt, enough for evolve from t=0 to t=T, where [0, T] is the considered horizon)
    TT = uu.shape[1]
    xx = zeros((len(xx0), TT))
    xx[:,0] = xx0
    for tt in range(TT-1):
        xx[:,tt+1] = discretizedDynamicFuntion(xx[:,tt], uu[:,tt], onlyZeroOrderDynamic = True)
    return xx

def dynamicC(xx):
    """ Given the state vector xx, this function returns the Coriolis and centrifugal forces matrix (2x1)"""
    return array([
        -m2*l1*r2*xx[3]*sin(xx[1])*(xx[3]+2*xx[2]),
        m2*l1*r2*sin(xx[1])*(xx[2])**2
    ])

def dynamicG(xx):
    """ Given the state vector xx, this function returns the gravity forces matrix (2x1)"""
    return array([
        g*(m1*r1+m2*l1)*sin(xx[0])+g*m2*r2*sin(xx[0]+xx[1]), 
        g*m2*r2*sin(xx[0]+xx[1])
    ])

def computeLocalLinearization(xx_traj, uu_traj):
    """
    Given a feasible state-input trajectory of states and inputs,
    this function computes the local linearization of the dynamics around that trajectory.
    Arguments:
    - xx_traj: nsxTT column vector state trajectory
    - uu_traj: nixTT column vector input trajectory
    Returns:
    - AA: nsxnsxTT tensor of jacobians of the dynamics w.r.t. the state at each time instant
    - BB: nsxnixTT tensor of jacobians of the dynamics w.r.t. the input at each time instant
    """
    xx_traj, uu_traj, ns, ni, TT = correctStateInputCurvesShapes(xx_traj, uu_traj)
    AA = zeros((ns, ns, TT))
    BB = zeros((ns, ni, TT))
    for tt in range(TT): 
        AA[:,:,tt], BB[:,:,tt] = discretizedDynamicFRA(xx_traj[:,tt], uu_traj[:,tt])[1:3]
    return AA, BB