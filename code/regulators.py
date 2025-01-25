
# Collection of the functions that implement the Regulators used in the Project (and correlated functions)

from miscellaneous import correctStateInputCurvesShapes
from numpy import zeros, std, random, array, abs, max
from cvxpy import Variable, Parameter, Minimize, Problem, quad_form, OSQP
from solver import solveLQP
import logger

def runLQRController(xx_traj, uu_traj, KK, discretizedDynamicFunction, xx0Disturbance = None, generateMeasureNoises = False):
    """
    Run the LQR controller (using the given feedback gain KK) on the given trajectory.
    Arguments:
    - xx_traj: The state reference trajectory
    - uu_traj: The input reference trajectory
    - KK: The feedback gain matrix (related to the LQR)
    - discretizedDynamicFunction: A function that represents the discretized dynamic of the system
    - xx0Disturbance: A disturbance to be eventually added to the initial state
    - generateMeasureNoises: A boolean flag that indicates if the measure noises should be included in the simulation
    Returns:
    - xx_track: The state trajectory tracked by the LQR controller
    - uu_track: The input trajectory applied by the LQR controller
    """
    xx_traj, uu_traj, ns, ni, TT = correctStateInputCurvesShapes(xx_traj, uu_traj)
    xx_real = zeros((ns, TT))
    uu_real = zeros((ni, TT))

    if generateMeasureNoises:
        xx_traj_maxs = max(abs(xx_traj), axis=1)
        percentage = 0.01/5
        xx_traj_noise_std = percentage*xx_traj_maxs

    xx_real[:,0] = xx_traj[:,0] + (xx0Disturbance if xx0Disturbance is not None else zeros(ns))

    # Evolving the dynamic of the system using the LQR (in a closed loop fashion)
    for tt in range(TT-1): 
        
        uu_real[:,tt] = uu_traj[:, tt] + KK[:,:,tt]@(xx_real[:,tt] - xx_traj[:,tt])
        xx_real[:,tt+1] = discretizedDynamicFunction(xx_real[:,tt], uu_real[:,tt], onlyZeroOrderDynamic = True)

        if generateMeasureNoises:
            # noises = [random.normal(loc=0.0, scale=abs(xx_real_val)/100/2, size=1) for xx_real_val in xx_real[:,tt+1]]
            noises = [random.normal(loc=0.0, scale=sc, size=1) for sc in xx_traj_noise_std]
            xx_real[:,tt+1] += array(noises).squeeze()

    return xx_real, uu_real

def runMPCController(xx_traj, uu_traj, AA, BB, QQ, RR, MPC_TT, discretizedDynamicFunction,
                    xx0Disturbance = None, generateMeasureNoises = False, useCVXSolver = True, considerAdditionalConstraints = True):
    """
    Run the MPC controller on the given trajectory.
    Arguments:
    - xx_traj: The state reference trajectory
    - uu_traj: The input reference trajectory
    - AA: Matrix A (alias obtained deriving w.r.t. the state) of the linearized dynamics around the reference trajectory (T.V.)
    - BB: Matrix B (alias obtained deriving w.r.t. the input) of the linearized dynamics around the reference trajectory (T.V.)
    - QQ, RR: Cost matrices (supposed to be time invariant)
    - MPC_TT: Prediction time horizon for the MPC (in terms of quantity of time instants)
    - discretizedDynamicFunction: A function that represents the discretized dynamic of the system
    - xx0Disturbance: A disturbance to be eventually added to the initial state
    - generateMeasureNoises: A boolean flag that indicates if the measure noises should be included in the simulation
    - considerAdditionalConstraints: A boolean flag that indicates if already-set-up additional constraints should be considered.
                                     In case, the cvxpy library is used (instead of the analytic solver) to solve the LQ problems involved
                                     in the MPC action (in order to make it possible to take into account the additional constraints)
    Returns:
    - xx_track: The state trajectory tracked by the MPC controller
    - uu_track: The input trajectory applied by the MPC controller
    """
    xx_traj, uu_traj, ns, ni, TT = correctStateInputCurvesShapes(xx_traj, uu_traj)
    xx_real = zeros((ns, TT))
    uu_real = zeros((ni, TT))

    if generateMeasureNoises:
        xx_traj_maxs = max(abs(xx_traj), axis=1)
        percentage = 0.01/5
        xx_traj_noise_std = percentage*xx_traj_maxs

    # CVXPY problem set-up
    cvxProblem, xx_real_param, uu_real_param, xx_traj_param, uu_traj_param, xx0_real_param, AAparam, BBparam = cvxpyProblemSetup(ns, ni, QQ, RR, QQ, MPC_TT, considerAdditionalConstraints)

    # Generating the real state-trajectory value at time t=0 (an eventually disturbed initial state condition)
    xx_real[:,0] = xx_traj[:,0] + (xx0Disturbance if xx0Disturbance is not None else 0)

    # Computing MPC parameters at each step
    progressPercentageCounter = 0
    for tt in range(TT-MPC_TT-1):

        xx0_real_val = xx_real[:,tt]
        AAval = AA[:,:,tt:tt+MPC_TT-1] # shape=(ns,ns,MPC_TT-1)
        BBval = BB[:,:,tt:tt+MPC_TT-1] # shape=(ns,ni,MPC_TT-1)
        xx_traj_val = xx_traj[:,tt:tt+MPC_TT]
        uu_traj_val = uu_traj[:,tt:tt+MPC_TT]

        if not useCVXSolver and not considerAdditionalConstraints:
            # In order to solve the LQP problem, relying on the analytic solver (uncostrained optimal control case)
            KK = solveLQP(AAval, BBval, QQ, RR, QQ, MPC_TT, xx0_real_val - xx_traj[:,tt])[0]
            uu_real[:,tt] = uu_traj[:, tt] + KK[:,:,0]@(xx_real[:,tt] - xx_traj[:,tt])
        else:
            # In order to solve the LQP problem, relying on the cvxpy library (potentially constrained optimal control case)
            uu_real[:,tt] = cvxpyProblemSolver(
                cvxProblem, xx_real_param, uu_real_param, xx0_real_param, AAparam, BBparam, xx_traj_param, uu_traj_param,
                xx0_real_val, AAval, BBval, xx_traj_val, uu_traj_val
            )[2]

        xx_real[:, tt+1] = discretizedDynamicFunction(xx_real[:, tt], uu_real[:,tt])[0]

        if generateMeasureNoises:
            # noises = [random.normal(loc=0.0, scale=abs(xx_real_val)/100/2, size=1) for xx_real_val in xx_real[:,tt+1]]
            noises = [random.normal(loc=0.0, scale=sc, size=1) for sc in xx_traj_noise_std]
            xx_real[:,tt+1] += array(noises).squeeze()

        # Loggin progress percentage
        progressPercentage = 100*(tt+1)/(TT-MPC_TT-1)
        if int(progressPercentage) % 5 == 0 and int(progressPercentage) > progressPercentageCounter:
            progressPercentageCounter = progressPercentage
            logger.log(f"Progress: {progressPercentage:.0f}%")

    return xx_real, uu_real

def cvxpyProblemSetup(ns, ni, QQ, RR, QQT, MPC_TT, considerAdditionalConstraints):
    """
    This function sets up the CVXPY problem for the MPC controller (if needed)
    Arguments:
    - ns, ni: Number of states and inputs
    - QQ, RR, QQT: States and input cost matrices (supposed to be time invariant)
    - MPC_TT: Prediction time horizon for the MPC (in terms of quantity of time instants)
    - considerAdditionalConstraints: A boolean flag that indicates if already-set-up additional constraints should be considered
    Returns:
    - problem: The set-up CVXPY problem
    - xx_real, uu_real: The state and input CVXPY Variables (that correspond to the prediction that the MPC will compute at each instant of time)
    - xx_traj, uu_traj: The CVXPY Parameters related to the reference trajectory (that the MPC aims to track)
    - xx0_real: The initial state CVXPY Parameter (that correspond to the actual real value of the state at current istant of time, beginning of the prediction horizon)
    - AA, BB: The list of CVXPY Parameters related to the local linearization matrices (on the prediction horizon)
    """

    #Variables
    dx = Variable((ns, MPC_TT))
    du = Variable((ni, MPC_TT))

    #Parameters
    xx0_real = Parameter(ns)  #Initial state
    AA = [Parameter((ns, ns)) for _ in range(MPC_TT-1)]
    BB = [Parameter((ns, ni)) for _ in range(MPC_TT-1)]
    xx_traj = Parameter((ns, MPC_TT))   
    uu_traj = Parameter((ni, MPC_TT))   

    # Definition of the cost functions and the equality constraints related to the dynamic    
    cost = 0
    constraints = []
    xx_real = dx + xx_traj
    uu_real = du + uu_traj
    for tt in range(MPC_TT-1):
        cost += quad_form(dx[:, tt], QQ) + quad_form(du[:, tt], RR)
        constraints += [dx[:, tt+1] == AA[tt]@dx[:, tt] + BB[tt]@du[:, tt]]
    # Definition of the terminal cost function
    cost += quad_form(dx[:,MPC_TT-1], QQT)
    # Definition the initial state constraint
    constraints += [dx[:,0] == xx0_real - xx_traj[:,0]]
    if considerAdditionalConstraints:
        # Additional constraints (related to the input, values in Nm)
        constraints += [uu_real <= 45.0, uu_real >= -10.0]
    # Finally, defintion of the CVXPY problem
    cvxProblem = Problem(Minimize(cost), constraints)

    return cvxProblem, xx_real, uu_real, xx_traj, uu_traj, xx0_real, AA, BB


def cvxpyProblemSolver(problem, xx_real, uu_real, xx0_real, AA, BB, xx_traj, uu_traj, xx0_real_val, AAval, BBval, xx_traj_val, uu_traj_val):
    """ Updating CVXPY parameters and solving the problem """

    # Updating parameters
    xx0_real.value = xx0_real_val
    xx_traj.value = xx_traj_val
    uu_traj.value = uu_traj_val
    for tt in range(len(AA)):
        AA[tt].value = AAval[:,:,tt]
        BB[tt].value = BBval[:,:,tt]

    # Solving the problem
    problem.solve(solver=OSQP, warm_start=True)

    xx_real_val = xx_real.value
    uu_real_val = uu_real.value
    return xx_real_val, uu_real_val, uu_real_val[:,0]

def generateInitialStateNoise(xx, noiseStdPercentage, gainK = 1, randomNumberGenerator = None):
    """
    Generation of a noise to be added to the initial state of the system.
    That noise in generated (for each single state) by taking samples from a N(0,1) normal distribution
    scaled in its standard deviation by the given percentage of the standard deviation (p) of the state itself,
    alias by taking samples from a N(0,p) distribution

    """
    ns = xx.shape[0]
    if not noiseStdPercentage or noiseStdPercentage <= 0: return zeros(ns)

    # Fix a local r.n.g. related to a local seed in order to generate the same noise (in the various Tasks)
    if randomNumberGenerator is None: randomNumberGenerator = random.default_rng(2828)  

    # Compute the standard deviation for each state (row-wise)
    stateSD = std(xx, axis = 1)
    # Scale the standard deviation by the percentage
    noiseSD = gainK*stateSD*noiseStdPercentage
    
    # Generate (and return) a Gaussian noise with personalized standard deviation taking
    # samples from a N(0,1) normal distribution and scaling them by noiseStdPercentage
    return randomNumberGenerator.normal(loc = 0.0, scale = noiseSD, size = ns)
