
# Flexible Robotic Arm Task3: after linearizing the dynamics of the FRA around a given trajectory,
# exploiting the LQR algorithm to define the optimal feedback controller to track the said trajectory.

from miscellaneous import correctStateInputCurvesShapes, saveDataOnFile, loadDataFromFile, TrjTrkCntrlData, RegulatorType
from dynamics import computeLocalLinearization
from solver import solveARE, solveLQP
from regulators import runLQRController, generateInitialStateNoise
from dynamics import discretizedDynamicFRA
from datetime import datetime
from numpy import diag, eye
import logger

taskName = "task3"
def task3(xx_traj, uu_traj, lazyExecution = False):
    
    if lazyExecution:
        data = loadDataFromFile(taskName)
        if data is not None :
            logger.log("Task3 data loaded from file")
            return data

    logger.log("Computing the linearization of the dynamics around the given trajectory...")
    xx_traj, uu_traj, _, ni, TT = correctStateInputCurvesShapes(xx_traj, uu_traj)
    AAlin, BBlin = computeLocalLinearization(xx_traj, uu_traj)

    logger.log("Defining cost matrices")
    QQ = diag([16.0,16.0,6.0,6.0])
    RR = 0.001*eye(ni)

    logger.log("Computing the terminal cost matrix as the solution of the ARE (alias Pinfinity)")
    QQT = solveARE(AAlin[:,:,-1], BBlin[:,:,-1], QQ, RR, None)

    logger.log("Computing the T.V. LQR (Linear Quadratic Regulator), optimal feedback controller")
    KK = solveLQP(AAlin, BBlin, QQ, RR, QQT, TT, xx_traj[:,0])[0]

    # Defining some initial disturbance levels (in %) and eventually add the generation of measure noises
    xx0disturbanceLevels = [0.0, 0.1, 0.2]
    generateMeasureNoises = False
    # xx0disturbanceLevels = [0.0]
    # generateMeasureNoises = True

    # Run the LQR on the given trajectory
    startComputingTime = datetime.now()
    xx0disturbances = []
    tracks = []
    for dp in xx0disturbanceLevels:
        logger.log(f'Running the LQR controller tracking the given trajectory (xx0Disturbance~N(0,p) with p={dp*100}% of state S.D.)...')
        xx0disturbance = generateInitialStateNoise(xx_traj, dp)
        xx0disturbances.append(xx0disturbance)
        tracks.append(runLQRController(xx_traj, uu_traj, KK, discretizedDynamicFRA, xx0disturbance, generateMeasureNoises))

    logger.log("Saving results on file and returning them")
    data = TrjTrkCntrlData(xx_traj, uu_traj, tracks, xx0disturbances, generateMeasureNoises, RegulatorType.LQR, startComputingTime, datetime.now())
    saveDataOnFile(data, taskName)
    return data
