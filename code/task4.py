
# Flexible Robotic Arm Task4: after linearizing the dynamics of the FRA around a
# given trajectory, exploiting an MPC algorithm to track the said trajectory.

from miscellaneous import correctStateInputCurvesShapes, saveDataOnFile, loadDataFromFile, TrjTrkCntrlData, RegulatorType
from dynamics import computeLocalLinearization
from regulators import runMPCController, generateInitialStateNoise
from dynamics import discretizedDynamicFRA
from datetime import datetime
from numpy import diag, eye, array, hstack
from parameters import discretizationStep as dt
import logger

taskName = "task4"
def task4(xx_traj, uu_traj, lazyExecution = False):
    
    if lazyExecution:
        data = loadDataFromFile(taskName)
        if data is not None:
            logger.log("Task4 data loaded from file")
            return data
        
    logger.log("Defining the prediction time horizon for the MPC (in terms of quantity of time instants)")
    MPC_TT = 400 # int(4/dt)

    logger.log("Computing the linearization of the dynamics around the given trajectory...")
    xx_traj, uu_traj, ns, ni, _ = correctStateInputCurvesShapes(xx_traj, uu_traj)
    # Adding a costant terminal part to the input-state trajectory of the same time-size of the prediction horizon
    xx_traj = hstack((xx_traj, array(xx_traj[:,-1]).reshape((ns, 1)).repeat(MPC_TT, axis=1)))
    uu_traj = hstack((uu_traj, array(uu_traj[:,-1]).reshape((ni, 1)).repeat(MPC_TT, axis=1)))
    AAlin, BBlin = computeLocalLinearization(xx_traj, uu_traj)

    logger.log("Defining cost matrices")
    QQ = diag([16.0,16.0,6.0,6.0])
    RR = 0.0001*eye(ni)

    # Defining some initial disturbance levels (in %) and eventually add the generation of measure noises
    xx0DisturbancesLevels = [0.0, 0.1, 0.2]
    generateMeasureNoises = False
    # xx0DisturbancesLevels = [0.0]
    # generateMeasureNoises = True
    considerAdditionalConstraints = False # True

    # Run the MPC on the given trajectory
    useCVXSolver = True
    startComputingTime = datetime.now()
    xx0Disturbances = []
    tracks = []
    for dp in xx0DisturbancesLevels:
        logger.log(f'Running the MPC controller tracking the given trajectory (xx0Disturbance~N(0,p) with p={dp*100}% of state S.D.)...')
        xx0Disturbance = generateInitialStateNoise(xx_traj, dp)
        xx0Disturbances.append(xx0Disturbance)
        tracks.append(runMPCController(xx_traj, uu_traj, AAlin, BBlin, QQ, RR, MPC_TT, discretizedDynamicFRA,
                                       xx0Disturbance, generateMeasureNoises, useCVXSolver, considerAdditionalConstraints))
    for i in range(len(tracks)):
        xx_track, uu_ttrack = tracks[i]
        xx_track = xx_track[:,:-MPC_TT]
        uu_ttrack = uu_ttrack[:,:-MPC_TT]
        tracks[i] = (xx_track, uu_ttrack)
    xx_traj = xx_traj[:,:-MPC_TT]
    uu_traj = uu_traj[:,:-MPC_TT]

    logger.log("Saving results on file and returning them")
    data = TrjTrkCntrlData(xx_traj, uu_traj, tracks, xx0Disturbances, generateMeasureNoises, RegulatorType.MPC, startComputingTime, datetime.now())
    saveDataOnFile(data, taskName)
    return data
