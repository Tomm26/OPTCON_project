
# The main file of the project! It contains the code for the execution of all project's tasks!

from miscellaneous import TrjTrkOCPData, TrjTrkCntrlData, initSavesFolder
from task1 import task1
from task2 import task2
from task3 import task3
from task4 import task4
from task5 import task5
import logger

# Initializing the project saves folder
# Be aware: if you change some parameters of the project, you of course should NOT rely on the old saved data!
# In this case, you should delete the folder "Data/Project/saves" before running the code again.
initSavesFolder()

# Tasks selection (from one to five, for each task, set zero to avoid the
# execution, any other value to instead execute the particualr task)
tasks = [1,1,1,1,1]

# Lazyness selection (from one to five, for each task, set zero to avoid the laziness, any other
# value to force the single task to load the results data from the last saved file (if it exists))
lazyness = [True,True,True,True,None]

# Correcting the tasks and lazyness lists (adding an empty element at zero index)
tasks = [None] + tasks
lazyness = [None] + lazyness

if (tasks[1]):
    logger.setActive("task1")
    task1Data: TrjTrkOCPData
    task1Data = task1(lazyness[1])
    logger.log("Plotting results (close all plots to make the code proceed with its execution)")
    task1Data.plotStateInputOptimalTrajectory()
    task1Data.plotStateInputOptimalTrajectoryEvolution([0,1,2,3,task1Data.K])
    task1Data.plotArmijo([4, 5, 8, 9])
    task1Data.plotDescentDirectionNormEvolution()
    task1Data.plotCostEvolution()

if (tasks[2]):
    logger.setActive("task2")
    task2Data: TrjTrkOCPData
    task2Data, _, _ = task2(lazyness[2])
    logger.log("Plotting results (close all plots to make the code proceed with its execution)")
    task2Data.plotStateInputOptimalTrajectory()
    task2Data.plotStateInputOptimalTrajectoryEvolution([0,1,2,3,task2Data.K])
    task2Data.plotArmijo([2, 3, 4])
    task2Data.plotDescentDirectionNormEvolution()
    task2Data.plotCostEvolution()

if (tasks[3]):
    logger.setActive("task3")
    logger.log("Recovering reference trajectory from Task2")
    task2Data, _, _ = task2(lazyExecution = lazyness[2])
    xx_traj, uu_traj = task2Data.getOptimalTrajectory()
    task3Data: TrjTrkCntrlData
    task3Data = task3(xx_traj, uu_traj, lazyness[3])
    logger.log("Plotting results (close all plots to make the code proceed with its execution)")
    for i in range(task3Data.getTracksLength()): task3Data.plotTrack(i)

if (tasks[4]):
    logger.setActive("task4")
    logger.log("Recovering reference trajectory from Task2")
    task2Data, _, _ = task2(lazyExecution = lazyness[2])
    xx_traj, uu_traj = task2Data.getOptimalTrajectory()
    task4Data: TrjTrkCntrlData
    task4Data = task4(xx_traj, uu_traj, lazyness[4])
    logger.log("Plotting results (close all plots to make the code proceed with its execution)")
    for i in range(task4Data.getTracksLength()): task4Data.plotTrack(i)

if (tasks[5]):
    logger.setActive("task5")

    logger.log("Recovering reference trajectory from Task2")
    task2Data, _, _ = task2(lazyExecution = lazyness[2])
    xx_traj, uu_traj = task2Data.getOptimalTrajectory()

    logger.log("Recovering tracked by LQR trajectory from Task3")
    task3Data = task3(xx_traj, uu_traj, lazyExecution = lazyness[3])
    xx_lqr, _ = task3Data.tracks[1] # Recovering the LQR tracked trajectory (the second one, alias with minor initial disturbances)

    logger.log("Animating FRA (close the animation to make the code end its execution)")
    task5(xx_traj, xx_lqr, "Reference Path", "Tracked By LQR Path")
    # task4Data = task4(task2Data.xx_des, task2Data.uu_des, xx_traj, uu_traj, lazyExecution = True)
    # xx_mpc, _ = task4Data.tracks[0]
    # task5(xx_traj, xx_mpc, "Reference Path", "Tracked By MPC Path")
