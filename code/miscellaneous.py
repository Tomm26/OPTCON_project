
# Some useful functions (and the classes TrjTrkOCPData and TrjTrkCntrlData) that are used in the project

import os, parameters as params
from numpy import linspace, zeros, zeros_like
import joblib
from parameters import discretizationStep as dt
from datetime import datetime
import plots as plotter
from enum import Enum
from numpy import pi

def getTimeDifferenceAsString(endingTime, startingTime):
    timeDifference = endingTime - startingTime
    hours, remainder = divmod(timeDifference.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return str(hours) + " hours, " + str(minutes) + " minutes, " + str(seconds) + " seconds"

def getTime(): return datetime.now().strftime(params.dateFormat)

def initSavesFolder():
    currentFileDirectory = os.path.dirname(os.path.abspath(__file__))
    params.savesFolder = os.path.join(currentFileDirectory, "saves")
    if not os.path.exists(params.savesFolder): os.makedirs(params.savesFolder)

def saveDataOnFile(data, filename):
    
    if not os.path.exists(params.savesFolder): os.makedirs(params.savesFolder)
    fullFileName = filename + "_" + datetime.now().strftime(params.dateFormat) + ".pkl"
    joblib.dump(data, os.path.join(params.savesFolder, fullFileName))

def loadDataFromFile(filename):

    if not os.path.exists(params.savesFolder): return None
    files = []
    for file in os.listdir(params.savesFolder):
        if file.startswith(filename):
            try:
                date = file[len(filename) + 1 : file.rfind('.')]
                dateStr = datetime.strptime(date, params.dateFormat)
                files.append((file, dateStr))
            except ValueError:
                continue
    if len(files) == 0: return None
    fullFileName = max(files, key=lambda x: x[1])[0]

    return joblib.load(os.path.join(params.savesFolder, fullFileName))

def correctStateInputCurvesShapes(xx, uu):
    
    if (uu.ndim == 1):
        ni = 1
        uu = uu.reshape(ni, uu.shape[0])
    else: ni = uu.shape[0]
    if (xx.ndim == 1):
        ns = 1
        xx = xx.reshape(ns, xx.shape[0])
    else: ns = xx.shape[0]
    # Retrive the TT value, alias, the number of time steps, each one of duration dt,
    # enough for evolve from t=0 to t=T, where [0, T] is the considered horizon)
    TT = xx.shape[1]; TT2 = uu.shape[1]
    if TT != TT2: raise ValueError(
        "The given state curve and input curve do not match in their second dimension, alias TT value ("+str(TT)+" VS "+str(TT2)+")"
    )
    return xx, uu, ns, ni, TT

def generateCleanedIndexCollection(dirtyIndexCollection, maxAmount, forceExtremes = True):
        if not dirtyIndexCollection: return list(range(maxAmount))
        indexesCollection = [int(i) for i in dirtyIndexCollection]
        filter(lambda x: x >= 0 and x < maxAmount, indexesCollection)
        indexesCollection.sort()
        if len(indexesCollection) > maxAmount: indexesCollection = indexesCollection[:maxAmount]
        if indexesCollection[0] != 0 and forceExtremes: indexesCollection[0] = 0
        if indexesCollection[-1] != maxAmount-1 and forceExtremes: indexesCollection[-1] = maxAmount-1
        return indexesCollection

class TrjTrkOCPData:

    def __init__(self, ns, ni, xx_des, uu_des, TT, maxIterations):
        self.startingTime = datetime.now()
        self.endingTime = None
        self.ns = ns
        self.ni = ni
        self.TT = TT
        self.T = dt*TT
        self.t = linspace(0, self.T, TT)
        self.xx_des = xx_des
        self.uu_des = uu_des
        self.xxCollection = zeros((ns, TT, maxIterations))
        self.uuCollection = zeros((ni, TT, maxIterations))
        self.grdJJCollection = zeros_like(self.uuCollection)
        self.costCollection = []
        self.armijoStepsizesCollection = []
        self.armijoCostsCollection = []
        self.armijoLinePendenceCollection = []
        self.armijoStepsizesCollectionPlot = []
        self.armijoCostsCollectionPlot = []
        self.K = None

    def setEndingTime(self):
        self.endingTime = datetime.now()
    def getElapsedTime(self):
        if self.endingTime == None: self.setEndingTime()
        return getTimeDifferenceAsString(self.endingTime, self.startingTime)
    def getNumberOfNeededIteration(self): return self.K
    
    def getOptimalTrajectory(self):
        return self.xxCollection[:,:,self.K], self.uuCollection[:,:,self.K]
    def getOptimalCostGradient(self):
        return self.grdJJCollection[:,:,self.K]
    def getOptimalTrajectoryErrorsAtFinalTime(self):
        x,u = self.getOptimalTrajectory()
        xd = self.xx_des
        ud = self.uu_des
        return x[:,-1]-xd[:,-1], u[:,-2]-ud[:,-2]
    
    def plotStateInputOptimalTrajectory(self):
        xx_opt, uu_opt = self.getOptimalTrajectory()
        return plotter.plotStateInputCurves(
            self.xx_des, self.uu_des, xx_opt, uu_opt,
            'desired', 'optimal', dt
        )
    
    def plotStateInputOptimalTrajectoryEvolution(self, itemsList = None):
        itemsList = generateCleanedIndexCollection(itemsList, self.K+1)
        xxCollectionCast = self.xxCollection[:,:,itemsList]
        uuCollectionCast = self.uuCollection[:,:,itemsList]
        return plotter.plotStateInputCurvesEvolution(
            self.xx_des, self.uu_des, xxCollectionCast, uuCollectionCast,
            'desired', 'optimal', dt,
            "States Trajectories" + f' for iterations: k∈{str(itemsList)}',
            "Input Trajectory" + f' for iterations: k∈{str(itemsList)}'
        ), itemsList
    
    def plotArmijo(self, itemsList = None):
        if itemsList is not None: itemsList = [int(x)-1 for x in itemsList]
        itemsList = generateCleanedIndexCollection(itemsList, self.K, forceExtremes = False)
        armijoStepsizesCollectionCast = [self.armijoStepsizesCollection[i] for i in itemsList]
        armijoCostsCollectionCast = [self.armijoCostsCollection[i] for i in itemsList]
        armijoStepsizesCollectionCastPlot = [self.armijoStepsizesCollectionPlot[i] for i in itemsList]
        armijoCostsCollectionCastPlot = [self.armijoCostsCollectionPlot[i] for i in itemsList]
        armijoLinePendenceCollectionCast = [self.armijoLinePendenceCollection[i] for i in itemsList]
        figs = []
        for i in range(len(itemsList)): figs.append(
            plotter.plotArmijo(
            armijoStepsizesCollectionCast[i],
            armijoCostsCollectionCast[i],
            armijoStepsizesCollectionCastPlot[i],
            armijoCostsCollectionCastPlot[i],
            armijoLinePendenceCollectionCast[i],
            f'Armijo\'s Rule Step Size Selection Behavior for iteration k={itemsList[i]+1}',
        ))
        return figs, itemsList
    
    def plotDescentDirectionNormEvolution(self, itemsList = None):
        itemsList = generateCleanedIndexCollection(itemsList, self.K+1)
        grdJJCollectionCast = self.grdJJCollection[:,:,itemsList]
        return plotter.plotDescentDirectionNormEvolution(grdJJCollectionCast, itemsList)
    
    def plotCostEvolution(self, itemsList = None):
        itemsList = generateCleanedIndexCollection(itemsList, self.K+1)
        return plotter.plotCostEvolution(self.costCollection, itemsList)

class RegulatorType(Enum): LQR = "LQR"; MPC = "MPC"
class TrjTrkCntrlData:

    def __init__(self, xx_traj, uu_traj, tracks, disturbances, measureNoisesIncluded, regulatorType, startComputingTime = None, endComputingTime = None):
        self.xx_traj = xx_traj
        self.uu_traj = uu_traj
        self.tracks = tracks
        self.initialStateDisturbances = disturbances
        self.measureNoisesIncluded = measureNoisesIncluded
        self.regulatorType = regulatorType
        self.startComputingTime = startComputingTime
        self.endComputingTime = endComputingTime

    def getTrack(self, index): return self.tracks[index][0:2]
    def getNoise(self, index): return self.initialStateDisturbances[index]
    def getTracksLength(self): return len(self.tracks)

    def plotTrack(self, index):

        x, u = self.getTrack(index)
        disturbance = ', '.join([f"{n:.2f}" for n in self.getNoise(index)/pi*180])
        withNoises = "\n(with measure noises)" if self.measureNoisesIncluded else ""
        return plotter.plotStateInputCurves(
            self.xx_traj, self.uu_traj, x, u,
            'reference', 'tracked', dt,
            f'States Trajectories Tracked with {self.regulatorType.value}\n(initial state disturbance in degrees Δxx0={disturbance}){withNoises}',
            f'Input Trajectory Tracked with {self.regulatorType.value}\n(initial state disturbance in degrees Δxx0={disturbance}){withNoises}'
        )



