import numpy as np

# discretization step
dt = 1e-2

#number of states
ns = 4
ni = 1

#parameters set 3
m1, m2, l1, l2, r1, r2, I1, I2, g, f1, f2 = 1.5, 1.5, 2, 2, 1, 1, 2, 2, 9.81, 0.1, 0.1

#newton's fixed stepsize
fixed_stepsize = 0.7

# Armijo parameters
stepsize_0 = 1.0
cc = 0.5
beta = 0.7
armijo_maxiters = 10

# Task1
Q1 = np.diag([20.0, 20.0, 1.0, 1.0]) # 15 40 15 13
R1 = 0.0005*np.eye(1)
QT1 = None

# Task2
Q2 = np.diag([15.0, 15.0, 1.0, 1.0])
R2 = 0.0005*np.eye(1)
QT2 = None

# LQR
Q3 = np.diag([15.0, 15.0, 1.0, 1.0])
R3 = 0.0005 * np.eye(1)
QT3 = np.diag([20.0, 20.0, 1.0, 1.0])

# MPC
Q4 = np.diag([30.0, 30.0, 1.0, 1.0])
R4 = 0.00001 * np.eye(1)
N4 = 10