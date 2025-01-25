
# Parameters of the Flexible Robotic Arm (FRA) system, as well as other parameters used in the code

# Flexible Robotic Arm Parameters
ns = 4      # number of states
ni = 1      # number of inputs
m1 = 1.5    # mass of the first link
m2 = 1.5    # mass of the second link 
l1 = 2.0    #length of the first link
l2 = 2.0    # length of the second link
r1 = 1.0    # distance from the pivot point of the first link to its center of mass
r2 = 1.0    # distance from the pivot point of the second link to its center of mass
I1 = 2.0    # inertia of the first link
I2 = 2.0    # inertia of the second link
g = 9.81    # gravity
f1 = 0.1    # friction associated to the pivot of the first link
f2 = 0.1    # friction associated to the pivot of the second link

# Armijo Rule for stepsize selection parameters
armijoBeta = 0.7
armijoC = 0.5

# The discretization step (used for example in the FRA discretized dynamic), in seconds
discretizationStep = 1e-2

# Absolute folder where to save the results of the various tasks
savesFolder = 'saves'

# Date format used in the name of the files
dateFormat = "%Y-%m-%d_%H-%M-%S"