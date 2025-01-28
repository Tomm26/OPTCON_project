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
stepsize_0 = 1
cc = 0.5
beta = 0.7
armijo_maxiters = 15 # number of Armijo iterations