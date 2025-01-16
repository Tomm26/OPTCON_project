# discretization step
dt = 1e-3

# number of states and inputs
ns = 4
ni = 1

# parameters set 3
m1 = 1.5  # mass of link 1
m2 = 1.5  # mass of link 2
l1 = 2.0  # length of link 1
l2 = 2.0  # length of link 2
r1 = 1.0  # distance to center of mass of link 1
r2 = 1.0  # distance to center of mass of link 2
I1 = 2.0  # inertia of link 1
I2 = 2.0  # inertia of link 2
g = 9.81  # gravity acceleration
f1 = 0.1  # friction coefficient 1
f2 = 0.1  # friction coefficient 2

# Export all variables
__all__ = ['dt', 'ns', 'ni', 'm1', 'm2', 'l1', 'l2', 'r1', 'r2', 'I1', 'I2', 'g', 'f1', 'f2']