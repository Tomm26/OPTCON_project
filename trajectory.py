#this will generate a specific trajectory
import numpy as np
from matplotlib import pyplot as plt
import equilibrium as eq

def stepFun(xx_in, xx_fin, TT, k):

    """
    Generates a step function using the sigmoid:
    xx_in:  starting values    (1 x ns)
    xx_fin: ending values      (1 x ns)
    TT:     number of instants (1 x 1 )
    k:      smoothness         (1 x 1 )  [1 for very smooth, around 10 for very edgy]
    """

    xx_ref = []
    for tt in range(TT):
        xx0 = sigmoid(2*tt/TT-1, xx_in[0], xx_fin[0], k)
        xx1 = sigmoid(2*tt/TT-1, xx_in[1], xx_fin[1], k)
        xx2 = xx_in[2]
        xx3 = xx_in[3]

        xx_ref.append([xx0, xx1, xx2, xx3])

    return np.array(xx_ref).T



def sigmoid(x, a, b, k):
    """
    Returns the value of the sigmoid in x, given (a,b) and the smoothness k
    """
    return (a-b) * 1/(1+np.exp(k*x)) + b


