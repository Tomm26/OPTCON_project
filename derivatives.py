import sympy as sy
import parameters as dyn
import numpy as np

x1,x2,x3,x4, u, a, b, c, d, f1, f2, g, e= sy.symbols('x1 x2 x3 x4 u a b c d f1 f2 g e')

x = sy.Matrix([[x1], [x2], [x3], [x4]])

#define all the useful matrices
M = sy.Matrix([[a + b*sy.cos(x2), c + d*sy.cos(x2)], 
                  [c + d*sy.cos(x2), c]])

C = sy.Matrix([[-d*x4*sy.sin(x2)*(x4 + 2*x3)], 
                  [d*sy.sin(x2)*x3*x3]])

G = sy.Matrix([[g*sy.sin(x1) + e*sy.sin(x1 + x2)],
                  [e*sy.sin(x1 + x2)]])

F = sy.Matrix([[f1, 0], 
                [0, f2]])

U = sy.Matrix([[u], [0]])

#define the dynamics
f34 = M.inv()*(U - C - F*sy.Matrix([[x3], [x4]]) - G)
f12 = sy.Matrix([[x3], [x4]])
f = sy.Matrix.vstack(f12, f34)

#compute gradient as transpose of jacobian
grad = f.jacobian(x)

#compute the gradient wrt u
grad_u = f.diff(u)

#compute the hessians 
hess_xx = [grad[:, i].jacobian(x).T for i in range(dyn.ns)]
hess_xu = sy.Matrix([grad[:,i].diff(u) for i in range(dyn.ns)]).reshape(4,4).T

hess_ux = grad_u.jacobian(x).T
hess_uu = grad_u.diff(u).T

#lambdify the gradients
pars = sy.Matrix([[x1], [x2], [x3], [x4], [u], [a], [b],  [c], [d],  [f1], [f2],  [g], [e]])

lgrad = sy.lambdify(pars, grad)
lgrad_u = sy.lambdify(pars, grad_u)

#lambdify the hessians
lhess_xx = [sy.lambdify(pars, hess_xx[i]) for i in range(dyn.ns)]
lhess_xu = sy.lambdify(pars,hess_xu)

lhess_ux = sy.lambdify(pars, hess_ux)
lhess_uu = sy.lambdify(pars, hess_uu)

#parameters
m1, m2, l1, l2, r1, r2, I1, I2, g, f1, f2 = dyn.m1, dyn.m2, dyn.l1, dyn.l2, dyn.r1, dyn.r2, dyn.I1, dyn.I2, dyn.g, dyn.f1, dyn.f2
a = I1 + I2 + m1*r1*r1 + m2*(l1*l1 + r2*r2)
b = 2*m2*l1*r2
c = I2 + m2*r2*r2
d = m2*l1*r2
g = g*(m1*r1 + m2*l1)
e = g*m2*r2

def gradient(x, u):
    #x and u are numpy arrays (4,1) and (1,1)
    gradx = np.identity(dyn.ns) + dyn.dt * np.array(lgrad(*x, *u, a,b,c,d,f1,f2,g,e))
    gradu = dyn.dt * np.array(lgrad_u(*x, *u, a,b,c,d,f1,f2,g,e))
    return  gradx, gradu

def hessian(x, u):
    #hessian 11, 12, 21, 22
    return (np.array([lhess_xx[i](*x, *u, a, b, c, d, f1, f2, g, e) for i in range(dyn.ns)]), 
            np.array(lhess_xu(*x, *u, a, b, c, d, f1, f2, g, e)), 
            np.array(lhess_ux(*x, *u, a, b, c, d, f1, f2, g, e)),
            np.array(lhess_uu(*x, *u, a, b, c, d, f1, f2, g, e)) )