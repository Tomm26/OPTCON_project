import numpy as np
from matplotlib import pyplot as plt
from ..parameters import *
from ..controllers.cost import stagecost, termcost
from ..dynamics import dynamics
from ..controllers.lqr import ltv_LQR
from tqdm import tqdm
from ..controllers.armijo import select_stepsize
from ..equilibrium import find_equilibrium, generate_reference_trajectory

def run_task1():
    # Time parameters
    tf = 5  # Total time in seconds (increased for smoother transition)
    TT = int(tf/dt) 
    max_iters = 500

    # Optimization parameters
    fixed_stepsize = 1e-3
    Armijo = True
    stepsize_0 = 1
    cc = 0.8
    beta = 0.2
    armijo_maxiters = 20

    # Find equilibrium points
    print("Computing equilibrium points...")
    x_eq1, u_eq1 = find_equilibrium(np.pi/4, -np.pi/4)
    x_eq2, u_eq2 = find_equilibrium(-np.pi/4, np.pi/4)
    
    # Generate smooth reference trajectory
    print("Generating reference trajectory...")
    xx_ref, uu_ref = generate_reference_trajectory(
        x_eq1, x_eq2, u_eq1, u_eq2, tf, dt)

    # Initialize trajectories
    xx_init = np.repeat(xx_ref[:,0].reshape(-1,1), TT, axis=1)
    uu_init = np.repeat(uu_ref[:,0].reshape(-1,1), TT, axis=1)

    # Arrays for optimization
    xx = np.zeros((ns, TT, max_iters))
    uu = np.zeros((ni, TT, max_iters))
    xx[:,:,0] = xx_init
    uu[:,:,0] = uu_init
    x0 = xx_ref[:,0]

    JJ = np.zeros(max_iters)
    deltau = np.zeros((ni,TT, max_iters))
    descent_arm = np.zeros(max_iters)
    lmbd = np.zeros((ns, TT, max_iters))

    print("Starting optimization...")
    for kk in tqdm(range(max_iters-1)):
        # Compute cost
        JJ[kk] = 0
        for tt in range(TT-1):
            JJ[kk] += stagecost(xx[:, tt, kk], uu[:, tt, kk], 
                               xx_ref[:, tt], uu_ref[:, tt])[0]
        JJ[kk] += termcost(xx[:, -1, kk], xx_ref[:, -1])[0]

        if kk % 10 == 0:
            print(f"Iteration {kk}, Cost: {JJ[kk]:.4f}")

        # Build LTV approximation matrices
        QQt = np.zeros((ns, ns, TT))
        RRt = np.zeros((ni, ni, TT-1))
        SSt = np.zeros((ni, ns, TT-1))

        # Get quadratic approximation of the cost
        for tt in range(TT-1):
            QQt[:,:,tt] = stagecost(xx[:,tt,kk], uu[:,tt,kk], 
                                   xx_ref[:,tt], uu_ref[:,tt])[3]
            RRt[:,:,tt] = stagecost(xx[:,tt,kk], uu[:,tt,kk], 
                                   xx_ref[:,tt], uu_ref[:,tt])[4]

        QQt[:,:,TT-1] = termcost(xx[:,TT-1,kk], xx_ref[:,TT-1])[2]

        # Get dynamics linearization
        grads_fx = np.zeros((ns, ns, TT-1))
        grads_fu = np.zeros((ns, ni, TT-1))
        for tt in range(TT-1):
            fx, fu = dynamics(xx[:,tt,kk], uu[:,tt,kk])[1:3]
            grads_fx[:,:,tt] = fx
            grads_fu[:,:,tt] = fu

        # Get cost gradients
        qqin = np.array([stagecost(xx[:,tt,kk], uu[:,tt,kk], 
                        xx_ref[:,tt], uu_ref[:,tt])[1] for tt in range(TT-1)]).T
        rrin = np.array([stagecost(xx[:,tt,kk], uu[:,tt,kk], 
                        xx_ref[:,tt], uu_ref[:,tt])[2] for tt in range(TT-1)]).T
        qqfin = termcost(xx[:,TT-1,kk], xx_ref[:,TT-1])[1]

        # Solve LTV-LQR problem
        KK, sigma, PP, xx_pred, uu_pred = ltv_LQR(
            grads_fx, grads_fu, QQt, RRt, SSt, QQt[:,:,TT-1], 
            TT, x0, qqin, rrin, qqfin)

        # Update with Armijo stepsize
        if Armijo:
            descent_arm[kk] = np.sum(np.linalg.norm(uu_pred, axis=0)**2)
            stepsize = select_stepsize(
                stepsize_0, armijo_maxiters, cc, beta,
                uu_pred, xx_ref, uu_ref, x0, 
                uu[:,:,kk], JJ[kk], descent_arm[kk], False)
        else:
            stepsize = fixed_stepsize

        # Update trajectories
        uu[:,:,kk+1] = uu[:,:,kk] + stepsize * uu_pred
        
        # Forward simulate
        xx[:,0,kk+1] = x0
        for tt in range(TT-1):
            xx[:,tt+1,kk+1] = dynamics(xx[:,tt,kk+1], uu[:,tt,kk+1])[0]

        # Check convergence
        if kk > 10 and abs(JJ[kk] - JJ[kk-1]) < 1e-6:
            print(f"Converged after {kk} iterations")
            break

    return xx, uu, xx_ref, uu_ref, JJ[:kk+1]

if __name__ == "__main__":
    # Run optimization
    xx, uu, xx_ref, uu_ref, JJ = run_task1()
    
    # Plot results
    plt.figure(figsize=(12,8))
    
    # Plot joint angles
    plt.subplot(311)
    plt.plot(xx_ref[0,:], 'r--', label='θ₁ ref')
    plt.plot(xx[0,:,-1], 'r-', label='θ₁ opt')
    plt.plot(xx_ref[1,:], 'b--', label='θ₂ ref')
    plt.plot(xx[1,:,-1], 'b-', label='θ₂ opt')
    plt.grid(True)
    plt.legend()
    plt.ylabel('Joint Angles [rad]')
    
    # Plot joint velocities
    plt.subplot(312)
    plt.plot(xx_ref[2,:], 'r--', label='ω₁ ref')
    plt.plot(xx[2,:,-1], 'r-', label='ω₁ opt')
    plt.plot(xx_ref[3,:], 'b--', label='ω₂ ref')
    plt.plot(xx[3,:,-1], 'b-', label='ω₂ opt')
    plt.grid(True)
    plt.legend()
    plt.ylabel('Angular Velocities [rad/s]')
    
    # Plot input torque
    plt.subplot(313)
    plt.plot(uu_ref[0,:], 'g--', label='u ref')
    plt.plot(uu[0,:,-1], 'g-', label='u opt')
    plt.grid(True)
    plt.legend()
    plt.ylabel('Input Torque [Nm]')
    plt.xlabel('Time steps')
    
    plt.tight_layout()
    plt.show()
    
    # Plot cost convergence
    plt.figure(figsize=(8,6))
    plt.semilogy(JJ)
    plt.grid(True)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost Convergence')
    plt.show()