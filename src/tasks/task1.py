import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

from ..parameters import dt, ns, ni
from ..controllers.cost import stagecost, termcost
from ..controllers.armijo import select_stepsize
from ..dynamics import dynamics
from ..equilibrium import find_equilibrium, generate_reference_trajectory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_stage_derivatives(xx, uu, xx_ref, uu_ref, lmbd_next=None):
    """
    Compute derivatives for Newton step using Gauss-Newton approximation with scaling
    """
    # Get dynamics derivatives
    fx, fu = dynamics(xx, uu)[1:3]
    
    # Get cost derivatives
    _, stage_grad_x, stage_grad_u, _, stage_hess_uu = stagecost(
        xx, uu, xx_ref, uu_ref)

    # Scale gradients if too large
    scale = 1.0
    if np.max(np.abs(stage_grad_u)) > 1e3:
        scale = 1e3/np.max(np.abs(stage_grad_u))
        stage_grad_u *= scale
        stage_hess_uu *= scale
    
    # Compute gradient with dynamics
    stage_g = stage_grad_u
    if lmbd_next is not None:
        dyn_grad = fu.T @ lmbd_next
        if np.max(np.abs(dyn_grad)) > 1e3:
            dyn_scale = 1e3/np.max(np.abs(dyn_grad))
            dyn_grad *= dyn_scale
        stage_g += dyn_grad
    
    # Compute Hessian approximation
    stage_H = stage_hess_uu
    if lmbd_next is not None:
        gn_term = fu.T @ fu  # Gauss-Newton term
        if np.max(np.abs(gn_term)) > 1e3:
            gn_scale = 1e3/np.max(np.abs(gn_term))
            gn_term *= gn_scale
        stage_H += gn_term
        
        # Make sure Hessian is well-conditioned
        eigs = np.linalg.eigvals(stage_H)
        if np.max(np.abs(eigs)) > 1e3:
            hess_scale = 1e3/np.max(np.abs(eigs))
            stage_H *= hess_scale
            
        # Ensure symmetry
        stage_H = 0.5 * (stage_H + stage_H.T)
    
    return stage_H, stage_g

def run_task1():
    """
    Compute optimal transition between two equilibrium points
    """
    # Optimization parameters
    max_iters = 100
    reg_eps = 1.0  # Increased regularization
    reg_min = 0.1  # Minimum regularization
    term_tol = 1e-4
    
    # Armijo parameters
    stepsize_0 = 0.01  # Smaller initial stepsize
    cc = 0.1
    beta = 0.5
    armijo_maxiters = 20
    
    # Time parameters
    tf = 5.0
    TT = int(tf/dt)
    
    try:
        # Find equilibrium points
        logger.info("Computing equilibrium points...")
        x_eq1, u_eq1 = find_equilibrium(np.pi/4, -np.pi/4)
        x_eq2, u_eq2 = find_equilibrium(-np.pi/4, np.pi/4)
        
        # Generate reference trajectory
        logger.info("Generating reference trajectory...")
        xx_ref, uu_ref = generate_reference_trajectory(
            x_eq1, x_eq2, u_eq1, u_eq2, tf, dt)
        
        # Initialize optimization variables
        xx = np.zeros((ns, TT, max_iters))
        uu = np.zeros((ni, TT, max_iters))
        
        # Initialize with reference
        xx[:,:,0] = xx_ref
        uu[:,:,0] = uu_ref
        
        x0 = xx_ref[:,0]
        xx[:,0,0] = x0
        
        JJ = np.zeros(max_iters)
        descent_norm = np.zeros(max_iters)
        
        logger.info("Starting Newton optimization...")
        for kk in tqdm(range(max_iters-1)):
            # Forward pass - compute states and cost
            xx[:,0,kk] = x0
            for tt in range(TT-1):
                xx[:,tt+1,kk] = dynamics(xx[:,tt,kk], uu[:,tt,kk])[0]
                
            # Compute cost
            JJ[kk] = 0
            for tt in range(TT-1):
                JJ[kk] += stagecost(xx[:,tt,kk], uu[:,tt,kk], 
                                  xx_ref[:,tt], uu_ref[:,tt])[0]
            JJ[kk] += termcost(xx[:,-1,kk], xx_ref[:,-1])[0]
            
            # Backward pass for costate with scaling
            lmbd = np.zeros((ns, TT))
            lmbd[:,-1] = termcost(xx[:,-1,kk], xx_ref[:,-1])[1]
            
            for tt in reversed(range(TT-1)):
                fx = dynamics(xx[:,tt,kk], uu[:,tt,kk])[1]
                stage_grad_x = stagecost(xx[:,tt,kk], uu[:,tt,kk], 
                                       xx_ref[:,tt], uu_ref[:,tt])[1]
                
                # Safe costate computation with clipping
                lmbd_temp = fx.T @ lmbd[:,tt+1]
                if np.max(np.abs(lmbd_temp)) > 1e3:
                    scale = 1e3/np.max(np.abs(lmbd_temp))
                    lmbd_temp *= scale
                lmbd[:,tt] = lmbd_temp + stage_grad_x
            
            # Build KKT system
            HH = np.zeros((TT*ni, TT*ni))
            grad = np.zeros(TT*ni)
            
            for tt in range(TT-1):
                idx = tt * ni
                stage_H, stage_g = compute_stage_derivatives(
                    xx[:,tt,kk], uu[:,tt,kk],
                    xx_ref[:,tt], uu_ref[:,tt],
                    lmbd[:,tt+1] if tt < TT-1 else None
                )
                
                HH[idx:idx+ni, idx:idx+ni] = stage_H
                grad[idx:idx+ni] = stage_g.flatten()
            
            # Add regularization
            HH_reg = HH + reg_eps * np.eye(TT*ni)
            
            # Solve KKT system
            try:
                deltau_flat = -np.linalg.solve(HH_reg, grad)
                deltau = deltau_flat.reshape(ni, TT)
                
                # Scale large updates
                max_update = np.max(np.abs(deltau))
                if max_update > 10:
                    deltau *= (10/max_update)
                    deltau_flat = deltau.flatten()
                
            except np.linalg.LinAlgError:
                logger.warning("Linear solver failed")
                xx[:,:,kk+1] = xx[:,:,kk]
                uu[:,:,kk+1] = uu[:,:,kk]
                reg_eps *= 2
                continue
            
            # Check descent direction
            descent_norm[kk] = -grad.T @ deltau_flat
            if descent_norm[kk] > -1e-6:
                logger.warning(f"Weak descent direction: {descent_norm[kk]}")
                # Fall back to scaled gradient descent
                deltau = -grad.reshape(ni, TT)
                if np.max(np.abs(deltau)) > 10:
                    deltau *= 10/np.max(np.abs(deltau))
                deltau_flat = deltau.flatten()
                descent_norm[kk] = -grad.T @ deltau_flat
            
            # Line search
            stepsize = select_stepsize(
                stepsize_0=stepsize_0,
                armijo_maxiters=armijo_maxiters,
                cc=cc,
                beta=beta,
                deltau=deltau,
                xx_ref=xx_ref,
                uu_ref=uu_ref,
                x0=x0,
                uu=uu[:,:,kk],
                JJ=JJ[kk],
                descent_arm=descent_norm[kk]
            )
            
            # Update
            if stepsize > 0:
                uu[:,:,kk+1] = uu[:,:,kk] + stepsize * deltau
                xx[:,0,kk+1] = x0
                for tt in range(TT-1):
                    xx[:,tt+1,kk+1] = dynamics(xx[:,tt,kk+1], uu[:,tt,kk+1])[0]
                    
                logger.info(f'Iteration {kk}: Cost = {JJ[kk]:.6f}, '
                          f'Descent = {descent_norm[kk]:.6e}, '
                          f'Stepsize = {stepsize:.6e}')
                
                # Check convergence
                if abs(descent_norm[kk]) < term_tol and kk > 10:
                    logger.info(f"Converged after {kk} iterations")
                    break
                    
                # Check relative improvement
                if kk > 0:
                    rel_improvement = (JJ[kk-1] - JJ[kk])/max(abs(JJ[kk-1]), 1e-8)
                    if rel_improvement < 1e-4 and kk > 20:
                        logger.info("Small relative improvement, stopping")
                        break
                
                # Adapt regularization based on progress
                if kk > 0 and JJ[kk] < JJ[kk-1]:
                    reg_eps = max(reg_min, reg_eps * 0.8)
            else:
                logger.warning("Zero step size, copying previous iteration")
                xx[:,:,kk+1] = xx[:,:,kk]
                uu[:,:,kk+1] = uu[:,:,kk]
                reg_eps *= 2
        
        return xx[:,:,:kk+1], uu[:,:,:kk+1], xx_ref, uu_ref, JJ[:kk+1]
    
    except Exception as e:
        logger.error(f"Error in optimization: {str(e)}")
        raise

def plot_results(xx, uu, xx_ref, uu_ref, JJ):
    """Plot optimization results"""
    tt_grid = np.arange(xx.shape[1])*dt
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12,8))
    
    # Plot angles
    ax1.plot(tt_grid, xx_ref[0,:], 'g--', linewidth=2, label='θ₁ ref')
    ax1.plot(tt_grid, xx_ref[1,:], 'b--', linewidth=2, label='θ₂ ref')
    ax1.plot(tt_grid, xx[0,:,-1], 'g-', linewidth=1.5, alpha=0.8, label='θ₁ opt')
    ax1.plot(tt_grid, xx[1,:,-1], 'b-', linewidth=1.5, alpha=0.8, label='θ₂ opt')
    ax1.grid(True)
    ax1.legend()
    ax1.set_ylabel('Joint Angles [rad]')
    
    # Plot velocities
    ax2.plot(tt_grid, xx_ref[2,:], 'g--', linewidth=2, label='ω₁ ref')
    ax2.plot(tt_grid, xx_ref[3,:], 'b--', linewidth=2, label='ω₂ ref')
    ax2.plot(tt_grid, xx[2,:,-1], 'g-', linewidth=1.5, alpha=0.8, label='ω₁ opt')
    ax2.plot(tt_grid, xx[3,:,-1], 'b-', linewidth=1.5, alpha=0.8, label='ω₂ opt')
    ax2.grid(True)
    ax2.legend()
    ax2.set_ylabel('Angular Velocities [rad/s]')
    
    # Plot input
    ax3.plot(tt_grid, uu_ref[0,:], 'r--', linewidth=2, label='u ref')
    ax3.plot(tt_grid, uu[0,:,-1], 'r-', linewidth=1.5, alpha=0.8, label='u opt')
    ax3.grid(True)
    ax3.legend()
    ax3.set_ylabel('Torque [Nm]')
    ax3.set_xlabel('Time [s]')
    
    plt.tight_layout()
    
    # Plot cost convergence
    plt.figure(figsize=(8,6))
    plt.semilogy(range(len(JJ)), JJ, 'b-', linewidth=2)
    plt.grid(True)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost Convergence')
    plt.show()

if __name__ == "__main__":
    # Run optimization
    xx, uu, xx_ref, uu_ref, JJ = run_task1()
    plot_results(xx, uu, xx_ref, uu_ref, JJ)