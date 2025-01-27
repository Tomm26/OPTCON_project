#
# Gradient Method for Optimal Control
# Armijo stepsize selection rule 
# Marco Falotico
# Bologna, 8/11/2024
#


import numpy as np
import matplotlib.pyplot as plt
import cost as cst
import dynamics as dyn


def select_stepsize(stepsize_0, armijo_maxiters, cc, beta, deltau, xx_ref, uu_ref,  x0, xx, uu, JJ, descent_arm, KK, sigma,TT, plot = False):

      """
      Computes the stepsize using Armijo's rule.
      input parameters:
            - stepsize_0 : initial stepsize guess,
            - armijo_maxiters : maximum number of iterations for armijo rule,
            - deltau : descending direction for the control action,
            - xx_ref : reference trajectory state,
            - uu_ref : reference trajectory input,
            - x0 : initial state,
            - uu : input at current iteration,
            - JJ : cost at current iteration,
            - descent_arm: armijo descent direction at current iteration,
            - plot: whether or not to show descent plot.

      output parameters:
            - stepsize
      """

      TT = uu.shape[1]


      stepsizes = []  # list of stepsizes
      costs_armijo = []

      stepsize = stepsize_0

      ns = xx_ref.shape[0]
      ni = uu_ref.shape[0]


      for ii in range(armijo_maxiters):

            # temp solution update

            xx_temp = np.zeros((ns,TT))
            uu_temp = np.zeros((ni,TT))

            xx_temp[:,0] = x0


            for tt in range(TT-1):
                uu_temp[:, tt] = uu[:, tt] + KK[:,:,tt]@(xx_temp[:,tt] - xx[:,tt]) + stepsize * sigma[:,tt]
                xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt], uu_temp[:, tt])[0]
                

            # temp cost calculation
            JJ_temp = 0

            for tt in range(TT-1):
                  JJ_temp += cst.stagecost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt])[0]

            temp_cost = cst.termcost(xx[:,-1], xx_ref[:,-1])[0]
            JJ_temp += temp_cost

            stepsizes.append(stepsize)      # save the stepsize
            costs_armijo.append(np.min([JJ_temp, 100*JJ]))    # save the cost associated to the stepsize

            #print(JJ_temp, JJ)
            if JJ_temp >= JJ  + cc*stepsize*descent_arm:
                  # update the stepsize
                  stepsize = beta*stepsize

            else:
                  print('Armijo stepsize = {:.3e}'.format(stepsize))
                  break
            
            if ii == armijo_maxiters -1:
                  print("WARNING: no stepsize was found with armijo rule!")
            
            
      ############################
      # Descent Plot
      ############################

      if plot:

            steps = np.linspace(0,stepsize_0,int(2e1))
            costs = np.zeros(len(steps))

            for ii in range(len(steps)):

                  step = steps[ii]

                  # temp solution update

                  xx_temp = np.zeros((ns,TT))
                  uu_temp = np.zeros((ni,TT))

                  xx_temp[:,0] = x0

                  for tt in range(TT-1):
                        uu_temp[:, tt] = uu[:, tt] + KK[:,:,tt]@(xx_temp[:,tt] - xx[:,tt]) + step * sigma[:,tt]
                        xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt], uu_temp[:, tt])[0]

                    # temp cost calculation
                  JJ_temp = 0

                  for tt in range(TT-1):
                        temp_cost = cst.stagecost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt])[0]
                        JJ_temp += temp_cost

                  temp_cost = cst.termcost(xx_temp[:,-1], xx_ref[:,-1])[0]
                  JJ_temp += temp_cost

                  costs[ii] = np.min([JJ_temp, 100*JJ])


            plt.figure(1)
            plt.clf()

      
            plt.plot(steps, costs, color='y', label='$J(\\mathbf{u}^k - stepsize*d^k)$')
            plt.plot(steps, JJ + descent_arm*steps, color='r', label='$J(\\mathbf{u}^k) - stepsize*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
            # plt.plot(steps, JJ - descent*steps, color='r', label='$J(\\mathbf{u}^k) - stepsize*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
            plt.plot(steps, JJ + cc*descent_arm*steps, color='g', linestyle='dashed', label='$J(\\mathbf{u}^k) - stepsize*c*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')

            plt.scatter(stepsizes, costs_armijo, marker='*') # plot the tested stepsize

            plt.grid()
            plt.xlabel('stepsize')
            plt.legend()
            plt.draw()

            plt.show()

      return stepsize
