
# Project Plotter

from numpy import squeeze, linspace, linalg
from matplotlib.pyplot import subplots, tight_layout, title, grid, show, figure, plot, xlabel, ylabel, legend, suptitle, scatter, yscale
import parameters as params

# Color map definition (for the different states)
colorMap = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

def plotStateInputCurves(xxFirst, uuFirst, xxSecond, uuSecond, labelFirst, labelSecond, dt, supTitleFirst = None, supTitleSecond = None, showFigures = True):
    return plotStateInputCurvesEvolution(xxFirst, uuFirst, xxSecond, uuSecond, labelFirst, labelSecond, dt, supTitleFirst, supTitleSecond, showFigures)

def plotStateInputCurvesEvolution(xxFirst, uuFirst, xxSecondCollection, uuSecondCollection, labelFirst, labelSecond, dt, supTitleFirst = None, supTitleSecond = None, showFigures = True):
    """
    Generates ns subplots for desired and optimal state trajectories and a separate plot for the desired and optimal (supposed single) input trajectory
    """
    # Preparation of the data for the plotting
    KK = xxSecondCollection.shape[2] if xxSecondCollection.ndim > 2 else 1
    xxFirst, uuFirst = squeeze(xxFirst), squeeze(uuFirst)
    xxSecond = squeeze(xxSecondCollection) if KK > 1 else squeeze(xxSecondCollection)[:,:,None]
    uuSecond = squeeze(uuSecondCollection) if KK > 1 else squeeze(uuSecondCollection)[:,None]
    TT = xxFirst.shape[1]
    ns, tx = xxFirst.shape[0], squeeze(linspace(0, dt*TT, TT))
    uuFirst = uuFirst[:-1]
    uuSecond = uuSecond[:-1, :] if uuSecond.ndim > 1 else uuSecond[:-1]
    tu = tx[:-1]
    KK = xxSecond.shape[2]
    alpha = linspace(0.33 if KK > 1 else 1, 1, KK)

    # Creating various subplots (one for each state)
    xFigureTitle = supTitleFirst if supTitleFirst else "States Trajectories"
    xFigure, axes = subplots(int(ns/2), 2, figsize = (10, 8))
    xFigure.canvas.manager.set_window_title(xFigureTitle if not supTitleFirst else supTitleFirst)
    if supTitleFirst: suptitle(xFigureTitle)
    axes = axes.flatten()  # Axes flattening for a simpler indexing
    for i in range(ns):
        ax = axes[i]
        color = colorMap[i]
        ax.plot(tx, xxFirst[i, :], '--', color = color, label = f'ϑ{i+1} {labelFirst}')
        for k in range(KK):
            ax.plot(tx, xxSecond[i, :, k], color = color, label = (None if k < KK-1 else f'ϑ{i+1} {labelSecond}'), alpha = alpha[k])
        ax.set_title(f'Theta{i} Trajectory')
        ax.grid(True)
        ax.legend()
        ax.set_xlabel('Time (s)')
        ax.set_ylabel("Value ("+("rad" if i < ns/2 else "rad/s")+")")
    tight_layout()
    if (showFigures): show()

    uFigureTitle = supTitleSecond if supTitleSecond else "Input Trajectory"
    uFigure = figure(figsize=(8, 6))
    uFigure.canvas.manager.set_window_title(uFigureTitle)
    title(uFigureTitle)
    line, = plot(tu, uuFirst, '--', label = f'{labelFirst} input')
    color = line.get_color()
    for k in range(KK):
        plot(tu, uuSecond[:, k], color = color, label = (None if k < KK-1 else f'{labelSecond} input'), alpha = alpha[k])
    xlabel('Time (s)')
    ylabel('Value (Nm)')
    legend()
    grid(True)
    if (showFigures): show()

    return xFigure, uFigure

def plotArmijo(armijoStepsizes, armijoCosts, armijoStepsizesPlot, armijoCostsPlot, armijoLinePendence, pTitle, showFigure = True):
    """ Generates a plot for the Armijo's Rule step size selection behavior"""
    if not armijoStepsizesPlot.any():
        ll = armijoCosts[-1]
        fig = figure(figsize=(8, 6))
        fig.canvas.manager.set_window_title(pTitle)
        title(pTitle)
        plot(armijoStepsizes, armijoCosts, color='k', label='$J(\\mathbf{u}^k+stepsize*d^k)$')
        plot(armijoStepsizes, ll + armijoLinePendence*armijoStepsizes, color='r', label='$J(\\mathbf{u}^k)+stepsize*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
        plot(armijoStepsizes, ll + params.armijoC*armijoLinePendence*armijoStepsizes, color='g', linestyle='dashed', label='$J(\\mathbf{u}^k)+stepsize*c*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
        scatter(armijoStepsizes, armijoCosts, marker='*')
        grid()
        xlabel('stepsize')
        legend()
        tight_layout()
        if (showFigure): show()
    else:
        ll = armijoCostsPlot[0]
        fig = figure(figsize=(8, 6))
        fig.canvas.manager.set_window_title(pTitle)
        title(pTitle)
        plot(armijoStepsizesPlot, armijoCostsPlot, color='k', label='$J(\\mathbf{u}^k+stepsize*d^k)$')
        plot(armijoStepsizesPlot, ll + armijoLinePendence*armijoStepsizesPlot, color='r', label='$J(\\mathbf{u}^k)+stepsize*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
        plot(armijoStepsizesPlot, ll + params.armijoC*armijoLinePendence*armijoStepsizesPlot, color='g', linestyle='dashed', label='$J(\\mathbf{u}^k)+stepsize*c*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
        scatter(armijoStepsizes, armijoCosts, marker='*', color='orange')
        grid()
        xlabel('stepsize')
        legend()
        tight_layout()
        if (showFigure): show()
    return fig

def plotDescentDirectionNormEvolution(grdJJCollection, showFigure = True):

    pTitle = "Descent Direction Norm Evolution"
    fig = figure(figsize=(8, 6))
    fig.canvas.manager.set_window_title(pTitle)
    title(pTitle)
    values = [linalg.norm(grdJJCollection[:,:,k]) for k in range(grdJJCollection.shape[2])]
    plot(range(1, len(values)+1), values)
    xlabel('$k$')
    ylabel('||$d\ell(z^k)||$')
    yscale('log')
    grid()
    if (showFigure): show()
    return fig

def plotCostEvolution(costs, showFigure = True):

    pTitle = "Cost Evolution"
    fig = figure(figsize=(8, 6))
    fig.canvas.manager.set_window_title(pTitle)
    title(pTitle)
    plot(range(1, len(costs)+1), costs)
    xlabel('$k$')
    ylabel('$\ell(z^k)$')
    yscale('log')
    grid()
    if (showFigure): show()
    return fig