
# Project Animator

from matplotlib.animation import FuncAnimation
from parameters import discretizationStep as dt, l1, l2, savesFolder
import matplotlib.pyplot as plt
from numpy import sin, cos
import os

def animateFRA(xx_star, xx_ref, lab1 = "path1", lab2 = "path2", showAnimation = True):

    TT = xx_star.shape[1]

    # Set up the figure and axis for the animation
    fig, ax = plt.subplots()
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-4.5, 4.5)

    # Adding grid and adjusting aspect ratio
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')

    # Plot elements
    pendulum_line1, = ax.plot([], [], 'o-', lw=3, color="blue", label=f'{lab1}')         
    pendulum_line2, = ax.plot([], [], 'o-', lw=3, color="blue")                                                          
    reference_line1, = ax.plot([], [], 'o-', lw=2, color="green", label=f'{lab2}')
    reference_line2, = ax.plot([], [], 'o-', lw=2, color="green") 
    time_text = ax.text(0.045, 0.875, '', transform=ax.transAxes)

    titl = f'Pendulum Trajectory: {lab1} VS {lab2}'
    ax.legend()
    ax.set_title(titl)
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    fig.canvas.manager.set_window_title(titl)

    # Initial setup function for the animation
    def init():
        pendulum_line1.set_data([], [])
        pendulum_line2.set_data([], [])
        reference_line1.set_data([], [])
        reference_line2.set_data([], [])
        time_text.set_text('')
        return pendulum_line1, pendulum_line2, reference_line1, reference_line2, time_text

    # Update function for each frame of the animation
    def update(frame):
        # Pendulum position (optimal solution)
        x1_opt = l1*sin(xx_star[0, frame])  
        y1_opt = -l1*cos(xx_star[0, frame])
        x2_opt = x1_opt + l2*sin(xx_star[0, frame]+xx_star[1, frame])
        y2_opt = y1_opt - l2*cos(xx_star[0, frame]+xx_star[1, frame])
        # Update pendulum line
        pendulum_line1.set_data([0, x1_opt], [0, y1_opt])            
        pendulum_line2.set_data([x1_opt, x2_opt], [y1_opt, y2_opt]) 
        # Reference position
        x1_ref = l1*sin(xx_ref[0, frame])  
        y1_ref = -l1*cos(xx_ref[0, frame])
        x2_ref = x1_ref + l2*sin(xx_ref[0, frame]+xx_ref[1, frame])
        y2_ref = y1_ref - l2*cos(xx_ref[0, frame]+xx_ref[1, frame])
        # Update reference line
        reference_line1.set_data([0, x1_ref], [0, y1_ref])            
        reference_line2.set_data([x1_ref, x2_ref], [y1_ref, y2_ref])
        # Update time text
        time_text.set_text(f'time = {frame*dt:.2f}s (over {TT*dt:.2f}s)')
        return pendulum_line1, pendulum_line2, reference_line1, reference_line2, time_text

    # Creating the animation
    an = FuncAnimation(fig, func = update, frames = TT, init_func = init, blit = True, interval = dt*1000)

    # Display the animation
    if (showAnimation): plt.show()

    # Saving the animation
    # an.save(savesFolder + os.sep + "Task5animationFRA.mp4")

    return an