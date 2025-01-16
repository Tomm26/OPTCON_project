# src/visualization/animate.py
import matplotlib.animation as animation 
import matplotlib.pyplot as plt 
import numpy as np
import sys
from pathlib import Path

# Aggiungi la directory src al path di Python
src_path = Path(__file__).parent.parent
sys.path.append(str(src_path))

try:
    # Prova prima l'import relativo per VS Code
    from ..dynamics import dynamics
    from ..parameters import l1, l2
except ImportError:
    # Fallback all'import assoluto per l'esecuzione diretta
    from dynamics import dynamics
    from parameters import l1, l2

# Animation parameters
num_frames = 200  
u0 = 5.0  
x = np.array([np.pi/6, 0, 0, 0])  

# Initialize figure
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-2 * (l1 + l2), 2 * (l1 + l2))
ax.set_ylim(-2 * (l1 + l2), 2 * (l1 + l2))
ax.set_aspect('equal')
ax.grid(True)
ax.set_title('Double Pendulum Animation')

# Lines representing the arm
line, = ax.plot([], [], 'b-o', lw=3, markersize=12, markerfacecolor='red')

# Function to initialize the plot
def init():
    line.set_data([], [])
    return line,

# Function to update the plot
def update(frame):
    global x
    # Compute multiple steps per frame for smoother animation
    for _ in range(5):  
        x = dynamics(x, u0)

    # Extract angles
    theta1, theta2 = x[0], x[1]

    # Compute joint positions
    x0, y0 = 0, 0  
    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)
    x2 = x1 + l2 * np.sin(theta1 + theta2)
    y2 = y1 - l2 * np.cos(theta1 + theta2)

    # Update line data
    line.set_data([x0, x1, x2], [y0, y1, y2])
    return line,

if __name__ == "__main__":
    # Create the animation with faster update interval
    ani = animation.FuncAnimation(fig, update, 
                                init_func=init,
                                frames=num_frames,
                                interval=10,
                                blit=True)

    # Display the animation
    plt.show()