import matplotlib.animation as animation 
import matplotlib.pyplot as plt 
import numpy as np
from ..dynamics import dynamics

l1,l2 = 2,2

num_steps = 3000
u0 = 300
x = np.array([np.pi/4, 0,0,0])

dt = 1e-3


# Initialize figure
fig, ax = plt.subplots()
ax.set_xlim(-2 * (l1 + l2), 2 * (l1 + l2))
ax.set_ylim(-2 * (l1 + l2), 2 * (l1 + l2))
ax.set_aspect('equal')

# Lines representing the arm
line, = ax.plot([], [], 'o-', lw=2)

# Function to initialize the plot
def init():
    line.set_data([], [])
    return line,

# Function to update the plot
def update(frame):
    global x
    # Compute next state
    x = dynamics.dynamics(x, u0)

    # Extract angles
    theta1, theta2 = x[0], x[1]

    # Compute joint positions
    x0, y0 = 0, 0  # Base of the arm
    x1, y1 = l1 * np.sin(theta1), -l1 * np.cos(theta1)
    x2, y2 = x1 + l2 * np.sin(theta1 + theta2), y1 - l2 * np.cos(theta1 + theta2)

    # Update line data
    line.set_data([x0, x1, x2], [y0, y1, y2])
    return line,

# Create the animation
ani = animation.FuncAnimation(fig, update, init_func=init, blit=True, interval=1e-3)

# Display the animation
plt.show()