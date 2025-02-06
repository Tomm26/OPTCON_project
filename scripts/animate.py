import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from parameters import l1, l2

class FlexibleRobotAnimator:
    def __init__(self, trajectory, dt):
        """
        Initialize the animator with a trajectory.
        
        Args:
            trajectory: Array of shape (T, 4) containing [theta1, theta2, dtheta1, dtheta2]
            dt: Time step between frames
        """
        self.trajectory = trajectory
        self.dt = dt
        
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_xlim(-4, 4)
        self.ax.set_ylim(-4, 4)
        self.ax.grid(True)
        self.ax.set_aspect('equal')
        self.ax.set_title('Flexible Robotic Arm Animation')
        
        # Initialize robot components
        self.line, = self.ax.plot([], [], 'b-', lw=4, label='Robot Arms')
        self.joints = [Circle((0, 0), 0.1, fc='k'),
                      Circle((0, 0), 0.1, fc='r'),
                      Circle((0, 0), 0.1, fc='r')]
        
        # Add joints to axis
        for joint in self.joints:
            self.ax.add_patch(joint)
            
        # Add trajectory trace
        self.trace, = self.ax.plot([], [], 'g--', alpha=0.5, label='End Effector Trace',lw=2.2)
        self.trace_x = []
        self.trace_y = []
        
        self.ax.legend()

    def forward_kinematics(self, theta1, theta2):
        """
        Compute the positions of the joints given the angles.
        
        Args:
            theta1: Angle of first link relative to vertical
            theta2: Angle of second link relative to first link
            
        Returns:
            List of (x, y) coordinates for each joint
        """
        # Base position
        x0, y0 = 0, 0
        
        # First joint position
        x1 = l1 * np.sin(theta1)
        y1 = -l1 * np.cos(theta1)
        
        # End effector position
        x2 = x1 + l2 * np.sin(theta1 + theta2)
        y2 = y1 - l2 * np.cos(theta1 + theta2)
        
        return [(x0, y0), (x1, y1), (x2, y2)]

    def init_animation(self):
        """Initialize the animation with empty plots."""
        self.line.set_data([], [])
        for joint in self.joints:
            joint.center = (0, 0)
        self.trace.set_data([], [])
        return [self.line] + self.joints + [self.trace]

    def update(self, frame):
        """Update the animation for each frame."""
        # Get current angles
        theta1 = self.trajectory[frame, 0]
        theta2 = self.trajectory[frame, 1]
        
        # Compute joint positions
        positions = self.forward_kinematics(theta1, theta2)
        
        # Update robot arm line
        x_points = [p[0] for p in positions]
        y_points = [p[1] for p in positions]
        self.line.set_data(x_points, y_points)
        
        # Update joint positions
        for joint, pos in zip(self.joints, positions):
            joint.center = pos
            
        # Update trace
        self.trace_x.append(positions[-1][0])
        self.trace_y.append(positions[-1][1])
        self.trace.set_data(self.trace_x, self.trace_y)
        
        return [self.line] + self.joints + [self.trace]

    def animate(self, save_path=None):
        """
        Create and display the animation.
        
        Args:
            save_path: If provided, save the animation to this path
        """
        anim = FuncAnimation(self.fig, self.update,
                           init_func=self.init_animation,
                           frames=len(self.trajectory),
                           interval=self.dt,
                           blit=True)
        
        if save_path:
            anim.save(save_path, writer='pillow')
            
        plt.show()

# Example usage:
if __name__ == "__main__":
    # Generate sample trajectory
    t = np.linspace(0, 10, 200)
    theta1 = np.pi/4 * np.sin(t)
    theta2 = np.pi/6 * np.cos(t)
    dtheta1 = np.pi/4 * np.cos(t)
    dtheta2 = -np.pi/6 * np.sin(t)
    
    trajectory = np.column_stack((theta1, theta2, dtheta1, dtheta2))
    
    # Create and run animation
    animator = FlexibleRobotAnimator(trajectory, dt=0.05)
    animator.animate()