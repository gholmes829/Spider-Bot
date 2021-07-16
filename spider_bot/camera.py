"""

"""
from icecream import ic
import pybullet as pb
import numpy as np

def rotation_generator(theta: float) -> callable:
    """Returns matrix to rotate tuple CCW by <theta> degrees"""
    theta_rad = np.radians(theta)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    return np.array([[cos_theta, -sin_theta],
                     [sin_theta, cos_theta]])

class Camera:
    def __init__(self, physics_client, initial_pos: tuple = (0, 0, 0)) -> None:
        assert initial_pos[2] >= 0, f'Invalid initial pos: {initial_pos}'
        self.physics_client = physics_client
        self.initial_pos = list(initial_pos)
        self.pos = initial_pos
        self.pos_func = self.get_pos
        self.yaw = 10
        self.pitch = -15
        self.dist = 1
        self.target_lock = False
        
        self.rotate_cw = rotation_generator(-90)
        
    def update(self):
        self.pos = self.pos_func()
        self.physics_client.resetDebugVisualizerCamera(cameraDistance = self.dist, cameraYaw = self.yaw, cameraPitch = self.pitch, cameraTargetPosition = self.pos)   
    
    def get_direction(self):
        return list(self.physics_client.getDebugVisualizerCamera()[5])[:-1]  # all but z axis
        
    def change_x(self, magnitude):
        direction = self.get_direction()
        # get left orthogonal vector
        orthogonal_direction = self.rotate_cw @ direction
        self.pos[0] += magnitude * orthogonal_direction[0]
        self.pos[1] += magnitude * orthogonal_direction[1]
        
    def change_y(self, magnitude):
        direction = self.get_direction()
        self.pos[0] += magnitude * direction[0]
        self.pos[1] += magnitude * direction[1]
        
    def change_global_z(self, magnitude):
        self.pos[2] = max(0, self.pos[2] + magnitude)
        
    def change_yaw(self, magnitude):
        self.yaw += magnitude
        
    def change_pitch(self, magnitude):
        self.pitch = np.clip(self.pitch + magnitude, -89.99, 89.99)
        
    def change_zoom(self, magnitude):
        self.dist = max(0, self.dist + magnitude)
        
    def track_target(self, pos_func):
        self.pos_func = pos_func
        self.dist = 1
        
    def clear_target(self):
        self.pos_func = self.get_pos
        
    def reset(self):
        self.pos = self.initial_pos
        self.pos_func = self.get_pos
        self.yaw = 10
        self.pitch = -15
        self.dist = 1
        self.target_lock = False
        
    def get_pos(self):
        return self.pos