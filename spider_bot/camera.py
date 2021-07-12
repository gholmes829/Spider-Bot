"""

"""
from icecream import ic
import pybullet as pb
import numpy as np

def rotation_generator(theta: float) -> callable:
    """Returns lambda func to rotate tuple CCW by <theta> degrees"""
    theta_rad = np.radians(theta)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    return lambda v: (
        round(v[0] * cos_theta - v[1] * sin_theta, 5),
        round(v[0] * sin_theta + v[1] * cos_theta, 5)
    )

class Camera:
    def __init__(self, initial_pos: tuple = (0, 0, 0)) -> None:
        assert initial_pos[2] >= 0, f'Invalid initial pos: {initial_pos}'
        self.initial_pos = list(initial_pos)
        self.pos = initial_pos
        self.pos_func = lambda: self.pos
        self.yaw = 10
        self.pitch = -15
        self.dist = 5
        self.target_lock = False
        
        self.rotate_cw = rotation_generator(-90)
        self.rotate_ccw = rotation_generator(90)
        
    def update(self):
        self.pos = self.pos_func()
        pb.resetDebugVisualizerCamera(cameraDistance = self.dist, cameraYaw = self.yaw, cameraPitch = self.pitch, cameraTargetPosition = self.pos)   
    
    def get_direction(self):
        return list(pb.getDebugVisualizerCamera()[5])[:-1]  # all but z axis
        
    def change_x(self, magnitude):
        direction = self.get_direction()
        # get left orthogonal vector
        orthogonal_direction = self.rotate_cw(direction)
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
        self.dist = 5
        
    def clear_target(self):
        self.pos_func = lambda: self.pos
        
    def reset(self):
        self.pos = self.initial_pos
        self.pos_func = lambda: self.pos
        self.yaw = 10
        self.pitch = -15
        self.dist = 5
        self.target_lock = False
        