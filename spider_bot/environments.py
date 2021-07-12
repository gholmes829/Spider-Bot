"""

"""

import pybullet as pb
import pybullet_data
from icecream import ic
import time
import numpy as np
from gym import spaces, Env

from spider_bot.spider_bot_model import SpiderBot
from spider_bot.camera import Camera

class SpiderBotSimulator(Env):
    def __init__(self, spider_bot_model_path: str, real_time: bool = False, gui: bool = True) -> None:
        Env.__init__(self)
        self.spider_bot_model_path = spider_bot_model_path
        self.real_time = real_time
        
        self.gui = gui
        self.physics_client = pb.connect(pb.GUI if self.gui else pb.DIRECT)  # pb.DIRECT for non-graphical version
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())  # get default URDFs like plane   
        
        pb.setRealTimeSimulation(self.real_time)     # make simulation decoupled from <pb.stepSimulation> and
        pb.setGravity(0, 0, -9.81)  # earth gravity  # based on internal asynchronous clock instead
        
        pb.loadURDF('plane.urdf')  # basic floor
        self.spider = SpiderBot(spider_bot_model_path)
        
        spider_pos = self.spider.get_pos()
        self.camera = Camera(initial_pos = spider_pos)
        self.camera_tracking = False
        
        # ToDo: define sample and actions spaces using <gym.spaces>
        
        self.i = 0
        self.t = 0
        
    def step(self, controls: list) -> tuple:
        if self.gui:
            self.update_camera()
        
        # right now this is just to test spider movements -- we will need to consider pos
        # and torque control as well. We will also be using all of the joints. This may
        # be a place to artificially limit joint range of motion to prevent clipping.
        # Alternitively, we could limit range of motion internally in spider bot class 
        target_joints = self.spider.outer_joints
        target_velocities = controls
        self.spider.set_joint_controls(target_joints, controlMode = pb.VELOCITY_CONTROL, targetVelocities = target_velocities)
        
        pb.stepSimulation()
        time.sleep(1 / 1000)
        self.i += 1
        observation = self.get_observation()
        reward = 0
        done = self.is_terminated()
        info = {}
        
        return observation, reward, done, info  # adhere to gym interface
        
    def get_observation(self) -> dict:
        return self.spider.get_joints_state(self.spider.joints_flat)
        
    def close(self) -> None:
        pb.disconnect()
        
    def is_terminated(self) -> bool:
        return False
        
    def reset(self) -> dict:
        pb.removeBody(self.spider.id)
        self.spider = SpiderBot(self.spider_bot_model_path)
        self.camera.reset()
        
        self.i = 0
        self.t = 0
        # also tried using <pb.resetSimulation()> and <pb.resetBasePositionAndOrientation(self.spider.id, (0, 0, 1), (0, 0, 0, 0))>
        return self.get_observation()
        
    def update_camera(self, verbose: bool = False) -> None:
        """
        shift + left arrow, right arrow to translate x
        shift + up arrow, down arrow to translate y
        a, d to translate global z
        left arrow, right arrow to yaw
        up arrow, down arrow to pitch
        q, e to zoom
        c to toggle target tracking
        
        verbose: bool -- if true, displays current key events
        """
        keys = pb.getKeyboardEvents()
        if keys and verbose: ic(keys)  # display keys that got pressed to see their id easily 
        
        self.camera.change_x(int(bool(keys.get(65306))) * (0.25 if keys.get(65296) else -0.25 if keys.get(65295) else 0))
        self.camera.change_y(int(bool(keys.get(65306))) * (0.25 if keys.get(65297) else -0.25 if keys.get(65298) else 0))
        self.camera.change_global_z(0.25 if keys.get(97) else -0.25 if keys.get(100) else 0)
        self.camera.change_yaw(int(not keys.get(65306)) * (1 if keys.get(65296) else -1 if keys.get(65295) else 0))
        self.camera.change_pitch(int(not keys.get(65306)) * (1 if keys.get(65298) else -1 if keys.get(65297) else 0))
        self.camera.change_zoom(0.25 if keys.get(113) else -0.25 if keys.get(101) else 0)
            
        
        if keys.get(99) == 3:
            self.camera_tracking = not self.camera_tracking
            if self.camera_tracking:
                self.camera.track_target(lambda: self.spider.get_pos())
            else:
                self.camera.clear_target()
            
        self.camera.update()
        