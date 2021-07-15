"""

"""

import numpy as np
import pybullet as pb
import pybullet_data
from icecream import ic
import time
import numpy as np
from gym import spaces, Env

from spider_bot.spider_bot_model import SpiderBot
from spider_bot.camera import Camera

class SpiderBotSimulator(Env):
    def __init__(self, spider_bot_model_path: str, real_time_enabled: bool = False, gui: bool = True, fast_mode = False) -> None:
        Env.__init__(self)
        self.spider_bot_model_path = spider_bot_model_path
        self.real_time_enabled = real_time_enabled
        
        self.gui = gui
        self.physics_client = pb.connect(pb.GUI if self.gui else pb.DIRECT)  # pb.DIRECT for non-graphical version
        self.fast_mode = fast_mode # no time.sleep
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())  # get default URDFs like plane   
        
        pb.setRealTimeSimulation(self.real_time_enabled)     # make simulation decoupled from <pb.stepSimulation> and based on internal asynchronous clock instead
        pb.setGravity(0, 0, -9.81)  # earth gravity
        
        self.plane_id = pb.loadURDF('plane.urdf')  # basic floor
        self.spider = SpiderBot(spider_bot_model_path)
        
        spider_pos = self.spider.get_pos()
        self.camera = Camera(initial_pos = spider_pos)
        self.camera_tracking = False
        
        self.action_space = spaces.Box(
            low = np.full(12, -1),
            high = np.full(12, 1)
        )
        
        self.observation_space = spaces.Box(
            low = np.array([
                    *np.full(12, -self.spider.max_joint_angle),
                    *np.full(12, -self.spider.nominal_joint_velocity)
                ]),
            high = np.array([
                    *np.full(12, self.spider.max_joint_angle),
                    *np.full(12, self.spider.nominal_joint_velocity)
                ]),
        )

        self.initial_position = self.spider.get_pos()
        self.last_position = None
        self.curr_position = self.initial_position
        
        self.i = 0
        self.t = 0
        
    def step(self, controls: list) -> tuple:
        if self.gui:
            self.update_camera()
        
        self.spider.set_joint_velocities(self.spider.outer_joints, controls[:4])
        self.spider.set_joint_velocities(self.spider.middle_joints, controls[4:8])
        self.spider.set_joint_velocities(self.spider.inner_joints, controls[8:])
        
        self.last_position = self.curr_position
        pb.stepSimulation()
        pb.performCollisionDetection()
        self.current_position = self.spider.get_pos()
        
        if not self.fast_mode:
            time.sleep(1 / 240)
        self.i += 1

        observation = self.get_observation()
        reward = self.get_prop_vel_proj_score()
        done = self.is_terminated()
        info = self.get_info()
        
        return observation, reward, done, info  # adhere to gym interface
        
    def get_observation(self) -> np.array:
        joint_info = self.spider.get_joints_state(self.spider.joints_flat)
        return np.array([
            *joint_info['pos'],
            *joint_info['vel']
        ])

    def get_distance_from_start(self):
        return np.sqrt( np.square(self.spider.get_pos()[0]) + np.square(self.spider.get_pos()[1]) )

    def get_ang_vel_proj_score(self) -> float:
        """
        score = ammount of speed's direction that is parallel to optimal speed direction
        
        This means the score will be highest when moving in the right direction, irregardless of speed.
        """
        velocity = (self.current_position - self.last_position)[:2]
        origin_to_bot = (self.last_position - self.initial_position)[:2]
        
        if not (origin_to_bot * velocity).sum():  # rewarded if moving
            return int(velocity.any())
        else:  # unit projection of velocity onto best direction
            return (velocity @ origin_to_bot) / (np.linalg.norm(origin_to_bot) * np.linalg.norm(velocity))

    def get_prop_vel_proj_score(self) -> float:
        """
        score = speed * (ammount of speed's direction that is parallel to optimal speed direction)
        
        This means the score will be highest when moving at large speed in the right direction.
        """
        velocity = (self.current_position - self.last_position)[:2]
        origin_to_bot = (self.last_position - self.initial_position)[:2]
        
        if not origin_to_bot.any():  # speed
            return np.linalg.norm(velocity)
        else:  # projection of velocity onto best direction
            return (velocity @ origin_to_bot) / np.linalg.norm(origin_to_bot)

    def get_info(self) -> dict:
        joint_info = self.spider.get_joints_state(self.spider.joints_flat)
        return {
            "body-pos":           self.spider.get_pos(),
            "orientation":        self.spider.get_orientation(),
            "joint-pos":          joint_info['pos'],
            "joint-vel":          joint_info['vel'],
            "joint-torques":      joint_info['motor_torques']
        }
    
    def spider_is_standing(self):
        # returns !(there exists some point of contact involving a spider link thats not an outer leg)
        return not any([p[3] not in self.spider.ankles for p in pb.getContactPoints(self.spider.id, self.plane_id)])
        
    def close(self) -> None:
        pb.disconnect()
        
    def is_terminated(self) -> bool:
        return not self.spider_is_standing()
        
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
        