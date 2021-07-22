"""

"""

import numpy as np
import pybullet as pb
from pybullet_utils import bullet_client as bc
import pybullet_data
from icecream import ic
import time
import numpy as np
import gym
from gym import spaces, Env
gym.logger.set_level(40)

from spider_bot.spider_bot_model import SpiderBot
from spider_bot.camera import Camera

class SpiderBotSimulator(Env):
    def __init__(self, spider_bot_model_path: str, real_time_enabled: bool = True, gui: bool = True, fast_mode = False) -> None:
        Env.__init__(self)
        self.spider_bot_model_path = spider_bot_model_path
        self.real_time_enabled = real_time_enabled
        
        self.gui = gui
        self.fast_mode = fast_mode # no time.sleep
 
        self.physics_client = bc.BulletClient(connection_mode=pb.GUI if self.gui else pb.DIRECT)
        self.physics_client.setAdditionalSearchPath(pybullet_data.getDataPath())  # get default URDFs like plane   
        
        self.physics_client.setRealTimeSimulation(self.real_time_enabled)     # make simulation decoupled from <pb.stepSimulation> and based on internal asynchronous clock instead
        self.physics_client.setGravity(0, 0, -9.81)  # earth gravity
        
        self.plane_id = self.physics_client.loadURDF('plane.urdf')  # basic floor
        self.spider = SpiderBot(spider_bot_model_path, self.physics_client)
        
        self.max_spider_vel = 5
        
        self.initial_position = self.spider.get_pos()
        self.last_position = None
        self.curr_position = self.initial_position
        self.velocity = np.zeros(3)
        
        self.camera = Camera(self.physics_client, initial_pos = self.initial_position)
        self.camera_tracking = False
        self.prev_cd = [True, True, True, True]
        self.is_stepping = [False, False, False, False]
        self.height_threshold = 0.03
        self.rising_edges = [[0] for _ in range(4)]
        self.steps = [0 for _ in range(4)]
        
        self.action_space = spaces.Box(
            low = np.full(12, -1),
            high = np.full(12, 1)
        )
        
        self.observation_space = spaces.Box(
            low = np.array([
                    *np.full(12, -self.spider.max_joint_angle),
                    *np.full(12, -self.spider.nominal_joint_velocity),
                    *np.full(3, -2*np.pi),
                    *np.full(3, -self.max_spider_vel)
                ]),
            high = np.array([
                    *np.full(12, self.spider.max_joint_angle),
                    *np.full(12, self.spider.nominal_joint_velocity),
                    *np.full(3, 2*np.pi),
                    *np.full(3, self.max_spider_vel)
                ]),
        )
        
        self.i = 0
        self.t = 0
        
    def step(self, controls: list) -> tuple:
        if self.gui:
            self.update_camera()
        num_bodies = self.physics_client.getNumBodies()
        assert num_bodies == 2, f'Expected 2, recieved {num_bodies} bodies'  # there should only be the spider and floor, helps debug multiprocessing
        
        # use control outputs to move robot
        self.spider.set_joint_velocities(self.spider.outer_joints, controls[:4])
        self.spider.set_joint_velocities(self.spider.middle_joints, controls[4:8])
        self.spider.set_joint_velocities(self.spider.inner_joints, controls[8:])
        
        # update state vars
        self.last_position = self.curr_position
        self.physics_client.stepSimulation()
        self.physics_client.performCollisionDetection()
        self.current_position = self.spider.get_pos()
        self.velocity = self.current_position - self.last_position
        
        if not self.fast_mode:
            time.sleep(1 / 240)
        self.i += 1

        observation = self.get_observation()
        reward = 0#self.get_prop_vel_proj_score()
        done = self.is_terminated()
        info = self.get_info()

        cd = info['contact-data']
        ankle_heights = np.array(info['ankle-pos']).T[2]
        #ic(np.round(ankle_heights, 2))
        assert len(cd) == 4
        for i in range(len(cd)):
            self.rising_edges[i].append(int(cd[i] == False and self.prev_cd[i]))
            if self.is_stepping[i] and cd[i] and not self.prev_cd[i]: # spider is stepping and just touched ground again
                self.is_stepping[i] = False 
                self.steps[i] += 1
                
                #self.has_stepped[i] = 
            #not (cd[i] and not self.prev_cd[i]) # reset step once foot touches ground
        self.prev_cd = cd

        for i, height in enumerate(ankle_heights):
            if height > self.height_threshold and not self.is_stepping[i]: 
                #self.steps[i] += 1
                self.is_stepping[i] = True
        #print(self.steps)
        # ankle_pos = info['ankle-pos']
        # for pos in ankle_pos:
        #     ic(pos)
        #     pb.addUserDebugLine([pos[0], pos[1], 0], pos, lineColorRGB = [1, 0, 0], lifeTime=0.1)
        self.spider.clamp_joints(verbose=False)
        
        return observation, reward, done, info  # adhere to gym interface
        
    def get_observation(self) -> np.array:
        joint_info = self.spider.get_joints_state(self.spider.joints_flat)
        orientation = self.spider.get_orientation()
        return np.array([
            *joint_info['pos'],
            *joint_info['vel'],
            *orientation,
            *self.velocity
        ])   

    def get_ang_vel_proj_score(self) -> float:
        """
        score = ammount of speed's direction that is parallel to optimal speed direction
        
        This means the score will be highest when moving in the right direction, irregardless of speed.
        """
        velocity = self.velocity[:2]
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

    def get_prop_vel_proj_score_gait_monitor(self) -> float:
        """
        score = speed * (ammount of speed's direction that is parallel to optimal speed direction) * (diagonal legs are in sync)
        
        This means the score will be highest when moving at large speed in the right direction and it is moving opposite legs in sync (ideally).

        """
        velocity = (self.current_position - self.last_position)[:2]
        origin_to_bot = (self.last_position - self.initial_position)[:2]
        cd = self.get_contact_data()
        reward_amplifier = 2.0 if (cd[0] == cd[3] and cd[1] == cd[2] and cd[0] != cd[1]) else 0.5
        
        if not origin_to_bot.any():  # speed
            return np.linalg.norm(velocity) * reward_amplifier
        else:  # projection of velocity onto best direction
            return ((velocity @ origin_to_bot) / np.linalg.norm(origin_to_bot)) * reward_amplifier

    def get_info(self) -> dict:
        joint_info = self.spider.get_joints_state(self.spider.joints_flat)
        binary_contact_data = self.get_contact_data()
        return {
            "body-pos":           self.spider.get_pos(),
            "orientation":        self.spider.get_orientation(),
            "joint-pos":          joint_info['pos'],
            "joint-vel":          joint_info['vel'],
            "joint-torques":      joint_info['motor_torques'],
            "contact-data":       binary_contact_data,
            "ankle-pos":          joint_info['ankle-pos']
        }

    def get_contact_data(self) -> list:
        contact_points = [p[3] for p in self.physics_client.getContactPoints(self.spider.id, self.plane_id)]
        return [3 in contact_points, 7 in contact_points, 11 in contact_points, 15 in contact_points]
    
    def spider_is_standing(self):
        # returns not (there exists some point of contact involving a spider link thats not an outer leg)
        return not any([p[3] not in self.spider.ankles + self.spider.outer_joints for p in self.physics_client.getContactPoints(self.spider.id, self.plane_id)])
        
    def close(self) -> None:
        self.physics_client.disconnect()
        
    def is_terminated(self) -> bool:
        return not self.spider_is_standing()
        
    def reset(self) -> dict:
        self.physics_client.removeBody(self.spider.id)
        self.spider = SpiderBot(self.spider_bot_model_path, self.physics_client)
        self.camera.reset()
        
        self.i = 0
        self.t = 0
        self.rising_edges = [[0] for _ in range(4)]
        self.steps = [0 for _ in range(4)]
        self.velocity = 0
        self.initial_position = self.spider.get_pos()
        self.last_position = None
        self.curr_position = self.initial_position
        self.velocity = np.zeros(3)
        
        self.prev_cd = [True, True, True, True]
        self.is_stepping = [False, False, False, False]

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
        keys = self.physics_client.getKeyboardEvents()
        if keys and verbose: ic(keys)  # display keys that got pressed to see their id easily 
        
        self.camera.change_x(int(bool(keys.get(65306))) * (0.1 if keys.get(65296) else -0.25 if keys.get(65295) else 0))
        self.camera.change_y(int(bool(keys.get(65306))) * (0.1 if keys.get(65297) else -0.25 if keys.get(65298) else 0))
        self.camera.change_global_z(0.25 if keys.get(97) else -0.1 if keys.get(100) else 0)
        self.camera.change_yaw(int(not keys.get(65306)) * (1 if keys.get(65296) else -1 if keys.get(65295) else 0))
        self.camera.change_pitch(int(not keys.get(65306)) * (1 if keys.get(65298) else -1 if keys.get(65297) else 0))
        self.camera.change_zoom(0.25 if keys.get(113) else -0.1 if keys.get(101) else 0)
            
        
        if keys.get(99) == 3:
            self.camera_tracking = not self.camera_tracking
            if self.camera_tracking:
                self.camera.track_target(lambda: self.spider.get_pos())
            else:
                self.camera.clear_target()
            
        self.camera.update()
        