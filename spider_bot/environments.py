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

        self.velocity = np.zeros(3)      #  These three used for health-based reward
        self.torques = np.zeros(12)      #
        self.V_ax = 0                    #
        
        self.camera = Camera(self.physics_client, initial_pos = self.initial_position) if self.gui else None
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

        
        # orange, green, yellow, purple        
        # self.inner_joints = [0, 4, 8, 12]
        # self.middle_joints = [1, 5, 9, 13]
        # self.outer_joints = [2, 6, 10, 14]
        # self.ankles = [3, 7, 11, 15]

        self.joint_pairs = [[0, 12],
                            [4, 8],
                            [2, 14],
                            [6, 10],
                            [5, 9],
                            [1, 13]]
        
        self.observation_space = spaces.Box(
            low = np.array([
                    *np.full(12, -self.spider.max_joint_angle),
                    *np.full(12, -self.spider.nominal_joint_velocity),
                    *np.full(3, -2*np.pi),
                    *np.full(3, -1)
                ]),
            high = np.array([
                    *np.full(12, self.spider.max_joint_angle),
                    *np.full(12, self.spider.nominal_joint_velocity),
                    *np.full(3, 2*np.pi),
                    *np.full(3, 1)
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
        #for i, control in zip(self.spider.joints_flat, controls):
        #ic(self.spider.joint_limits)
        # for i, control in enumerate(controls):
        #     control = np.clip(control, -1, 1)
        #     target_position = self.spider.joint_limits[self.joint_pairs[i][0]]['center'] + 0.5 * control * self.spider.joint_limits[self.joint_pairs[i][0]]['range']
        #     self.spider.set_joint_position(self.joint_pairs[i][0], target_position)
        #     self.spider.set_joint_position(self.joint_pairs[i][1], target_position)
            #ic(self.joint_pairs[i][0], target_position)
            #ic(self.joint_pairs[i][1], target_position)

        #controls = self.filter_controls(controls.copy())
        for i, pair in enumerate(self.joint_pairs):
            sign = 1 if i < 4 else -1
            self.spider.set_joint_velocities(pair, [controls[i], sign * controls[i]])
        # self.spider.set_joint_velocities(self.spider.outer_joints, controls[i])
        # self.spider.set_joint_velocities(self.spider.middle_joints, controls[i])
        # self.spider.set_joint_velocities(self.spider.inner_joints, controls[i])
        
        # update state vars
        self.last_position = self.curr_position
        self.physics_client.stepSimulation()
        self.physics_client.performCollisionDetection()
        self.current_position = self.spider.get_pos()
        self.velocity = self.physics_client.getBaseVelocity(self.spider.id)[0]
        
        if not self.fast_mode:
            time.sleep(1 / 256)
        self.i += 1

        observation = self.get_observation()
        done = self.is_terminated()
        info = self.get_info()
        reward = self.get_health_score(info['body-pos'], info['orientation'], info['joint-torques'])

        cd = info['contact-data']
        ankle_heights = np.array(info['ankle-pos']).T[2]
        assert len(cd) == 4
        self.update_steps(cd, ankle_heights)
        #self.spider.clamp_joints(verbose=False)

        # self.prev_velocity = self.velocity               # Used for health-based reward
        # self.torques = np.array(info['joint-torques'])   #
        
        return observation, reward, done, info  # adhere to gym interface
        
    def get_observation(self) -> np.array:
        joint_info = self.spider.get_joints_state(self.spider.joints_flat)
        orientation = self.spider.get_orientation()
        periods = [32, 64, 128, 256]
        return np.array([
            *[np.sin((2 * np.pi * self.i) / period) for period in periods]            
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

    def get_health_score(self, body_pos, orientation, tau) -> float:
        z = body_pos[2]
        R, P = orientation[:2]
        V_x = self.velocity[0]
        H = 0 if z > 0.125 and np.abs(R) < 0.7 and np.abs(P) < 0.7 else -1
        U = -(R**2 + P**2)
        V_ax = (V_x * 1/256) + (self.V_ax * (1 - 1/256))
        self.V_ax = V_ax
        V_d = -np.abs(V_x - V_ax) if V_ax >= 0.3 else 0
        d_tau = -1 * np.sqrt(np.sum((tau - self.torques)**2))
        reward = H + U + V_ax + V_d + d_tau
        #ic(H, U, V_ay, V_d, d_tau, reward)

        return reward

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

    def update_steps(self, cd: list, ankle_heights: np.array) -> None:
        for i in range(len(cd)):
            self.rising_edges[i].append(int(cd[i] == False and self.prev_cd[i]))
            if self.is_stepping[i] and cd[i] and not self.prev_cd[i]: # spider is stepping and just touched ground again
                self.is_stepping[i] = False 
                self.steps[i] += 1
        self.prev_cd = cd

        for i, height in enumerate(ankle_heights):
            if height > self.height_threshold and not self.is_stepping[i]: 
                self.is_stepping[i] = True
        
    def close(self) -> None:
        self.physics_client.disconnect()
        
    def is_terminated(self) -> bool:
        return not self.spider_is_standing()
        
    def reset(self) -> dict:
        self.physics_client.removeBody(self.spider.id)
        self.spider = SpiderBot(self.spider_bot_model_path, self.physics_client)
        if self.gui:
            self.camera.reset()
        
        for _ in range(10):
            self.physics_client.stepSimulation()
            
        for i in self.spider.all_joints:
            self.physics_client.setJointMotorControl2(self.spider.id, i, controlMode=pb.VELOCITY_CONTROL, force=0)

        
        self.i = 0
        self.t = 0
        self.rising_edges = [[0] for _ in range(4)]
        self.steps = [0 for _ in range(4)]
        self.velocity = 0
        self.initial_position = self.spider.get_pos()
        self.last_position = None
        self.curr_position = self.initial_position
        self.velocity = np.zeros(3)
        self.prev_velocity = np.zeros(3)
        
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
        