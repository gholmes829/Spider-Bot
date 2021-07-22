"""

"""

from icecream import ic
import pybullet as pb
import numpy as np

class SpiderBot:
    def __init__(self, model_path: str, physics_client, initial_pos: tuple = (0, 0, 0.17)) -> None:
        self.model_path = model_path
        self.physics_client = physics_client
        self.id = self.physics_client.loadURDF(self.model_path, initial_pos, flags=pb.URDF_USE_SELF_COLLISION)
        self.num_joints = self.physics_client.getNumJoints(self.id)

        # orange, green, yellow, purple        
        self.inner_joints = [0, 4, 8, 12]
        self.middle_joints = [1, 5, 9, 13]
        self.outer_joints = [2, 6, 10, 14]
        self.ankles = [3, 7, 11, 15]
        
        self.max_joint_angle = 1.571
        self.max_angle_range = 3.142
        self.nominal_joint_velocity = 13.09
        
        self.joints = [self.inner_joints, self.middle_joints, self.outer_joints]
        self.joints_flat = self.inner_joints + self.middle_joints + self.outer_joints
        
        # move joints to initial pos
        self.reset_joints_state(self.outer_joints, np.full(4, -0.5))
        self.reset_joints_state(self.middle_joints, np.full(4, -1))
        self.reset_joints_state(self.inner_joints, np.full(4, 1))
        
        self.change_lateral_friction(self.ankles, np.full(4, 2))
        self.set_max_joint_velocities(self.joints_flat + self.ankles, np.full(12, self.nominal_joint_velocity))
        
        #self.debug_joints()
        
        # JOINT INDICES -- UPDATED
        # 0 is orange inner
        # 1 is orange middle
        # 2 is orange outer
        # 3 is orange ankle
        
        # 4 is green inner
        # 5 is green middle
        # 6 is green outer
        # 7 is green ankle
        
        # 8 is yellow inner
        # 9 is yellow middle
        # 10 is yellow outer
        # 11 is yellow ankle
        
        # 12 is purple inner
        # 13 is purple middle
        # 14 is purple outer
        # 15 is purple ankle
        
        # there may be more that are unused, check URDF
        
    def get_id(self):
        return self.id
    
    def get_pos(self):
        return np.array(self.physics_client.getBasePositionAndOrientation(self.id)[0])
    
    def get_orientation(self):
        return np.array(self.physics_client.getEulerFromQuaternion(self.physics_client.getBasePositionAndOrientation(self.id)[1]))
    
    def set_joint_velocities(self, joint_indices, target_velocities):
        for i, target_velocity in zip(joint_indices, target_velocities):
            self.physics_client.setJointMotorControl2(
                self.id,
                i,
                controlMode = pb.VELOCITY_CONTROL,
                targetVelocity = target_velocity,
                maxVelocity = self.nominal_joint_velocity
            )
            
    def debug_joints(self, verbose=True, terminate=False):
        for i in range(self.physics_client.getNumJoints(self.id)):
            joint_info = self.physics_client.getJointInfo(self.id, i)
            state = self.get_joints_state([i])
            curr_pos = round(state['pos'][0], 5)
            index = joint_info[0]
            name = joint_info[1]
            limits = joint_info[8:10]
            if verbose:
                ic(index, name, limits, curr_pos)
            try:
                assert round(np.clip(curr_pos, *limits), 3) == round(curr_pos, 3), f'{name, limits, curr_pos, np.clip(curr_pos, *limits)}'
            except AssertionError as ae:
                if terminate:
                    raise ae
                else:
                    print(f'Joint pos bounds error: {ae}')

    def clamp_joints(self, verbose=True):
        for i in range(self.physics_client.getNumJoints(self.id)):
            joint_info = self.physics_client.getJointInfo(self.id, i)
            state = self.get_joints_state([i])
            curr_pos = round(state['pos'][0], 5)
            #index = joint_info[0]
            #name = joint_info[1]
            limits = joint_info[8:10]
            if curr_pos < limits[0]:
                self.reset_joints_state([i], [limits[0]])
                if verbose:
                    print(f'Clamping up', curr_pos, 'is less than', limits[0], flush=True)
            elif curr_pos > limits[1]:
                self.reset_joints_state([i], [limits[1]])
                if verbose:
                    print(f'Clamping down', curr_pos, 'is greater than', limits[1], flush=True)

    def get_joints_state(self, joint_indices):
        state = self.physics_client.getJointStates(self.id, joint_indices)
        ankles = self.physics_client.getLinkStates(self.id, self.ankles)
        #for i, joint in enumerate(state): assert joint[0] == np.clip(joint[0], -self.max_angle_range, self.max_angle_range), f'Joint {i} is outside expected position ({joint[0]})' 
        return {
            'pos': [joint[0] for joint in state],
            'vel': [joint[1] for joint in state],
            'reaction_forces': [joint[2] for joint in state],
            'motor_torques': [joint[3] for joint in state],
            'ankle-pos': [joint[0] for joint in ankles]
        }
        
    def reset_joints_state(self, joint_indices, target_pos):
        for i, pos in zip(joint_indices, target_pos):
            self.physics_client.resetJointState(self.id, i, pos)
            
    def change_lateral_friction(self, link_indices, target_lateral_frictions):
        for i, lateral_friction in zip(link_indices, target_lateral_frictions):
            self.physics_client.changeDynamics(self.id, i, lateralFriction=lateral_friction)
        
    def set_max_joint_velocities(self, link_indices, max_velocities):
        for i, max_joint_velocity in zip(link_indices, max_velocities):
            self.physics_client.changeDynamics(self.id, i, maxJointVelocity=max_joint_velocity)