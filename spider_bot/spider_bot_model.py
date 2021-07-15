"""

"""

from icecream import ic
import pybullet as pb
import numpy as np

class SpiderBot:
    def __init__(self, model_path: str, initial_pos: tuple = (0, 0, 0.1625)) -> None:
        self.model_path = model_path
        self.id = pb.loadURDF(self.model_path, initial_pos, flags=pb.URDF_USE_SELF_COLLISION)
        self.num_joints = pb.getNumJoints(self.id)

        # orange, green, yellow, purple        
        self.inner_joints = [0, 4, 8, 12]
        self.middle_joints = [1, 5, 9, 13]
        self.outer_joints = [2, 6, 10, 14]
        self.ankles = [3, 7, 11, 15]
        
        self.max_joint_angle = 1.571
        self.max_angle_range = 3.142
        self.nominal_joint_velocity = 6  # hard code for now to improve normalization # 13.09
        
        self.joints = [self.inner_joints, self.middle_joints, self.outer_joints]
        self.joints_flat = self.inner_joints + self.middle_joints + self.outer_joints
        
        # move joints to initial pos
        self.reset_joints_state(self.outer_joints, np.full(4, -0.5))
        self.reset_joints_state(self.middle_joints, np.full(4, -1))
        self.reset_joints_state(self.inner_joints, np.full(4, 1))
        
        self.change_lateral_friction(self.ankles, np.full(4, 2))
        self.set_max_joint_velocities(self.joints_flat + self.ankles, np.full(12, self.nominal_joint_velocity))
        
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
        
    def get_id(self):
        return self.id
    
    def get_pos(self):
        return np.array(pb.getBasePositionAndOrientation(self.id)[0])
    
    def get_orientation(self):
        return np.array(pb.getEulerFromQuaternion(pb.getBasePositionAndOrientation(self.id)[1]))
    
    def set_joint_velocities(self, joint_indices, target_velocities):
        for i, target_velocity in zip(joint_indices, target_velocities):
            pb.setJointMotorControl2(
                self.id,
                i,
                controlMode = pb.VELOCITY_CONTROL,
                targetVelocity = target_velocity,
                maxVelocity = self.nominal_joint_velocity
            )

    def get_joints_state(self, joint_indices):
        state = pb.getJointStates(self.id, joint_indices)
        return {
            'pos': [joint[0] for joint in state],
            'vel': [joint[1] for joint in state],
            'reaction_forces': [joint[2] for joint in state],
            'motor_torques': [joint[3] for joint in state]
        }
        
    def reset_joints_state(self, joint_indices, target_pos):
        for i, pos in zip(joint_indices, target_pos):
            pb.resetJointState(self.id, i, pos)
            
    def change_lateral_friction(self, link_indices, target_lateral_frictions):
        for i, lateral_friction in zip(link_indices, target_lateral_frictions):
            pb.changeDynamics(self.id, i, lateralFriction=lateral_friction)
        
    def set_max_joint_velocities(self, link_indices, max_velocities):
        for i, max_joint_velocity in zip(link_indices, max_velocities):
            pb.changeDynamics(self.id, i, maxJointVelocity=max_joint_velocity)