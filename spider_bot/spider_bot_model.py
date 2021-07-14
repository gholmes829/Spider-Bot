"""

"""

import pybullet as pb
import numpy as np

class SpiderBot:
    def __init__(self, model_path: str, initial_pos: tuple = (0, 0, 1.5625)) -> None:
        self.model_path = model_path
        self.id = pb.loadURDF(self.model_path, initial_pos, flags=pb.URDF_USE_SELF_COLLISION)
        self.num_joints = pb.getNumJoints(self.id)
        
        # orange, green, yellow, purple        
        self.inner_joints = [0, 3, 6, 9]
        self.middle_joints = [1, 4, 7, 10]
        self.outer_joints = [2, 5, 8, 11]
        self.joints = [self.inner_joints, self.middle_joints, self.outer_joints]
        self.joints_flat = self.inner_joints + self.middle_joints + self.outer_joints
        
        # move joints to initial pos
        self.reset_joints_state(self.outer_joints, np.full(4, -0.5))
        self.reset_joints_state(self.middle_joints, np.full(4, -1))
        self.reset_joints_state(self.inner_joints, np.full(4, 1))
        
        self.change_lateral_friction(self.outer_joints, np.full(4, 1))
        
        # JOINT INDICES
        # 0 is orange inner leg to body
        # 1 is orange inner leg to middle leg
        # 2 is orange middle leg to outer leg

        # 3 is green inner leg to body
        # 4 is green inner leg to middle leg
        # 5 is green middle leg to outer leg

        # 6 is yellow inner leg to body
        # 7 is yellow inner leg to middle leg
        # 8 is yellow middle leg to outer leg

        # 9 is purple inner leg to body
        # 10 is purple inner leg to middle leg
        # 11 is purple middle leg to outer leg
        
    def get_id(self):
        return self.id
    
    def get_pos(self):
        return list(pb.getBasePositionAndOrientation(self.id)[0])
    
    def get_orientation(self):
        return list(pb.getBasePositionAndOrientation(self.id)[1])
    
    def set_joint_controls(self, *args, **kwargs):
        pb.setJointMotorControlArray(self.id, *args, **kwargs)

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
        