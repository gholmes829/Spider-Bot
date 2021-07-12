"""

"""

from icecream import ic
import pybullet as pb
import pybullet_data
import time
from random import randint
import numpy as np

def rotationGenerator(theta: float) -> callable:
    """Returns lambda func to rotate tuple CCW by <theta> degrees"""
    thetaRad = np.radians(theta)
    cosTheta = np.cos(thetaRad)
    sinTheta = np.sin(thetaRad)
    return lambda v: (
        round(v[0] * cosTheta - v[1] * sinTheta, 5),
        round(v[0] * sinTheta + v[1] * cosTheta, 5)
    )

class Driver:
    def __init__(self) -> None:
        physicsClient = pb.connect(pb.GUI)  # or p.DIRECT for non-graphical version
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())        
        pb.setGravity(0, 0, -9.81)
        #pb.setRealTimeSimulation(1)
        
        self.planeId = pb.loadURDF("plane.urdf")
        self.botId = pb.loadURDF("urdfs/spiderbot.urdf", (0, 0, 1))
        self.numBotJoints = pb.getNumJoints(self.botId)
        
        self.rotate_cw = rotationGenerator(-90)
        self.rotate_ccw = rotationGenerator(90)
        
        #for i in range(self.numBotJoints):
            #ic(pb.getJointInfo(self.botId, i))
        #ic(self.numBotJoints)
        
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
        
        self.active_joints = set(range(12))
        self.inner_joints = {0, 3, 6, 9}
        self.middle_joints = {1, 4, 7, 10}
        self.outer_joints = {2, 5, 8, 11}
        
        cube_pos, cube_orientation = pb.getBasePositionAndOrientation(self.botId)        
        
        # camera
        self.cpos = list(cube_pos)
        self.cyaw = 10
        self.cpitch = -15
        self.cdist = 5
        self.cameraFollow = False

    def update_keys(self):
        keys = pb.getKeyboardEvents()
        #if keys:  # display keys that got pressed to see their id easily
        #    ic(keys)
        
        direction = list(pb.getDebugVisualizerCamera()[5])[:-1]  # all but z axis

        ic(direction, np.linalg.norm(direction))
        
        if keys.get(97):   # a
            self.cpos[2] -= 0.25
        if keys.get(100):   # d
            self.cpos[2] += 0.25
            
        # yawing
        if keys.get(65296):  # right arrow
            if keys.get(65306):  # left shift
                # get right orthogonal vector
                target_direction = self.rotate_cw(direction)
                self.cpos[0] += 0.25 * target_direction[0]
                self.cpos[1] += 0.25 * target_direction[1]
            else:
                self.cyaw += 1
                
        if keys.get(65295):   # left arrow
            if keys.get(65306):  # left shift
                # get left orthogonal vector
                target_direction = self.rotate_ccw(direction)
                self.cpos[0] += 0.25 * target_direction[0]
                self.cpos[1] += 0.25 * target_direction[1]
            else:
                self.cyaw -= 1
            
        # pitching
        if keys.get(65298):   # down arrow
            if keys.get(65306):  # left shift
                self.cpos[0] -= 0.25 * direction[0]
                self.cpos[1] -= 0.25 * direction[1]
            else:
                self.cpitch +=  1
        if keys.get(65297):  # up arrow
            if keys.get(65306):  # left shift
                self.cpos[0] += 0.25 * direction[0]
                self.cpos[1] += 0.25 * direction[1]
            else:
                self.cpitch -= 1
            
        # zooming
        if keys.get(113):  # q
            self.cdist += 0.1
        if keys.get(101):  # e
            self.cdist -= 0.1
        if keys.get(99) == 3:
            self.cameraFollow = not self.cameraFollow
            
        pb.resetDebugVisualizerCamera(cameraDistance = self.cdist, cameraYaw = self.cyaw, cameraPitch = self.cpitch, cameraTargetPosition = self.cpos)
    
    def run(self):
        i = 0
        while True:
            #ic(pb.getDebugVisualizerCamera())
            cube_pos, cube_orientation = pb.getBasePositionAndOrientation(self.botId) 
            self.update_keys() 
            if self.cameraFollow:
                self.cpos = list(cube_pos)
            #ic(self.cameraFollow, self.cpos)

            for j in self.active_joints & (self.outer_joints | self.middle_joints):
                pb.setJointMotorControl2(self.botId, j, controlMode = pb.VELOCITY_CONTROL, targetVelocity = -5 * np.sin(i / 10))
            
            pb.stepSimulation()
            time.sleep(1 / 1000)
            i += 1

        pb.disconnect()

if __name__ == '__main__':
    driver = Driver()
    driver.run()