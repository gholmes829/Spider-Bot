"""

"""

from icecream import ic
import pybullet as pb
import pybullet_data
import time
from random import randint
import numpy as np

class Driver:
    def __init__(self) -> None:
        physicsClient = pb.connect(pb.GUI)  # or p.DIRECT for non-graphical version
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())        
        pb.setGravity(0, 0, -9.81)
        pb.setRealTimeSimulation(1)
        
        self.planeId = pb.loadURDF("plane.urdf")
        self.botId = pb.loadURDF("urdfs/spiderbot.urdf")
        self.numBotJoints = pb.getNumJoints(self.botId)
        cube_pos, cube_orientation = pb.getBasePositionAndOrientation(self.botId)        
        
        # camera
        self.cpos = list(cube_pos)
        self.cyaw = 10
        self.cpitch = -15
        self.cdist = 5
        self.cameraFollow = False

    def update_keys(self):
        keys = pb.getKeyboardEvents()
        #ic(keys)
        
        # moving (note: this is very glitchy for some reason on my device)
        if keys.get(105):  # i
            self.cpos[1] += 0.5
        if keys.get(106):   # j
            self.cpos[0] -= 0.5
        if keys.get(107):  # k
            self.cpos[1] -= 0.5
        if keys.get(108):   # l
            self.cpos[0] += 0.5
        
        if keys.get(117):   # l
            self.cpos[2] -= 0.5
        if keys.get(111):   # l
            self.cpos[2] += 0.5
            
        # yawing
        if keys.get(65296):  # right arrow
            self.cyaw += 1
        if keys.get(65295):   # right arrow
            self.cyaw -= 1
            
        # pitching
        if keys.get(65298):   # down arrow
            self.cpitch +=  1
        if keys.get(65297):  # up arrow
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
            for j in range(self.numBotJoints):
                pb.setJointMotorControl2(self.botId, j, controlMode = pb.VELOCITY_CONTROL, targetVelocity = -np.sin(i/10))
            
            pb.stepSimulation()
            time.sleep(1 / 240)
            i += 1

        pb.disconnect()

if __name__ == '__main__':
    driver = Driver()
    driver.run()