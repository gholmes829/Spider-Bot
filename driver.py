"""

"""

from typing import ChainMap
from icecream import ic
import pybullet as pb
import pybullet_data
import time

class Driver:
    def __init__(self) -> None:
        self.cyaw = 10
        self.cpitch = -15
        self.cdist = 5
        self.cameraFollow = False


    def updateKeys(self):
        keys = pb.getKeyboardEvents()
        #ic(keys)
                
        if keys.get(65296):  # right arrow
            self.cyaw += 1
        if keys.get(65295):   # right arrow
            self.cyaw -= 1
        if keys.get(65298):   # down arrow
            self.cpitch +=  1
        if keys.get(65297):  # up arrow
            self.cpitch -= 1
        if keys.get(113):  # q
            self.cdist += 0.1
        if keys.get(101):  # e
            self.cdist -= 0.1
        if keys.get(99):
            self.cameraFollow = not self.cameraFollow
    
    def run(self):
        physicsClient = pb.connect(pb.GUI)  # or p.DIRECT for non-graphical version
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.setGravity(0,0,-10)
        planeId = pb.loadURDF("plane.urdf")
        botId = pb.loadURDF("urdfs/spiderbot.urdf")
        lastCubePos, initialCubeOrn = pb.getBasePositionAndOrientation(botId)
        
        pb.setGravity(0, 0, -9.81)
        pb.setRealTimeSimulation(1)

        while True:
            #ic(pb.getDebugVisualizerCamera())
            cubePos, cubeOrn = pb.getBasePositionAndOrientation(botId)
            
            if self.cameraFollow:
                lastCubePos = cubePos
            
            target_pos = cubePos if self.cameraFollow else lastCubePos 
            pb.resetDebugVisualizerCamera(cameraDistance = self.cdist, cameraYaw = self.cyaw, cameraPitch = self.cpitch, cameraTargetPosition = target_pos)
            self.updateKeys()
            
            
            pb.stepSimulation()
            time.sleep(1 / 240)

        pb.disconnect()

if __name__ == '__main__':
    driver = Driver()
    driver.run()