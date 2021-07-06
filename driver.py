"""

"""

from icecream import ic
import pybullet as pb
import pybullet_data
import time

def main():
    physicsClient = pb.connect(pb.GUI)  # or p.DIRECT for non-graphical version
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.setGravity(0,0,-10)
    planeId = pb.loadURDF("plane.urdf")
    botId = pb.loadURDF("urdfs/spiderbot.urdf")
    
    pb.setGravity(0, 0, -9.81)
    pb.setRealTimeSimulation(1)
    #Point the camera at the robot at the desired angle and distance
    pb.resetDebugVisualizerCamera( cameraDistance=1.5, cameraYaw=-30, cameraPitch=-30, cameraTargetPosition=[0.0, 0.0, 0.25])
    
    #cubeStartPos = [0, 0, 1]
    #cubeStartOrientation = pb.getQuaternionFromEuler([0, 0, 0])
    #boxId = pb.loadURDF("r2d2.urdf", cubeStartPos, cubeStartOrientation)
    for i in range (10000):
        pb.stepSimulation()
        time.sleep(1 / 240)

    pb.disconnect()

if __name__ == '__main__':
    main()