import pybullet as p
import os
from kinova_env import KinovaEnv
import time


class ThreeBlocksEnv(KinovaEnv):
    def __init__(self):
        super().__init__()
        # set time step
        self.dt = 1 / 10000
        p.setTimeStep(self.dt)
        # load URDF file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        robot_path = os.path.join(script_dir, "square_obstacle.urdf")

        # load obstacle
        block1 = p.loadURDF(robot_path, useFixedBase=True)
        block1_pos = [0.45, 0.4, 1.7]
        p.resetBasePositionAndOrientation(block1, block1_pos, [0, 0, 0, 1])

        block2 = p.loadURDF(robot_path, useFixedBase=True)
        block2_pos = [0.55, 0.3, 1.3]
        p.resetBasePositionAndOrientation(block2, block2_pos, [0, 0, 0, 1])

        block3 = p.loadURDF(robot_path, useFixedBase=True)
        block3_pos = [0.7, 0.05, 1.1]
        p.resetBasePositionAndOrientation(block3, block3_pos, [0, 0, 0, 1])
