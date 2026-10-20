"""Isolated Bullet scene matching the fixed-base Pinocchio coursework model."""

import numpy as np
import pinocchio as pin
import pybullet as pb
import pybullet_data
from pybullet_utils.bullet_client import BulletClient

from config import (NEXTAGE_URDF, TABLE_URDF, OBSTACLE_URDF, CUBE_URDF,
                    ROBOT_PLACEMENT, TABLE_PLACEMENT, OBSTACLE_PLACEMENT,
                    CUBE_PLACEMENT, DT)


class Simulation:
    def __init__(self, robot, gui=False, cube_placement=CUBE_PLACEMENT):
        self.p = BulletClient(connection_mode=pb.GUI if gui else pb.DIRECT)
        if gui:
            # Bullet's built-in R/Delete shortcuts can erase the scene while
            # the coursework script is still using its body IDs.
            self.p.configureDebugVisualizer(pb.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
            # macOS otherwise redraws expensive panels during every physics
            # query. The runner explicitly requests only 25 visible frames/s.
            self.p.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
            self.p.configureDebugVisualizer(pb.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.p.setGravity(0, 0, -9.81)
        self.p.setTimeStep(DT)
        self.p.setPhysicsEngineParameter(numSolverIterations=100)
        self.floor = self.p.loadURDF('plane.urdf')
        self.robot = self._load(NEXTAGE_URDF, ROBOT_PLACEMENT, True)
        self.tableId = self._load(TABLE_URDF, TABLE_PLACEMENT, True)
        self.obstacleId = self._load(OBSTACLE_URDF, OBSTACLE_PLACEMENT, True)
        self.cubeId = self._load(CUBE_URDF, cube_placement, False)
        names = {self.p.getJointInfo(self.robot, i)[1].decode(): i
                 for i in range(self.p.getNumJoints(self.robot))}
        self.bulletCtrlJointsInPinOrder = [names[n] for n in robot.model.names[1:]]
        self.hand_links = [names['LARM_EFF'], names['RARM_EFF']]
        # Empty frame links have no physical material. Bullet otherwise assigns
        # them 1 kg each, unlike Pinocchio; remove only those spurious masses.
        for index in self.hand_links:
            self.p.changeDynamics(self.robot, index, mass=0)
        for index in range(self.p.getNumJoints(self.cubeId)):
            self.p.changeDynamics(self.cubeId, index, mass=0)
        for index in range(self.p.getNumJoints(self.robot)):
            self.p.changeDynamics(self.robot, index, jointDamping=0,
                                  linearDamping=0, angularDamping=0)
        self.p.setJointMotorControlArray(self.robot, self.bulletCtrlJointsInPinOrder,
                                        pb.VELOCITY_CONTROL, forces=[0.] * robot.model.nv)
        self.p.resetDebugVisualizerCamera(1.5, 90, -25, [0.2, 0, 0.95])

    def _load(self, filename, placement, fixed):
        pose = pin.SE3ToXYZQUAT(placement)
        return self.p.loadURDF(filename, pose[:3], pose[3:], useFixedBase=fixed,
                               flags=pb.URDF_USE_INERTIA_FROM_FILE)

    def getpybulletstate(self):
        states = self.p.getJointStates(self.robot, self.bulletCtrlJointsInPinOrder)
        return np.array([s[0] for s in states]), np.array([s[1] for s in states])

    def setqsim(self, q):
        for index, value in zip(self.bulletCtrlJointsInPinOrder, q):
            self.p.resetJointState(self.robot, index, float(value))

    def step(self, torques):
        self.p.setJointMotorControlArray(self.robot, self.bulletCtrlJointsInPinOrder,
                                        pb.TORQUE_CONTROL, forces=torques)
        self.p.stepSimulation()

    def close(self):
        if self.p.isConnected():
            self.p.disconnect()
