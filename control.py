#!/usr/bin/env python3
"""Time parametrisation and inverse-dynamics torque control for CW2."""

import numpy as np
import pinocchio as pin

from config import LEFT_HAND, LEFT_HOOK, RIGHT_HAND, RIGHT_HOOK


Kp = 300.0
Kv = 2.0 * np.sqrt(Kp)
SQUEEZE_FORCE = 8.0  # N per hand, maintained through physical contact
MAX_TORQUE = 180.0


class PiecewiseQuinticTrajectory:
    """A C2 trajectory that follows every configuration in a planned path.

    A fifth-order smooth-step has zero velocity and acceleration at both ends.
    Joining one curve per path edge therefore creates a C2 trajectory with a
    deliberate pause at each RRT waypoint, which is conservative but robust in
    the contact simulator.
    """

    def __init__(self, path, max_joint_speed=0.40):
        if len(path) < 1:
            raise ValueError("a trajectory needs at least one configuration")
        self.path = [np.asarray(q, dtype=float).copy() for q in path]
        if any(q.ndim != 1 or not np.all(np.isfinite(q)) for q in self.path):
            raise ValueError("trajectory configurations must be finite vectors")
        if any(q.shape != self.path[0].shape for q in self.path[1:]):
            raise ValueError("all trajectory configurations must have the same size")
        changes = [np.max(np.abs(b - a)) for a, b in zip(self.path[:-1], self.path[1:])]
        if max_joint_speed <= 0:
            raise ValueError("max_joint_speed must be positive")
        # Quintic smoothstep reaches a peak slope of 1.875, not 1.
        self.durations = np.maximum(0.45, 1.875 * np.asarray(changes) / max_joint_speed)
        self.breakpoints = np.concatenate(([0.0], np.cumsum(self.durations)))
        self.duration = float(self.breakpoints[-1])

    def _segment(self, t):
        if not self.durations.size:
            return 0, 0.0
        t = float(np.clip(t, 0.0, self.duration))
        index = min(np.searchsorted(self.breakpoints, t, side="right") - 1, len(self.durations) - 1)
        local_time = t - self.breakpoints[index]
        return index, np.clip(local_time / self.durations[index], 0.0, 1.0)

    def evaluate(self, t):
        """Return desired position, velocity and acceleration at time ``t``."""
        if not self.durations.size:
            zeros = np.zeros_like(self.path[0])
            return self.path[0].copy(), zeros, zeros
        index, s = self._segment(t)
        delta = self.path[index + 1] - self.path[index]
        duration = self.durations[index]
        blend = 10 * s**3 - 15 * s**4 + 6 * s**5
        blend_dot = (30 * s**2 - 60 * s**3 + 30 * s**4) / duration
        blend_ddot = (60 * s - 180 * s**2 + 120 * s**3) / duration**2
        return (
            self.path[index] + blend * delta,
            blend_dot * delta,
            blend_ddot * delta,
        )


def _reference_at(trajs, tcurrent):
    """Accept this solution's trajectory or the starter tuple of Bezier curves."""
    if hasattr(trajs, "evaluate"):
        return trajs.evaluate(tcurrent)
    return tuple(trajectory(tcurrent) for trajectory in trajs)


def _world_cube_hook(cube, hook_name, cube_placement):
    """Use the static URDF hook offset with the cube's simulated world pose."""
    pin.framesForwardKinematics(cube.model, cube.data, cube.q0)
    hook_in_cube = cube.data.oMf[cube.model.getFrameId(hook_name)]
    return cube_placement * hook_in_cube


def _grasp_wrench_torque(sim, robot, cube, q, vq):
    """Squeeze opposing faces and support weight through hand joint torques.

    A spring aimed at the exact hook gives zero force when aligned, so it
    cannot maintain a friction grasp. A small inward force supplies preload.
    Forces are applied ONLY to robot joints; the cube moves through contacts.
    """
    position, quaternion = sim.p.getBasePositionAndOrientation(sim.cubeId)
    cube_placement = pin.XYZQUATToSE3(np.asarray([*position, *quaternion]))
    mass = sum(sim.p.getDynamicsInfo(sim.cubeId, i)[0]
               for i in range(-1, sim.p.getNumJoints(sim.cubeId)))
    pin.computeJointJacobians(robot.model, robot.data, q)
    pin.framesForwardKinematics(robot.model, robot.data, q)

    total = np.zeros(robot.model.nv)
    for hand_name, sign in ((LEFT_HAND, -1), (RIGHT_HAND, 1)):
        frame_id = robot.model.getFrameId(hand_name)
        jacobian = pin.getFrameJacobian(
            robot.model, robot.data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        force = sign * SQUEEZE_FORCE * cube_placement.rotation[:, 1]
        force += np.array([0., 0., mass * 9.81 / 2])
        # This frame Jacobian uses linear rows followed by angular rows.
        total += jacobian.T @ np.concatenate((force, np.zeros(3)))
    return total


def controllaw(sim, robot, trajs, tcurrent, cube):
    """Run one computed-torque control step and advance Bullet by one tick."""
    q, vq = sim.getpybulletstate()
    q_desired, v_desired, a_desired = _reference_at(trajs, tcurrent)

    # Inverse dynamics converts the desired closed-loop acceleration into a
    # torque command.  nle includes gravity, Coriolis and centrifugal terms.
    closed_loop_acceleration = a_desired + Kp * (q_desired - q) + Kv * (v_desired - vq)
    torques = pin.rnea(robot.model, robot.data, q, vq, closed_loop_acceleration)
    torques += _grasp_wrench_torque(sim, robot, cube, q, vq)
    limits = np.minimum(MAX_TORQUE, robot.model.effortLimit)
    sim.step(np.clip(torques, -limits, limits).tolist())



# The simulator and runner live here so the three coursework files can be
# copied into the unchanged course repository without extra modules.
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import pybullet as pb
import pybullet_data
from pybullet_utils.bullet_client import BulletClient
from config import (NEXTAGE_URDF, TABLE_URDF, OBSTACLE_URDF, CUBE_URDF,
                    ROBOT_PLACEMENT, TABLE_PLACEMENT, OBSTACLE_PLACEMENT,
                    CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, DT)
from setup_pinocchio import setuppinocchio
from inverse_geometry import computeqgrasppose
from path import computepath

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


def run_demo(gui=False, start=None, goal=None, path=None, record=None, q0=None, qe=None):
    """Plan, initialise once, then use torque/contact physics until settled.

    start/goal are cube SE3 poses; explicit q0/qe override supplied path ends.
    Recording uses the system ffmpeg executable only; it is not needed to run.
    """
    if gui and sys.platform == 'darwin' and 'ipykernel' in sys.modules:
        raise RuntimeError(
            'PyBullet GUI can freeze inside a macOS Jupyter kernel. '
            'Run `python control.py` from Terminal to watch the physical carry; '
            'use run_demo(gui=False) inside Jupyter.'
        )
    start = CUBE_PLACEMENT if start is None else start
    goal = CUBE_PLACEMENT_TARGET if goal is None else goal
    robot, _, _, cube = setuppinocchio()
    override_start, override_goal = q0 is not None, qe is not None
    if q0 is None:
        q0, ok0 = computeqgrasppose(robot, robot.q0, cube, start)
    else:
        ok0 = True
    if qe is None:
        qe, oke = computeqgrasppose(robot, q0, cube, goal)
    else:
        oke = True
    if not (ok0 and oke):
        raise RuntimeError('Cannot run: collision-free endpoint IK failed')
    if path is None:
        path = computepath(q0, qe, start, goal, robot, cube)
    else:
        path = [np.asarray(q, dtype=float).copy() for q in path]
        if not path:
            raise ValueError('The supplied path must contain at least one configuration')
        if override_start:
            path[0] = np.asarray(q0, dtype=float).copy()
        if override_goal:
            path[-1] = np.asarray(qe, dtype=float).copy()
    trajectory = PiecewiseQuinticTrajectory(path)
    sim = Simulation(robot, gui=gui, cube_placement=start)
    encoder = None
    try:
        if record:
            ffmpeg = shutil.which('ffmpeg')
            if not ffmpeg:
                raise RuntimeError('Recording requires ffmpeg; omit --record to run without it')
            target = Path(record)
            if target.exists():
                raise FileExistsError(f'Refusing to overwrite video: {target}')
            target.parent.mkdir(parents=True, exist_ok=True)
            encoder = subprocess.Popen([ffmpeg, '-v', 'error', '-f', 'rawvideo',
                '-pix_fmt', 'rgb24', '-s', '640x480', '-r', '25', '-i', '-',
                '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(target)],
                stdin=subprocess.PIPE)
        sim.setqsim(path[0])
        max_height = start.translation[2]
        lifted_time = 0.
        both_contact_time = 0.
        max_tracking_error = 0.
        forbidden_contacts = 0
        joint_limit_samples = 0
        left_links = {i for i in range(sim.p.getNumJoints(sim.robot))
                      if sim.p.getJointInfo(sim.robot, i)[12].decode().startswith('LARM')}
        right_links = {i for i in range(sim.p.getNumJoints(sim.robot))
                       if sim.p.getJointInfo(sim.robot, i)[12].decode().startswith('RARM')}
        view = sim.p.computeViewMatrixFromYawPitchRoll([0.25, 0, 1.0], 2.0, 90, -27, 0, 2)
        projection = sim.p.computeProjectionMatrixFOV(50, 640/480, .05, 10)
        for k in range(int((trajectory.duration + 2.5) / DT)):
            tick_start = time.perf_counter()
            t = k * DT
            if gui and k % 5000 == 0:
                print(f'Physics: {t:.1f} / {trajectory.duration + 2.5:.1f} seconds', flush=True)
            # Half a second to establish normal force, then follow the path.
            controllaw(sim, robot, trajectory, max(0., t - .5), cube)
            pos, _ = sim.p.getBasePositionAndOrientation(sim.cubeId)
            max_height = max(max_height, pos[2])
            elevated = pos[2] > start.translation[2] + .05
            lifted_time += DT if elevated else 0
            contacts = sim.p.getContactPoints(sim.robot, sim.cubeId)
            active = {c[3] for c in contacts if c[9] > 0}
            if elevated and active & left_links and active & right_links:
                both_contact_time += DT
            q, _ = sim.getpybulletstate()
            ref, _, _ = trajectory.evaluate(max(0., t - .5))
            max_tracking_error = max(max_tracking_error, float(np.max(np.abs(ref - q))))
            joint_limit_samples += int(np.any(q < robot.model.lowerPositionLimit - 1e-5)
                                       or np.any(q > robot.model.upperPositionLimit + 1e-5))
            # The fixed pedestal may touch the floor; only tabletop/obstacle
            # contacts and cube/obstacle impacts are unexpected in this task.
            forbidden_contacts += int(any(sim.p.getContactPoints(a, b) for a, b in (
                (sim.robot, sim.tableId), (sim.robot, sim.obstacleId), (sim.cubeId, sim.obstacleId))))
            if encoder is not None and k % 40 == 0:
                rgba = sim.p.getCameraImage(640, 480, view, projection,
                                           renderer=sim.p.ER_TINY_RENDERER)[2]
                encoder.stdin.write(np.asarray(rgba, dtype=np.uint8)[:, :, :3].tobytes())
            if gui:
                if k % 40 == 0:
                    sim.p.configureDebugVisualizer(sim.p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
                time.sleep(max(0., DT - (time.perf_counter() - tick_start)))
        target_error = float(np.linalg.norm(np.asarray(pos) - goal.translation))
        result = dict(waypoints=len(path), cube_final=list(pos), target_error_m=target_error,
                      max_lift_m=float(max_height-start.translation[2]),
                      lifted_seconds=lifted_time, bimanual_contact_seconds=both_contact_time,
                      max_joint_tracking_error_rad=max_tracking_error,
                      forbidden_contact_ticks=forbidden_contacts,
                      joint_limit_violation_ticks=joint_limit_samples,
                      constraints=sim.p.getNumConstraints())
        result['success'] = bool(target_error < .02 and both_contact_time > 1
                                 and forbidden_contacts == 0 and joint_limit_samples == 0)
        return result
    finally:
        sim.close()
        if encoder is not None:
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError('Video encoder failed')


def main(path=None, q0=None, qe=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--record', help='New .mp4 path; optional ffmpeg is required')
    args = parser.parse_args()
    result = run_demo(gui=not args.headless, record=args.record, path=path, q0=q0, qe=qe)
    print(json.dumps(result, indent=2))
    if not result['success']:
        raise SystemExit('Physical carry did not meet the verification thresholds')



if __name__ == '__main__':
    # Set these to explicit configurations or a list of configurations to
    # experiment. None asks the code to calculate that value automatically.
    q0 = None
    qe = None
    path = None
    main(path=path, q0=q0, qe=qe)
