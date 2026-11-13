#!/usr/bin/env python3
import numpy as np
import pinocchio as pin

from config import LEFT_HAND, RIGHT_HAND


Kp = 300.0
Kv = 2.0 * np.sqrt(Kp)
SQUEEZE_FORCE = 8.0 
MAX_TORQUE = 180.0

#turn path into motion 
class PiecewiseQuinticTrajectory:

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

# trying to smooth the robot instead of stopping at each one 
class OptimizedTrajectory(PiecewiseQuinticTrajectory):

    def __init__(self, path, robot, max_joint_speed=0.40):
        import quadprog
        if not np.isfinite(max_joint_speed) or max_joint_speed <= 0:
            raise ValueError('max_joint_speed must be finite and positive')
        super().__init__(path, max_joint_speed=max_joint_speed)
        count = len(self.path)
        dims = self.path[0].size
        if dims != robot.model.nq:
            raise ValueError('path configurations do not match the robot model')
        if any(np.any(q < robot.model.lowerPositionLimit - 1e-10) or
               np.any(q > robot.model.upperPositionLimit + 1e-10) for q in self.path):
            raise ValueError('path contains a configuration outside joint limits')
        self.velocities = np.zeros((count, dims))
        self.accelerations = np.zeros((count, dims))
        self.reference_objective = 0.0
        self.optimized_objective = 0.0
        self.max_joint_speed = max_joint_speed
        if count < 2:
            return

        for joint in range(dims):
            n = 2 * count
            rows, targets, columns, bounds = [], [], [], []
            lower = robot.model.lowerPositionLimit[joint]
            upper = robot.model.upperPositionLimit[joint]
            for index, T in enumerate(self.durations):
                q0, q1 = self.path[index][joint], self.path[index + 1][joint]
                delta = q1 - q0
                for u in (0.0, 0.25, 0.5, 0.75, 1.0):
                    s = 10*u**3 - 15*u**4 + 6*u**5
                    ds = 30*u*u - 60*u**3 + 30*u**4
                    dds = 60*u - 180*u*u + 120*u**3
                    h0 = u - 6*u**3 + 8*u**4 - 3*u**5
                    h1 = -4*u**3 + 7*u**4 - 3*u**5
                    h2 = (u*u - 3*u**3 + 3*u**4 - u**5) / 2
                    h3 = (u**3 - 2*u**4 + u**5) / 2
                    dh0 = 1 - 18*u*u + 32*u**3 - 15*u**4
                    dh1 = -12*u*u + 28*u**3 - 15*u**4
                    dh2 = u - 4.5*u*u + 6*u**3 - 2.5*u**4
                    dh3 = 1.5*u*u - 4*u**3 + 2.5*u**4
                    ddh0 = -36*u + 96*u*u - 60*u**3
                    ddh1 = -24*u + 84*u*u - 60*u**3
                    ddh2 = 1 - 9*u + 18*u*u - 10*u**3
                    ddh3 = 3*u - 12*u*u + 10*u**3
                    ids = [index, index + 1, count + index, count + index + 1]
                    pos_row, vel_row, acc_row = np.zeros(n), np.zeros(n), np.zeros(n)
                    pos_row[ids] = [T*h0, T*h1, T*T*h2, T*T*h3]
                    vel_row[ids] = [dh0, dh1, T*dh2, T*dh3]
                    acc_row[ids] = [ddh0/T, ddh1/T, ddh2, ddh3]
                    base_q = q0 + s*delta
                    base_v = ds*delta/T
                    base_a = dds*delta/T**2
                    columns.extend((vel_row, -vel_row))
                    bounds.extend((-max_joint_speed - base_v,
                                   -max_joint_speed + base_v))
                    if 0.0 < u < 1.0:
                        if np.isfinite(lower):
                            columns.append(pos_row); bounds.append(lower - base_q)
                        if np.isfinite(upper):
                            columns.append(-pos_row); bounds.append(base_q - upper)
                        # Penalise departing from the safe reference path.
                        rows.append(np.sqrt(1000.0) * pos_row); targets.append(0.0)
                        # Penalise acceleration, encouraging motion through joins.
                        weight = np.sqrt(0.01 * T / 3)
                        rows.append(weight * acc_row); targets.append(-weight * base_a)
            A, b = np.asarray(rows), np.asarray(targets)
            G = 2 * (A.T @ A + 1e-6 * np.eye(n))
            linear = 2 * A.T @ b
            identity = np.eye(n)
            equalities = [identity[0], identity[count - 1], identity[count], identity[-1]]
            C = np.column_stack(equalities + columns)
            limits = np.asarray([0.0] * 4 + bounds)
            try:
                result = quadprog.solve_qp(G, linear, C, limits, meq=4)[0]
            except ValueError as error:
                raise RuntimeError(f'QP infeasible for joint {joint}: {error}') from error
            self.velocities[:, joint] = result[:count]
            self.accelerations[:, joint] = result[count:]
            self.reference_objective += float(b @ b)
            residual = A @ result - b
            self.optimized_objective += float(residual @ residual + 1e-6 * (result @ result))

        peak = 0.0
        for index, T in enumerate(self.durations):
            coeff = self._coefficients(index)
            for joint in range(dims):
                c = coeff[:, joint]

                pos_poly = [5*c[5], 4*c[4], 3*c[3], 2*c[2], c[1]]
                pos_roots = np.roots(np.trim_zeros(pos_poly, 'f')) if np.any(np.abs(pos_poly) > 1e-14) else []
                pos_points = [0.0, 1.0] + [float(root.real) for root in pos_roots
                                           if abs(root.imag) < 1e-9 and 0 < root.real < 1]
                for u in pos_points:
                    value = sum(c[k] * u**k for k in range(6))
                    if (value < robot.model.lowerPositionLimit[joint] - 1e-10 or
                            value > robot.model.upperPositionLimit[joint] + 1e-10):
                        raise RuntimeError(f'QP trajectory crosses joint {joint} limits between samples')
                acc_poly = [20*c[5], 12*c[4], 6*c[3], 2*c[2]]
                roots = np.roots(np.trim_zeros(acc_poly, 'f')) if np.any(np.abs(acc_poly) > 1e-14) else []
                points = [0.0, 1.0] + [float(root.real) for root in roots
                                       if abs(root.imag) < 1e-9 and 0 < root.real < 1]
                for u in points:
                    speed = abs((c[1] + 2*c[2]*u + 3*c[3]*u**2 + 4*c[4]*u**3 + 5*c[5]*u**4)/T)
                    peak = max(peak, speed)
        factor = max(1.0, peak / max_joint_speed)
        self.durations *= factor
        self.velocities /= factor
        self.accelerations /= factor**2
        self.breakpoints = np.concatenate(([0.0], np.cumsum(self.durations)))
        self.duration = float(self.breakpoints[-1])

    def _coefficients(self, index):
        T = self.durations[index]
        q0, q1 = self.path[index:index + 2]
        v0, v1 = T * self.velocities[index:index + 2]
        a0, a1 = T*T * self.accelerations[index:index + 2]
        delta = q1 - q0
        return np.asarray([q0, v0, a0/2,
            10*delta - 6*v0 - 4*v1 - 1.5*a0 + 0.5*a1,
            -15*delta + 8*v0 + 7*v1 + 1.5*a0 - a1,
            6*delta - 3*v0 - 3*v1 - 0.5*a0 + 0.5*a1])

    def evaluate(self, t):
        if not self.durations.size:
            zero = np.zeros_like(self.path[0])
            return self.path[0].copy(), zero, zero
        index, u = self._segment(t)
        T = self.durations[index]
        c = self._coefficients(index)
        q = c[0] + c[1]*u + c[2]*u**2 + c[3]*u**3 + c[4]*u**4 + c[5]*u**5
        v = (c[1] + 2*c[2]*u + 3*c[3]*u**2 + 4*c[4]*u**3 + 5*c[5]*u**4)/T
        a = (2*c[2] + 6*c[3]*u + 12*c[4]*u**2 + 20*c[5]*u**3)/T**2
        return q, v, a

# checking if the trajectory is valid with no collision and within joint limits
def validate_trajectory(robot, cube, trajectory, samples_per_segment=51):
    from config import LEFT_HOOK, RIGHT_HOOK
    from inverse_geometry import _cube_hook_target
    from tools import collision, setcubeplacement
    left_id = robot.model.getFrameId(LEFT_HAND)
    right_id = robot.model.getFrameId(RIGHT_HAND)
    left_hook = _cube_hook_target(cube, pin.SE3.Identity(), LEFT_HOOK)
    right_hook = _cube_hook_target(cube, pin.SE3.Identity(), RIGHT_HOOK)
    max_error, samples = 0.0, 0
    intervals = list(zip(trajectory.breakpoints[:-1], trajectory.breakpoints[1:]))
    if not intervals:
        intervals = [(0.0, 0.0)]
    for first, last in intervals:
        for t in np.linspace(first, last, samples_per_segment):
            q, _, _ = trajectory.evaluate(t)
            pin.framesForwardKinematics(robot.model, robot.data, q)
            cube_pose = robot.data.oMf[left_id] * left_hook.inverse()
            error = float(np.linalg.norm(pin.log6(
                robot.data.oMf[right_id].inverse() * cube_pose * right_hook).vector))
            max_error = max(max_error, error)
            setcubeplacement(robot, cube, cube_pose)
            if (error >= 0.005 or
                    np.any(q < robot.model.lowerPositionLimit - 1e-10) or
                    np.any(q > robot.model.upperPositionLimit + 1e-10) or
                    collision(robot, q)):
                raise RuntimeError('Optimized trajectory failed a sampled grasp, joint-limit or collision check')
            samples += 1
    return dict(trajectory_samples_checked=samples, max_grasp_residual=max_error)


def _reference_at(trajs, tcurrent):
    if hasattr(trajs, "evaluate"):
        return trajs.evaluate(tcurrent)
    return tuple(trajectory(tcurrent) for trajectory in trajs)

#holding the cube 
def _grasp_wrench_torque(sim, robot, q):
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
    q, vq = sim.getpybulletstate()
    q_desired, v_desired, a_desired = _reference_at(trajs, tcurrent)

    closed_loop_acceleration = a_desired + Kp * (q_desired - q) + Kv * (v_desired - vq)
    torques = pin.rnea(robot.model, robot.data, q, vq, closed_loop_acceleration)
    torques += _grasp_wrench_torque(sim, robot, q)
    limits = np.minimum(MAX_TORQUE, robot.model.effortLimit)
    sim.step(np.clip(torques, -limits, limits).tolist())




import argparse
import json
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
# the physics simulation 
class Simulation:
    def __init__(self, robot, gui=False, cube_placement=CUBE_PLACEMENT):
        self.p = BulletClient(connection_mode=pb.GUI if gui else pb.DIRECT)
        if gui:
            self.p.configureDebugVisualizer(pb.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
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

#running the task 
def run_demo(gui=False, start=None, goal=None, path=None, q0=None, qe=None,
             trajectory_method='qp'):
    if gui and sys.platform == 'darwin' and 'ipykernel' in sys.modules:
        raise RuntimeError(
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
    if trajectory_method == 'qp':
        trajectory = OptimizedTrajectory(path, robot)
        checked = validate_trajectory(robot, cube, trajectory)
    elif trajectory_method == 'quintic':
        trajectory = PiecewiseQuinticTrajectory(path)
        checked = {}
    else:
        raise ValueError("trajectory_method must be 'qp' or 'quintic'")
    sim = Simulation(robot, gui=gui, cube_placement=start)
    try:
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
            forbidden_contacts += int(any(sim.p.getContactPoints(a, b) for a, b in (
                (sim.robot, sim.tableId), (sim.robot, sim.obstacleId), (sim.cubeId, sim.obstacleId))))
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
        result.update(checked)
        result['trajectory_method'] = trajectory_method
        if trajectory_method == 'qp':
            result['qp_reference_objective'] = trajectory.reference_objective
            result['qp_optimized_objective'] = trajectory.optimized_objective
        result['success'] = bool(target_error < .02 and both_contact_time > 1
                                 and forbidden_contacts == 0 and joint_limit_samples == 0)
        return result
    finally:
        sim.close()


def main(path=None, q0=None, qe=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--trajectory', choices=('qp', 'quintic'), default='qp',
                        help='Use the QP trajectory (default) or original quintic timing')
    args = parser.parse_args()
    result = run_demo(gui=not args.headless, path=path, q0=q0, qe=qe,
                      trajectory_method=args.trajectory)
    print(json.dumps(result, indent=2))
    if not result['success']:
        raise SystemExit('Physical carry did not meet the verification thresholds')



if __name__ == '__main__':
    q0 = None
    qe = None
    path = None
    main(path=path, q0=q0, qe=qe)
