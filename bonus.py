#!/usr/bin/env python3


import argparse
import json
import time

import numpy as np
import pinocchio as pin

from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, DT
from control import Kp, Kv, MAX_TORQUE, PiecewiseQuinticTrajectory, Simulation, _grasp_wrench_torque
from inverse_geometry import _solve_grasp, computeqgrasppose
from path import computepath
from setup_pinocchio import setuppinocchio


GRAVITY = 9.81
LIFT = 0.12
ARC_CLEARANCE = 0.12
STROKE_TIME = 0.04

def random_target():
    rng = np.random.default_rng()
    target = CUBE_PLACEMENT_TARGET.copy()
    center = CUBE_PLACEMENT_TARGET.translation.copy()
    target.translation[:2] = center[:2] + np.array([
        rng.uniform(-0.03, 0.03),
        rng.uniform(-0.08, 0.08),
    ])
    return target


TARGET = random_target()

def gravity_throw_velocity(start, target):
    peak = max(start.translation[2], target.translation[2]) + ARC_CLEARANCE
    vz = np.sqrt(2.0 * GRAVITY * (peak - start.translation[2]))
    dz = target.translation[2] - start.translation[2]
    flight_time = (vz + np.sqrt(vz * vz - 2.0 * GRAVITY * dz)) / GRAVITY
    velocity = (target.translation - start.translation) / flight_time
    velocity[2] = vz
    return velocity, flight_time

def solve_along_segment(robot, cube, q_start, pose_start, pose_end, step=0.035):
    distance = np.linalg.norm(pose_end.translation - pose_start.translation)
    count = max(1, int(np.ceil(distance / step)))
    q = q_start
    for i in range(1, count + 1):
        alpha = i / count
        pose = pose_start.copy()
        pose.translation = (1.0 - alpha) * pose_start.translation + alpha * pose_end.translation
        q, success = computeqgrasppose(robot, q, cube, pose)
        if not success:
            return None
    return q

def open_hands(robot, q, cube_pose):
    pin.framesForwardKinematics(robot.model, robot.data, q)
    outward = cube_pose.rotation[:, 1]
    targets = []
    for name, sign in (("LARM_EFF", -1.0), ("RARM_EFF", 1.0)):
        target = robot.data.oMf[robot.model.getFrameId(name)].copy()
        target.translation += sign * 0.02 * outward + np.array([0.0, 0.0, 0.01])
        targets.append((name, target))
    q_open, success = _solve_grasp(robot, q, targets, q, 180, 5e-3)
    return q_open if success else q.copy()

def throw_trajectory(q0, q1, duration):
    def evaluate(t):
        delta = q1 - q0
        s = np.clip(float(t) / duration, 0.0, 1.0)
        h0 = 2.0 * s**3 - 3.0 * s**2 + 1.0
        h1 = -2.0 * s**3 + 3.0 * s**2
        dh0 = (6.0 * s**2 - 6.0 * s) / duration
        dh1 = (-6.0 * s**2 + 6.0 * s) / duration
        ddh0 = (12.0 * s - 6.0) / duration**2
        ddh1 = (-12.0 * s + 6.0) / duration**2
        return h0 * q0 + h1 * q1, dh0 * q0 + dh1 * q1, ddh0 * q0 + ddh1 * q1
    return evaluate

def torque_step(sim, robot, cube, q_desired, v_desired, a_desired, grasping):
    q, vq = sim.getpybulletstate()
    acceleration = a_desired + Kp * (q_desired - q) + Kv * (v_desired - vq)
    torque = pin.rnea(robot.model, robot.data, q, vq, acceleration)
    if grasping:
        torque += _grasp_wrench_torque(sim, robot, q)
    limits = np.minimum(MAX_TORQUE, robot.model.effortLimit)
    sim.step(np.clip(torque, -limits, limits).tolist())

def release_cube(sim):
    for robot_link in range(-1, sim.p.getNumJoints(sim.robot)):
        for cube_link in range(-1, sim.p.getNumJoints(sim.cubeId)):
            sim.p.setCollisionFilterPair(sim.robot, sim.cubeId, robot_link, cube_link, 0)

def draw_target(sim):
    point = TARGET.translation
    top = point + np.array([0.0, 0.0, 0.05])
    red = [1.0, 0.0, 0.0]
    sim.p.addUserDebugLine(point.tolist(), top.tolist(), red, lineWidth=4)
    sim.p.addUserDebugLine((top + [-0.06, 0.0, 0.0]).tolist(), (top + [0.06, 0.0, 0.0]).tolist(), red, lineWidth=3)
    sim.p.addUserDebugLine((top + [0.0, -0.06, 0.0]).tolist(), (top + [0.0, 0.06, 0.0]).tolist(), red, lineWidth=3)

def run_throw(gui=False):
    robot, _, _, cube = setuppinocchio()
    print(f"Random target: {TARGET.translation.tolist()}", flush=True)
    q_start, success = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT)
    if not success:
        raise RuntimeError("initial grasp IK failed")

    release_pose = TARGET.copy()
    release_pose.translation[2] += LIFT
    q_release, success = computeqgrasppose(robot, q_start, cube, release_pose)
    if not success:
        raise RuntimeError("release-pose IK failed")
    approach_path = computepath(q_start, q_release, CUBE_PLACEMENT, release_pose, robot, cube)

    velocity, flight_time = gravity_throw_velocity(release_pose, TARGET)
    stroke_pose = release_pose.copy()
    stroke_pose.translation += velocity * STROKE_TIME
    q_stroke = solve_along_segment(robot, cube, q_release, release_pose, stroke_pose)
    if q_stroke is None:
        raise RuntimeError("launch-stroke IK failed")

    q_open = open_hands(robot, q_stroke, stroke_pose)
    approach = PiecewiseQuinticTrajectory(approach_path, max_joint_speed=0.55)
    launch = throw_trajectory(q_release, q_stroke, STROKE_TIME)
    retreat = PiecewiseQuinticTrajectory([q_stroke, q_open], max_joint_speed=1.0)

    sim = Simulation(robot, gui=gui, cube_placement=CUBE_PLACEMENT)
    sim.p.removeBody(sim.obstacleId)
    if gui:
        draw_target(sim)
    try:
        sim.setqsim(q_start)
        approach_time = approach.duration + 0.35
        release_time = approach_time + STROKE_TIME
        total_time = release_time + retreat.duration + flight_time + 0.35
        released = False
        release_position = release_velocity = None
        maximum_height = CUBE_PLACEMENT.translation[2]

        for k in range(int(np.ceil(total_time / DT))):
            tick = time.perf_counter()
            t = k * DT
            if t < approach_time:
                qd, vd, ad = approach.evaluate(max(0.0, t - 0.35))
                torque_step(sim, robot, cube, qd, vd, ad, True)
            elif t < release_time:
                qd, vd, ad = launch(t - approach_time)
                torque_step(sim, robot, cube, qd, vd, ad, True)
            else:
                if not released:
                    release_cube(sim)
                    released = True
                    release_position = np.asarray(sim.p.getBasePositionAndOrientation(sim.cubeId)[0])
                    release_velocity = np.asarray(sim.p.getBaseVelocity(sim.cubeId)[0])
                qd, vd, ad = retreat.evaluate(t - release_time)
                torque_step(sim, robot, cube, qd, vd, ad, False)

            position = np.asarray(sim.p.getBasePositionAndOrientation(sim.cubeId)[0])
            maximum_height = max(maximum_height, position[2])
            if gui:
                if k % 40 == 0:
                    sim.p.configureDebugVisualizer(sim.p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
                time.sleep(max(0.0, DT - (time.perf_counter() - tick)))

        final_position = np.asarray(sim.p.getBasePositionAndOrientation(sim.cubeId)[0])
        error = float(np.linalg.norm(final_position - TARGET.translation))
        if gui:
            print(f"Throw finished; target error = {error:.3f} m", flush=True)
            time.sleep(2.0)
        return {
            "target": TARGET.translation.tolist(),
            "release_position": release_position.tolist(),
            "release_velocity": release_velocity.tolist(),
            "planned_velocity": velocity.tolist(),
            "max_height_m": float(maximum_height),
            "cube_final": final_position.tolist(),
            "target_error_m": error,
            "success": bool(error < 0.05),
        }
    finally:
        sim.close()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    result = run_throw(gui=not args.headless)
    print(json.dumps(result, indent=2))
    if not result["success"]:
        raise SystemExit("Throw missed the target")


if __name__ == "__main__":
    main()
