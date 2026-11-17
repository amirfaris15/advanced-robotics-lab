#!/usr/bin/env python3
"""Physical two-hand ballistic throw with an offset launch point."""


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
LIFT = 0.20
ARC_CLEARANCE = 0.06
STROKE_TIME = 0.10
LAUNCH_OFFSETS = ((0.0, -0.20), (0.0, -0.18), (0.0, -0.22),
                  (0.05, 0.0), (0.07, 0.0), (0.09, 0.0))
OPEN_DISTANCE = 0.18
OPEN_TIME = 0.04
MIN_FLIGHT_DISTANCE = 0.15
MAX_FLIGHT_DISTANCE = 0.25
MIN_AIR_TIME = 0.15
SEPARATION_TIME = 0.020

# Pick a landing spot near the configured target.
def random_target(rng):
    target = CUBE_PLACEMENT_TARGET.copy()
    center = CUBE_PLACEMENT_TARGET.translation.copy()
    target.translation[:2] = center[:2] + np.array([
        rng.uniform(-0.03, 0.03),
        rng.uniform(-0.08, 0.08),
    ])
    return target


# Work out the launch velocity and flight time for the chosen arc.
def gravity_throw_velocity(start, target):
    peak = max(start.translation[2], target.translation[2]) + ARC_CLEARANCE
    vz = np.sqrt(2.0 * GRAVITY * (peak - start.translation[2]))
    dz = target.translation[2] - start.translation[2]
    flight_time = (vz + np.sqrt(vz * vz - 2.0 * GRAVITY * dz)) / GRAVITY
    velocity = (target.translation - start.translation) / flight_time
    velocity[2] = vz
    return velocity, flight_time

# Solve IK at small steps along the launch path.
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

# Find a pose with both hands moved away from the cube.
def open_hands(robot, q, cube_pose):
    pin.framesForwardKinematics(robot.model, robot.data, q)
    outward = cube_pose.rotation[:, 1]
    targets = []
    for name, sign in (("LARM_EFF", 1.0), ("RARM_EFF", -1.0)):
        target = robot.data.oMf[robot.model.getFrameId(name)].copy()
        target.translation += sign * OPEN_DISTANCE * outward
        targets.append((name, target))
    q_open, success = _solve_grasp(robot, q, targets, q, 180, 5e-3)
    if not success:
        raise RuntimeError("outward hand-opening IK failed")
    return q_open


# Open both hands across the grasp axis after the throw.
def open_hands_followthrough(robot, q, cube_pose, velocity):
    pin.framesForwardKinematics(robot.model, robot.data, q)
    outward = cube_pose.rotation[:, 1]
    # A forward movement can push one hand back towards the cube.
    forward = np.zeros(3)
    targets = []
    for name, sign in (("LARM_EFF", 1.0), ("RARM_EFF", -1.0)):
        target = robot.data.oMf[robot.model.getFrameId(name)].copy()
        target.translation += forward + sign * OPEN_DISTANCE * outward
        targets.append((name, target))
    q_open, success = _solve_grasp(robot, q, targets, q, 220, 5e-3)
    if not success:
        raise RuntimeError("forward-and-outward hand-opening IK failed")
    return q_open

# Find joint speeds that give both hands the desired launch velocity.
def launch_joint_velocity(robot, q, velocity):
    pin.computeJointJacobians(robot.model, robot.data, q)
    pin.framesForwardKinematics(robot.model, robot.data, q)
    jacobians = [pin.getFrameJacobian(
        robot.model, robot.data, robot.model.getFrameId(name),
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    ) for name in ("LARM_EFF", "RARM_EFF")]
    jacobian = np.vstack(jacobians)
    desired = np.tile(np.r_[velocity, np.zeros(3)], 2)
    joint_velocity = np.linalg.lstsq(jacobian, desired, rcond=None)[0]
    if np.linalg.norm(jacobian @ joint_velocity - desired) > 0.02:
        raise RuntimeError("launch velocity cannot be achieved by both hands")
    return joint_velocity


# Connect two joint poses with the chosen start and end speeds.
def throw_trajectory(q0, q1, duration, v0=None, v1=None):
    v0 = np.zeros_like(q0) if v0 is None else v0
    v1 = np.zeros_like(q1) if v1 is None else v1
    def evaluate(t):
        if t > duration:
            return q1.copy(), np.zeros_like(q1), np.zeros_like(q1)
        s = np.clip(float(t) / duration, 0.0, 1.0)
        h00, h10 = 2*s**3 - 3*s**2 + 1, s**3 - 2*s**2 + s
        h01, h11 = -2*s**3 + 3*s**2, s**3 - s**2
        dh00, dh10 = 6*s**2 - 6*s, 3*s**2 - 4*s + 1
        dh01, dh11 = -dh00, 3*s**2 - 2*s
        ddh00, ddh10 = 12*s - 6, 6*s - 4
        ddh01, ddh11 = -ddh00, 6*s - 2
        return (h00*q0 + h10*duration*v0 + h01*q1 + h11*duration*v1,
                (dh00*q0 + dh10*duration*v0 + dh01*q1 + dh11*duration*v1) / duration,
                (ddh00*q0 + ddh10*duration*v0 + ddh01*q1 + ddh11*duration*v1) / duration**2)
    return evaluate


# Solve the hand poses along the accelerating launch stroke.
def cartesian_stroke_path(robot, cube, q_start, pose_start, pose_end,
                          duration, velocity, step=0.005):
    distance = float(np.linalg.norm(pose_end.translation - pose_start.translation))
    count = max(2, int(np.ceil(distance / step)))
    path = [np.asarray(q_start).copy()]
    q = path[0]
    for i in range(1, count + 1):
        s = i / count
        blend = s**3 - 0.5 * s**4
        pose = pose_start.copy()
        pose.translation = pose_start.translation + blend * duration * velocity
        q, success = computeqgrasppose(robot, q, cube, pose)
        if not success:
            return None
        path.append(q.copy())
    return np.asarray(path)


# Turn the sampled IK poses into a smooth joint trajectory.
def sampled_joint_trajectory(path, duration, end_velocity):
    path = np.asarray(path, dtype=float)
    count = len(path)
    times = np.linspace(0.0, duration, count)
    velocities = np.zeros_like(path)
    accelerations = np.zeros_like(path)
    if count > 2:
        velocities[1:-1] = (path[2:] - path[:-2]) / (times[2:, None] - times[:-2, None])
        accelerations[1:-1] = 2.0 * ((path[2:] - path[1:-1]) / (times[2:, None] - times[1:-1, None]) -
                                     (path[1:-1] - path[:-2]) / (times[1:-1, None] - times[:-2, None])) / (times[2:, None] - times[:-2, None])
    velocities[-1] = end_velocity

    def evaluate(t):
        if t >= duration:
            return path[-1].copy(), np.zeros_like(path[-1]), np.zeros_like(path[-1])
        t = max(0.0, float(t))
        index = min(int(t / duration * (count - 1)), count - 2)
        h = times[index + 1] - times[index]
        u = (t - times[index]) / h
        q0, q1 = path[index], path[index + 1]
        v0, v1 = velocities[index], velocities[index + 1]
        a0, a1 = accelerations[index], accelerations[index + 1]
        c0 = q0
        c1 = h * v0
        c2 = 0.5 * h * h * a0
        d = q1 - c0 - c1 - c2
        e = h * v1 - c1 - 2.0 * c2
        f = h * h * a1 - 2.0 * c2
        c3 = 10*d - 4*e + 0.5*f
        c4 = -15*d + 7*e - f
        c5 = 6*d - 3*e + 0.5*f
        q = c0 + c1*u + c2*u**2 + c3*u**3 + c4*u**4 + c5*u**5
        v = (c1 + 2*c2*u + 3*c3*u**2 + 4*c4*u**3 + 5*c5*u**4) / h
        a = (2*c2 + 6*c3*u + 12*c4*u**2 + 20*c5*u**3) / (h*h)
        return q, v, a
    return evaluate

# Track the desired joint motion and add grip force while holding the cube.
def torque_step(sim, robot, cube, q_desired, v_desired, a_desired, grasping):
    q, vq = sim.getpybulletstate()
    acceleration = a_desired + Kp * (q_desired - q) + Kv * (v_desired - vq)
    torque = pin.rnea(robot.model, robot.data, q, vq, acceleration)
    if grasping:
        torque += _grasp_wrench_torque(sim, robot, q)
    limits = np.minimum(MAX_TORQUE, robot.model.effortLimit)
    sim.step(np.clip(torque, -limits, limits).tolist())

# Draw the target and the predicted flight path.
def draw_target(sim, target, release_pose, velocity, flight_time):
    point = target.translation
    top = point + np.array([0.0, 0.0, 0.05])
    red = [1.0, 0.0, 0.0]
    sim.p.addUserDebugLine(point.tolist(), top.tolist(), red, lineWidth=4)
    sim.p.addUserDebugLine((top + [-0.06, 0.0, 0.0]).tolist(), (top + [0.06, 0.0, 0.0]).tolist(), red, lineWidth=3)
    sim.p.addUserDebugLine((top + [0.0, -0.06, 0.0]).tolist(), (top + [0.0, 0.06, 0.0]).tolist(), red, lineWidth=3)
    ring_z = point[2] - 0.045
    ring = [np.array([point[0] + 0.055*np.cos(a), point[1] + 0.055*np.sin(a), ring_z])
            for a in np.linspace(0.0, 2.0*np.pi, 25)]
    for a, b in zip(ring[:-1], ring[1:]):
        sim.p.addUserDebugLine(a.tolist(), b.tolist(), red, lineWidth=2)

    # The dashed line shows the predicted path before the throw.
    previous = release_pose.translation.copy()
    for i in range(1, 41):
        t0 = (i - 1) * flight_time / 40
        t1 = i * flight_time / 40
        current = release_pose.translation + velocity*t1 + np.array([0.0, 0.0, -0.5*GRAVITY*t1*t1])
        if i % 2:
            sim.p.addUserDebugLine(previous.tolist(), current.tolist(), [0.42, 0.58, 0.72], lineWidth=1)
        previous = current
    camera_target = [0.5*(release_pose.translation[0] + point[0]),
                     0.5*(release_pose.translation[1] + point[1]), 1.06]
    sim.p.resetDebugVisualizerCamera(1.5, 55, -20, camera_target)


def set_throw_status(sim, target, label, item_id=None):
    return sim.p.addUserDebugText(label, (target.translation + [0.0, 0.0, 0.20]).tolist(),
                                  textColorRGB=[1.0, 1.0, 1.0], textSize=1.4,
                                  lifeTime=0, replaceItemUniqueId=-1 if item_id is None else item_id)

# Plan the approach, launch the cube, and check where it lands.
def run_throw(gui=False, seed=15, target=None, randomize=False):
    if target is None:
        target = random_target(np.random.default_rng(seed)) if randomize else CUBE_PLACEMENT_TARGET.copy()
    else:
        target = target.copy()
    robot, _, _, cube = setuppinocchio()
    print(f"Target (seed {seed}): {target.translation.tolist()}", flush=True)
    q_start, success = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT)
    if not success:
        raise RuntimeError("initial grasp IK failed")

    # Try a few release points in case the first one is blocked or unreachable.
    planning_error = None
    for x_offset, y_offset in LAUNCH_OFFSETS:
        release_pose = target.copy()
        release_pose.translation[:2] += [-x_offset, y_offset]
        release_pose.translation[2] += LIFT
        velocity, flight_time = gravity_throw_velocity(release_pose, target)
        stroke_start = release_pose.copy()
        stroke_start.translation -= 0.5 * STROKE_TIME * velocity
        q_stroke_start, success = computeqgrasppose(robot, q_start, cube, stroke_start)
        if not success:
            planning_error = "launch-stroke start IK failed"
            continue
        stroke_path = cartesian_stroke_path(robot, cube, q_stroke_start,
                                            stroke_start, release_pose,
                                            STROKE_TIME, velocity)
        if stroke_path is None:
            planning_error = "launch-stroke IK failed"
            continue
        q_release = stroke_path[-1]
        try:
            v_release = launch_joint_velocity(robot, q_release, velocity)
            q_open = open_hands_followthrough(robot, q_release, release_pose, velocity)
            approach_path = computepath(q_start, q_stroke_start, CUBE_PLACEMENT, stroke_start, robot, cube)
        except RuntimeError as exc:
            planning_error = str(exc)
            continue
        break
    else:
        raise RuntimeError(f"no feasible launch plan: {planning_error}")

    approach = PiecewiseQuinticTrajectory(approach_path, max_joint_speed=0.55)
    launch = sampled_joint_trajectory(stroke_path, STROKE_TIME, v_release)
    retreat = throw_trajectory(q_release, q_open, OPEN_TIME, v0=v_release)

    sim = Simulation(robot, gui=gui, cube_placement=CUBE_PLACEMENT)
    sim.p.removeBody(sim.obstacleId)
    if gui:
        draw_target(sim, target, release_pose, velocity, flight_time)
    try:
        sim.setqsim(q_start)
        approach_time = approach.duration + 0.35
        release_time = approach_time + STROKE_TIME
        total_time = release_time + flight_time + 0.90
        released = False
        release_position = release_velocity = None
        separation_position = separation_velocity = None
        separation_candidate = None
        separation_candidate_velocity = None
        separation_candidate_time = None
        separation_clear_ticks = 0
        recontact_before_landing = False
        landed = False
        landing_position = None
        landing_time = None
        flight_max_height = None
        status_phase = "HOLDING · APPROACH"
        status_id = set_throw_status(sim, target, status_phase) if gui else None
        trail_last_position = None
        last_trail_tick = -10

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
                    released = True
                    release_position = np.asarray(sim.p.getBasePositionAndOrientation(sim.cubeId)[0])
                    release_velocity = np.asarray(sim.p.getBaseVelocity(sim.cubeId)[0])
                qd, vd, ad = retreat(t - release_time)
                torque_step(sim, robot, cube, qd, vd, ad, False)

            position = np.asarray(sim.p.getBasePositionAndOrientation(sim.cubeId)[0])
            robot_contact = bool(sim.p.getContactPoints(bodyA=sim.robot, bodyB=sim.cubeId))
            if released:
                table_touch = bool(sim.p.getContactPoints(bodyA=sim.tableId, bodyB=sim.cubeId))
                if separation_position is None:
                    if robot_contact:
                        if separation_candidate is not None:
                            recontact_before_landing = True
                        separation_candidate = None
                        separation_candidate_velocity = None
                        separation_candidate_time = None
                        separation_clear_ticks = 0
                    elif table_touch:
                        separation_candidate = None
                        separation_clear_ticks = 0
                    else:
                        if separation_candidate is None:
                            separation_candidate = position.copy()
                            separation_candidate_velocity = np.asarray(sim.p.getBaseVelocity(sim.cubeId)[0])
                            separation_candidate_time = t
                            flight_max_height = float(position[2])
                        flight_max_height = max(flight_max_height, float(position[2]))
                        separation_clear_ticks += 1
                        if separation_clear_ticks >= int(np.ceil(SEPARATION_TIME / DT)):
                            separation_position = separation_candidate
                            separation_velocity = separation_candidate_velocity
                            separation_tick = separation_candidate_time
                elif not landed:
                    if robot_contact:
                        recontact_before_landing = True
                    if table_touch:
                        landed = True
                        landing_position = position.copy()
                        landing_time = t
                    else:
                        flight_max_height = max(flight_max_height, float(position[2]))
                if gui and separation_position is not None and not landed and k - last_trail_tick >= 5:
                    if trail_last_position is not None:
                        sim.p.addUserDebugLine(trail_last_position.tolist(), position.tolist(),
                                               [0.0, 0.95, 1.0], lineWidth=4)
                    trail_last_position = position.copy()
                    last_trail_tick = k
            if gui:
                phase = ("LANDED" if landed else "RECONTACT" if recontact_before_landing else
                         "FREE FLIGHT" if separation_position is not None else
                         "RELEASING" if released else "THROW STROKE" if t >= approach_time else
                         "HOLDING · APPROACH")
                if phase != status_phase:
                    status_phase = phase
                    status_id = set_throw_status(sim, target, phase, status_id)
            if gui:
                if k % 16 == 0:
                    sim.p.configureDebugVisualizer(sim.p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
                time.sleep(max(0.0, DT - (time.perf_counter() - tick)))

        final_position = np.asarray(sim.p.getBasePositionAndOrientation(sim.cubeId)[0])
        linear_velocity, angular_velocity = sim.p.getBaseVelocity(sim.cubeId)
        table_contact = bool(sim.p.getContactPoints(bodyA=sim.tableId, bodyB=sim.cubeId))
        settled = bool(np.linalg.norm(linear_velocity) < 0.05 and np.linalg.norm(angular_velocity) < 0.5)
        error = float(np.linalg.norm(final_position - target.translation))
        if gui:
            print(f"Throw finished; target error = {error:.3f} m", flush=True)
            time.sleep(2.0)
        return {
            "target": target.translation.tolist(),
            "planned_release_position": release_pose.translation.tolist(),
            "release_position": release_position.tolist(),
            "release_velocity": release_velocity.tolist(),
            "planned_velocity": velocity.tolist(),
            "separation_position": separation_position.tolist() if separation_position is not None else None,
            "separation_velocity": separation_velocity.tolist() if separation_velocity is not None else None,
            "airborne_horizontal_distance_m": (float(np.linalg.norm(landing_position[:2] - separation_position[:2]))
                                                   if landing_position is not None and separation_position is not None else 0.0),
            "air_duration_s": max(0.0, (landing_time - separation_tick)) if landing_position is not None else 0.0,
            "recontact_before_landing": recontact_before_landing,
            "landing_position": landing_position.tolist() if landing_position is not None else None,
            "max_height_m": float(flight_max_height if flight_max_height is not None else CUBE_PLACEMENT.translation[2]),
            "cube_final": final_position.tolist(),
            "table_contact": table_contact,
            "settled": settled,
            "target_error_m": error,
            "success": bool(error < 0.05 and separation_position is not None and landing_position is not None and
                            MIN_FLIGHT_DISTANCE <= np.linalg.norm(landing_position[:2] - separation_position[:2]) <= MAX_FLIGHT_DISTANCE and
                            (landing_time - separation_tick) >= MIN_AIR_TIME and not recontact_before_landing and flight_max_height >= separation_position[2] + 0.025 and
                            table_contact and settled),
        }
    finally:
        sim.close()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--random-target", action="store_true",
                        help="choose a reproducible nearby target using --seed")
    parser.add_argument("--seed", type=int, default=15)
    args = parser.parse_args()
    result = run_throw(gui=not args.headless, seed=args.seed, randomize=args.random_target)
    print(json.dumps(result, indent=2))
    if not result["success"]:
        raise SystemExit("Throw missed the target")


if __name__ == "__main__":
    main()
