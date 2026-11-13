#!/usr/bin/env python3  # Run this file with Python 3.


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
    """Cubic joint motion ending with the launch velocity."""
    def evaluate(t):
        delta = q1 - q0  # Total joint displacement.
        s = np.clip(float(t) / duration, 0.0, 1.0)  # Normalize time to [0, 1].
        h0 = 2.0 * s**3 - 3.0 * s**2 + 1.0  # Cubic position coefficient at q0.
        h1 = -2.0 * s**3 + 3.0 * s**2  # Cubic position coefficient at q1.
        dh0 = (6.0 * s**2 - 6.0 * s) / duration  # Velocity coefficient at q0.
        dh1 = (-6.0 * s**2 + 6.0 * s) / duration  # Velocity coefficient at q1.
        ddh0 = (12.0 * s - 6.0) / duration**2  # Acceleration coefficient at q0.
        ddh1 = (-12.0 * s + 6.0) / duration**2  # Acceleration coefficient at q1.
        return h0 * q0 + h1 * q1, dh0 * q0 + dh1 * q1, ddh0 * q0 + ddh1 * q1  # q, qdot, qddot.
    return evaluate  # Return a function that evaluates the launch trajectory.


def torque_step(sim, robot, cube, q_desired, v_desired, a_desired, grasping):
    q, vq = sim.getpybulletstate()  # Read measured joint position and velocity.
    acceleration = a_desired + Kp * (q_desired - q) + Kv * (v_desired - vq)  # PD corrected acceleration.
    torque = pin.rnea(robot.model, robot.data, q, vq, acceleration)  # Convert acceleration to torque.
    if grasping:
        torque += _grasp_wrench_torque(sim, robot, q)  # Add squeeze and weight support.
    limits = np.minimum(MAX_TORQUE, robot.model.effortLimit)  # Respect both torque limits.
    sim.step(np.clip(torque, -limits, limits).tolist())  # Apply torque and advance Bullet.


def release_cube(sim):
    """Open the simulated gripper by disabling robot/cube contacts."""
    for robot_link in range(-1, sim.p.getNumJoints(sim.robot)):
        for cube_link in range(-1, sim.p.getNumJoints(sim.cubeId)):
            sim.p.setCollisionFilterPair(sim.robot, sim.cubeId, robot_link, cube_link, 0)  # Remove contact pair.


def draw_target(sim):
    point = TARGET.translation  # Target position on the table.
    top = point + np.array([0.0, 0.0, 0.05])  # Lift marker above the table.
    red = [1.0, 0.0, 0.0]  # Marker color.
    sim.p.addUserDebugLine(point.tolist(), top.tolist(), red, lineWidth=4)  # Vertical marker stem.
    sim.p.addUserDebugLine((top + [-0.06, 0.0, 0.0]).tolist(), (top + [0.06, 0.0, 0.0]).tolist(), red, lineWidth=3)  # X crossbar.
    sim.p.addUserDebugLine((top + [0.0, -0.06, 0.0]).tolist(), (top + [0.0, 0.06, 0.0]).tolist(), red, lineWidth=3)  # Y crossbar.


def run_throw(gui=False):
    robot, _, _, cube = setuppinocchio()  # Build robot, cube, table, and collision models.
    print(f"Random target: {TARGET.translation.tolist()}", flush=True)  # Show selected target.
    q_start, success = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT)  # Find initial grasp.
    if not success:
        raise RuntimeError("initial grasp IK failed")  # Stop if the start cannot be grasped.

    release_pose = TARGET.copy()  # Release directly above the random target.
    release_pose.translation[2] += LIFT  # Move release pose upward.
    q_release, success = computeqgrasppose(robot, q_start, cube, release_pose)  # Solve release IK.
    if not success:
        raise RuntimeError("release-pose IK failed")  # Stop if release pose is unreachable.
    approach_path = computepath(q_start, q_release, CUBE_PLACEMENT, release_pose, robot, cube)  # Plan RRT.

    velocity, flight_time = gravity_throw_velocity(release_pose, TARGET)  # Calculate projectile motion.
    stroke_pose = release_pose.copy()  # Copy release pose for the launch endpoint.
    stroke_pose.translation += velocity * STROKE_TIME  # Move endpoint using planned velocity.
    q_stroke = solve_along_segment(robot, cube, q_release, release_pose, stroke_pose)  # Solve launch IK.
    if q_stroke is None:
        raise RuntimeError("launch-stroke IK failed")  # Stop if launch pose is unreachable.

    q_open = open_hands(robot, q_stroke, stroke_pose)  # Compute a small hand-opening motion.
    approach = PiecewiseQuinticTrajectory(approach_path, max_joint_speed=0.55)  # Time-parameterize RRT path.
    launch = throw_trajectory(q_release, q_stroke, STROKE_TIME)  # Time-parameterize launch stroke.
    retreat = PiecewiseQuinticTrajectory([q_stroke, q_open], max_joint_speed=1.0)  # Time-parameterize retreat.

    sim = Simulation(robot, gui=gui, cube_placement=CUBE_PLACEMENT)  # Create Bullet simulation.
    sim.p.removeBody(sim.obstacleId)  # Remove obstacle only in this bonus run.
    if gui:
        draw_target(sim)  # Draw the random target marker.
    try:
        sim.setqsim(q_start)  # Initialize Bullet joints at the IK start configuration.
        approach_time = approach.duration + 0.35  # Add time for contact to settle.
        release_time = approach_time + STROKE_TIME  # Time at which the cube is released.
        total_time = release_time + retreat.duration + flight_time + 0.35  # Total simulation duration.
        released = False  # Tracks whether collision contacts have been removed.
        release_position = release_velocity = None  # Values recorded at release.
        maximum_height = CUBE_PLACEMENT.translation[2]  # Initial cube height.

        for k in range(int(np.ceil(total_time / DT))):  # Simulate one Bullet tick at a time.
            tick = time.perf_counter()  # Start timing this tick for GUI pacing.
            t = k * DT  # Convert tick index to simulation time.
            if t < approach_time:
                qd, vd, ad = approach.evaluate(max(0.0, t - 0.35))  # Desired approach state.
                torque_step(sim, robot, cube, qd, vd, ad, True)  # Track while holding cube.
            elif t < release_time:
                qd, vd, ad = launch(t - approach_time)  # Desired launch state.
                torque_step(sim, robot, cube, qd, vd, ad, True)  # Track while launching.
            else:
                if not released:
                    release_cube(sim)  # Remove robot/cube contacts once.
                    released = True  # Prevent releasing again.
                    release_position = np.asarray(sim.p.getBasePositionAndOrientation(sim.cubeId)[0])  # Save position.
                    release_velocity = np.asarray(sim.p.getBaseVelocity(sim.cubeId)[0])  # Save velocity.
                qd, vd, ad = retreat.evaluate(t - release_time)  # Desired hand retreat state.
                torque_step(sim, robot, cube, qd, vd, ad, False)  # Move hands without grasp force.

            position = np.asarray(sim.p.getBasePositionAndOrientation(sim.cubeId)[0])  # Read cube position.
            maximum_height = max(maximum_height, position[2])  # Track peak cube height.
            if gui:
                if k % 40 == 0:
                    sim.p.configureDebugVisualizer(sim.p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)  # Refresh GUI.
                time.sleep(max(0.0, DT - (time.perf_counter() - tick)))  # Run GUI in real time.

        final_position = np.asarray(sim.p.getBasePositionAndOrientation(sim.cubeId)[0])  # Read landing position.
        error = float(np.linalg.norm(final_position - TARGET.translation))  # Calculate landing error.
        if gui:
            print(f"Throw finished; target error = {error:.3f} m", flush=True)  # Report GUI result.
            time.sleep(2.0)  # Leave the final frame visible briefly.
        return {
            "target": TARGET.translation.tolist(),  # Random target coordinates.
            "release_position": release_position.tolist(),  # Actual release position.
            "release_velocity": release_velocity.tolist(),  # Actual release velocity.
            "planned_velocity": velocity.tolist(),  # Gravity-calculated velocity.
            "max_height_m": float(maximum_height),  # Highest cube position.
            "cube_final": final_position.tolist(),  # Final cube coordinates.
            "target_error_m": error,  # Distance from target after landing.
            "success": bool(error < 0.05),  # Success requires less than 5 cm error.
        }
    finally:
        sim.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)  # Create command-line parser.
    parser.add_argument("--headless", action="store_true")  # Add optional no-window mode.
    args = parser.parse_args()  # Read command-line arguments.
    result = run_throw(gui=not args.headless)  # Run GUI unless --headless was supplied.
    print(json.dumps(result, indent=2))  # Print all measured results.
    if not result["success"]:
        raise SystemExit("Throw missed the target tolerance")  # Return failure status.


if __name__ == "__main__":  # Only run automatically when executed as a script.
    main()  # Start the program.
