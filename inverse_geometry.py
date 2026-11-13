#!/usr/bin/env python3

import numpy as np
import pinocchio as pin

from config import EPSILON, LEFT_HAND, LEFT_HOOK, RIGHT_HAND, RIGHT_HOOK
from tools import collision, jointlimitsviolated, projecttojointlimits, setcubeplacement

# viewer using Meshcat
def setup_viewer():
    import zmq
    from config import MESHCAT_URL
    from tools import setupwithmeshcat
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    try:
        socket.connect(MESHCAT_URL)
        socket.send(b'url')
        ready = bool(socket.poll(500))
        if ready:
            socket.recv()
    finally:
        socket.close()
        context.term()
    return setupwithmeshcat(url=MESHCAT_URL if ready else None)


def open_viewer(viz):
    import sys
    if sys.stdin.isatty():
        viz.viewer.open()


def keep_viewer_open():
    import sys
    if sys.stdin.isatty():
        try:
            input("Meshcat is open. Press Enter to finish. ")
        except (EOFError, KeyboardInterrupt):
            print()



def _cube_hook_target(cube, cube_placement, hook_name):
    pin.framesForwardKinematics(cube.model, cube.data, cube.q0)
    hook_id = cube.model.getFrameId(hook_name)
    return cube_placement * cube.data.oMf[hook_id]

# Jacobian and caculating how far the robot is from the target 
def _pose_error_and_jacobian(robot, q, targets):
    pin.computeJointJacobians(robot.model, robot.data, q)
    pin.framesForwardKinematics(robot.model, robot.data, q)

    errors, jacobians = [], []
    for frame_name, target in targets:
        frame_id = robot.model.getFrameId(frame_name)
        current = robot.data.oMf[frame_id]
        errors.append(pin.log6(current.inverse() * target).vector)
        jacobians.append(
            pin.getFrameJacobian(robot.model, robot.data, frame_id, pin.ReferenceFrame.LOCAL)
        )
    return np.concatenate(errors), np.vstack(jacobians)


def _damped_least_squares(jacobian, error, damping):
    jj_t = jacobian @ jacobian.T
    return jacobian.T @ np.linalg.solve(jj_t + damping * np.eye(jj_t.shape[0]), error)

    #checking how good the grasp is 
def _grasp_residual(robot, cube, q, cube_placement):
    targets = (
        (LEFT_HAND, _cube_hook_target(cube, cube_placement, LEFT_HOOK)),
        (RIGHT_HAND, _cube_hook_target(cube, cube_placement, RIGHT_HOOK)),
    )
    error, _ = _pose_error_and_jacobian(robot, q, targets)
    return float(np.linalg.norm(error))

    # display the robot and cube 
def display_grasp(robot, cube, q, viz, cube_placement=None):
    pin.framesForwardKinematics(robot.model, robot.data, q)
    if cube_placement is None:
        cube_to_hook = _cube_hook_target(cube, pin.SE3.Identity(), LEFT_HOOK)
        hand = robot.data.oMf[robot.model.getFrameId(LEFT_HAND)]
        cube_placement = hand * cube_to_hook.inverse()
    setcubeplacement(robot, cube, cube_placement)
    viz.display(q)
    for hand_name, hook_name in ((LEFT_HAND, LEFT_HOOK), (RIGHT_HAND, RIGHT_HOOK)):
        hand = robot.data.oMf[robot.model.getFrameId(hand_name)]
        viz.applyConfiguration(hand_name + "frame", hand)
        # Use the non-mutating helper: reading a hook must not move the cube.
        viz.applyConfiguration(
            hand_name + "HookFrame", _cube_hook_target(cube, cube_placement, hook_name)
        )

    # Solver: repeating small steps until both hand reach the target 
def _solve_grasp(robot, q_start, targets, posture, max_iterations, tolerance):
    q = np.asarray(q_start, dtype=float).copy()
    posture = np.asarray(posture, dtype=float)
    if (q.ndim != 1 or posture.ndim != 1 or q.shape != posture.shape or
            q.size != robot.model.nq or not np.all(np.isfinite(q)) or
            not np.all(np.isfinite(posture))):
        return q, False
    q = projecttojointlimits(robot, q)
    posture = projecttojointlimits(robot, posture)
    damping = 1e-5

    for _ in range(max_iterations):
        error, jacobian = _pose_error_and_jacobian(robot, q, targets)
        if np.linalg.norm(error) < tolerance:
            return q, True

        primary_step = _damped_least_squares(jacobian, error, damping)
        null_space = np.eye(robot.model.nv) - np.linalg.pinv(jacobian) @ jacobian
        posture_step = null_space @ pin.difference(robot.model, q, posture)
        velocity = primary_step + 0.08 * posture_step
        scale = max(1.0, np.linalg.norm(velocity) / 0.20)
        q = pin.integrate(robot.model, q, velocity / scale)
        q = projecttojointlimits(robot, q)

    error, _ = _pose_error_and_jacobian(robot, q, targets)
    return q, np.linalg.norm(error) < tolerance

#sequence of starting pose 
def _restart_postures(robot, qcurrent, count=12):
    qcurrent = projecttojointlimits(robot, qcurrent)
    yield qcurrent
    yield projecttojointlimits(robot, robot.q0.copy())

    rng = np.random.default_rng(2026)
    lower, upper = robot.model.lowerPositionLimit, robot.model.upperPositionLimit
    for _ in range(count):
        candidate = robot.q0 + rng.normal(0.0, 0.35, size=robot.model.nq)
        yield np.minimum(np.maximum(candidate, lower), upper)

#return the pose that grabs the cube or return original if fails 
def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    qcurrent = np.asarray(qcurrent, dtype=float)
    if (qcurrent.ndim != 1 or qcurrent.size != robot.model.nq or
            not np.all(np.isfinite(qcurrent))):
        raise ValueError("qcurrent must be a finite robot configuration")
    setcubeplacement(robot, cube, cubetarget)
    targets = (
        (LEFT_HAND, _cube_hook_target(cube, cubetarget, LEFT_HOOK)),
        (RIGHT_HAND, _cube_hook_target(cube, cubetarget, RIGHT_HOOK)),
    )

    for posture in _restart_postures(robot, qcurrent):
        q, converged = _solve_grasp(
            robot, posture, targets, posture, max_iterations=250, tolerance=5 * EPSILON
        )
        if (converged and
                _grasp_residual(robot, cube, q, cubetarget) < 5 * EPSILON and
                not jointlimitsviolated(robot, q) and not collision(robot, q)):
            if viz is not None:
                viz.display(q)
            return q, True

    return qcurrent.copy(), False


if __name__ == "__main__":
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET

    robot, cube, viz = setup_viewer()
    q0, initial_ok = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT, viz)
    qe, final_ok = computeqgrasppose(robot, q0, cube, CUBE_PLACEMENT_TARGET, viz)
    print(f"initial grasp: {initial_ok}; target grasp: {final_ok}")
    if initial_ok:
        display_grasp(robot, cube, q0, viz, CUBE_PLACEMENT)
    open_viewer(viz)
    keep_viewer_open()
