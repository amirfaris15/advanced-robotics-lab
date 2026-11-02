#!/usr/bin/env python3
"""Collision-checked bimanual inverse geometry for Coursework 2."""

import numpy as np
import pinocchio as pin

from config import EPSILON, LEFT_HAND, LEFT_HOOK, RIGHT_HAND, RIGHT_HOOK
from tools import collision, jointlimitsviolated, projecttojointlimits, setcubeplacement
# Keep the relative hand placement locally; the original course solution.py
# does not provide a helper for storing it.
HAND_RELATION = None

def remember_hand_relation(left, right):
    global HAND_RELATION
    HAND_RELATION = left.inverse() * right
    return HAND_RELATION


def setup_viewer():
    """Use an existing Meshcat server, or start one when none responds."""
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



def _frame_placement(robot, q, frame_name):
    """Update kinematics and return the world placement of one named frame."""
    pin.framesForwardKinematics(robot.model, robot.data, q)
    return robot.data.oMf[robot.model.getFrameId(frame_name)]


def _cube_hook_target(cube, cube_placement, hook_name):
    """Return a hook's world pose without mutating its geometry placement."""
    pin.framesForwardKinematics(cube.model, cube.data, cube.q0)
    hook_id = cube.model.getFrameId(hook_name)
    return cube_placement * cube.data.oMf[hook_id]


def _pose_error_and_jacobian(robot, q, targets):
    """Stack the two 6-D local SE(3) errors and their matching Jacobians."""
    pin.computeJointJacobians(robot.model, robot.data, q)
    pin.framesForwardKinematics(robot.model, robot.data, q)

    errors, jacobians = [], []
    for frame_name, target in targets:
        frame_id = robot.model.getFrameId(frame_name)
        current = robot.data.oMf[frame_id]
        # log6(current^-1 target) is a 6-D body displacement taking the hand
        # from its current pose to the matching cube hook.
        errors.append(pin.log6(current.inverse() * target).vector)
        jacobians.append(
            pin.getFrameJacobian(robot.model, robot.data, frame_id, pin.ReferenceFrame.LOCAL)
        )
    return np.concatenate(errors), np.vstack(jacobians)


def _damped_least_squares(jacobian, error, damping):
    """Solve J dq = error robustly near a singular arm configuration."""
    jj_t = jacobian @ jacobian.T
    return jacobian.T @ np.linalg.solve(jj_t + damping * np.eye(jj_t.shape[0]), error)


def _grasp_residual(robot, cube, q, cube_placement):
    """Measure the full two-hand SE(3) error for a candidate grasp."""
    targets = (
        (LEFT_HAND, _cube_hook_target(cube, cube_placement, LEFT_HOOK)),
        (RIGHT_HAND, _cube_hook_target(cube, cube_placement, RIGHT_HOOK)),
    )
    error, _ = _pose_error_and_jacobian(robot, q, targets)
    return float(np.linalg.norm(error))


def display_grasp(robot, cube, q, viz, cube_placement=None):
    """Draw the robot, carried cube and frame markers in one consistent pose.

    When no explicit cube pose is supplied, infer it from the left hand:
    world_T_cube = world_T_hand * inverse(cube_T_hook). This is geometric
    playback of an already planned grasp, not a physical Bullet simulation.
    """
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


def _solve_grasp(robot, q_start, targets, posture, max_iterations, tolerance):
    """Run redundant, two-hand IK from one seed and return ``(q, success)``."""
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
        # The null space leaves the hand task unchanged to first order.  It
        # biases the redundant torso/arms back to a comfortable configuration.
        null_space = np.eye(robot.model.nv) - np.linalg.pinv(jacobian) @ jacobian
        posture_step = null_space @ pin.difference(robot.model, q, posture)
        velocity = primary_step + 0.08 * posture_step
        # A bounded integration step prevents a single poor linearisation from
        # jumping through joint limits.
        scale = max(1.0, np.linalg.norm(velocity) / 0.20)
        q = pin.integrate(robot.model, q, velocity / scale)
        q = projecttojointlimits(robot, q)

    error, _ = _pose_error_and_jacobian(robot, q, targets)
    return q, np.linalg.norm(error) < tolerance


def _restart_postures(robot, qcurrent, count=12):
    """Yield deterministic, legal posture references for IK restarts."""
    qcurrent = projecttojointlimits(robot, qcurrent)
    yield qcurrent
    yield projecttojointlimits(robot, robot.q0.copy())

    rng = np.random.default_rng(2026)
    lower, upper = robot.model.lowerPositionLimit, robot.model.upperPositionLimit
    for _ in range(count):
        # Mild perturbations retain a human-like folded-arm posture much better
        # than sampling the full joint box.
        candidate = robot.q0 + rng.normal(0.0, 0.35, size=robot.model.nq)
        yield np.minimum(np.maximum(candidate, lower), upper)


def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    """Return ``(q, success)`` for a collision-free grasp at ``cubetarget``.

    Both end effectors are constrained in position *and* orientation to their
    corresponding cube hooks.  No fixed joint values or hard-coded cube poses
    are used, so this also supports a perturbed initial/target cube placement.
    """
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
        # Re-evaluate the residual after the final integration and limit
        # projection.  This guards against accepting a numerically converged
        # step whose projection moved it outside the requested grasp tolerance.
        if (converged and
                _grasp_residual(robot, cube, q, cubetarget) < 5 * EPSILON and
                not jointlimitsviolated(robot, q) and not collision(robot, q)):
            left = _frame_placement(robot, q, LEFT_HAND)
            right = _frame_placement(robot, q, RIGHT_HAND)
            remember_hand_relation(left, right)
            if viz is not None:
                viz.display(q)
            return q, True

    # Returning the caller's valid seed makes failure explicit without
    # accidentally commanding an arbitrary configuration in the simulator.
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
