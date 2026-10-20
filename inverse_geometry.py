#!/usr/bin/env python3
"""Collision-checked bimanual inverse geometry for Coursework 2."""

import numpy as np
import pinocchio as pin

from config import EPSILON, LEFT_HAND, LEFT_HOOK, RIGHT_HAND, RIGHT_HOOK
from tools import collision, projecttojointlimits, setcubeplacement
from solution import remember_hand_relation


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
    q = projecttojointlimits(robot, np.asarray(q_start, dtype=float).copy())
    posture = projecttojointlimits(robot, np.asarray(posture, dtype=float))
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
    setcubeplacement(robot, cube, cubetarget)
    targets = (
        (LEFT_HAND, _cube_hook_target(cube, cubetarget, LEFT_HOOK)),
        (RIGHT_HAND, _cube_hook_target(cube, cubetarget, RIGHT_HOOK)),
    )

    for posture in _restart_postures(robot, qcurrent):
        q, converged = _solve_grasp(
            robot, posture, targets, posture, max_iterations=250, tolerance=5 * EPSILON
        )
        if converged and not collision(robot, q):
            left = _frame_placement(robot, q, LEFT_HAND)
            right = _frame_placement(robot, q, RIGHT_HAND)
            remember_hand_relation(left, right)
            if viz is not None:
                viz.display(q)
            return q, True

    # Returning the caller's valid seed makes failure explicit without
    # accidentally commanding an arbitrary configuration in the simulator.
    return np.asarray(qcurrent, dtype=float).copy(), False


if __name__ == "__main__":
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
    from viewer import setupwithmeshcat

    robot, cube, viz = setupwithmeshcat()
    q0, initial_ok = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT, viz)
    qe, final_ok = computeqgrasppose(robot, q0, cube, CUBE_PLACEMENT_TARGET, viz)
    print(f"initial grasp: {initial_ok}; target grasp: {final_ok}")
    if initial_ok:
        display_grasp(robot, cube, q0, viz, CUBE_PLACEMENT)
    
    
    
