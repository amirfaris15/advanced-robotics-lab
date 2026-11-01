#!/usr/bin/env python3
"""Projected RRT planner for a rigid bimanual grasp."""

from dataclasses import dataclass
import time

import numpy as np
import pinocchio as pin

from inverse_geometry import computeqgrasppose, display_grasp, _cube_hook_target
from tools import collision, setcubeplacement
from tools import jointlimitsviolated
from config import LEFT_HAND, RIGHT_HAND, LEFT_HOOK, RIGHT_HOOK, EPSILON


@dataclass
class _Node:
    """One RRT node: its cube pose, grasping robot pose, and parent index."""

    cube_pose: pin.SE3
    q: np.ndarray
    parent: int | None


def _interpolate_pose(start, end, alpha):
    """Interpolate a full SE(3) cube pose rather than only its translation."""
    displacement = pin.log6(start.inverse() * end).vector
    return start * pin.exp6(displacement * alpha)


def _cube_distance(first, second):
    """Use metres plus a small rotation penalty as the RRT distance metric."""
    translation = np.linalg.norm(first.translation - second.translation)
    rotation = np.linalg.norm(pin.log3(first.rotation.T @ second.rotation))
    return translation + 0.10 * rotation


def _project_segment(robot, cube, q_start, cube_start, cube_end, step=0.025):
    """Project a cube-space edge onto valid grasping configurations.

    Each intermediate point is re-solved using the preceding solution as its
    seed.  Consequently every returned configuration satisfies the bimanual
    grasp constraint; ordinary joint-space linear interpolation would not.
    """
    distance = _cube_distance(cube_start, cube_end)
    samples = max(1, int(np.ceil(distance / step)))
    path, q_previous = [], q_start

    for index in range(1, samples + 1):
        target = _interpolate_pose(cube_start, cube_end, index / samples)
        q_next, success = computeqgrasppose(robot, q_previous, cube, target)
        if not success or collision(robot, q_next):
            return path, False
        path.append(q_next)
        q_previous = q_next
    return path, True


def _sample_cube_pose(rng, start, goal):
    """Sample the useful portion of the tabletop, with clearance around ends."""
    lo = np.minimum(start.translation, goal.translation) - np.array([0.20, 0.25, 0.0])
    hi = np.maximum(start.translation, goal.translation) + np.array([0.20, 0.25, 0.0])
    # Include clearance above the table: a height-only obstacle cannot be
    # traversed if every random sample is restricted to tabletop height.
    # Clamp each interval without assuming that a caller's target is inside
    # the default table bounds.  ``uniform(a, b)`` raises when a > b, which
    # made the planner fail before it had a chance to report an unreachable
    # target.  For an out-of-bounds target, use the target-side interval.
    def bounded_interval(lower, upper, scene_lower, scene_upper):
        a, b = max(scene_lower, lower), min(scene_upper, upper)
        if a <= b:
            return a, b
        midpoint = 0.5 * (lower + upper)
        return midpoint, midpoint

    x0, x1 = bounded_interval(lo[0], hi[0], 0.18, 0.65)
    y0, y1 = bounded_interval(lo[1], hi[1], -0.55, 0.45)
    translation = np.array([
        rng.uniform(x0, x1),
        rng.uniform(y0, y1),
        rng.uniform(min(start.translation[2], goal.translation[2]),
                    max(start.translation[2], goal.translation[2]) + 0.20),
    ])
    rotation = _interpolate_pose(start, goal, rng.uniform()).rotation
    return pin.SE3(rotation, translation)


def _finish_at_goal(robot, cube, path, qgoal, goal_pose):
    """Connect the IK branch to the requested exact endpoint, or reject it.

    Equivalent grasps can have different torso/elbow postures. For a nearby
    endpoint, explicitly check a finely sampled connecting segment for both
    hook alignment and collision; never silently return another configuration.
    """
    if not path:
        return None
    qgoal = np.asarray(qgoal, dtype=float)
    if qgoal.ndim != 1 or not np.all(np.isfinite(qgoal)):
        return None
    setcubeplacement(robot, cube, goal_pose)
    count = max(1, int(np.ceil(np.max(np.abs(qgoal - path[-1])) / 0.005)))
    tail = []
    targets = [(robot.model.getFrameId(hand), _cube_hook_target(cube, goal_pose, hook))
               for hand, hook in ((LEFT_HAND, LEFT_HOOK), (RIGHT_HAND, RIGHT_HOOK))]
    for alpha in np.linspace(0., 1., count + 1)[1:]:
        q = (1 - alpha) * path[-1] + alpha * qgoal
        pin.framesForwardKinematics(robot.model, robot.data, q)
        if jointlimitsviolated(robot, q) or collision(robot, q):
            return None
        if any(np.linalg.norm(pin.log6(robot.data.oMf[fid].inverse() * target).vector)
               > 5 * EPSILON for fid, target in targets):
            return None
        tail.append(q)
    # Validate the exact endpoint as well; assigning it after the loop would
    # otherwise allow a caller to receive a colliding or out-of-limit qgoal.
    pin.framesForwardKinematics(robot.model, robot.data, qgoal)
    if jointlimitsviolated(robot, qgoal) or collision(robot, qgoal):
        return None
    if any(np.linalg.norm(pin.log6(robot.data.oMf[fid].inverse() * target).vector)
           > 5 * EPSILON for fid, target in targets):
        return None
    tail[-1] = qgoal.copy()
    return path + tail


def projectpath(robot, cube, q_start, cube_start, cube_end, step=0.025):
    """Project a cube-space edge into valid bimanual robot configurations.

    This is the coursework path-projection primitive made public so it can be
    tested independently of the RRT.  The returned list excludes ``q_start``
    and contains only successfully projected samples.
    """
    if step <= 0:
        raise ValueError("step must be positive")
    q_start = np.asarray(q_start, dtype=float)
    if q_start.ndim != 1 or not np.all(np.isfinite(q_start)):
        raise ValueError("q_start must be a finite one-dimensional configuration")
    return _project_segment(robot, cube, q_start, cube_start, cube_end, step)


def smoothpath(robot, cube, nodes, chain, rng=None, attempts=80):
    """Shortcut and re-project an RRT chain, preserving the grasp constraint.

    ``nodes`` is the planner's node sequence and ``chain`` contains node
    indices in start-to-goal order.  Smoothing is accepted only when the full
    replacement edge passes IK and collision checks.
    """
    if not chain:
        raise ValueError("chain must contain at least one node")
    if attempts < 0:
        raise ValueError("attempts must be non-negative")
    if rng is None:
        rng = np.random.default_rng(2026)
    smoothed = _shortcut_chain(robot, cube, nodes, chain, rng, attempts)
    projected = _project_chain(robot, cube, nodes, smoothed)
    if projected is None:
        raise RuntimeError("smoothed chain could not be re-projected")
    return projected


def _trace_to_root(nodes, node_index):
    """Recover RRT node indices in start-to-goal order."""
    indices = []
    while node_index is not None:
        indices.append(node_index)
        node_index = nodes[node_index].parent
    return list(reversed(indices))


def _shortcut_chain(robot, cube, nodes, chain, rng, attempts=80):
    """Remove RRT waypoints when a projected collision-free shortcut exists.

    This is the usual coursework-1 path-smoothing post-process, applied in
    cube pose space.  A shortcut is accepted only after the complete edge has
    been projected through two-hand IK, so smoothing cannot discard the
    grasping constraint or the collision checks performed by the planner.
    """
    chain = list(chain)
    if len(chain) < 3:
        return chain

    for _ in range(attempts):
        first = int(rng.integers(0, len(chain) - 2))
        last = int(rng.integers(first + 2, len(chain)))
        start, end = nodes[chain[first]], nodes[chain[last]]
        _, valid = _project_segment(
            robot, cube, start.q, start.cube_pose, end.cube_pose
        )
        if valid:
            chain[first + 1:last] = []
            if len(chain) < 3:
                break
    return chain


def _project_chain(robot, cube, nodes, chain):
    """Rebuild a smoothed chain while propagating each projected IK seed."""
    result = [nodes[chain[0]].q]
    q_previous = nodes[chain[0]].q
    for first_index, second_index in zip(chain[:-1], chain[1:]):
        edge, valid = _project_segment(
            robot,
            cube,
            q_previous,
            nodes[first_index].cube_pose,
            nodes[second_index].cube_pose,
        )
        if not valid:
            return None
        result.extend(edge)
        q_previous = edge[-1]
    return result


def _try_clearance_route(robot, cube, qinit, cube_start, cube_goal):
    """Try a simple, collision-aware lift-carry-lower motion first.

    In the supplied scene the fixed obstacle is a low bar. Sampling only at
    tabletop height makes RRT spend most of its time proposing colliding edges;
    lifting the cube above the bar is both natural and a useful deterministic
    connection attempt. Different clearances are tried because an excessively
    high lift can exceed the robot's workspace.
    """
    for clearance in (0.18, 0.15, 0.20):
        start_above = pin.SE3(
            cube_start.rotation, cube_start.translation + np.array([0.0, 0.0, clearance])
        )
        goal_above = pin.SE3(
            cube_goal.rotation, cube_goal.translation + np.array([0.0, 0.0, clearance])
        )
        lift, lift_ok = _project_segment(robot, cube, qinit, cube_start, start_above)
        if not lift_ok:
            continue
        carry, carry_ok = _project_segment(robot, cube, lift[-1], start_above, goal_above)
        if not carry_ok:
            continue
        lower, lower_ok = _project_segment(robot, cube, carry[-1], goal_above, cube_goal)
        if lower_ok:
            return [qinit] + lift + carry + lower
    return None


def computepath(qinit, qgoal, cubeplacementq0, cubeplacementqgoal, robot=None, cube=None):
    """Return a collision-free, grasp-preserving path from ``qinit`` to ``qgoal``.

    The four required positional arguments preserve the supplied coursework
    API.  ``robot`` and ``cube`` are optional dependency-injection arguments so
    callers that already have a scene do not need to create a second one.
    """
    if robot is None or cube is None:
        from setup_pinocchio import setuppinocchio
        robot, _, _, cube = setuppinocchio()

    qinit, qgoal = np.asarray(qinit, dtype=float), np.asarray(qgoal, dtype=float)
    setcubeplacement(robot, cube, cubeplacementq0)
    if qinit.ndim != 1 or qgoal.ndim != 1 or qinit.shape != qgoal.shape:
        raise ValueError("qinit and qgoal must be same-size configuration vectors")
    if not np.all(np.isfinite(qinit)) or not np.all(np.isfinite(qgoal)):
        raise ValueError("qinit and qgoal must be finite")
    if jointlimitsviolated(robot, qinit) or collision(robot, qinit):
        raise ValueError("qinit is invalid; compute a valid grasp before planning")
    setcubeplacement(robot, cube, cubeplacementqgoal)
    if jointlimitsviolated(robot, qgoal) or collision(robot, qgoal):
        raise ValueError("qgoal is invalid; compute a valid grasp before planning")

    # Always test the cheap direct edge first.  It is common for nearby,
    # perturbed placements and avoids RRT randomness when it is already valid.
    direct_path, direct_ok = _project_segment(
        robot, cube, qinit, cubeplacementq0, cubeplacementqgoal
    )
    if direct_ok:
        completed = _finish_at_goal(robot, cube, [qinit] + direct_path, qgoal, cubeplacementqgoal)
        if completed is not None:
            return completed

    # The coursework obstacle can be cleared by carrying the cube over it.
    # This is attempted before random exploration so the standard scene is
    # repeatable and fast, while RRT remains available for perturbed layouts.
    clearance_path = _try_clearance_route(
        robot, cube, qinit, cubeplacementq0, cubeplacementqgoal
    )
    if clearance_path is not None:
        completed = _finish_at_goal(robot, cube, clearance_path, qgoal, cubeplacementqgoal)
        if completed is not None:
            return completed

    nodes = [_Node(cubeplacementq0, qinit, None)]
    rng = np.random.default_rng(2026)  # reproducible runs are easier to assess
    max_iterations, step_length = 500, 0.075

    for iteration in range(max_iterations):
        # Goal bias accelerates connection around the obstacle. This bounded
        # sampler is heuristic; it does not guarantee a solution in all scenes.
        sample = cubeplacementqgoal if iteration % 8 == 0 else _sample_cube_pose(
            rng, cubeplacementq0, cubeplacementqgoal
        )
        nearest_index = min(
            range(len(nodes)), key=lambda i: _cube_distance(nodes[i].cube_pose, sample)
        )
        nearest = nodes[nearest_index]
        distance = _cube_distance(nearest.cube_pose, sample)
        alpha = min(1.0, step_length / max(distance, 1e-12))
        candidate_pose = _interpolate_pose(nearest.cube_pose, sample, alpha)
        edge, edge_ok = _project_segment(
            robot, cube, nearest.q, nearest.cube_pose, candidate_pose
        )
        if not edge_ok:
            continue

        nodes.append(_Node(candidate_pose, edge[-1], nearest_index))
        newest_index = len(nodes) - 1
        if _cube_distance(candidate_pose, cubeplacementqgoal) > step_length:
            continue

        chain = _trace_to_root(nodes, newest_index)
        try:
            result = smoothpath(robot, cube, nodes, chain, rng)
        except RuntimeError:  # Defensive: protects against mutable-scene mistakes.
            continue

        # Recompute the goal connection from the smoothed chain's actual IK
        # endpoint.  A projected shortcut can choose a different redundant
        # arm posture at the same cube pose.
        final_edge, final_ok = _project_segment(
            robot, cube, result[-1], nodes[chain[-1]].cube_pose, cubeplacementqgoal
        )
        if not final_ok:
            continue
        result.extend(final_edge)
        completed = _finish_at_goal(robot, cube, result, qgoal, cubeplacementqgoal)
        if completed is not None:
            return completed

    raise RuntimeError("RRT could not find a valid path within 500 expansions")


def displaypath(robot, path, dt, viz, cube=None):
    """Replay a validated path; pass cube to animate its rigid grasp too.

    The optional argument preserves the original robot-only display API.
    Each frame recomputes the cube pose, so replay starts correctly even when
    planning or a previous animation left the scene at the goal.
    """
    for q in path:
        if cube is None:
            viz.display(q)
        else:
            display_grasp(robot, cube, q, viz)
        time.sleep(dt)


if __name__ == "__main__":
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
    from viewer import keep_viewer_open, open_viewer, setupwithmeshcat

    robot, cube, viz = setupwithmeshcat()
    q0, initial_ok = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT, viz)
    qe, final_ok = computeqgrasppose(robot, q0, cube, CUBE_PLACEMENT_TARGET, viz)
    if not (initial_ok and final_ok):
        raise RuntimeError("Cannot plan because one grasping configuration failed")
    path = computepath(q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, robot, cube)
    print(f"planned {len(path)} grasping configurations")
    open_viewer(viz)
    displaypath(robot, path, dt=0.05, viz=viz, cube=cube)
    keep_viewer_open()
