#!/usr/bin/env python3

from __future__ import annotations

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

    cube_pose: pin.SE3
    q: np.ndarray
    parent: int | None


def _interpolate_pose(start, end, alpha):
    displacement = pin.log6(start.inverse() * end).vector
    return start * pin.exp6(displacement * alpha)


def _cube_distance(first, second):
    translation = np.linalg.norm(first.translation - second.translation)
    rotation = np.linalg.norm(pin.log3(first.rotation.T @ second.rotation))
    return translation + 0.10 * rotation

#move the cube from start to end in small steps, check if it can be done without collision or limit violation
def _project_segment(robot, cube, q_start, cube_start, cube_end, step=0.025):
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

#random cube position sampling for rrt, within bounds
def _sample_cube_pose(rng, start, goal):
    lo = np.minimum(start.translation, goal.translation) - np.array([0.04, 0.05, 0.0])
    hi = np.maximum(start.translation, goal.translation) + np.array([0.04, 0.05, 0.0])
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
        rng.uniform(min(start.translation[2], goal.translation[2]) + 0.16,
                    max(start.translation[2], goal.translation[2]) + 0.20),
    ])
    rotation = _interpolate_pose(start, goal, rng.uniform()).rotation
    return pin.SE3(rotation, translation)

# make sure the path ends at the goal
def _finish_at_goal(robot, cube, path, qgoal, goal_pose):
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

    pin.framesForwardKinematics(robot.model, robot.data, qgoal)
    if jointlimitsviolated(robot, qgoal) or collision(robot, qgoal):
        return None
    if any(np.linalg.norm(pin.log6(robot.data.oMf[fid].inverse() * target).vector)
           > 5 * EPSILON for fid, target in targets):
        return None
    tail[-1] = qgoal.copy()
    return path + tail


def projectpath(robot, cube, q_start, cube_start, cube_end, step=0.025):
    if step <= 0:
        raise ValueError("step less than zero")
    q_start = np.asarray(q_start, dtype=float)
    if q_start.ndim != 1 or not np.all(np.isfinite(q_start)):
        raise ValueError("q_start error")
    return _project_segment(robot, cube, q_start, cube_start, cube_end, step)

# smooth the path by removing detours, but only if the shortcut is valid
def smoothpath(robot, cube, nodes, chain, rng=None, attempts=80):
    if not chain:
        raise ValueError("chain has no nodes")
    if attempts < 0:
        raise ValueError("attempts less than zero")
    if rng is None:
        rng = np.random.default_rng(15)
    smoothed = _shortcut_chain(robot, cube, nodes, chain, rng, attempts)
    projected = _project_chain(robot, cube, nodes, smoothed)
    if projected is None:
        raise RuntimeError("smoothed chain error")
    return projected


def _trace_to_root(nodes, node_index):
    indices = []
    while node_index is not None:
        indices.append(node_index)
        node_index = nodes[node_index].parent
    return list(reversed(indices))

#straightens the route
def _shortcut_chain(robot, cube, nodes, chain, rng, attempts=80):
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

#check if the path is valid
def computepath(qinit, qgoal, cubeplacementq0, cubeplacementqgoal, robot=None, cube=None):
    if robot is None or cube is None:
        from setup_pinocchio import setuppinocchio
        robot, _, _, cube = setuppinocchio()

    qinit, qgoal = np.asarray(qinit, dtype=float), np.asarray(qgoal, dtype=float)
    setcubeplacement(robot, cube, cubeplacementq0)
    if qinit.ndim != 1 or qgoal.ndim != 1 or qinit.shape != qgoal.shape:
        raise ValueError("qinit and qgoal size mismatch")
    if not np.all(np.isfinite(qinit)) or not np.all(np.isfinite(qgoal)):
        raise ValueError("qinit and qgoal must be finite")
    if jointlimitsviolated(robot, qinit) or collision(robot, qinit):
        raise ValueError("qinit is invalid")
    setcubeplacement(robot, cube, cubeplacementqgoal)
    if jointlimitsviolated(robot, qgoal) or collision(robot, qgoal):
        raise ValueError("qgoal is invalid")

    nodes = [_Node(cubeplacementq0, qinit, None)]
    rng = np.random.default_rng(15)  
    max_iterations, step_length = 500, 0.075

    for iteration in range(max_iterations):
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
        except RuntimeError:  
            continue
        final_edge, final_ok = _project_segment(
            robot, cube, result[-1], nodes[chain[-1]].cube_pose, cubeplacementqgoal
        )
        if not final_ok:
            continue
        result.extend(final_edge)
        completed = _finish_at_goal(robot, cube, result, qgoal, cubeplacementqgoal)
        if completed is not None:
            return completed

    raise RuntimeError("RRT could not find a valid path within 500 tries")


def displaypath(robot, path, dt, viz, cube=None):
    for q in path:
        if cube is None:
            viz.display(q)
        else:
            display_grasp(robot, cube, q, viz)
        time.sleep(dt)


if __name__ == "__main__":
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
    from inverse_geometry import keep_viewer_open, open_viewer, setup_viewer

    robot, cube, viz = setup_viewer()
    q0, initial_ok = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT, viz)
    qe, final_ok = computeqgrasppose(robot, q0, cube, CUBE_PLACEMENT_TARGET, viz)
    if not (initial_ok and final_ok):
        raise RuntimeError("Grasping configuration failed")
    path = computepath(q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, robot, cube)
    print(f"planned {len(path)} grasping configurations")
    open_viewer(viz)
    displaypath(robot, path, dt=0.05, viz=viz, cube=cube)
    keep_viewer_open()
