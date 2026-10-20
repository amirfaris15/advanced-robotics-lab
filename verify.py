"""Reproducible geometry, trajectory and contact-physics regression checks."""

import argparse
import json
from pathlib import Path

import numpy as np
import pinocchio as pin

from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, LEFT_HAND, RIGHT_HAND, LEFT_HOOK, RIGHT_HOOK
from tools import collision, jointlimitsviolated, setcubeplacement
from setup_pinocchio import setuppinocchio
from inverse_geometry import computeqgrasppose, _cube_hook_target
from path import computepath
from control import PiecewiseQuinticTrajectory
from demo import run_demo


def check_geometry(start, goal):
    robot, _, _, cube = setuppinocchio()
    q0, ok0 = computeqgrasppose(robot, robot.q0, cube, start)
    qe, oke = computeqgrasppose(robot, q0, cube, goal)
    assert ok0 and oke, 'Endpoint IK failed'
    path = computepath(q0, qe, start, goal, robot, cube)
    assert isinstance(path, list) and np.array_equal(path[0], q0)
    assert np.array_equal(path[-1], qe), 'Planner did not return exact qgoal'
    max_error, samples = 0., 0
    left_id = robot.model.getFrameId(LEFT_HAND)
    right_id = robot.model.getFrameId(RIGHT_HAND)
    cube_left = _cube_hook_target(cube, pin.SE3.Identity(), LEFT_HOOK)
    cube_right = _cube_hook_target(cube, pin.SE3.Identity(), RIGHT_HOOK)
    for first, second in zip(path[:-1], path[1:]):
        for alpha in np.linspace(0, 1, 11):
            q = (1-alpha)*first + alpha*second
            pin.framesForwardKinematics(robot.model, robot.data, q)
            pose = robot.data.oMf[left_id] * cube_left.inverse()
            error = np.linalg.norm(pin.log6(robot.data.oMf[right_id].inverse() * pose * cube_right).vector)
            max_error = max(max_error, float(error))
            setcubeplacement(robot, cube, pose)
            assert error < .005, f'Grasp drift {error}'
            assert not collision(robot, q), 'Collision on dense path sample'
            assert not jointlimitsviolated(robot, q), 'Joint limit violation'
            samples += 1
    trajectory = PiecewiseQuinticTrajectory(path)
    for t in trajectory.breakpoints:
        _, v, a = trajectory.evaluate(t)
        assert np.max(np.abs(v)) < 1e-10 and np.max(np.abs(a)) < 1e-10
    for t in (trajectory.breakpoints[:-1]+trajectory.breakpoints[1:])/2:
        _, v, _ = trajectory.evaluate(t)
        assert np.max(np.abs(v)) <= .40+1e-10, 'Speed exceeds declared limit'
    return dict(waypoints=len(path), dense_samples=samples, max_grasp_residual=max_error,
                exact_goal=True, collision_free=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', help='New JSON report path')
    parser.add_argument('--geometry-only', action='store_true')
    args = parser.parse_args()
    cases = [('default', CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)]
    translated = CUBE_PLACEMENT_TARGET.copy()
    translated.translation += np.array([-.01, .01, 0.])
    cases.append(('translated_goal', CUBE_PLACEMENT, translated))
    report = {}
    for name, start, goal in cases:
        entry = dict(geometry=check_geometry(start, goal))
        if not args.geometry_only:
            entry['physics'] = run_demo(start=start, goal=goal)
            assert entry['physics']['success'], f'Physical carry failed: {name}'
        report[name] = entry
        print(name, json.dumps(entry), flush=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x') as stream:
            json.dump(report, stream, indent=2)


if __name__ == '__main__':
    main()
