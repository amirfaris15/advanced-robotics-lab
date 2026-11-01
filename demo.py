"""Run and measure the complete physical carry, with optional MP4 recording."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, DT
from setup_pinocchio import setuppinocchio
from inverse_geometry import computeqgrasppose
from path import computepath
from control import PiecewiseQuinticTrajectory, controllaw


def run_demo(gui=False, start=None, goal=None, path=None, record=None):
    """Plan, initialise once, then use torque/contact physics until settled.

    start/goal are cube SE3 poses; an optional supplied path is used unchanged.
    Recording uses the system ffmpeg executable only; it is not needed to run.
    """
    if gui and sys.platform == 'darwin' and 'ipykernel' in sys.modules:
        raise RuntimeError(
            'PyBullet GUI can freeze inside a macOS Jupyter kernel. '
            'Run `python control.py` from Terminal to watch the physical carry; '
            'use run_demo(gui=False) inside Jupyter.'
        )
    # Physics is optional for geometry-only tools such as verify.py
    # --geometry-only.  Delay this import so IK/RRT checks do not require
    # PyBullet to be importable.
    from simulation import Simulation

    start = CUBE_PLACEMENT if start is None else start
    goal = CUBE_PLACEMENT_TARGET if goal is None else goal
    robot, _, _, cube = setuppinocchio()
    q0, ok0 = computeqgrasppose(robot, robot.q0, cube, start)
    qe, oke = computeqgrasppose(robot, q0, cube, goal)
    if not (ok0 and oke):
        raise RuntimeError('Cannot run: collision-free endpoint IK failed')
    if path is None:
        path = computepath(q0, qe, start, goal, robot, cube)
    trajectory = PiecewiseQuinticTrajectory(path)
    sim = Simulation(robot, gui=gui, cube_placement=start)
    encoder = None
    try:
        if record:
            ffmpeg = shutil.which('ffmpeg')
            if not ffmpeg:
                raise RuntimeError('Recording requires ffmpeg; omit --record to run without it')
            target = Path(record)
            if target.exists():
                raise FileExistsError(f'Refusing to overwrite video: {target}')
            target.parent.mkdir(parents=True, exist_ok=True)
            encoder = subprocess.Popen([ffmpeg, '-v', 'error', '-f', 'rawvideo',
                '-pix_fmt', 'rgb24', '-s', '640x480', '-r', '25', '-i', '-',
                '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(target)],
                stdin=subprocess.PIPE)
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
        view = sim.p.computeViewMatrixFromYawPitchRoll([0.25, 0, .95], 1.65, 90, -27, 0, 2)
        projection = sim.p.computeProjectionMatrixFOV(50, 640/480, .05, 10)
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
            # The fixed pedestal may touch the floor; only tabletop/obstacle
            # contacts and cube/obstacle impacts are unexpected in this task.
            forbidden_contacts += int(any(sim.p.getContactPoints(a, b) for a, b in (
                (sim.robot, sim.tableId), (sim.robot, sim.obstacleId), (sim.cubeId, sim.obstacleId))))
            if encoder is not None and k % 40 == 0:
                rgba = sim.p.getCameraImage(640, 480, view, projection,
                                           renderer=sim.p.ER_TINY_RENDERER)[2]
                encoder.stdin.write(np.asarray(rgba, dtype=np.uint8)[:, :, :3].tobytes())
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
        result['success'] = bool(target_error < .02 and both_contact_time > 1
                                 and forbidden_contacts == 0 and joint_limit_samples == 0)
        return result
    finally:
        sim.close()
        if encoder is not None:
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError('Video encoder failed')


def main(path=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--record', help='New .mp4 path; optional ffmpeg is required')
    args = parser.parse_args()
    result = run_demo(gui=not args.headless, record=args.record, path=path)
    print(json.dumps(result, indent=2))
    if not result['success']:
        raise SystemExit('Physical carry did not meet the verification thresholds')


if __name__ == '__main__':
    main()
