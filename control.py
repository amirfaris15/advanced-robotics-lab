#!/usr/bin/env python3
"""Time parametrisation and inverse-dynamics torque control for CW2."""

import numpy as np
import pinocchio as pin

from config import LEFT_HAND, LEFT_HOOK, RIGHT_HAND, RIGHT_HOOK


Kp = 300.0
Kv = 2.0 * np.sqrt(Kp)
SQUEEZE_FORCE = 8.0  # N per hand, maintained through physical contact
MAX_TORQUE = 180.0


class PiecewiseQuinticTrajectory:
    """A C2 trajectory that follows every configuration in a planned path.

    A fifth-order smooth-step has zero velocity and acceleration at both ends.
    Joining one curve per path edge therefore creates a C2 trajectory with a
    deliberate pause at each RRT waypoint, which is conservative but robust in
    the contact simulator.
    """

    def __init__(self, path, max_joint_speed=0.40):
        if len(path) < 1:
            raise ValueError("a trajectory needs at least one configuration")
        self.path = [np.asarray(q, dtype=float).copy() for q in path]
        if any(q.ndim != 1 or not np.all(np.isfinite(q)) for q in self.path):
            raise ValueError("trajectory configurations must be finite vectors")
        if any(q.shape != self.path[0].shape for q in self.path[1:]):
            raise ValueError("all trajectory configurations must have the same size")
        changes = [np.max(np.abs(b - a)) for a, b in zip(self.path[:-1], self.path[1:])]
        if max_joint_speed <= 0:
            raise ValueError("max_joint_speed must be positive")
        # Quintic smoothstep reaches a peak slope of 1.875, not 1.
        self.durations = np.maximum(0.45, 1.875 * np.asarray(changes) / max_joint_speed)
        self.breakpoints = np.concatenate(([0.0], np.cumsum(self.durations)))
        self.duration = float(self.breakpoints[-1])

    def _segment(self, t):
        if not self.durations.size:
            return 0, 0.0
        t = float(np.clip(t, 0.0, self.duration))
        index = min(np.searchsorted(self.breakpoints, t, side="right") - 1, len(self.durations) - 1)
        local_time = t - self.breakpoints[index]
        return index, np.clip(local_time / self.durations[index], 0.0, 1.0)

    def evaluate(self, t):
        """Return desired position, velocity and acceleration at time ``t``."""
        if not self.durations.size:
            zeros = np.zeros_like(self.path[0])
            return self.path[0].copy(), zeros, zeros
        index, s = self._segment(t)
        delta = self.path[index + 1] - self.path[index]
        duration = self.durations[index]
        blend = 10 * s**3 - 15 * s**4 + 6 * s**5
        blend_dot = (30 * s**2 - 60 * s**3 + 30 * s**4) / duration
        blend_ddot = (60 * s - 180 * s**2 + 120 * s**3) / duration**2
        return (
            self.path[index] + blend * delta,
            blend_dot * delta,
            blend_ddot * delta,
        )


def _reference_at(trajs, tcurrent):
    """Accept this solution's trajectory or the starter tuple of Bezier curves."""
    if hasattr(trajs, "evaluate"):
        return trajs.evaluate(tcurrent)
    return tuple(trajectory(tcurrent) for trajectory in trajs)


def _world_cube_hook(cube, hook_name, cube_placement):
    """Use the static URDF hook offset with the cube's simulated world pose."""
    pin.framesForwardKinematics(cube.model, cube.data, cube.q0)
    hook_in_cube = cube.data.oMf[cube.model.getFrameId(hook_name)]
    return cube_placement * hook_in_cube


def _grasp_wrench_torque(sim, robot, cube, q, vq):
    """Squeeze opposing faces and support weight through hand joint torques.

    A spring aimed at the exact hook gives zero force when aligned, so it
    cannot maintain a friction grasp. A small inward force supplies preload.
    Forces are applied ONLY to robot joints; the cube moves through contacts.
    """
    position, quaternion = sim.p.getBasePositionAndOrientation(sim.cubeId)
    cube_placement = pin.XYZQUATToSE3(np.asarray([*position, *quaternion]))
    mass = sum(sim.p.getDynamicsInfo(sim.cubeId, i)[0]
               for i in range(-1, sim.p.getNumJoints(sim.cubeId)))
    pin.computeJointJacobians(robot.model, robot.data, q)
    pin.framesForwardKinematics(robot.model, robot.data, q)

    total = np.zeros(robot.model.nv)
    for hand_name, sign in ((LEFT_HAND, -1), (RIGHT_HAND, 1)):
        frame_id = robot.model.getFrameId(hand_name)
        jacobian = pin.getFrameJacobian(
            robot.model, robot.data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        force = sign * SQUEEZE_FORCE * cube_placement.rotation[:, 1]
        force += np.array([0., 0., mass * 9.81 / 2])
        # This frame Jacobian uses linear rows followed by angular rows.
        total += jacobian.T @ np.concatenate((force, np.zeros(3)))
    return total


def controllaw(sim, robot, trajs, tcurrent, cube):
    """Run one computed-torque control step and advance Bullet by one tick."""
    q, vq = sim.getpybulletstate()
    q_desired, v_desired, a_desired = _reference_at(trajs, tcurrent)

    # Inverse dynamics converts the desired closed-loop acceleration into a
    # torque command.  nle includes gravity, Coriolis and centrifugal terms.
    closed_loop_acceleration = a_desired + Kp * (q_desired - q) + Kv * (v_desired - vq)
    torques = pin.rnea(robot.model, robot.data, q, vq, closed_loop_acceleration)
    torques += _grasp_wrench_torque(sim, robot, cube, q, vq)
    limits = np.minimum(MAX_TORQUE, robot.model.effortLimit)
    sim.step(np.clip(torques, -limits, limits).tolist())


if __name__ == '__main__':
    try:
        import pybullet  # noqa: F401
    except ModuleNotFoundError as error:
        if error.name == 'pybullet':
            raise SystemExit(
                "PyBullet is not installed in this Python environment. "
                "Create the course environment first with: "
                "conda env update -f environment.yml && conda activate aro2026; "
                "then run `python control.py` from this directory."
            ) from error
        raise

    # Keep this file as the single user-facing entry point.  The runner owns
    # scene creation, endpoint IK, path planning, simulation and CLI parsing;
    # duplicating those steps here made `python control.py` solve the task
    # twice and could leave the visual and physical scenes out of sync.
    from demo import main
    main()
    
    
    
