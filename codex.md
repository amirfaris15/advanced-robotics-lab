# Coursework implementation notes

The submission implementation lives in `inverse_geometry.py`, `path.py` and
`control.py`. The geometry solver finds hand-to-hook grasps, the projected RRT
in `path.py` plans a collision-checked geometric path, and `control.py` turns
that path into a trajectory and simulates the carry in PyBullet. Keep the course
interfaces and editable start, goal and path working. Do not assume this local
notes file is part of the coursework submission.

The default control trajectory is `OptimizedTrajectory`: a `quadprog` QP finds
shared waypoint velocities and accelerations for a C2 piecewise-quintic path.
It fits the input path and penalises acceleration, constrains sampled joint
positions and velocities, checks exact polynomial joint extrema, and stretches
time if necessary for speed bounds. The geometric grasp and collision check is
sampled; it is not a continuous collision guarantee. The previous quintic
interpolation remains available with `--trajectory quintic`.
The control runner does not include video recording or an ffmpeg dependency.

`bonus.py` is the throwing experiment from `origin/bonus-folder`; its
simulation import points to `control.Simulation` because this worktree has no
`simulation.py`, and its grasp-wrench call uses that controller's signature.
It samples target offsets uniformly within ±0.03 m in x and ±0.08 m in y of
`CUBE_PLACEMENT_TARGET`, uses a quintic approach, and can fail for particular
target samples. Test the bonus independently of the default QP carry; preserve
the three coursework-file submission rules.

After control changes, check both the default and a changed reachable goal in
headless PyBullet, check waypoint interpolation, joint limits, grasp/collision
samples and physical carry metrics, and exercise the graphical entry point where
the platform permits. `quadprog` is already a course dependency. Do not treat a
successful local run as evidence of a guaranteed mark or DICE compatibility.
