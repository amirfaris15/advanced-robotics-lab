# CW2: grasp, plan and carry

Use Python 3.11 and the course dependencies: `conda env update -f environment.yml`,
then `conda activate aro2026`. No extra Python packages are needed. Optional MP4
recording requires the system `ffmpeg` executable; playback does not.

From the lab root, run `python inverse_geometry.py`, `python path.py`, then
`python control.py`. Meshcat starts automatically if needed; control opens
Bullet and closes its connection on completion/error. `q0`, `qe` and `path` are
editable in the control entrypoint. Use `python control.py --headless` to run
the same physics without a window.

Inverse geometry uses damped two-hand SE(3) IK with postural bias, genuine seed
restarts, joint limits and collision checks. Planning tries a direct projected
edge, a lift-carry-lower route, then cube-space RRT with height/orientation
sampling. A checked final connection reaches the exact requested joint goal.
Quintic timing respects a peak joint-speed bound. Inverse-dynamics control adds
opposing hand squeeze and weight support; contact friction carries the cube.

The isolated Bullet scene anchors the robot base, uses authored inertias,
removes Bullet's invented masses on empty coordinate-marker links, and sets
joint/link damping to zero. Solid masses, collision geometry, friction and
gravity remain unchanged. No attachment constraints or cube teleportation
are used during motion.

Verify: `python verify.py`. Default and translated-goal carries
passed; broader robustness is limited. `successful_run.mp4` records the default
run. No trajectory-optimisation bonus is claimed.

Open `cw2_solution.ipynb` with the ARO Python 3.11 kernel; restart after edits.
Its Bullet cell defaults to headless execution. Missing imports indicate the
wrong environment. Unreachable grasps and exhausted planning budgets raise errors.
