# Advanced Robotics Coursework 2

The Nextage robot grasps a cube with both hands, plans a collision-checked path
around the obstacle, and carries it in PyBullet. The optional `bonus.py`
experiment throws the cube towards a random target.

## Setup

On DICE or another machine with Conda installed, from the project folder:

```bash
conda env create -f environment.yml
conda activate aro2026
```

The course environment includes Python 3.11, Pinocchio, Meshcat, PyBullet,
SciPy and quadprog. Run from this project directory:

```bash
python inverse_geometry.py  # initial and goal grasps in Meshcat
python path.py              # grasp-preserving path in Meshcat
python control.py           # physical carry in a PyBullet window
python bonus.py             # optional throwing experiment
```

For a non-visual physics check, use `python control.py --headless` or
`python bonus.py --headless`. Start and goal configurations and a path can be
edited at the bottom of `control.py`.

## Methods and limits

Inverse geometry uses damped Jacobian least squares for both hands, a posture
bias and collision/joint-limit checks. Path planning uses a projected RRT and
collision-checked shortcutting. Control uses inverse-dynamics torque tracking;
a quadratic programme fits a smooth joint trajectory to the planned path.
Collision and grasp checks along the trajectory use finite samples. The bonus
calculates a ballistic throw to a random target and releases the cube by
disabling robot–cube contacts in the simulator. Some target placements fail IK
or the landing check.

Local tests have passed headlessly on macOS; DICE has not been verified. A
PyBullet GUI run may lose its connection on macOS. A successful-run video is
required alongside this project for coursework submission.
