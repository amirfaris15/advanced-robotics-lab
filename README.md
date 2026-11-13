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

Inverse geometry: Jocobian IK on both hands 

Motion planning: RRT over cube placements. EAch edge is projected by re-solving IK every 2.5cm, so every configuration grasps the cube. Shortcut smoothing, then goal 
