# Advanced Robotics Coursework 2

## DICE setup

Extract the ZIP, then enter the extracted folder containing `environment.yml` (replace the path below with its actual location):

```bash
cd /path/to/extracted/project
```

If Conda is not initialized in Bash, do this once:

```bash
/opt/conda/bin/conda init bash
source ~/.bashrc
```

Create and activate the project environment:

```bash
conda env create -f environment.yml
conda activate aro2026
```

Activate `aro2026` again in each new terminal. If it already exists, update it with `conda env update -f environment.yml`.

## Coursework files

The first three files implement the core grasping, planning, and control tasks described in the coursework brief. `bonus.py` is an optional throwing experiment.

- **`inverse_geometry.py`** — Uses damped Jacobian inverse geometry to align both hands with the cube hooks at the start and target poses. A posture bias favours comfortable configurations, which are checked for joint limits and collisions. Run: `python inverse_geometry.py`
- **`path.py`** — Implements `computepath` using a projected RRT. It searches for a path around the obstacle, projecting motions to preserve the two-hand grasp and checking for collisions. Shortcutting reduces unnecessary movement before Meshcat displays the result. Run: `python path.py`
- **`control.py`** — Fits a smooth, timed trajectory to the planned path using a quadratic programme, then uses inverse-dynamics torque control to carry the cube in PyBullet. It reports the final target error, contact time and other simulation results. Run: `python control.py`
- **`bonus.py`** — Calculates a ballistic throw toward a randomly chosen target and releases the cube in PyBullet. It reports the landing error; some targets may fail the grasp or landing checks. Run: `python bonus.py`

For Meshcat displays from the first two programs, run `meshcat-server` in a second terminal with `aro2026` activated. To run physics without a window, add `--headless` to `control.py` or `bonus.py`.
