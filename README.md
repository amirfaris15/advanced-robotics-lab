# Advanced Robotics Coursework 2

## DICE setup

Extract the ZIP and open Bash in the folder containing `environment.yml`. The project uses Python 3.11, Pinocchio, NumPy, SciPy, Meshcat, PyBullet and quadprog.

If `conda` is unavailable, use `/opt/conda/bin/conda` in the first command.

```bash
eval "$(conda shell.bash hook)"
conda env create -f environment.yml
conda activate aro2026
python -m pip install -r requirements.txt
```

If `aro2026` already exists, use `conda env update -f environment.yml` instead of `create`. Activate it in each new terminal.

## Coursework files

- `python inverse_geometry.py` finds the initial and final grasps with damped Jacobian inverse kinematics. Both hands align with the cube hooks while respecting joint limits and avoiding collisions.
- `python path.py` uses a projected RRT to carry the cube around the obstacle. It solves for a grasp along each edge and checks shortcuts before keeping them. Meshcat plays back the resulting path.
- `python control.py` fits a smooth joint trajectory with a quadratic programme and tracks it with inverse dynamics and grip torques in PyBullet. To try a different configuration or path, edit `q0`, `qe` or `path` at the bottom of the file.
- `python bonus.py` tries an upward throw to a randomly chosen target. It releases the cube by disabling contact with the hands in PyBullet. Some targets fail inverse kinematics or miss the landing tolerance.

For Meshcat, open the address printed in the terminal. If you start `meshcat-server` separately, run it in the same environment. Use `python control.py --headless` for a terminal-only check and look for `"success": true` in its output. Use `--headless` with `bonus.py` for the same display-free mode.

The submission ZIP includes `successful_run.mp4`, `planned_path.mp4` and `bonus_throw.mp4`.
