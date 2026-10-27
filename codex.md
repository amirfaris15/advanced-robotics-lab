# Coding notes for the CW2 lab

## Architecture

- `inverse_geometry.py` solves the two-hand grasp at a requested cube SE(3) pose. It must respect both hook frames, joint limits and collisions; do not hard-code the coursework's default target.
- `path.py` plans and validates the grasp-constrained geometric carry. `displaypath` is a viewer playback, not the physical simulation.
- `control.py`, `simulation.py` and `demo.py` execute the planned carry in PyBullet. The headless demo is the reliable automated verification path on macOS.
- `viewer.py` probes the configured Meshcat ZMQ endpoint and starts a temporary server when none responds. Meshcat's temporary server is killed at Python interpreter exit, so interactive terminal entrypoints must open the viewer and keep their process alive until the user presses Enter.

## Workflow and constraints

- Use the `aro2026` Python 3.11 environment and run commands from the lab root.
- On the project owner's Mac, `conda` and `python` may be absent from the shell PATH. The existing environment can be called directly as `../aro2026/bin/python` from the lab root.
- Keep notebook and direct terminal entrypoints working. Restart a notebook kernel after Python source changes because imports are cached.
- In the notebook, use Meshcat for geometry and headless PyBullet for physics. Run `python control.py` from Terminal for the visible PyBullet carry on macOS.
- Preserve the original course files and avoid writing generated outputs or videos into version control.
- After behavior or setup changes, update this file and `README.md` to match the code.

## Verification

- Run `python verify.py --geometry-only` for IK, path and trajectory checks.
- Run `python verify.py` for the physical carry checks when touching physics or control.
- For viewer changes, run `python inverse_geometry.py` and `python path.py` in an interactive terminal, check that each live Meshcat URL contains the robot scene, then press Enter to close the temporary server.
