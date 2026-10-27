# Advanced Robotics - Coursework 2

The CW2 implementation now lives directly in the lab root. The original starter
versions remain in Git history. Open [cw2_solution.ipynb](cw2_solution.ipynb) for
the commented Jupyter runner, or run the Python entrypoints below.

## Setup

Use Python 3.11 and the supplied course environment from this directory:

```sh
conda env update -f environment.yml
conda activate aro2026
python -m ipykernel install --user --name aro2026 --display-name "ARO CW2 (Python 3.11)"
jupyter notebook .
```

Select **ARO CW2 (Python 3.11)** in Jupyter. Restart the kernel after source
changes: old imported modules stay cached until restarted. No new Python
dependencies are required. Optional MP4 recording needs the system `ffmpeg`
executable; running the simulation does not. There are no migrations, database
or deployment steps; this is a local robotics project.

On this Mac, the existing course environment is in `../aro2026`. If `python`
or `conda` is not found in Terminal, run the scripts with its interpreter
directly from the lab folder, for example:

```sh
../aro2026/bin/python path.py
```

## Run and understand the code

| File | Purpose |
| --- | --- |
| `inverse_geometry.py` | Find collision-free starting/goal grasps using damped two-hand IK, posture bias and seed restarts. |
| `solution.py` | Store the relative hand transform for diagnostics. |
| `path.py` | Plan a projected direct, lift-carry-lower or RRT path; check the exact requested endpoint. |
| `control.py` | Quintic time parameterisation and inverse-dynamics tracking with inward squeeze and weight support. |
| `simulation.py` | Isolated Bullet connection and scene setup. |
| `demo.py` | Execute a supplied/planned path, collect metrics, record optional video and clean up. |
| `viewer.py` | Connect to Meshcat, or start it if no server responds. |
| `verify.py` | Geometry, trajectory and contact-physics regression checks. |
| `package_submission.py` | Build a clean archive with models, runnable root scripts, short README and video. |

```sh
python inverse_geometry.py
python path.py
python control.py
```

The first two open Meshcat in a browser when run from an interactive Terminal.
Keep the Terminal command running while viewing the robot; press Enter in that
Terminal to close its temporary viewer server. If you opened an older Meshcat
tab and see only a blue grid, use the URL printed by the current run. The last
command opens Bullet, runs the carry, then closes its connection. The control
entrypoint exposes editable `q0`, `qe`, and `path`;
the simulator uses that supplied path. Use `python control.py --headless` for
the same physics without a window. Visible rendering runs at 25 Hz with 1 kHz
physics, hidden debug panels and disabled scene-reset keyboard shortcuts.

The notebook resets the cube after endpoint IK and moves it with the hands
during path playback. Its physics cell runs headlessly. On macOS, the PyBullet
GUI can fail inside a Jupyter kernel and leave its window unresponsive; run
`python control.py` from Terminal to watch the physical carry. The GUI process
closes automatically when the run finishes.
The separate `lab_instructions.ipynb` retains the course notes and a simplified
robot/table view; its older setup cells expect an already running Meshcat server.

## Model assumptions and limits

The Bullet scene anchors the robot base, uses authored inertias, removes
spurious masses on empty hand/hook coordinate frames, and disables joint/link
damping to match the inverse-dynamics model. Solid masses, geometry, friction
and gravity are unchanged. The cube is carried through contact friction:
no artificial attachments or cube teleportation during motion.

The starter's disabled Bullet self-contact setting is retained. Planned self
collisions are checked using Pinocchio/SRDF. Dense collision checks are finite
samples, not a proof of continuous-time collision freedom. Some perturbed
targets fail IK; no general-layout robustness or trajectory-optimisation bonus
is claimed. DICE execution has not been verified.

## Verification

```sh
python verify.py
python verify.py --geometry-only
python demo.py --headless --record artifacts/new_run.mp4
```

The verifier checks endpoint IK, exact qgoal, dense grasp/collision/joint-limit
conditions, trajectory boundary derivatives and peak velocity. It then tests
default and translated-goal physical carries. A pass requires <2 cm final
translation error, >1 second of airborne contact with both arms, and no measured
table/obstacle contact or joint-limit violations. Existing evidence/video and
historical review notes are in the local, Git-ignored `artifacts/` directory.
The tested default run lifted the cube 18 cm and ended about 1 mm from target.

Common failures: missing `pinocchio`/`pybullet` means the wrong environment;
an import error after an edit can mean a cached kernel; failed IK/RRT can mean
an infeasible target or exhausted search budget. Meshcat setup in `viewer.py`
probes the server before connecting, avoiding an indefinite wait on a dead
server. If the GUI fails, headless verification separates physics from rendering.

## Version control and submission

See [the commit guide](docs/VERSION_CONTROL.md) for a dependency-aware breakdown
of the work and a workflow for future changes. The first two groups were
committed separately; the remaining implementation is grouped into one
integration commit at the owner's request.
Keep source, models, environment files and useful notebook cells in Git.
Generated videos, ZIPs, verification reports, caches and macOS metadata are
ignored. Review unrelated existing edits separately.
Fresh clones do not include the local `artifacts/` directory. To package a new
submission, record a successful run first and pass that video's path below.

```sh
python package_submission.py --output artifacts/STUDENT1NAME_STUDENT2NAME.zip --video artifacts/successful_run.mp4
```

Replace both placeholders with the students' family names. The builder refuses
overwrite and substitutes the <=300-word `SUBMISSION_README.md` inside the ZIP.
Older archives in artifacts use the earlier folder layout; build a new archive
when submitting current changes. The ZIP is a deliverable, not a Git commit.
The project retains the supplied LICENSE.
