# Committing the CW2 work

All implementation files are now in the lab root. The repository already has
Git history and an origin remote; do not run git init or create another repo.
The first two groups below were committed separately. The remaining work was
subsequently grouped into one integration commit at the owner's request.
The breakdown remains a reference for understanding dependencies and organising
future changes; it is not a list of commits still waiting to be made.

Small coherent commits make review easier. There is no rule that an initial
implementation must be split into artificial steps: these groups reflect real
dependencies in the work already completed. Use accurate messages and dates.

## Start a feature branch

From the lab directory:

```sh
git status --short
git switch -c cw2-next-change
```

Your uncommitted changes stay on the new branch. If the name already exists,
inspect it first or choose another descriptive branch name.

## Suggested commit groups, in order

| Group | Files to stage | Suggested message |
| --- | --- | --- |
| 1 | `.gitignore` | Ignore local metadata and generated coursework artifacts |
| 2 | `solution.py inverse_geometry.py viewer.py` | Implement collision-checked bimanual inverse geometry |
| 3 | `path.py` | Add grasp-preserving path planning and cube playback |
| 4 | `control.py simulation.py demo.py` | Implement contact-based cube carrying in Bullet |
| 5 | `verify.py` | Add geometry and physical-carry regression checks |
| 6 | `cw2_solution.ipynb` | Add a commented notebook runner for the CW2 pipeline |
| 7 | `package_submission.py SUBMISSION_README.md docs/VERSION_CONTROL.md` | Add reproducible coursework packaging and workflow documentation |

Stage relevant README sections alongside each feature using `git add -p README.md`.
If a README hunk mixes multiple sections, use the interactive split/edit options,
or include the final consolidated README in group 7. Review the notebook output
before committing it; remove stale outputs if they contain old code or local paths.

Example for group 2:

```sh
git add solution.py inverse_geometry.py viewer.py
git add -p README.md
git diff --cached --stat
git diff --cached --check
git diff --cached
git commit -m "Implement collision-checked bimanual inverse geometry"
```

Repeat with the listed files for each group. When several changes share a file,
`git add -p FILE` stages selected hunks. `git restore --staged FILE` unstages a
mistake without changing your working copy. Avoid indiscriminate `git add .`
when unrelated edits are present. Review notebooks such as
`lab_instructions.ipynb` and `lab_0_my_work.ipynb` before including their changes.

Keep `.DS_Store`, caches, videos, archives and generated reports out of Git.
They remain available locally under artifacts; the ignore rules do not delete
them. The review ZIPs contain the older layout; build a fresh ZIP for submission.

## Verify and share

```sh
python verify.py
git status --short
git log --oneline -8
git push -u origin cw2-next-change
```

The final command publishes your branch only when you choose to run it. Open a
pull request into main so your partner can review the changes. Check the course
rules before making coursework public. Never force-push shared history merely
to make a sequence of commits look cleaner.

Tests use working-tree files, so running them with uncommitted later features
does not prove an earlier commit works independently. For that check, use a
temporary detached worktree at the desired commit and run only its relevant
entrypoints/tests, or let CI test each pushed commit.
