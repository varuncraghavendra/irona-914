# Live run artefacts

- `isaac_fetch_mission.json` — the mission report from the last `--mode fetch`
  run: every state transition with its timestamp, the camera estimate of the vase,
  the aim point actually used, IK residuals, palm-correction count, the height the
  vase was lifted and whether the grasp assist engaged.
- `isaac_fetch_trajectory.json`, `isaac_walk_trajectory.json`,
  `isaac_stand_trajectory.json` — 20 Hz samples of pelvis pose, joint positions,
  centre of mass, tilt and (for fetch) the vase position and mission state.

These are overwritten by each run. `docs/ISAAC_RUN_NOTES.md` explains what the
numbers mean and the limits behind them.
