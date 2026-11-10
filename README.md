# Tetris Studio · RL Arcade

Windows desktop Tetris game. The interface runs in a native WinForms window; it does not open a browser. The local Python runtime runs the existing Tetris rules and frozen model inference. No training or checkpoint writes happen during play, and there is no audio.

## Start

Double-click `START_DEMO.cmd`. The desktop window starts its local game engine automatically. Press `Alt+F4` to exit; the engine is stopped with the app.

This build targets .NET 10 for Windows Desktop. If Windows reports a missing .NET runtime, install the .NET 10 Desktop Runtime. Python, NumPy, and Gymnasium are bundled under `runtime`; no Python installation, PyTorch, CUDA, internet, or browser is required.

## Play

- Select **Survival Run** for an endless empty-board game or **Challenge Levels** for eight garbage-clearing levels.
- Select **You · Keyboard** or **DDQN Agent**. Challenge uses the newest DDQN + Sparse checkpoint; Survival uses DDQN + Danger from `demo_realtime_cpu`.
- When **DDQN Agent** is selected, the speed slider offers **×1, ×2, ×3, ×4, ×5** and applies immediately while the agent plays. Target placement intervals are **0.480 / 0.240 / 0.160 / 0.120 / 0.096 seconds**, respectively. Actual speed depends on CPU inference time. The multiplier is preserved when starting or advancing levels.
- Human play keeps its separate gravity slider in seconds. Changing the controller in setup does not change the current game until Start is pressed. Pause and stop controls remain available during a run.
- During human play, use the six on-screen controls below the board or the keyboard shortcuts.
- After changing mode, controller, or level, press **Start / Restart Game** to apply the new setup. Speed applies immediately.
- Hold starts empty. Press **C** or **HOLD** during human play to save the current piece; another swap is allowed after locking a piece. The frozen agents select placements and do not use Hold.
- Keyboard: `←` / `→` move, `↓` soft drop, `↑` or `X` rotate, `Z` counter-rotate, `C` hold, `Space` hard drop, `P` or `Esc` pause.
- In Challenge, the objective counts only original garbage provenance; clearing ordinary rows does not complete the level.

## Policies and checkpoints

| Mode | Policy | Included model |
|---|---|---|
| Survival / Agent | DDQN + Danger, 13-feature MLP | `models/survival_ddqn_danger.npz` |
| Challenge / Agent | DDQN + Sparse, 477-feature CNN | `models/challenge_ddqn_sparse.npz` for CPU inference; `.pt` is an archival copy |

Challenge checkpoint source: `results/runs/garbage_focused_wells_v6_5000_20261030/seed_101/ddqn__sparse/checkpoint_best_validation.pt`. It solved 87.5% of 40 paired held-out boards in the recorded evaluation. The training checkpoint remains untouched in the original results folder.

The desktop project source is in `Desktop/`. Rebuild the app with:

```cmd
dotnet build Desktop\TetrisStudio.csproj -c Release -o Desktop\publish_v10
```
