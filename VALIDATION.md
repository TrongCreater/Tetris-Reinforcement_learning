# Desktop repair — 2026-10-30

Launcher: `START_DEMO.cmd` → `Desktop/publish_v10/TetrisStudioDesktop.exe`.

## Cause and fix

- `PostAsJsonAsync` used chunked HTTP requests. The Python handler read only `Content-Length` and treated those bodies as empty. This caused survival/human defaults and ignored action/speed fields.
- `GameApiClient.cs` now buffers JSON with `StringContent` to send a known length. The backend rejects unsupported framing and invalid setup rather than silently applying defaults.
- Commands run in order; a poll started before a command cannot overwrite its response.
- TrackBar has `AutoSize = false` and a separate layout row. Header, cards, and preview sections use layout containers. Preview shapes are cropped to occupied cells before sizing.
- Context menus are disposed when replaced or when their owner is disposed, never during the `Closed` callback (which precedes completion of item-click handling).

## Verification

- Release build: 0 errors, 0 warnings.
- .NET integration check uses the same `GameApiClient.cs` and the bundled Python runtime over localhost.
- Verified left/right, rotation, hold payload, hard drop, live speed, pause/resume, all eight challenge levels with both controllers, and successful placement by both checkpoint policies.
- Original chunked request explicitly returns HTTP 400 rather than silently starting the wrong mode.
- Native UI: selected challenge, started a board with 36 original cells, used C/Left/P, confirmed held piece is visible, used on-screen Drop, changed speed to 1.20s, selected level 3 and confirmed 45 cells, and started DDQN Agent.
- The native agent run then completed level 3: 17 placements, 5 lines, 0 original garbage cells remaining, and the NEXT LEVEL button appeared.

Hold is empty before the first human hold action. Current frozen policies choose placements and do not use the hold action; the UI explains this rather than showing a fabricated piece.

## Agent playback multiplier

- Five discrete settings: ×1–×5, targeting a placement interval of 0.48 / multiplier seconds.
- Human gravity and agent speed selections are stored separately in the UI.
- Verified all five live settings, ×5 at start (0.096 seconds), retaining ×5 on next level, paused gameplay while changing speed, and rejection of out-of-range/fractional multipliers.
- Compared actual checkpoint playback at ×1 and ×5 using identical seed and duration; ×5 placed more pieces. Inference throughput and timer granularity can limit the effective multiplier.
