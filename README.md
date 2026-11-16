# RL_Alien_Survival

A university survival game project where a human player faces a reinforcement learning Alien agent.
The game runs on a 2D world and shows it in first person as 3D surfaces built from that map.

The full 40x40 ship deck is playable against an Alien trained from scratch with MaskablePPO.

## Setup

Windows x64 with [Git](https://git-scm.com/download/win) and [Python 3.12.5](https://www.python.org/downloads/release/python-3125/) exactly, because the trained Alien loads only on the versions it was trained with.

```powershell
git clone https://github.com/sorravit-te/RL_Alien_Survival.git
cd RL_Alien_Survival
powershell -ExecutionPolicy Bypass -File tools\bootstrap.ps1
```

Clone into a short path such as `C:\RL_Alien_Survival`: a deep folder runs into Windows' 260-character path limit during the install. The script makes `.venv`, installs the locked packages (about 1 GB) and downloads the free sounds, the view from the bridge and the mood. If `py -3.12` finds another 3.12, add `-Python <path to python.exe 3.12.5>`.

The ship's textures come from a purchased pack that cannot be shared here, and the game does not start without them. Owners of the pack put it in `assets/stradex/` before running the script; anyone given the baked textures puts `ship_textures.npz` and `ship_textures.json` in `assets/stradex/prepared/`.

## Play

Run from the repo root in PowerShell. The trained Alien is kept in `trained/`.

Against the trained Alien:

```powershell
.venv\Scripts\python.exe -B -m app.play --controller rl --run-dir trained/ship_deck_full_v3_scratch_s3/run --checkpoint trained/ship_deck_full_v3_scratch_s3/run/checkpoints/best.zip --sampling-seed 310003 --scenario-seed 310002 --fps-limit 60
```

Against the scripted Alien, or alone on the ship:

```powershell
.venv\Scripts\python.exe -B -m app.play --scenario ship_deck_v1 --scenario-seed 1 --fps-limit 60
.venv\Scripts\python.exe -B -m app.play --scenario ship_deck_v1 --scenario-seed 1 --fps-limit 60 --no-alien
```

- The game opens paused. Press N for a new round, since a command's seeds always deal the same first round. R replays the last round exactly, Esc pauses, and Q quits from the pause menu.
- WASD to move, Shift to run, E to interact, the mouse to look and the arrow keys to turn.
- `--windowed` stays in a window, `--fov` sets the view from 60 to 120 degrees (default 100), and `--scenario-seed <n>` picks a round.
- To record every round, add `--record-dir experiments/human_dev_<name>` to the trained Alien command. Review it with `.venv\Scripts\python.exe -m tools.review_recording experiments/human_dev_<name>`.
- In VS Code, Terminal > Run Task > "Alien RL: Play against trained Alien" runs the first command.

## Assets

[Stradex - Sci-Fi Tech Hall Texture Pack](https://stradex.itch.io/scifi-tech-hall-texture-pack)

Earth and Moon imagery: NASA Earth Observatory and NASA's Scientific Visualization Studio. This project is not endorsed by NASA.

Alien: 3D Horror Game Monster on OpenGameArt (CC0). Sounds: CC0 recordings from Freesound and OpenGameArt, listed by the sound bake.

Lens dirt: zisongbr (CC0). Smoke: Kenney (CC0). Color grade: REEBOO Labs Color Grading LUT Pack.
