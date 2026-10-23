"""Verify dependencies/assets and optionally exercise RL libraries, not the game itself."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_graphics() -> dict:
    # This verifies SDL/image operations only. It does not test a real game window or FPS.
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    import pygame

    manifest_path = ROOT / "assets" / "stradex" / "prepared" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(manifest["textures"]) != 4:
        raise RuntimeError("The setup requires four prepared textures.")
    pygame.display.init()
    try:
        screen = pygame.display.set_mode((320, 180))
        for entry in manifest["textures"]:
            source = ROOT / Path(manifest["source_root"]) / entry["source"]
            output = ROOT / entry["output"]
            for path, key in [(source, "source_sha256"), (output, "output_sha256")]:
                if hashlib.sha256(path.read_bytes()).hexdigest() != entry[key]:
                    raise RuntimeError(f"Texture hash mismatch: {path}")
            texture = pygame.image.load(str(output)).convert()
            if list(texture.get_size()) != entry["output_size"]:
                raise RuntimeError(f"Texture dimensions mismatch: {output}")
            screen.blit(texture, (0, 0))
        pygame.display.flip()
    finally:
        pygame.display.quit()
    return {"status": "passed", "driver": "dummy", "textures_checked": 4}


def check_rl() -> dict:
    import numpy as np
    import torch
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.envs import InvalidActionEnvDiscrete
    from sb3_contrib.common.maskable.utils import get_action_masks
    from stable_baselines3.common.env_checker import check_env

    torch.set_num_threads(1)
    env = InvalidActionEnvDiscrete(dim=8, n_invalid_actions=4)
    check_env(env, warn=True)
    model = MaskablePPO(
        "MlpPolicy", env, device="cpu", seed=7, n_steps=32,
        batch_size=16, n_epochs=1, policy_kwargs={"net_arch": [16, 16]},
        verbose=0,
    )
    before = {key: value.detach().clone() for key, value in model.policy.state_dict().items()}
    model.learn(total_timesteps=64)
    if not any(not torch.equal(before[key], value)
               for key, value in model.policy.state_dict().items()):
        raise RuntimeError("The RL setup smoke run did not update model weights.")
    with tempfile.TemporaryDirectory(prefix="alien_rl_setup_") as temp_dir:
        checkpoint = Path(temp_dir) / "setup_only"
        model.save(checkpoint)
        loaded = MaskablePPO.load(checkpoint, device="cpu")
        obs, _ = env.reset(seed=19)
        for _ in range(32):
            mask = get_action_masks(env)
            expected, _ = model.predict(obs, action_masks=mask, deterministic=True)
            actual, _ = loaded.predict(obs, action_masks=mask, deterministic=True)
            if not np.array_equal(expected, actual) or not bool(mask[int(actual)]):
                raise RuntimeError("Checkpoint round-trip or masked prediction failed.")
            obs, _, terminated, truncated, _ = env.step(int(actual))
            if terminated or truncated:
                obs, _ = env.reset()
    env.close()
    return {
        "status": "passed", "environment": "SB3 InvalidActionEnvDiscrete (not Alien game)",
        "training_steps": 64, "masked_predictions": 32, "weights_updated": True,
        "checkpoint_roundtrip": True, "device": "cpu",
        "evidence_boundary": "Library setup only; no game learning or FPS claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl-smoke", action="store_true")
    args = parser.parse_args()
    packages = ["torch", "pygame", "gymnasium", "stable-baselines3", "sb3-contrib",
                "numpy", "Pillow", "pytest", "ruff"]
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "executable": sys.executable,
        "platform": platform.platform(), "logical_cpus": os.cpu_count(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "graphics": check_graphics(),
        "rl": check_rl() if args.rl_smoke else {"status": "not requested"},
    }
    destination = ROOT / "experiments" / "setup" / "environment_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Setup verification passed. Report: {destination}")


if __name__ == "__main__":
    main()
