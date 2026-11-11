"""Where the repo and the purchased Stradex pack live; every bake tool starts from here."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolve_source_root(source_root: Path | None = None) -> Path:
    if source_root is None:
        config_path = ROOT / "configs" / "assets.local.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            source_root = Path(config["source_root"])
        else:
            source_root = ROOT / "assets" / "stradex"
    source_root = source_root.expanduser()
    if not source_root.is_absolute():
        source_root = ROOT / source_root
    return source_root.resolve()
