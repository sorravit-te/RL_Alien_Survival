"""Prepare local-use copies of purchased textures, preserving source files and aspect ratio."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TEXTURES = {
    "wall": "wall1.png",
    "floor": "floor1.png",
    "ceiling": "ceiling1.png",
    "door": "door1.png",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def prepare(max_size: int = 256, source_root: Path | None = None) -> dict:
    if max_size < 16:
        raise ValueError("max-size must be at least 16 pixels")
    source_root = resolve_source_root(source_root)
    source_dir = source_root / "Flat"
    license_path = source_root / "LICENSE.txt"
    sources = {role: source_dir / filename for role, filename in TEXTURES.items()}
    for path in [license_path, *sources.values()]:
        if not path.is_file():
            raise FileNotFoundError(f"Required Stradex source is missing: {path}")
    destination = ROOT / "assets" / "stradex" / "prepared"
    destination.mkdir(parents=True, exist_ok=True)
    entries = []
    for role, source in sources.items():
        source_hash = digest(source)
        target = destination / source.name
        with Image.open(source) as original:
            source_size = list(original.size)
            converted = original.convert("RGB")
            converted.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            converted.save(target, format="PNG")
            target_size = list(converted.size)
        if digest(source) != source_hash:
            raise RuntimeError(f"Source changed during preparation: {source}")
        entries.append({
            "role": role,
            "source": source.relative_to(source_root).as_posix(),
            "source_sha256": source_hash,
            "source_size": source_size,
            "output": target.relative_to(ROOT).as_posix(),
            "output_sha256": digest(target),
            "output_size": target_size,
        })
    shutil.copyfile(license_path, destination / "STRADEX_LICENSE.txt")
    source_reference = (
        source_root.relative_to(ROOT).as_posix()
        if source_root.is_relative_to(ROOT)
        else str(source_root)
    )
    manifest = {
        "creator": "Stradex / Maximiliano Ruben Viamonte",
        "source_url": "https://stradex.itch.io/scifi-tech-hall-texture-pack",
        "source_root": source_reference,
        "use": "Local project assets. Do not redistribute as standalone artwork.",
        "max_size": max_size,
        "textures": entries,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-size", type=int, default=256)
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    manifest = prepare(args.max_size, args.source_root)
    for entry in manifest["textures"]:
        print(f"{entry['role']}: {entry['source_size']} -> {entry['output_size']}")
    print("Prepared four local textures; source files are unchanged.")


if __name__ == "__main__":
    main()
