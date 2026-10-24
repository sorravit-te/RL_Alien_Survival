"""Summarize frozen-policy diagnostics; no model training or test-set access."""

import argparse
import ast
import csv
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np


def load_rows(output):
    return [
        json.loads(line)
        for line in (output / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def paired(first, second):
    groups = []
    for rows in (first, second):
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["id"]].append(row["outcome"] == "captured")
        groups.append({k: float(np.mean(v)) for k, v in grouped.items()})
    assert groups[0].keys() == groups[1].keys()
    delta = np.array([groups[0][key] - groups[1][key] for key in sorted(groups[0])])
    rng = np.random.default_rng(904)
    means = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(axis=1)
    return {
        "difference": float(delta.mean()),
        "paired_case_bootstrap95": np.quantile(means, [0.025, 0.975]).tolist(),
        "case_clusters": len(delta),
        "bootstrap_draws": 10000,
    }


def summarize(rows):
    outcomes = Counter(row["outcome"] for row in rows)
    decisions = sum(row["decisions"] for row in rows)
    return {
        "episodes": len(rows),
        "case_count": len({r["id"] for r in rows}),
        "captured": outcomes["captured"],
        "escaped": outcomes["escaped"],
        "unresolved": outcomes["unresolved"],
        "catch_rate": outcomes["captured"] / len(rows),
        "mean_return": float(np.mean([r["return"] for r in rows])),
        "mean_decisions": float(np.mean([r["decisions"] for r in rows])),
        "mean_seconds": float(np.mean([r["elapsed_ticks"] / 30 for r in rows])),
        "decisions": decisions,
        "mean_entropy": sum(r["entropy_sum"] for r in rows) / decisions,
        "mean_max_probability": sum(r["max_probability_sum"] for r in rows) / decisions,
        **{
            f"{k}_rate": sum(r.get(k, 0) for r in rows) / decisions
            for k in (
                "wait",
                "visible",
                "last_seen",
                "repeat_action",
                "repeat_target",
                "hold",
                "stalled",
                "alternation",
            )
        },
        "other_while_visible_rate": sum(r["other_while_visible"] for r in rows)
        / max(1, sum(r["visible_available"] for r in rows)),
        "action_histogram": {str(i): sum(r.get(f"action_{i}", 0) for r in rows) for i in range(32)},
        "zero_net_alien_capture_episodes": sum(r["stationary_capture"] for r in rows),
    }


def trace(output, row):
    with gzip.open(output / row["trace"], "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hashes(directory):
    return {
        str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def load_verified_geometry(input_dir):
    """Load verified static map/LoS source only; no environment, policy or checkpoint load."""
    protocol = json.loads((input_dir / "protocol.json").read_text(encoding="utf-8"))
    digest = json.loads((input_dir / "protocol_digest.json").read_text(encoding="utf-8"))["sha256"]
    if canonical_hash(protocol) != digest:
        raise ValueError("Diagnostic protocol digest mismatch")
    original = Path(protocol["original_export"]).resolve()
    files = sorted(p for folder in ("game", "ai", "rl") for p in (original / folder).glob("*.py"))
    files += [original / "requirements.lock.txt"]
    source_hash = hashlib.sha256(
        b"".join(p.relative_to(original).as_posix().encode() + p.read_bytes() for p in files)
    ).hexdigest()
    if source_hash != protocol["source"]["source_sha256"]:
        raise ValueError("Original geometry source hash mismatch")

    def static_module(filename):
        path = original / "game" / filename
        name = "_diagnostic_static_" + path.stem + "_" + source_hash[:12]
        module = ModuleType(name)
        module.__file__ = str(path)
        sys.modules[name] = module  # dataclasses resolve their own module name
        # Compile verified bytes directly: never create __pycache__ in the protected export.
        exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
        return module

    world = static_module("map.py").small_test_map()
    geometry_hash = canonical_hash(asdict(world))
    if geometry_hash != protocol["checkpoints"]["best"]["map"]["sha256"]:
        raise ValueError("Original map geometry hash mismatch")
    state = ast.parse((original / "game/state.py").read_text(encoding="utf-8"))
    config = next(
        node for node in state.body if isinstance(node, ast.ClassDef) and node.name == "CoreConfig"
    )
    radii = {
        node.target.id: ast.literal_eval(node.value)
        for node in config.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id in ("radius", "alien_radius", "capture_padding")
    }
    if set(radii) != {"radius", "alien_radius", "capture_padding"}:
        raise ValueError("Original capture geometry is incomplete")
    return SimpleNamespace(
        world=world,
        segment_clear=static_module("interaction.py").segment_clear,
        source_hash=source_hash,
        geometry_hash=geometry_hash,
        radii=radii,
    )


def connector_segments(points, geometry):
    """Schematic straight chords only; omit any chord crossing/touching solid tiles."""
    return [
        (a, b)
        for a, b in zip(points, points[1:])
        if a != b and geometry.segment_clear(geometry.world, *a, *b)
    ]


def draw_samples(ax, points, geometry, color, label):
    xs, ys = zip(*points, strict=True)
    ax.scatter(xs, ys, s=7, color=color, label=label)
    segments = connector_segments(points, geometry)
    for a, b in segments:
        ax.plot([a[0], b[0]], [a[1], b[1]], linestyle="--", linewidth=0.8, color=color, alpha=0.7)
    ax.scatter(xs[0], ys[0], color=color, marker="o", s=35)
    ax.scatter(xs[-1], ys[-1], color=color, marker="x", s=55)
    return len(segments)


def trajectory_plot(input_dir, case_id, rows, geometry, *, output_dir=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = input_dir if output_dir is None else output_dir
    world = geometry.world
    chosen = [
        next(
            r
            for r in rows
            if r["split"] == "validation"
            and r["id"] == case_id
            and r["mode"] == mode
            and r["repeat"] == 0
        )
        for mode in ("random", "best_deterministic", "best_stochastic")
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    counts = {}
    for ax, row in zip(axes, chosen, strict=True):
        records = trace(input_dir, row)
        ax.imshow(
            [[tile == "#" for tile in line] for line in world.rows],
            cmap="Greys",
            vmin=0,
            vmax=1,
            extent=(0, world.width, world.height, 0),
        )
        for actor, color in (("survivor", "tab:blue"), ("alien", "tab:red")):
            points = [records[0]["logger_truth"][f"before_{actor}"]] + [
                r["logger_truth"][f"after_{actor}"] for r in records
            ]
            drawn = draw_samples(ax, points, geometry, color, actor)
            counts[f"{row['mode']}/{actor}"] = {"samples": len(points), "connectors_drawn": drawn}
        for index, (x, y) in enumerate(world.patrol_waypoints):
            ax.text(x + 0.1, y + 0.1, str(index), color="darkorange", fontsize=8)
        for obj in world.objects:
            ax.scatter(
                obj.x, obj.y, marker="*" if obj.kind == "keycard" else "s", color="green", s=45
            )
        ax.set(
            xlim=(0, world.width),
            ylim=(world.height, 0),
            aspect="equal",
            title=f"{row['mode']}\n{row['outcome']}, {row['decisions']} decisions",
        )
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(
        f"{case_id}: decision-endpoint samples; schematic dashed connectors, NOT physics trajectories\n"
        "Circles=start, crosses=end; omit chords crossing/touching walls; orange=waypoint, green=card/exit"
    )
    fig.tight_layout()
    fig.savefig(destination / f"paths_{case_id}.png", dpi=150)
    plt.close(fig)
    return counts


def wait_capture_plot(input_dir, rows, geometry, output_dir, *, case_id="validation_0043"):
    import matplotlib

    matplotlib.use("Agg")
    from math import dist

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    row = next(
        r
        for r in rows
        if r["split"] == "validation"
        and r["id"] == case_id
        and r["mode"] == "random"
        and r["repeat"] == 0
    )
    final = trace(input_dir, row)[-1]
    truth = final["logger_truth"]
    if not (
        final["action"] == 30
        and final["outcome"] == "captured"
        and final["net_alien_movement"] < 1e-8
        and final["net_survivor_movement"] > 0
    ):
        raise ValueError("Selected trace is not a stationary WAIT capture")
    expected_distance = sum(geometry.radii.values())
    separation = dist(truth["after_survivor"], truth["after_alien"])
    if abs(separation - expected_distance) > 1e-6:
        raise ValueError("Logged capture separation disagrees with verified source radii")
    y = truth["before_alien"][1]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axhline(y, color="gray", alpha=0.15)
    for point, radius, color, label in (
        (truth["before_survivor"], geometry.radii["radius"], "tab:blue", "Survivor before"),
        (truth["after_survivor"], geometry.radii["radius"], "deepskyblue", "Survivor at capture"),
        (truth["after_alien"], geometry.radii["alien_radius"], "tab:red", "Alien fixed (WAIT)"),
    ):
        ax.add_patch(Circle(point, radius, fill=False, color=color, linewidth=2, label=label))
    ax.annotate(
        "",
        xy=truth["after_survivor"],
        xytext=truth["before_survivor"],
        arrowprops={"arrowstyle": "->", "color": "tab:blue"},
    )
    xs = [truth[key][0] for key in ("before_survivor", "after_survivor", "after_alien")]
    ax.set(
        xlim=(min(xs) - 0.4, max(xs) + 0.4),
        ylim=(y - 0.45, y + 0.45),
        aspect="equal",
        title=f"{case_id} random: logged WAIT capture endpoints; Alien net movement 0\n"
        f"Survivor advances {final['net_survivor_movement']:.3f}; separation {separation:.3f} = capture radii",
    )
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / f"approach_{case_id}.png", dpi=150)
    plt.close(fig)
    return {
        "case_id": case_id,
        "decision": final["decision"],
        "action": final["action"],
        "separation": separation,
        "radii": geometry.radii,
        "source_trace": row["trace"],
    }


def plot_saved_traces(input_dir, output_dir):
    """Post-analysis visualization only; fresh output and byte-identical input required."""
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    if (
        input_dir == output_dir
        or output_dir.is_relative_to(input_dir)
        or input_dir.is_relative_to(output_dir)
    ):
        raise ValueError(
            "Input/output trees must be separate; do not overwrite historical evidence"
        )
    if output_dir.exists():
        raise FileExistsError(f"New output directory required: {output_dir}")
    before = file_hashes(input_dir)
    geometry = load_verified_geometry(input_dir)
    protocol = json.loads((input_dir / "protocol.json").read_text(encoding="utf-8"))
    original = Path(protocol["original_export"]).resolve()
    protected = file_hashes(original)  # static export only; never open historical pilot/test files
    rows = load_rows(input_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    counts = trajectory_plot(input_dir, "validation_0003", rows, geometry, output_dir=output_dir)
    capture = wait_capture_plot(input_dir, rows, geometry, output_dir)
    if before != file_hashes(input_dir) or protected != file_hashes(original):
        raise RuntimeError("Visualization modified protected input bytes")
    snapshot = {
        "world": asdict(geometry.world),
        "source_sha256": geometry.source_hash,
        "geometry_sha256": geometry.geometry_hash,
        "capture_radii": geometry.radii,
    }
    (output_dir / "geometry_snapshot.json").write_text(
        json.dumps(snapshot, indent=2) + "\n", encoding="utf-8"
    )
    provenance = {
        "purpose": "Post-analysis visualization correction from existing traces only",
        "input_directory": str(input_dir),
        "input_sha256": before,
        "plotter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "original_source_sha256": geometry.source_hash,
        "geometry_sha256": geometry.geometry_hash,
        "connector_counts": counts,
        "wait_capture": capture,
        "input_unchanged": True,
        "original_static_source_files_unchanged": True,
        "episodes_run": 0,
        "optimizer_updates": 0,
        "checkpoint_loads": 0,
        "trajectory_note": "Decision endpoints with schematic dashed unobstructed chords, not exact physics trajectories",
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "review.txt").write_text(
        "Post-analysis reproducible visualization correction\n"
        "The original diagnostic plots/review/protocol/source_snapshot/raw traces remain untouched.\n"
        "Earlier final pictures required ad-hoc plotting; this saved command now reproduces their stated conventions.\n"
        "Uses only logged decision endpoints and hash-verified static original map/LoS/radii source.\n"
        "No environments, policies, checkpoints, rollouts, learn(), prepare/run or held-out test evaluation.\n"
        "Circles=start, crosses=end, dots=decision endpoints, dashed connectors=schematic straight chords.\n"
        "A connector is omitted if it crosses or touches a wall. These are NOT exact physics trajectories.\n"
        "WAIT zoom uses logged actor positions and capture radii extracted from verified original source.\n"
        "Diagnostic input and static-export hashes were checked before/after; original pilot/test files are never opened.\n"
        "provenance.json records these checks and zero episode/model loads.\n\n"
        "From repository root, use a NEW output directory:\n"
        f".venv/Scripts/python.exe -B -m tools.policy_diagnostic_report --input {input_dir.as_posix()} --output NEW_OUTPUT_DIRECTORY\n"
        "Output: paths_validation_0003.png, approach_validation_0043.png, geometry_snapshot.json, provenance.json, review.txt.\n"
        "Stop for read-only review; no training or next feature work.\n",
        encoding="utf-8",
    )
    return provenance


def report(output, runtime):
    from tools.policy_diagnostics import PILOT

    rows = load_rows(output)
    groups = defaultdict(list)
    for row in rows:
        groups[f"{row['split']}/{row['mode']}"].append(row)
    summaries = {key: summarize(value) for key, value in groups.items()}
    comparisons = {}
    for first, second in (
        ("best_stochastic", "best_deterministic"),
        ("last_stochastic", "last_deterministic"),
        ("best_deterministic", "random"),
        ("last_deterministic", "random"),
        ("best_stochastic", "random"),
        ("last_stochastic", "random"),
        ("best_stochastic", "untrained_stochastic"),
        ("last_stochastic", "untrained_stochastic"),
    ):
        comparisons[f"{first}_minus_{second}"] = paired(
            groups[f"validation/{first}"], groups[f"validation/{second}"]
        )
    # Reproduction controls use historical VALIDATION CSVs only, never results.json/test outputs.
    recorded = {
        "random": "baseline_random.csv",
        "untrained_deterministic": "baseline_untrained.csv",
        "best_deterministic": "best_reloaded.csv",
        "last_deterministic": "step_00020224.csv",
        "heuristic": "baseline_heuristic.csv",
    }
    for mode, filename in recorded.items():
        with (PILOT / "validation" / filename).open(newline="", encoding="utf-8") as handle:
            prior = list(csv.DictReader(handle))
        actual = groups[f"validation/{mode}"]
        assert len(prior) == len(actual) == 50
        for a, b in zip(actual, prior, strict=True):
            assert a["id"] == b["id"] and a["outcome"] == b["outcome"]
            assert a["action_sha256"] == b["action_sha256"]
            assert a["elapsed_ticks"] == int(b["elapsed_ticks"])
            assert (
                a["decisions"] == int(b["decision_steps"])
                and abs(a["return"] - float(b["return"])) < 1e-10
            )
    ids = [case["id"] for case in runtime.manifest["splits"]["validation"]]
    lookup = {(r["mode"], r["id"], r["repeat"]): r for r in rows if r["split"] == "validation"}
    conditions = {
        "deterministic_fails_sampling_catches": lambda c: (
            lookup["best_deterministic", c, 0]["outcome"] != "captured"
            and lookup["best_stochastic", c, 0]["outcome"] == "captured"
        ),
        "random_catches_deterministic_fails": lambda c: (
            lookup["random", c, 0]["outcome"] == "captured"
            and lookup["best_deterministic", c, 0]["outcome"] != "captured"
        ),
    }
    examples = {
        name: next((c for c in ids if predicate(c)), None) for name, predicate in conditions.items()
    }
    geometry = load_verified_geometry(output)
    for case in sorted({c for c in examples.values() if c}):
        trajectory_plot(output, case, rows, geometry)
    stationary = next(
        (r for r in rows if r["split"] == "validation" and r["stationary_capture"]), None
    )
    if stationary:
        final = trace(output, stationary)[-1]
        examples["first_zero_net_alien_capture"] = {
            "case": stationary["id"],
            "mode": stationary["mode"],
            "repeat": stationary["repeat"],
            "action": final["action"],
            "target": final["target"],
            "net_alien_movement": final["net_alien_movement"],
            "net_survivor_movement": final["net_survivor_movement"],
            "survivor_approach": final["survivor_approach"],
            "truth": final["logger_truth"],
        }
    spawns = {key: dict(Counter(str(r["spawn"]) for r in value)) for key, value in groups.items()}
    # Explain failure states from all deterministic failures, not only favorable examples.
    failure_behaviors = {}
    for mode in ("best_deterministic", "last_deterministic"):
        failed = [r for r in groups[f"validation/{mode}"] if r["outcome"] != "captured"]
        failure_behaviors[mode] = summarize(failed) if failed else None
    train_rows = []
    with (PILOT / "train_episodes.csv").open(newline="", encoding="utf-8") as handle:
        train_rows = list(csv.DictReader(handle))
    training_counts = Counter(r["outcome"] for r in train_rows)
    result = {
        "groups": summaries,
        "paired_comparisons": comparisons,
        "examples": examples,
        "spawn_counts": spawns,
        "deterministic_failures": failure_behaviors,
        "recorded_validation_actions_and_episodes_reproduced": True,
        "recorded_online_training": {
            "episodes": len(train_rows),
            "outcomes": dict(training_counts),
            "note": "Changing stochastic policy during optimization; not a fixed checkpoint evaluation.",
        },
    }
    runtime.p.write_json(output / "summary.json", result)
    fields = list(next(iter(summaries.values())).keys())
    with (output / "summary.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group", *fields])
        writer.writeheader()
        writer.writerows({"group": key, **value} for key, value in summaries.items())
    lines = [
        "Frozen policy diagnostics - evidence report",
        f"Output: {output}",
        "Only original TRAIN/VALIDATION cases used. No model updates and no held-out TEST evaluation.",
        "Three sampling repeats per stochastic case are NOT three training seeds. One trained policy lineage.",
        "Protocol/cases/seeds/checkpoint hashes were frozen before the800-episode collection.",
        "Runtime is the verified ORIGINAL export, not the renamed current runtime. Original guards remain active.",
        "",
        "Measured groups: C/E/U; return is raw reward; entropy is decision-weighted masked entropy in nats.",
        "group | episodes | C/E/U | return | decisions/episode | entropy | maxP | repeat-target | hold | wait",
    ]
    for key, s in summaries.items():
        lines.append(
            f"{key} | {s['episodes']} | {s['captured']}/{s['escaped']}/{s['unresolved']} | "
            f"{s['mean_return']:.3f} | {s['mean_decisions']:.2f} | {s['mean_entropy']:.3f} | "
            f"{s['mean_max_probability']:.3f} | {s['repeat_target_rate']:.3f} | {s['hold_rate']:.3f} | {s['wait_rate']:.3f}"
        )
    lines += [
        "",
        "Paired differences in catch probability (case-cluster bootstrap95%; all3 repeats retained in each case):",
    ]
    for key, comparison in comparisons.items():
        low, high = comparison["paired_case_bootstrap95"]
        lines.append(
            f"{key}: {comparison['difference']:+.3f} [{low:+.3f},{high:+.3f}], n=50 case clusters"
        )
    lines += [
        "",
        "Predeclared examples: " + json.dumps(examples, ensure_ascii=False),
        "",
        "Evidence boundary / definitions",
        "Policy inputs are the original float32[352] allowed observation and real mask; heuristic receives AllowedAlienView.",
        "All truth snapshots are logger-only AFTER action selection. Logging never supplies policy/bot inputs or changes actions.",
        "Probabilities are inspected after prediction without drawing RNG; replay test compares inspection on/off.",
        "Net movement measures decision-endpoint displacement, not exact within-tick path length. Progress is Euclidean to chosen target.",
        "ABAB action alternation is a switching indicator, not proof of physical looping. Hold is already-at-target plus zero net motion.",
        "Random entropy is uniform mask entropy; heuristic entropy is an artificial point mass. Do not confuse these with policy certainty.",
        "",
        "Source observations (not inferred from win statistics)",
        "Explorer SurvivorView contains self x/y/yaw, has_keycard and visible_objects; no Alien position/action/visibility.",
        "Explorer.command chooses known mission objects or geometry waypoints; it has no evade/flee branch.",
        "Both actors use written A* navigation and collision. Choosing a reachable waypoint delegates movement/pathfinding to helpers.",
        "Six static Alien waypoints cover this tiny fixed room/corridor, including card vicinity and a Survivor spawn vicinity.",
        "No claim that trained/random policies learned A*, sensing, movement or evasion; these are existing helpers.",
        "",
        "Interpretation still requires the attached failure/example traces; summary.json contains all group and failure statistics.",
        "No single example establishes population-level causality. Same-checkpoint same-case action-mode comparison isolates sampling",
        "from weights/case changes under this simulator, but individual trajectories diverge and generalization is not established.",
        "Train-vs-validation deterministic comparison uses the predeclared first50 train cases only; map/spawns are not new layouts.",
        "Recorded online training capture counts: "
        + json.dumps(dict(training_counts))
        + f", n={len(train_rows)}.",
        "",
        "Reproduction from repo root (NEW output directory required):",
        r".venv\Scripts\python.exe -B -m tools.policy_diagnostics --phase self-check",
        r".venv\Scripts\python.exe -B -m tools.policy_diagnostics --phase prepare --output experiments/policy_diagnostics/NEW_RUN_NAME",
        r".venv\Scripts\python.exe -u -B -m tools.policy_diagnostics --phase run --output experiments/policy_diagnostics/NEW_RUN_NAME",
        "Do not rerun/overwrite this completed directory. Reporter hashes are frozen too; do not edit tools between prepare/run.",
        "protocol.json/protocol_digest.json freeze inputs; episodes.csv/jsonl and traces/*.jsonl.gz hold complete per-decision records.",
        "summary.json/csv include all results; paths_*.png use drawn geometry only, no purchased artwork.",
        "execution.json records real episode count/wall clock and post-run weight/artifact integrity.",
        "",
        "Stop for Astra read-only review. Recommendations must be reviewed before any training/task changes.",
    ]
    (output / "review.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Render saved diagnostic traces only; no episodes or model loads"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = plot_saved_traces(args.input, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "episodes_run": result["episodes_run"],
                "checkpoint_loads": result["checkpoint_loads"],
                "input_unchanged": True,
            }
        )
    )


if __name__ == "__main__":
    main()
