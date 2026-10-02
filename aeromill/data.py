"""Frozen family manifests and open-loop train/validation windows; no test signals."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from .config import CANDIDATES, ToolProfile, WINDOW_SAMPLES
from .contracts import AppliedState
from .features import FEATURE_NAMES, FeatureStream
from .scenarios import BASE_SCENARIOS, EngagementSegment, Region, Scenario
from .simulator import Simulator

VERSIONS = dict(
    generator="engagement-v2", features="ordered-23-v1", model="rf-200-depth10-v1"
)


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def source_hashes():
    root = Path(__file__).parent
    return {
        name: file_hash(root / name)
        for name in (
            "simulator.py",
            "scenarios.py",
            "features.py",
            "config.py",
            "data.py",
        )
    }


def varied_scenario(base, rng, engaged):
    regions = tuple(
        replace(
            r,
            start_mm=float(rng.uniform(60, 100)) if i == 0 else r.start_mm,
            widths=tuple(
                float(rng.uniform(100, 180)) if w < 2000 else w for w in r.widths
            ),
        )
        for i, r in enumerate(base.regions)
    )
    segments = []
    if engaged:
        # Equal slots prevent overlap; lengths and levels are drawn without RF feedback.
        count = int(rng.integers(1, 4))
        for i in range(count):
            slot = (base.path_length_mm - 60) / count
            start = 60.0 + i * slot
            length = float(rng.uniform(40, min(120, slot)))
            level = (
                float(rng.uniform(1.3, 2.0))
                if rng.random() < 0.75
                else float(rng.uniform(0.8, 1.0))
            )
            segments.append(EngagementSegment(start, start + length, level))
    return replace(
        base,
        regions=regions,
        engagement=tuple(segments),
        version=VERSIONS["generator"],
        impact_positions_mm=(120.0, 280.0) if base.name == "stable" else (),
    )


def scenario_from_dict(value):
    return Scenario(
        **{
            **value,
            "regions": tuple(Region(**r) for r in value["regions"]),
            "engagement": tuple(EngagementSegment(**r) for r in value["engagement"]),
        }
    )


def make_manifest():
    rng = np.random.default_rng(4203)
    episodes = []
    for scenario_index, (name, base) in enumerate(BASE_SCENARIOS.items()):
        # sklearn's installed _split.py: split unique groups, then expand to samples.
        groups = np.repeat(np.arange(10), 5)
        train, held = next(
            GroupShuffleSplit(n_splits=1, train_size=0.6, random_state=42).split(
                groups, groups=groups
            )
        )
        val, test = next(
            GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=42).split(
                held, groups=groups[held]
            )
        )
        splits = {
            int(g): s
            for s, ids in [
                ("train", train),
                ("validation", held[val]),
                ("test", held[test]),
            ]
            for g in groups[ids]
        }
        for family in range(10):
            s = varied_scenario(base, rng, family < 5)
            profile = asdict(ToolProfile(h=float(rng.uniform(0.8, 2.0))))
            noise = float(rng.uniform(0.02, 0.10))
            frequency = float(rng.uniform(900, 1800))
            initial = (
                asdict(AppliedState())
                if family % 2 == 0
                else asdict(
                    AppliedState(
                        float(rng.uniform(2400, 4200)), float(rng.uniform(600, 1200))
                    )
                )
            )
            schedule = [
                dict(
                    time_s=t,
                    candidate=str(k),
                    rpm=CANDIDATES[k].rpm,
                    feed_mm_min=CANDIDATES[k].feed_mm_min,
                )
                for t, k in zip(
                    (4, 8, 12), rng.permutation(list(CANDIDATES)), strict=True
                )
            ]
            for variant in range(5):
                seed = 100000 + scenario_index * 1000 + family * 10 + variant
                episodes.append(
                    dict(
                        split=splits[family],
                        family_id=f"windows-{name}-{family}",
                        seed=seed,
                        base_scenario=name,
                        scenario=asdict(s),
                        profile=profile,
                        noise_std=noise,
                        chatter_hz=frequency,
                        initial_state=initial,
                        initial_phase=rng.uniform(0, 2 * np.pi, 2).tolist(),
                        schedule=schedule,
                        duration_s=40 if name == "repeat_after_c" else 20,
                    )
                )
    end_to_end = {}
    controllable = ("recover_a", "retry_b", "recover_c", "repeat_after_c")
    for split, seed_base in [
        ("validation", 200000),
        ("final", 300000),
        ("reserve", 400000),
    ]:
        entries = []
        for i in range(60):
            kind = "controllable" if i < 30 else "stable"
            name = controllable[i % 4] if i < 30 else "stable"
            s = varied_scenario(BASE_SCENARIOS[name], rng, i % 30 < 15)
            entries.append(
                dict(
                    split=split,
                    kind=kind,
                    family_id=f"e2e-{split}-{i}",
                    seed=seed_base + i,
                    base_scenario=name,
                    scenario=asdict(s),
                    profile=asdict(ToolProfile(h=float(rng.uniform(0.8, 2.0)))),
                    noise_std=float(rng.uniform(0.02, 0.10)),
                    chatter_hz=float(rng.uniform(900, 1800)),
                    initial_state=asdict(AppliedState()),
                    initial_phase=rng.uniform(0, 2 * np.pi, 2).tolist(),
                    schedule=[],
                    duration_s=65,
                )
            )
        end_to_end[split] = entries
    return dict(
        versions=VERSIONS,
        feature_order=FEATURE_NAMES,
        split_seed=42,
        source_hashes=source_hashes(),
        episodes=episodes,
        end_to_end=end_to_end,
    )


def config_hash(manifest):
    config = {
        k: v
        for k, v in manifest.items()
        if k not in ("artifact_hashes", "config_sha256")
    }
    return sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_manifest(manifest):
    entries = manifest["episodes"] + sum(manifest["end_to_end"].values(), [])
    seeds = [e["seed"] for e in entries]
    if len(set(seeds)) != len(seeds) or any(s >= 900000 for s in seeds):
        raise ValueError("duplicate or reserved seed")
    owners = {}
    for e in entries:
        owner = ("window" if e in manifest["episodes"] else "e2e", e["split"])
        previous = owners.setdefault(e["family_id"], owner)
        if owner != previous:
            raise ValueError("family crosses splits")


def window_label(states):
    if np.all(states == "stable"):
        return 0
    if np.all(states == "chatter"):
        return 1
    return -1


def generate_episode(entry):
    if entry["split"] == "test":
        from .freeze import require_active

        require_active("test")
    elif entry["split"] not in ("train", "validation"):
        raise ValueError("withheld split: manifests only")
    profile = ToolProfile(**entry["profile"])
    target = AppliedState(**entry["initial_state"])
    sim = Simulator(
        scenario_from_dict(entry["scenario"]),
        seed=entry["seed"],
        profile=profile,
        noise_std=entry["noise_std"],
        chatter_hz=entry["chatter_hz"],
        initial_state=target,
        initial_phase=entry["initial_phase"],
    )
    stream = FeatureStream(profile)
    states = np.empty(0, dtype="U10")
    position = np.empty(0)
    rpms = np.empty(0)
    feeds = np.empty(0)
    rows = []
    labels = []
    regimes = []
    engaged = []
    for tick in range(1, entry["duration_s"] * 10 + 1):
        for command in entry["schedule"]:
            if tick - 1 == command["time_s"] * 10:
                target = AppliedState(command["rpm"], command["feed_mm_min"])
        step = sim.step(0.1, target)
        t = step.telemetry
        states = np.concatenate((states, step.truth.states))[-WINDOW_SAMPLES:]
        position = np.concatenate((position, step.truth.x_mm))[-WINDOW_SAMPLES:]
        rpms = np.concatenate((rpms, t.rpm))[-WINDOW_SAMPLES:]
        feeds = np.concatenate((feeds, t.feed_mm_min))[-WINDOW_SAMPLES:]
        f = stream.push(t)
        if f is not None:
            rows.append(f)
            labels.append(window_label(states))
            # A window is during a command if any of its samples overlap the slew.
            command_times = [
                c["time_s"] for c in entry["schedule"] if c["time_s"] < tick / 10
            ]
            last = command_times[-1] if command_times else None
            if last is None:
                regime = "before"
            elif (
                t.time_s[-1] - 0.25 < last
                or np.any(abs(rpms - target.rpm) > 1)
                or np.any(abs(feeds - target.feed_mm_min) > 1)
            ):
                regime = "during"
            else:
                regime = "after"
            regimes.append(regime)
            engaged.append(
                any(
                    np.any((position > s.start_mm) & (position < s.end_mm))
                    for s in sim.scenario.engagement
                )
            )
        if t.xyz_mm[-1, 0] >= sim.scenario.path_length_mm:
            break
    return dict(
        features=np.asarray(rows),
        label=np.asarray(labels, dtype=np.int8),
        regime=np.asarray(regimes),
        engagement=np.asarray(engaged),
        family=np.full(len(rows), entry["family_id"]),
        seed=np.full(len(rows), entry["seed"]),
        split=np.full(len(rows), entry["split"]),
    )


def build(directory=Path("data"), workers=4):
    started = perf_counter()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = make_manifest()
    validate_manifest(manifest)
    path = directory / "manifest.json"
    # Freeze all assignments before the first simulator call; no model feedback.
    if path.exists() and config_hash(json.loads(path.read_text())) != config_hash(
        manifest
    ):
        raise ValueError(
            "manifest differs: use a new data directory and record the change"
        )
    manifest["config_sha256"] = config_hash(manifest)
    write_json(path, manifest)
    allowed = [e for e in manifest["episodes"] if e["split"] in ("train", "validation")]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        batches = list(pool.map(generate_episode, allowed))
    coverage = {}
    for split in ("train", "validation"):
        parts = [b for b in batches if b["split"][0] == split]
        arrays = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        np.savez_compressed(directory / f"{split}.npz", **arrays)
        counts = Counter(
            zip(
                arrays["label"].tolist(),
                arrays["regime"].tolist(),
                arrays["engagement"].tolist(),
            )
        )
        coverage[split] = [
            dict(label=l, regime=r, engagement=e, windows=counts[l, r, e])
            for l in (-1, 0, 1)
            for r in ("before", "during", "after")
            for e in (False, True)
        ]
    manifest["artifact_hashes"] = {
        s: file_hash(directory / f"{s}.npz") for s in ("train", "validation")
    }
    write_json(path, manifest)
    result = dict(
        build_seconds=perf_counter() - started,
        coverage=coverage,
        manifest_sha256=file_hash(path),
        artifact_hashes=manifest["artifact_hashes"],
        withheld="test/final/reserve: manifests only",
    )
    write_json(directory / "build.json", result)
    print(
        json.dumps(
            dict(
                build_seconds=result["build_seconds"],
                episodes_generated=len(allowed),
                manifest=str(path),
            )
        ),
        flush=True,
    )
    return result
