"""Canonical experiment-family directories and historical path resolution."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
TARGET_DROP_EXPERIMENT_DIR = EXPERIMENTS_ROOT / "target_drop"
DISTANCE_QUARTILE_EXPERIMENT_DIR = EXPERIMENTS_ROOT / "distance_quartiles"

TARGET_DROP_EXPERIMENT_RELATIVE_DIR = Path("experiments") / "target_drop"

# Historical JSON/state assets are intentionally not rewritten during the
# filesystem migration. Resolve their recorded paths when they are consumed.
_HISTORICAL_EXPERIMENT_ROOTS = {
    "experiments3": DISTANCE_QUARTILE_EXPERIMENT_DIR,
}


def resolve_recorded_experiment_path(value):
    """Resolve a canonical or historically recorded experiment asset path."""
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        try:
            relative_path = path.resolve().relative_to(PROJECT_ROOT)
        except ValueError:
            return path.resolve()
    else:
        relative_path = path

    parts = relative_path.parts
    if parts and parts[0] in _HISTORICAL_EXPERIMENT_ROOTS:
        return (
            _HISTORICAL_EXPERIMENT_ROOTS[parts[0]]
            .joinpath(*parts[1:])
            .resolve()
        )
    return (PROJECT_ROOT / relative_path).resolve()
