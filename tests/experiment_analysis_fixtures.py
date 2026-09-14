from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
from unittest import mock

from scripts.experiments import evaluate_quartile_experiments as quartile_launcher
from scripts.experiments.analysis import discovery


@contextmanager
def temporary_checkpoint_records(rows):
    """Exercise real checkpoint-state discovery with local fixtures."""
    with tempfile.TemporaryDirectory() as temporary_dir:
        root = Path(temporary_dir)
        reports = root / "original_reports"
        for weather, weather_rows in rows.items():
            records = {}
            for row in weather_rows:
                identity = discovery.row_identity(row)
                if identity is None:
                    continue
                group, seed = identity
                for branch in ("source", "target"):
                    task_id = f"{weather}_{group}_seed{seed}_{branch}"
                    checkpoint = root / "checkpoints" / task_id
                    checkpoint.mkdir(parents=True)
                    for epoch in range(5, 25):
                        (checkpoint / f"epoch_{epoch:03d}.pth").write_bytes(
                            b"fixture"
                        )
                    records[task_id] = {
                        "group": group,
                        "seed": seed,
                        "branch": branch,
                        "status": "completed",
                        "updated_at": "2026-01-02",
                        "checkpoint_root": str(checkpoint),
                        "report_path": str(
                            reports / weather / f"{task_id}.txt"
                        ),
                    }
                    records[task_id + "_stale"] = dict(
                        records[task_id],
                        updated_at="2026-01-01",
                        report_path=str(reports / "stale.txt"),
                    )
                    records[task_id + "_pending"] = dict(
                        records[task_id],
                        status="pending",
                        updated_at="2026-01-03",
                    )
            (root / f".{weather}_experiments.txt.queue_state.json").write_text(
                json.dumps({"tasks": records}),
                encoding="utf-8",
            )
        with mock.patch.object(
            quartile_launcher,
            "SOURCE_EXPERIMENT_DIR",
            root,
        ), mock.patch.object(
            quartile_launcher,
            "SOURCE_REPORT_ROOT",
            reports,
        ):
            yield root
