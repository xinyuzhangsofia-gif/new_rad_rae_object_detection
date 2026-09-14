"""Stable data structures for table-driven domain-shift experiments."""

from dataclasses import dataclass


VALID_BRANCHES = ("source", "target")
QUEUE_TASK_IDENTITY_VERSION = 5


@dataclass(frozen=True)
class DomainShiftExperiment:
    row_number: int
    name: str
    seed: int
    shared_sequences: tuple
    source_sequences: tuple
    target_sequences: tuple
    test_sequences: tuple
    shared_parts: tuple
    source_parts: tuple
    target_parts: tuple
    test_parts: tuple
    half_selection: tuple
    source_complete: bool
    target_complete: bool

    def branch_complete(self, branch):
        if branch == "source":
            return self.source_complete
        if branch == "target":
            return self.target_complete
        raise ValueError(f"Unsupported experiment branch: {branch!r}")


@dataclass(frozen=True)
class ExperimentQueueTask:
    ordinal: int
    experiment: DomainShiftExperiment
    branch: str

    @property
    def label(self):
        return f"{self.experiment.name}/{self.branch}"
