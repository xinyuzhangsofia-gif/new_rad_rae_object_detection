"""Run-specific recovery state retained for interrupted historical jobs."""


EXPERIMENT_QUEUE_RESUME_CHECKPOINTS = {
    "overcast_022_group11_seed43_target": (
        "checkpoints/overcast/0807_train_seq9_13_test_seq22/"
        "0808_epoch_018.pth"
    ),
    "overcast_023_group12_seed43_source": (
        "checkpoints/overcast/0807_train_seq12_11_test_seq22/"
        "0808_epoch_014.pth"
    ),
}


HISTORICAL_EXPERIMENT_QUEUE_OVERRIDES = {
    "experiment_queue_resume_checkpoints": EXPERIMENT_QUEUE_RESUME_CHECKPOINTS,
}
