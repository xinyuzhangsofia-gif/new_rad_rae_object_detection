"""Machine and process settings shared by training and evaluation."""


TRAIN_RUNTIME_CONFIG = {
    "num_workers": 0,
    "gpu_ids": "0,",
    "post_training_eval_min_free_memory_mb": 4096,
}


EXPERIMENT_QUEUE_RUNTIME_CONFIG = {
    # One batch-size-32 training process per GPU.
    "experiment_queue_train_workers": 3,
    "experiment_queue_gpu_strategy": "isolated",
    "experiment_queue_train_gpu_slots": ("0", "1", "2"),
    # Evaluation begins only after training for the active seed completes.
    "experiment_queue_eval_workers": 9,
    "experiment_queue_eval_gpu_pool": "0,1,2",
    "experiment_queue_eval_max_per_gpu": 3,
    "experiment_queue_eval_batch_size": 32,
    "experiment_queue_eval_min_free_memory_mb": 1500,
    "experiment_queue_eval_reservation_memory_mb": 2500,
    "experiment_queue_poll_seconds": 1.0,
}


EVALUATION_RUNTIME_CONFIG = {
    "batch_size": 32,
    "num_workers": 0,
    "gpu_ids": "0,1,2",
    "cuda": "cuda:1",
}
