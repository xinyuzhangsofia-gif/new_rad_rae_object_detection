import torch


NUM_CLASSES = 2


def parse_gpu_ids(gpu_ids_text):
    return [
        int(gpu_id.strip())
        for gpu_id in gpu_ids_text.split(",")
        if gpu_id.strip() != ""
    ]


def select_device_and_gpus(gpu_ids_text):
    gpu_ids = parse_gpu_ids(gpu_ids_text)
    if torch.cuda.is_available() and len(gpu_ids) > 0:
        available_gpu_count = torch.cuda.device_count()
        unavailable_gpu_ids = [
            gpu_id for gpu_id in gpu_ids
            if gpu_id < 0 or gpu_id >= available_gpu_count
        ]
        if len(unavailable_gpu_ids) > 0:
            raise ValueError(
                f"Requested GPU ids {unavailable_gpu_ids}, "
                f"but only {available_gpu_count} CUDA device(s) are available."
            )
        return torch.device(f"cuda:{gpu_ids[0]}"), gpu_ids

    return torch.device("cpu"), []
