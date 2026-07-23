"""Train with chronological internal validation from train_cfg_v2.py."""

from train import main
from train_cfg_v2 import TRAIN_CONFIG_V2


if __name__ == "__main__":
    main(TRAIN_CONFIG_V2)

