from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import main as train_main


def main():
    if len(sys.argv) > 1:
        raise ValueError(
            "sedan_only/train_sedan_only.py is now a compatibility wrapper. "
            "Edit train_cfg.py, then run: python train.py"
        )
    train_main()


if __name__ == "__main__":
    main()
