from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_resume import main as train_resume_main


def main():
    if len(sys.argv) > 1:
        raise ValueError(
            "sedan_only/train_resume.py is now a compatibility wrapper. "
            "Edit train_cfg.py, then run: python train_resume.py"
        )
    train_resume_main()


if __name__ == "__main__":
    main()
