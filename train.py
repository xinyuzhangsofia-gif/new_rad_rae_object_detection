"""Compatibility training CLI.

The implementation lives in :mod:`training_utils.runner`.  This module stays
at the repository root so existing commands, notebooks, queue workers, and
saved experiment instructions can continue to use ``python train.py`` and
``from train import ...`` unchanged.
"""

from training_utils.runner import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
