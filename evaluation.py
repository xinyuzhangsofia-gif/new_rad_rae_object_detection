"""Compatibility evaluation CLI and public facade.

The standalone workflow lives in :mod:`eval.workflow`.  The root module keeps
the historical command and broad import surface available while callers migrate
to the focused modules in :mod:`eval`.
"""

# These imports intentionally preserve the historical public surface of
# ``evaluation.py``.  Several scripts and notebooks import helpers from this
# facade, even though their implementations are now in ``eval``.
import json
import os
from pathlib import Path

import tqdm

from eval.checkpoints import *  # noqa: F401,F403
from eval.decoding import *  # noqa: F401,F403
from eval.evaluation_config import *  # noqa: F401,F403
from eval.inference import *  # noqa: F401,F403
from eval.metrics_runner import *  # noqa: F401,F403
from eval.reporting import *  # noqa: F401,F403
from eval.runner import *  # noqa: F401,F403
from eval.workflow import main


if __name__ == "__main__":
    main()
