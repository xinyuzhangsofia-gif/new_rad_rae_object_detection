"""Verify that root CLI modules delegate to their canonical workflows."""

import unittest

import evaluation
import train
import train_resume
import visualize
from eval.workflow import main as evaluation_workflow_main
from training.runner import main as training_workflow_main
from training.resume import main as resume_training_workflow_main
from visualization.workflow import main as visualization_workflow_main


class EntrypointTests(unittest.TestCase):
    def test_root_entrypoints_expose_their_canonical_main(self):
        self.assertIs(train.main, training_workflow_main)
        self.assertIs(evaluation.main, evaluation_workflow_main)
        self.assertIs(train_resume.main, resume_training_workflow_main)
        self.assertIs(visualize.main, visualization_workflow_main)


if __name__ == "__main__":
    unittest.main()
