import ast
import gc
import hashlib
import inspect
import json
import unittest
from pathlib import Path

import torch

from models.factory import MODEL_TYPES, build_model


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIRECTORY = REPOSITORY_ROOT / "models"

EXPECTED_MODELS = [f"model{index}" for index in range(1, 17)]

EXPECTED_PUBLIC_MODELS = {
    "model1": (
        "RADRAEStageCenterPointModel",
        "(d_in=64, e_in=37, num_classes=2, decoder_hidden_channels=128)",
        142,
        "647713e781484f5bff5afbe102fe25c288fb522402d5efa5ada6d0c1f398598d",
    ),
    "model2": (
        "RADRAEBiFPNCenterPointModel",
        "(d_in=64, e_in=37, num_classes=2, decoder_hidden_channels=128, "
        "bifpn_channels=128, num_bifpn_blocks=1)",
        224,
        "651f4b222a2a604a535e47e7460b0e79f103bd0a5ad39e134edd406e3f590d0e",
    ),
    "model3": (
        "RADRAEFPNNoDeformCenterPointModel",
        "(d_in=64, e_in=37, num_classes=2, decoder_hidden_channels=128, "
        "fpn_channels=128)",
        238,
        "575d8a33c94c587bbe2e1e08d7a16e275b9a338e6fde643ca6864069eab5582c",
    ),
    "model4": (
        "RADRAEStageDeformCenterPointModel",
        "(d_in=64, e_in=37, num_classes=2, decoder_hidden_channels=128)",
        150,
        "c6dda581839bbfcebb2ecbfba7cd8e4970970bbe1d38d919952c25b2ea9871ff",
    ),
    "model5": (
        "RADRAEFPNDeformCenterPointModel",
        "(d_in=64, e_in=37, num_classes=2, decoder_hidden_channels=128, "
        "fpn_channels=128)",
        250,
        "cedad3463039aa1c439707c1201ca231d6268bdb7e78bee0c80caa4246bd0df4",
    ),
    "model6": (
        "RADRAEFPNQualityCenterPointModel",
        "(d_in=64, e_in=37, num_classes=2, decoder_hidden_channels=128, "
        "fpn_channels=128)",
        258,
        "74e2f2232805fd267f6117868f4360b66d4e1c116cde6c6fb954914b2cd2ef1b",
    ),
    "model7": (
        "RADRAESwinFPNCenterPointModel",
        "(d_in=64, e_in=37, num_classes=2, decoder_hidden_channels=128, "
        "fpn_channels=128, box_coordinate_mode='polar', loss_mode='auto')",
        338,
        "f31cb8db1451ce392176a167517a63123d5dee0eab673bb1295d41b154caf959",
    ),
    "model8": (
        "RADRAEFPNCFECenterPointModel",
        "(d_in=64, e_in=37, num_classes=2, decoder_hidden_channels=128, "
        "fpn_channels=128, return_features=False)",
        766,
        "e0e56ed7d00f2f3aecdd848a83b7515c3f494d8705586d7a683dd3f1876b6520",
    ),
    "model9": (
        "RADRAECFEBiFPNCenterPointModel",
        "(d_in=64, e_in=37, num_classes=2, decoder_hidden_channels=128, "
        "bifpn_channels=128, num_bifpn_blocks=1, return_features=False)",
        748,
        "19a8315f37922788fc14346e1713ce5996e7e30b4653d4e3b4ddd5e553c59cca",
    ),
    "model10": (
        "RADRAEFPNMultiFeatureCenterPointModel",
        "(d_in=64, e_in=37, num_classes=2, decoder_hidden_channels=128, "
        "fpn_channels=128)",
        286,
        "583139edc098be965250ffb54ddaa79396437c67772def4da8300e60e9ed321e",
    ),
    "model11": (
        "RADRAEQFLFPNCenterPointModel",
        "(d_in=64, e_in=37, num_classes=2, decoder_hidden_channels=128, "
        "fpn_channels=128)",
        251,
        "90f7724d1f4ddb75d88691a645da19a1fa2dbf02402222ae6d456f61ccd1d6c6",
    ),
    "model12": (
        "RADRAEYOLOXFPNCenterPointModel",
        "(d_in=64, e_in=37, num_classes=2, decoder_hidden_channels=64, "
        "fpn_channels=64)",
        265,
        "71f1263379baa8255614d890841088ee563bc4694068a499f4e1f762fd2d3757",
    ),
    "model13": (
        "RADRAERADENetCenterPointModel",
        "(d_in=64, e_in=37, num_classes=2, decoder_hidden_channels=128, "
        "dropout=0.0)",
        177,
        "4f9308224949dc3cc3d709696aa6d10a79440b40942be7d82160f43b71cc0177",
    ),
    "model14": (
        "RADRAESwinYOLOXCenterPointModel",
        "(d_in=64, e_in=37, num_classes=2, decoder_hidden_channels=96, "
        "fpn_channels=96)",
        325,
        "df302e3a526bcf4b41852c4f3d1637393e16cd6dfc155a308fa00b401a3611ad",
    ),
    "model15": (
        "RADRAERADENetOfficialModel",
        "(d_in=64, e_in=37, num_classes=2, decoder_hidden_channels=128, "
        "dropout=0.0)",
        177,
        "0eeaa1ec332d137ac9b27b7335be1c499592b3a18dfa5b3a3c6708f3fbc53c42",
    ),
    "model16": (
        "RADRAESwinRADENetOfficialModel",
        "(d_in=64, e_in=37, num_classes=2, decoder_hidden_channels=128, "
        "fpn_channels=128)",
        339,
        "63400b286fd6cc1cfb4a6e41cf5d02406455fa8879e37ac019335873f6c4af68",
    ),
}

EXPECTED_FORWARD_KEYS = {
    "model12": [
        "rad_feat",
        "rae_feat",
        "fused_feat",
        "cls_logits",
        "objectness_logits",
        "center_offset",
        "center_height",
        "size",
        "yaw",
        "box_reg",
    ],
    "model14": [
        "rad_feat",
        "rae_feat",
        "fused_feat",
        "cls_logits",
        "objectness_logits",
        "center_offset",
        "center_height",
        "size",
        "yaw",
        "box_reg",
    ],
    "model15": [
        "backbone_feat",
        "fused_feat",
        "heatmap",
        "regression",
    ],
    "model16": [
        "rad_feat",
        "rae_feat",
        "fused_feat",
        "heatmap",
        "regression",
    ],
}


def _model_implementation_files():
    return sorted(MODEL_DIRECTORY.glob("model*.py"))


def _is_model_implementation_import(node):
    if isinstance(node, ast.ImportFrom):
        module = (node.module or "").lstrip(".")
        return module.startswith("model_") or module.startswith("models.model_")
    if isinstance(node, ast.Import):
        return any(
            alias.name.startswith("models.model_")
            for alias in node.names
        )
    return False


def _state_dict_signature(model):
    entries = [
        (name, list(value.shape), str(value.dtype))
        for name, value in model.state_dict().items()
    ]
    payload = json.dumps(entries, separators=(",", ":")).encode()
    return len(entries), hashlib.sha256(payload).hexdigest()


class ModelArchitectureIndependenceTests(unittest.TestCase):
    def test_every_model_implementation_is_free_of_model_to_model_imports(self):
        files = _model_implementation_files()
        self.assertEqual(len(files), 16)

        violations = []
        for path in files:
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if _is_model_implementation_import(node):
                    violations.append(f"{path.name}:{node.lineno}")

        self.assertEqual(violations, [])

    def test_factory_constructors_and_state_dict_layouts_are_stable(self):
        self.assertEqual(
            sorted(MODEL_TYPES, key=lambda item: int(item[5:])),
            EXPECTED_MODELS,
        )

        for model_type in EXPECTED_MODELS:
            with self.subTest(model_type=model_type):
                expected = EXPECTED_PUBLIC_MODELS[model_type]
                model = build_model(model_type, torch.device("cpu"))
                self.assertEqual(type(model).__name__, expected[0])
                self.assertEqual(str(inspect.signature(type(model))), expected[1])
                self.assertEqual(_state_dict_signature(model), expected[2:])
                del model
                gc.collect()

    def test_representative_head_families_preserve_forward_output_keys(self):
        rad = torch.zeros(1, 64, 32, 32)
        rae = torch.zeros(1, 37, 32, 32)

        for model_type, expected_keys in EXPECTED_FORWARD_KEYS.items():
            with self.subTest(model_type=model_type):
                model = build_model(model_type, torch.device("cpu")).eval()
                with torch.no_grad():
                    outputs = model(rad, rae)
                self.assertEqual(list(outputs), expected_keys)
                del model
                gc.collect()


if __name__ == "__main__":
    unittest.main()
