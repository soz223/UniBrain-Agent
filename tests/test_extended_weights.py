import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from unibrain_agent.full_extended import build_extended_model
from unibrain_agent.full_graph_candidates import build_graph_candidate
from unibrain_agent.paths import SOURCES
from unibrain_agent.replacement_training import parser, preflight


def host_fixture(*args, **kwargs):
    host = nn.Module()
    host.if_pred_aal = True
    return host, {}, {"unchanged_modules": {}}


class ExtendedWeightTests(unittest.TestCase):
    @unittest.skipUnless((SOURCES / "FBNETGEN/model/model.py").is_file(), "Pinned FBNETGEN required")
    def test_full_wrapper_checkpoint_loads_strictly(self):
        expected, _ = build_graph_candidate("fbnetgen")
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "candidate.pt"
            torch.save({"model_state_dict": expected.state_dict()}, checkpoint)
            with patch("unibrain_agent.full_extended.build_full_model", side_effect=host_fixture):
                actual, _, architecture = build_extended_model(Path(temporary), "fbnetgen", weights=checkpoint)
            self.assertEqual(set(expected.state_dict()), set(actual.graph_gen.state_dict()))
            for key, value in expected.state_dict().items():
                self.assertTrue(torch.equal(value, actual.graph_gen.state_dict()[key]))
            self.assertIsNotNone(architecture["weights_sha256"])

    @unittest.skipUnless((SOURCES / "FBNETGEN/model/model.py").is_file(), "Pinned FBNETGEN required")
    def test_bare_core_is_not_silently_relabelled_as_wrapper_weights(self):
        expected, _ = build_graph_candidate("brainnetcnn")
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "bare_core.pt"
            torch.save({"model_state_dict": expected.core.state_dict()}, checkpoint)
            with patch("unibrain_agent.full_extended.build_full_model", side_effect=host_fixture):
                with self.assertRaises(RuntimeError):
                    build_extended_model(Path(temporary), "brainnetcnn", weights=checkpoint)

    @unittest.skipUnless((SOURCES / "reg_ernet/model.py").is_file(), "Pinned ERNet required")
    def test_ernet_checkpoint_retains_both_native_modules(self):
        from unibrain_agent.full_anatomy_candidates import build_anatomy_modules
        modules, _ = build_anatomy_modules("ernet")
        expected = nn.ModuleDict(modules).state_dict()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "ernet_components.pt"
            torch.save({"model_state_dict": expected}, checkpoint)
            with patch("unibrain_agent.full_extended.build_full_model", side_effect=host_fixture):
                actual, _, _ = build_extended_model(Path(temporary), "ernet", weights=checkpoint)
            restored = nn.ModuleDict({name: getattr(actual, name) for name in actual.candidate_modules}).state_dict()
            self.assertEqual(set(expected), set(restored))
            self.assertTrue(any(key.startswith("ext_net.") for key in expected))
            self.assertTrue(any(key.startswith("reg_net.") for key in expected))
            self.assertTrue(all(torch.equal(value, restored[key]) for key, value in expected.items()))

    def test_extended_pretraining_is_explicitly_blocked_by_the_trainer(self):
        args = parser().parse_args(["--source", "source", "--data", "data", "--output", "output",
                                    "--method", "brainnetcnn", "--setting", "independent_finetuned"])
        with self.assertRaisesRegex(ValueError, "joint training only"):
            preflight(args)
        args.setting, args.initialization = "joint", "training_fold"
        with self.assertRaisesRegex(ValueError, "separate round trip validation"):
            preflight(args)


if __name__ == "__main__":
    unittest.main()
