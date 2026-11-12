import io
import unittest

import torch
from torch import nn

from unibrain_agent.candidate_integration import GeneratedCore
from unibrain_agent.replacement_training import candidate_export_core, parser, preflight


class CandidateExportTests(unittest.TestCase):
    def test_all_standalone_exports_strictly_reload_native_parameter_names(self):
        for method in ("synthstrip", "voxelmorph", "lapirn", "nnunet"):
            native, model = nn.Linear(2, 2), nn.Module()
            if method in ("voxelmorph", "lapirn"):
                model.reg_net = nn.Module()
                model.reg_net.core = GeneratedCore(native, "def adapt(core, moving, fixed):\n    return core(moving, fixed)\n", method)
            elif method == "nnunet":
                model.seg_net = GeneratedCore(native, "def adapt(core, image):\n    return core(image)\n", method)
            else:
                model.ext_net = nn.Module()
                model.ext_net.core = native
            exported = candidate_export_core(model, method)
            self.assertIs(exported, native)
            buffer = io.BytesIO()
            torch.save({"model_state_dict": exported.state_dict()}, buffer)
            buffer.seek(0)
            restored = nn.Linear(2, 2)
            restored.load_state_dict(torch.load(buffer, weights_only=True)["model_state_dict"], strict=True)
            value = torch.randn(3, 2)
            torch.testing.assert_close(native(value), restored(value))

    def test_synthstrip_new_only_rejects_missing_trained_host_before_data_loading(self):
        args = parser().parse_args(["--source", "missing", "--data", "missing", "--output", "out",
                                    "--method", "synthstrip", "--setting", "joint", "--trainable", "new"])
        with self.assertRaisesRegex(ValueError, "shared trained host"):
            preflight(args)


if __name__ == "__main__":
    unittest.main()
