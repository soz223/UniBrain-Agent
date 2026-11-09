import json
import ast
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from unibrain_agent.builds import generated_code, verify_build
from unibrain_agent.catalog import DEFAULTS, ADAPTER_DEFAULTS, catalog, validate_selection
from unibrain_agent.data import digest
from unibrain_agent.pipeline import loss_terms
from unibrain_agent.server import create_server
from unibrain_agent.sources import Compatibility
from unibrain_agent.report import png
from unibrain_agent.validation import spatial_checks


class CompositionTests(unittest.TestCase):
    def test_dropout_patch_leaves_module_calls_unchanged(self):
        tree = ast.parse("F.dropout(x, p=0.5)\nself.dropout(x)")
        patched = Compatibility().visit(tree)
        calls = [node.value for node in patched.body]
        self.assertEqual([key.arg for key in calls[0].keywords], ["p", "training"])
        self.assertEqual(calls[1].keywords, [])

    def test_catalog_preserves_fifteen_papers(self):
        library = catalog()
        ids = {option["id"] for stage in library["stages"] for option in stage["options"] if option["research"]}
        self.assertEqual(ids, {f"M{i:02}" for i in range(1, 16)})
        self.assertTrue(all(option["research"]["code"].startswith("https://github.com/")
                            for stage in library["stages"] for option in stage["options"] if option["research"]))

    def test_contracts_reject_wrong_stage_and_missing_mesh(self):
        for change in ({"registration": "M12"}, {"extraction": "M03"}, {"surface": "M15"}):
            with self.assertRaises(ValueError):
                validate_selection({**DEFAULTS, **change})
        self.assertEqual(validate_selection({**DEFAULTS, "extraction": "M03", "registration": "M03"})["extraction"], "M03")

    def test_generated_code_cannot_change_after_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            code = folder / "pipeline.py"
            code.write_text(generated_code(ADAPTER_DEFAULTS), encoding="utf-8")
            record = {"status": "ready", "validation": {"passed": True}, "selection": ADAPTER_DEFAULTS,
                      "code_sha256": digest(code), "implementation_hashes": {}}
            (folder / "build.json").write_text(json.dumps(record), encoding="utf-8")
            with patch("unibrain_agent.builds.verify", return_value=[]):
                verify_build(folder)
                code.write_text(code.read_text() + "\nprint('changed')", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "differs"):
                    verify_build(folder)

    def test_auxiliary_heads_receive_supervision(self):
        first = torch.randn(1, 4, 8, 8, 8, requires_grad=True)
        second = torch.randn(1, 4, 4, 4, 4, requires_grad=True)
        par_loss = torch.tensor(1., requires_grad=True)
        output = {"mask": torch.full((1, 1, 8, 8, 8), .5), "warped": torch.ones(1),
                  "tissue_logits": first, "tissue_aux": [second], "tissue_target": torch.ones(1, 1, 8, 8, 8),
                  "parcellation_loss": par_loss, "logits": torch.randn(1, 2, requires_grad=True)}
        sample = {"raw_img_brain_mask_np": torch.ones_like(output["mask"]), "label": torch.tensor([0])}
        core = type("Core", (), {"GCC": lambda self: lambda a, b: (a - b).square().mean()})()
        with patch("unibrain_agent.pipeline.unibrain_core", return_value=core):
            terms = loss_terms(output, sample, {"image": torch.zeros(1)})
            terms["segmentation"].backward()
        self.assertGreater(float(first.grad.norm()), 0)
        self.assertGreater(float(second.grad.norm()), 0)
        self.assertAlmostEqual(float(par_loss.grad), .2, places=6)

    def test_folded_grid_fails_even_with_valid_affine(self):
        theta = torch.eye(3, 4).unsqueeze(0)
        image = torch.zeros(1, 1, 96, 96, 96)
        grid = F.affine_grid(theta, image.shape, align_corners=True)
        output = {"mask": image, "warped": image, "affine": theta, "atlas": torch.ones(1, 1, 96, 96, 96),
                  "logits": torch.zeros(1, 2), "edges": torch.ones(1, 2, 2), "inverse_grid": grid,
                  "forward_grid": grid}
        self.assertTrue(spatial_checks(output)["passed"])
        output["forward_grid"] = grid * torch.tensor([-1., 1., 1.])
        self.assertFalse(spatial_checks(output)["passed"])


class ServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = create_server(0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_catalog_and_unknown_build(self):
        with urlopen(self.base + "/api/catalog") as response:
            self.assertEqual(json.load(response)["method_count"], 15)
        with self.assertRaises(HTTPError) as error:
            urlopen(self.base + "/api/builds/../../anything")
        self.assertEqual(error.exception.code, 404)

    def test_cross_origin_write_is_rejected(self):
        request = Request(self.base + "/api/prepare", data=b"{}", headers={"Origin": "https://example.org", "Content-Type": "application/json"})
        with self.assertRaises(HTTPError) as error:
            urlopen(request)
        self.assertEqual(error.exception.code, 403)

    def test_invalid_payload_cannot_start_run(self):
        request = Request(self.base + "/api/run", data=b'{"steps": true}', headers={"Content-Type": "application/json"})
        with self.assertRaises(HTTPError) as error:
            urlopen(request)
        self.assertEqual(error.exception.code, 400)
        self.assertFalse(self.server.state["running"])

    def test_volume_api_preserves_adjacent_slices_and_bounds(self):
        with tempfile.TemporaryDirectory() as directory, patch("unibrain_agent.server.RUNS", Path(directory)):
            folder = Path(directory) / "sample_run"
            folder.mkdir()
            volume = np.arange(5 * 6 * 7, dtype=np.float32).reshape(5, 6, 7) / 210
            np.savez(folder / "candidate_preview.npz", **{key: volume[None, None] for key in ("input", "mask", "extracted", "warped")})
            url = self.base + "/api/volume?run=sample_run&branch=candidate&stage=input&index="
            for index in range(5):
                with urlopen(url + str(index)) as response:
                    value = json.load(response)
                self.assertEqual(value["index"], index)
                self.assertEqual(value["shape"], [5, 6, 7])
                self.assertEqual(value["planes"], [png(np.rot90(volume[index, :, :])),
                                                  png(np.rot90(volume[:, index, :])), png(np.rot90(volume[:, :, index]))])
            for bad in ("-1", "5", "1.5", "0&index=1"):
                with self.assertRaises(HTTPError):
                    urlopen(url + bad)
            for bad in ("../sample_run", "%2Fsample_run", "sample_run%5C.."):
                with self.assertRaises(HTTPError):
                    urlopen(url.replace("run=sample_run", "run=" + bad) + "0")


if __name__ == "__main__":
    unittest.main()
