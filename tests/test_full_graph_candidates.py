import unittest

import torch
from torch import nn
from torch.nn import functional as F

from unibrain_agent.full_graph_candidates import build_graph_candidate
from unibrain_agent.paths import SOURCES


def has_source(repository, filename):
    return (SOURCES / repository / filename).is_file()


class FullGraphCandidateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)

    def check_update(self, model, tensor, objective):
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        loss = objective()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(tensor.grad)
        self.assertTrue(torch.isfinite(tensor.grad).all())
        self.assertGreater(tensor.grad.abs().sum().item(), 0)
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)
        before = [p.detach().clone() for p in model.parameters()]
        optimizer.step()
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())))

    @unittest.skipUnless(has_source("FBNETGEN", "model/model.py"), "Pinned FBNETGEN source required")
    def test_native_pair_mlp_forward_backward_and_update(self):
        model, evidence = build_graph_candidate("fbnetgen")
        self.assertEqual(model.core.fc_out.in_features, 256)
        self.assertEqual(model.core.fc_out.out_features, 128)
        self.assertEqual(evidence["parameters"], 33025)
        self.assertEqual(model.core.rel_rec.shape, (116 * 116, 116))
        self.assertIn("core.rel_rec", dict(model.named_buffers()))
        self.assertEqual(model.core.fc_cat.bias.shape, (1,))
        inputs = torch.randn(1, 116, 128, requires_grad=True)
        outputs = model(inputs)
        self.assertEqual(outputs.shape, (1, 116, 116))
        self.assertTrue(torch.isfinite(outputs).all())
        self.assertTrue(torch.equal(outputs, model.core(inputs).squeeze(-1)))
        self.check_update(model, inputs, lambda: model(inputs).square().mean())

    @unittest.skipUnless(has_source("FBNETGEN", "model/model.py"), "Pinned FBNETGEN source required")
    def test_native_brainnetcnn_forward_backward_and_update(self):
        model, evidence = build_graph_candidate("brainnetcnn")
        self.assertEqual(model.core.e2econv1.cnn1.out_channels, 32)
        self.assertEqual(model.core.e2econv2.cnn1.out_channels, 64)
        self.assertEqual(model.core.N2G.out_channels, 256)
        self.assertEqual(model.core.dense2.out_features, 30)
        inputs = torch.randn(1, 116, 116, requires_grad=True)
        features = torch.randn(1, 116, 128)
        self.assertEqual(model(inputs, features).shape, (1, 2))
        self.check_update(model, inputs, lambda: F.cross_entropy(model(inputs, features), torch.tensor([1])))
        model.eval()
        self.assertTrue(torch.equal(model(inputs), model(inputs)))
        self.assertIn("Third-party", evidence["provenance"])

    @unittest.skipUnless(has_source("BrainNetworkTransformer", "source/models/BNT/bnt.py"), "Pinned BNT source required")
    def test_native_bnt_forward_backward_and_update(self):
        model, evidence = build_graph_candidate("bnt")
        self.assertEqual(len(model.core.attention_list), 2)
        for layer in model.core.attention_list:
            self.assertEqual(layer.transformer.linear1.out_features, 1024)
            self.assertEqual(layer.transformer.self_attn.num_heads, 4)
        self.assertEqual(model.core.attention_list[-1].dec.cluster_number, 100)
        self.assertEqual(model.core.fc[0].in_features, 800)
        self.assertFalse(model.core.attention_list[-1].dec.assignment.cluster_centers.requires_grad)
        inputs = torch.randn(1, 116, 116, requires_grad=True)
        self.assertEqual(model(inputs).shape, (1, 2))
        self.check_update(model, inputs, lambda: F.cross_entropy(model(inputs), torch.tensor([1])))
        self.assertEqual(evidence["pooling_sizes"], [116, 100])
        self.assertLess(evidence["trainable_parameters"], evidence["parameters"])

    @unittest.skipUnless(has_source("FBNETGEN", "model/model.py"), "Pinned FBNETGEN source required")
    def test_rejects_compact_feature_and_node_interfaces(self):
        pair, _ = build_graph_candidate("fbnetgen")
        with self.assertRaises(ValueError):
            pair(torch.randn(1, 116, 16))
        classifier, _ = build_graph_candidate("brainnetcnn")
        with self.assertRaises(ValueError):
            classifier(torch.randn(1, 8, 8))


if __name__ == "__main__":
    unittest.main()
