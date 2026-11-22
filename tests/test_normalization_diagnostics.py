import unittest

import torch

from unibrain_agent.normalization_diagnostics import sample_statistics


class NormalizationDiagnosticsTests(unittest.TestCase):
    def test_matches_training_statistics_and_restores_state_after_error(self):
        layer = torch.nn.BatchNorm3d(2).eval()
        layer.running_mean.fill_(10.)
        state = {k: v.clone() for k, v in layer.state_dict().items()}
        image = torch.randn(1, 2, 4, 4, 4)
        expected = torch.nn.functional.batch_norm(image, None, None, layer.weight,
                                                  layer.bias, training=True, eps=layer.eps)
        with self.assertRaisesRegex(RuntimeError, 'probe'):
            with sample_statistics(layer):
                torch.testing.assert_close(layer(image), expected)
                raise RuntimeError('probe')
        self.assertFalse(layer.training)
        self.assertTrue(layer.track_running_stats)
        for key, value in layer.state_dict().items():
            torch.testing.assert_close(value, state[key])


if __name__ == '__main__':
    unittest.main()
