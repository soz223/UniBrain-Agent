import unittest
import numpy as np
from unibrain_agent.research_metrics import region_metrics


class MetricTests(unittest.TestCase):
    def test_spacing_and_direction(self):
        a, b = np.zeros((5, 5, 5)), np.zeros((5, 5, 5))
        a[1, 2, 2], b[2, 2, 2] = 1, 1
        row = region_metrics(a, b, (2, 1, 1))["regions"][0]
        self.assertEqual(row["hd95"], 2)
        self.assertEqual(row["assd"], 2)
        self.assertEqual(row["dice"], 0)

    def test_empty_masks_are_counted_not_perfect(self):
        a, b = np.zeros((5, 5, 5)), np.zeros((5, 5, 5))
        a[1, 2, 2] = 1
        result = region_metrics(a, b, (1, 1, 1), labels=[1, 2])
        self.assertEqual(result["one_empty_regions"], 1)
        self.assertIsNone(result["regions"][1]["dice"])
        self.assertIsNone(result["mean"]["hd95"])

    def test_identical_mask_and_invalid_spacing(self):
        a = np.ones((5, 5, 5))
        self.assertEqual(region_metrics(a, a, (1, 1, 1))["mean"], {"dice": 1., "hd95": 0., "assd": 0.})
        with self.assertRaises(ValueError):
            region_metrics(a, a, (2, 1, 1), "voxel")


if __name__ == "__main__":
    unittest.main()
