import base64
import struct
import tempfile
import types
import unittest
import zlib
from pathlib import Path

import numpy as np
import torch

from scripts.classroom_inference import install_cpu_affines, load_image, png_data


class ClassroomInferenceTests(unittest.TestCase):
    def test_cpu_affines_preserve_identity_and_composition(self):
        module = types.SimpleNamespace()
        install_cpu_affines(module)
        actual = module.get_batch_identity_theta_4_4(2)
        torch.testing.assert_close(actual, torch.eye(4).expand(2, 4, 4))
        self.assertEqual(actual.device.type, 'cpu')
        torch.testing.assert_close(module.get_batch_identity_theta_3_4(2), actual[:, :3])
        transform = actual.clone()
        transform[:, :3, 3] = torch.tensor([1., 2., 3.])
        torch.testing.assert_close(actual @ transform, transform)

    def test_single_image_excludes_identifiers_and_rejects_multiple_cases(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'sample.npy'
            data = np.zeros(1, dtype=[('id', 'U30'), ('raw_img_np', 'f4', (96, 96, 96))])
            data['id'] = 'private_identifier'
            data['raw_img_np'][0, 1, 2, 3] = 4
            np.save(path, data)
            loaded = load_image(path)
            self.assertEqual(loaded.shape, (96, 96, 96))
            self.assertEqual(loaded[1, 2, 3], 4)
            self.assertIsNone(loaded.dtype.names)
            np.save(path, np.repeat(data, 2))
            with self.assertRaises(ValueError):
                load_image(path)
            image = np.zeros((96, 96, 96), np.float32)
            image[0, 0, 0] = np.nan
            np.save(path, image)
            with self.assertRaises(ValueError):
                load_image(path)

    def test_generated_png_is_a_real_rgb_slice(self):
        volume = np.zeros((96, 96, 96), np.float32)
        volume[10:30, 20:40, 48] = 2
        for categorical in (False, True):
            result = png_data(volume, categorical)
            data = base64.b64decode(result.split(',', 1)[1])
            self.assertEqual(data[:8], b'\x89PNG\r\n\x1a\n')
            self.assertEqual(struct.unpack('>IIBB', data[16:26]), (288, 288, 8, 2))
            offset, payload = 8, b''
            while offset < len(data):
                length = struct.unpack('>I', data[offset:offset + 4])[0]
                kind = data[offset + 4:offset + 8]
                if kind == b'IDAT':
                    payload += data[offset + 8:offset + 8 + length]
                offset += length + 12
            pixels = np.frombuffer(zlib.decompress(payload), dtype=np.uint8).reshape(288, 865)
            self.assertTrue((pixels[:, 0] == 0).all())
            self.assertGreater(pixels[:, 1:].std(), 0)


if __name__ == '__main__':
    unittest.main()
