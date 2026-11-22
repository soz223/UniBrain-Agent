import unittest

from scripts.verify_code_release import verify


class CodeReleaseTests(unittest.TestCase):
    def test_accepts_matching_release(self):
        state = {'code': {'trainer.py': 'a'}, 'upstream_lf': {'model.py': 'b'}, 'packages': {'torch': '2.5.1'}}
        verify(state, state)

    def test_rejects_changed_missing_extra_code_and_dependencies(self):
        expected = {'code': {'trainer.py': 'a'}, 'upstream_lf': {'model.py': 'b'}, 'packages': {'torch': '2.5.1'}}
        for section, replacement in [('code', {}), ('code', {'trainer.py': 'c'}),
                                     ('code', {'trainer.py': 'a', 'extra.py': 'd'}),
                                     ('upstream_lf', {'model.py': 'c'}), ('packages', {'torch': '2.6'})]:
            with self.subTest(section=section, replacement=replacement):
                with self.assertRaisesRegex(ValueError, 'Release mismatch'):
                    verify({**expected, section: replacement}, expected)


if __name__ == '__main__':
    unittest.main()
