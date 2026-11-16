import json
import tempfile
import unittest
from pathlib import Path

from unibrain_agent.grouped_provenance import validate_protocol


class GroupedProvenanceTests(unittest.TestCase):
    def test_shared_family_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'split_protocol.json'
            record = {'version': 1, 'kind': 'family_grouped', 'master_sha256': 'a' * 64,
                      'family_metadata_sha256': 'b' * 64, 'family_disjoint_verified': True,
                      'members': [{'id': 'a', 'group': 0}, {'id': 'b', 'group': 1}, {'id': 'c', 'group': 2}],
                      'folds': [{'round': 1}]}
            path.write_text(json.dumps(record))
            indices = {'train': [0], 'val': [1], 'test': [2]}
            validate_protocol(path, 'a' * 64, 1, indices, ['a', 'b', 'c'])
            record['members'][2]['group'] = 0
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'Related groups'):
                validate_protocol(path, 'a' * 64, 1, indices)

    def test_missing_metadata_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'split_protocol.json'
            path.write_text(json.dumps({'version': 1, 'kind': 'family_grouped',
                                       'master_sha256': 'a' * 64, 'family_disjoint_verified': True}))
            with self.assertRaisesRegex(ValueError, 'family metadata'):
                validate_protocol(path, 'a' * 64, 1, {})
