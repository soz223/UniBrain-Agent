"""Validate group membership and index bindings for a versioned split."""

import json

from .cohort_provenance import file_hash, checksum


def validate_protocol(path, master_hash, fold, indices, subjects=None):
    record = json.loads(path.read_text(encoding='utf-8'))
    family = record.get('family_disjoint_verified')
    if (record.get('version') != 1 or record.get('master_sha256') != master_hash
            or type(family) is not bool or record.get('kind') != ('family_grouped' if family else 'subject_grouped')):
        raise ValueError('Split protocol identity mismatch')
    if family and not checksum(record.get('family_metadata_sha256')):
        raise ValueError('Missing family metadata provenance')
    members = record['members']
    if len(set(m['id'] for m in members)) != len(members):
        raise ValueError('Duplicate subject in split protocol')
    if subjects is not None and [m['id'] for m in members] != subjects:
        raise ValueError('Protocol subject order differs from master')
    if sorted(i for part in indices.values() for i in part) != list(range(len(members))):
        raise ValueError('Protocol indices do not partition the master')
    declared = record['folds'][fold - 1]
    if declared['round'] != fold:
        raise ValueError('Protocol fold mismatch')
    groups = []
    for split, index in indices.items():
        groups.append({members[int(i)]['group'] for i in index})
    if any(groups[a] & groups[b] for a in range(3) for b in range(a)):
        raise ValueError('Related groups cross split boundaries')
    return record
