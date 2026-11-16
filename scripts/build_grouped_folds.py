"""Create versioned subject and family folds without changing source data."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


MISSING = {'', 'nan', 'na', 'none', 'unknown', '-999', '-9999'}


def value(raw):
    text = str(raw).strip()
    return None if text.lower() in MISSING else text


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            digest.update(block)
    return digest.hexdigest()


def components(rows, family=False):
    parent = list(range(len(rows)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owners = {}
    for i, row in enumerate(rows):
        if not value(row.get('subject', '')):
            raise ValueError('Missing subject ID')
        if family and not value(row.get('family', '')):
            raise ValueError('Missing family metadata; never assume an unrelated singleton')
        fields = ['subject', 'image_sha256'] + (['family', 'mother', 'father'] if family else [])
        for field in fields:
            item = value(row.get(field, ''))
            if item is not None:
                token = (field, item)
                if token in owners:
                    parent[root(i)] = root(owners[token])
                else:
                    owners[token] = i
    return np.array([root(i) for i in range(len(rows))])


def make_folds(rows, family=False, seed=17):
    groups = components(rows, family)
    labels = np.array([row['label'] for row in rows])
    if len(set(groups)) < 5:
        raise ValueError('At least five independent groups are required')
    buckets = np.full(len(rows), -1, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for k, (_, test) in enumerate(splitter.split(np.zeros(len(rows)), labels, groups)):
        buckets[test] = k
    folds = []
    for k in range(5):
        sets = {'test': np.flatnonzero(buckets == k), 'val': np.flatnonzero(buckets == (k + 1) % 5),
                'train': np.flatnonzero((buckets != k) & (buckets != (k + 1) % 5))}
        assert sorted(np.concatenate(list(sets.values())).tolist()) == list(range(len(rows)))
        names = list(sets)
        for a in range(3):
            for b in range(a):
                assert not set(groups[sets[names[a]]]) & set(groups[sets[names[b]]])
        if any(len(set(labels[index])) != 2 for index in sets.values()):
            raise ValueError('Every split must contain both classes')
        folds.append(sets)
    return folds, groups


def write_folds(destination, rows, folds, groups):
    summaries = []
    for k, sets in enumerate(folds, 1):
        folder = destination / 'rounds' / f'round{k}'
        folder.mkdir(parents=True)
        summary = {'round': k}
        for split, indices in sets.items():
            path = folder / f'{split}_indices.npy'
            np.save(path, indices)
            with (folder / f'{split}_index.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['index', 'id', 'label'])
                writer.writerows((int(i), rows[i]['subject'], rows[i]['label']) for i in indices)
            summary[split] = {'count': len(indices), 'groups': len(set(groups[indices])),
                              'class_counts': {str(c): sum(rows[i]['label'] == c for i in indices) for c in (0, 1)},
                              'sha256': sha(path)}
        summaries.append(summary)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cohort', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--family-csv', type=Path)
    parser.add_argument('--subject-column', default='Subject')
    parser.add_argument('--family-column', default='Family_ID')
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output must be a new versioned directory')
    master_path = args.cohort / 'unibrain_master.npy'
    master = np.load(master_path, mmap_mode='r', allow_pickle=False)
    before = master_path.stat()
    rows = []
    for i in range(len(master)):
        sid = master[i]['id']
        sid = sid.decode() if isinstance(sid, bytes) else str(sid)
        rows.append({'subject': sid, 'label': int(master[i]['label']),
                     'image_sha256': hashlib.sha256(master[i]['raw_img_np'].tobytes()).hexdigest()})
    if any(row['label'] not in (0, 1) for row in rows):
        raise ValueError('Expected binary sex labels')
    family_hash = None
    if args.family_csv:
        family_hash = sha(args.family_csv)
        with args.family_csv.open(encoding='utf-8-sig') as stream:
            reader = csv.DictReader(stream)
            if not {args.subject_column, args.family_column} <= set(reader.fieldnames):
                raise ValueError('Required family columns are absent')
            lookup = {}
            for r in reader:
                sid = r[args.subject_column].strip()
                record = {key: r.get(col, '') for key, col in [('family', args.family_column),
                          ('mother', 'Mother_ID'), ('father', 'Father_ID')]}
                if sid in lookup and lookup[sid] != record:
                    raise ValueError('Conflicting family records')
                lookup[sid] = record
        for row in rows:
            row.update(lookup.get(row['subject'], {}))
    missing = sum(not value(r.get('family', '')) for r in rows)
    old = []
    for k in range(1, 6):
        indices = {s: np.load(args.cohort / 'rounds' / f'round{k}' / f'{s}_indices.npy') for s in ('train', 'val', 'test')}
        valid = sorted(np.concatenate(list(indices.values())).tolist()) == list(range(len(rows)))
        result = {'round': k, 'exact_partition': valid}
        if not valid:
            raise ValueError('Existing indices do not partition the master')
        for field in ['subject', 'image_sha256'] + ([] if missing else ['family']):
            sets = [{rows[i][field] for i in index} for index in indices.values()]
            result[field + '_cross_split_overlap'] = sum(len(sets[a] & sets[b]) for a in range(3) for b in range(a))
        old.append(result)
    args.output.mkdir(parents=True)
    report = {'source_master_sha256': sha(master_path), 'subjects': len(set(r['subject'] for r in rows)),
              'rows': len(rows), 'seed': args.seed, 'old_folds': old, 'family_metadata_sha256': family_hash,
              'missing_family_rows': missing, 'source_modified': False, 'policies': {
                  'subject': 'All sessions, visits, modalities, crops and augmentations inherit their subject group.',
                  'duplicate': 'Identical input arrays are grouped even across different subject IDs.',
                  'family': 'Connected components of family, mother, father, subject and duplicate image identities.',
                  'temporal': 'Within-subject visits cannot cross splits. Prospective date holdout requires actual dates.',
                  'preprocessing': 'Population-fitted transforms, feature selection, harmonization and calibration fit training only.',
                  'evaluation': 'Validation selects checkpoints. Test never selects settings. Retrain from fresh initialization.',
                  'external': 'Audit pretraining cohort overlap and external template provenance separately.'}}
    for name, family in [('subject_grouped', False), ('family_grouped', True)]:
        if family and missing:
            report[name] = {'status': 'blocked', 'reason': 'Missing verified family metadata; no folds emitted'}
            continue
        folds, groups = make_folds(rows, family, args.seed)
        summary = write_folds(args.output / name, rows, folds, groups)
        protocol = {'version': 1, 'kind': name, 'seed': args.seed,
                    'master_sha256': report['source_master_sha256'],
                    'family_metadata_sha256': family_hash if family else None,
                    'family_disjoint_verified': family, 'folds': summary,
                    'members': [{'id': row['subject'], 'group': int(groups[i])} for i, row in enumerate(rows)]}
        (args.output / name / 'split_protocol.json').write_text(json.dumps(protocol, indent=2))
        report[name] = {'status': 'created', 'family_disjoint_verified': family, 'groups': len(set(groups)), 'folds': summary}
    after = master_path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Source changed while auditing')
    (args.output / 'audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'output': str(args.output), 'family_status': report['family_grouped']['status']}))


if __name__ == '__main__':
    main()
