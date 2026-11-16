import importlib.util
from pathlib import Path

import unittest


spec = importlib.util.spec_from_file_location('grouped', Path(__file__).parents[1] / 'scripts/build_grouped_folds.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_transitive_family_and_duplicate_links():
    rows = [dict(subject='a', family='f1', mother='m1'),
            dict(subject='b', family='f2', mother='m1', image_sha256='same'),
            dict(subject='c', family='f3', image_sha256='same'),
            dict(subject='d', family='f4')]
    groups = module.components(rows, True)
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]


def test_missing_family_rejected():
    with unittest.TestCase().assertRaisesRegex(ValueError, 'Missing family'):
        module.components([dict(subject='a', family='')], True)


def test_all_visits_and_families_stay_together():
    rows = [dict(subject=str(i // 2), family=str(i // 4), label=i % 2) for i in range(80)]
    folds, groups = module.make_folds(rows, True)
    tested = []
    for split in folds:
        tested.extend(split['test'].tolist())
        sets = [set(groups[index]) for index in split.values()]
        assert all(not sets[a] & sets[b] for a in range(3) for b in range(a))
    assert sorted(tested) == list(range(80))


if __name__ == '__main__':
    suite = unittest.TestSuite(unittest.FunctionTestCase(test) for test in
                               (test_transitive_family_and_duplicate_links, test_missing_family_rejected,
                                test_all_visits_and_families_stay_together))
    result = unittest.TextTestRunner().run(suite)
    raise SystemExit(not result.wasSuccessful())
