"""Allow only audited normalization and coordinator-release migrations."""

import copy

PAPER = '8aeb39c34935d9c720a5f780b1c2ea2bb7991d3bd39d7764de245a76b2a04e3f'
RELEASES = {
    'replacement_training.py': {'7882d54516b57cde663e9ee2a39d0fc9104323b8e1d6dd6b6ec3371a3df66666',
                                'acfbf81c0209494e0d51faad04b80c695bb77fba0c19ea91d4fbacd8f5acd35e'},
    'anchored_taskforce.py': {'1f40a28c771140814b38a61ef8efaf07936293fbbfeeffef51e38a495dd07207',
                             '95f555b6932e94c193b24446057ada33cad9a3b4564391ef420e96fd5940a711'},
    'transfer_reward.py': {'9f1dfbb22e427c8a427d6f513c3ab9321cec14309bd853b25ea4d4cc17bafe06',
                          '717923e8c98b47c7b67671642ef5003702c5ec2a93a3d10dffebe09722bd62f9'},
}


def verify_migration(previous, current):
    if (not previous.get('paper_original') or previous.get('paper_implementation') != PAPER
            or previous.get('registration_normalization', 'stored') != 'stored'
            or current.get('registration_normalization') != 'per_image'):
        raise ValueError('Unsupported normalization migration source')
    migrated = copy.deepcopy(previous)
    migrated['paper_implementation'] = current['paper_implementation']
    migrated['registration_normalization'] = 'per_image'
    hashes = migrated['implementation_hashes']
    for name, allowed in RELEASES.items():
        if name in hashes:
            if hashes[name] not in allowed:
                raise ValueError('Unaudited migration source: ' + name)
            hashes[name] = current['implementation_hashes'][name]
    for name in ['registration_normalization.py', 'normalization_migration.py']:
        hashes[name] = current['implementation_hashes'][name]
    if previous.get('coordination') == 'none' and current.get('coordination') == 'taskforce':
        if current.get('taskforce_gradient_normalization') != 'anchored' or current.get('taskforce_profile') != 'paper':
            raise ValueError('Warm-start comparator requires the audited anchored coordinator')
        for key in ('coordination', 'taskforce_profile', 'taskforce_gradient_normalization',
                    'taskforce_action_bound', 'coordination_reward', 'coordination_bootstrap_epochs'):
            if key in current:
                migrated[key] = current[key]
        for name in ('taskforce_paper.py', 'training_stability.py', 'anchored_taskforce.py', 'transfer_reward.py'):
            hashes[name] = current['implementation_hashes'][name]
    if migrated.get('coordination_reward') == 'cross_subject':
        migrated['coordination_reward'] = 'classification_advantage'
    if migrated != current:
        changed = [key for key in migrated.keys() | current.keys() if migrated.get(key) != current.get(key)]
        raise ValueError('Migration cannot change unrelated configuration: ' + ', '.join(changed))
