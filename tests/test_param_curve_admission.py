import pytest
from exp05_fixture import exp05_fixture
from tools import param_curve as pc
from pathlib import Path


@pytest.mark.parametrize('name,legacy', [('CURVE_K8', False), ('CURVE_K1', True), ('YAW_K8_SEED42', False)])
def test_shared_admission_fixture_contract(exp05_fixture, name, legacy):
    fixture = exp05_fixture(name, legacy)
    groups = [(a, fixture.profile['num_shot'], fixture.paths[a['role']]) for a in fixture.profile['arms']]
    admitted = pc.admit_runs(fixture.profile, groups, fixture.approved, exploratory=True,
                            producer=fixture.producer, producer_key='producer_param_curve')
    assert [len(g) for g in admitted['groups']] == [len(fixture.profile['eval_seeds'])]*6
    assert all(d.endswith((': mutable_inputs names', ': evaluator approved closure', ' seed set'))
               for d in admitted['deviations'])


@pytest.mark.parametrize('legacy', [False, True])
def test_tier_contract_exp05_and_approved_legacy_m(exp05_fixture, legacy):
    f = exp05_fixture(m_exp04=legacy)
    for arm in f.profile['arms']:
        directory = f.paths[arm['role']][0]
        contract = pc.run_contract(directory, arm, f.profile, f.pins)
        expected = 'tools.exp04_eval' if legacy and arm['tier'] == 'M' else 'tools.exp05_eval'
        assert contract['evaluator'] == expected
        assert len(contract['waivers']) <= 1


@pytest.mark.parametrize('kind', ['tier', 'count', 'legacy', 'config_float', 'args_binding',
                                 'args_bytes', 'extra_binding', 'decomposition', 'unapproved_M'])
def test_extension_refusals(exp05_fixture, kind):
    f = exp05_fixture(m_exp04=kind == 'unapproved_M')
    arm = f.profile['arms'][2 if kind == 'unapproved_M' else 0]
    directory = Path(f.paths[arm['role']][0])
    fields = f.read(directory / 'eval_manifest.json')
    if kind in ('tier', 'legacy', 'config_float', 'count'):
        key, value = {'tier': ('tier', 'L'), 'legacy': ('legacy_M', True),
            'config_float': ('vit_dim', 256.0), 'count': ('param_counts', {})}[kind]
        fields[key] = value
    elif kind == 'args_binding':
        fields['mutable_inputs']['train_args']['path'] = str(directory / 'args.json')
    elif kind == 'args_bytes':
        Path(fields['mutable_inputs']['train_args']['path']).write_text('{}')
    elif kind == 'extra_binding':
        fields['mutable_inputs']['extra'] = fields['mutable_inputs']['train_args']
    elif kind == 'decomposition':
        fields['decomposition_batches'] = True
    else:
        f.pins['closures']['evaluator_exp04'] = None
    f.replace(directory / 'eval_manifest.json', fields)
    with pytest.raises((ValueError, OSError), match='tier|args|binding|decomposition|evaluator'):
        pc.run_contract(directory, arm, f.profile, f.pins)


@pytest.mark.parametrize('name,legacy', [('CURVE_K8', False), ('TARGETS_K1', True), ('YAW_K8_SEED42', False)])
def test_complete_admission(exp05_fixture, name, legacy):
    f = exp05_fixture(name, legacy)
    admitted = pc.admit(f.directories[::-1], f.profile, f.approved, f.producer)
    assert not admitted['deviations']
    assert len(admitted['groups']) == 6
    assert len(admitted['compatibility']) == len(f.directories)
    assert all(len(g) == len(f.profile['eval_seeds']) for g in admitted['groups'])


@pytest.mark.parametrize('kind', ['missing_seed', 'query', 'K', 'grid', 'checkpoint', 'sidecar',
                                 'producer', 'null', 'epoch', 'M_evaluator', 'duplicate'])
def test_fail_closed_admission(exp05_fixture, kind):
    f = exp05_fixture()
    path = Path(f.paths['M_simple' if kind == 'M_evaluator' else 'S_simple'][0])
    fields = f.read(path / 'eval_manifest.json')
    if kind == 'missing_seed':
        f.directories.pop()
    elif kind == 'duplicate':
        f.directories.append(f.directories[0])
    elif kind == 'sidecar':
        (path / 'completion.json').unlink()
    elif kind in ('producer', 'null'):
        f.pins['closures']['producer_param_curve'] = None if kind == 'null' else '0'*64
    elif kind == 'epoch':
        f.pins['checkpoints']['S_simple']['epoch'] = 11
    elif kind == 'query':
        sample = f.read(path / 'per_sample_yaw.json')
        sample['query'][0] = 'Room/different/query'
        f.rebind(path, sample=sample)
    else:
        if kind == 'M_evaluator':
            fields['source_closures']['entrypoint']['sha256'] = '0'*64
        else:
            key, value = {'K': ('num_shot', 1), 'grid': ('yaw_cols', [0, 32]),
                          'checkpoint': ('checkpoint_sha256', '0'*64)}[kind]
            fields[key] = value
        f.rebind(path, manifest=fields)
    with pytest.raises(ValueError, match='re-evaluate M' if kind == 'M_evaluator' else 'admission|approved'):
        pc.admit(f.directories, f.profile, f.approved, f.producer)
    admitted = pc.admit(f.directories, f.profile, f.approved, f.producer, exploratory=True)
    assert admitted['deviations']


def test_tier_bytes_cannot_change_between_contract_and_shared_admission(exp05_fixture, monkeypatch):
    f = exp05_fixture()
    original = pc.admit_runs
    def mutate(*args, **kwargs):
        directory = Path(f.paths['S_simple'][0])
        payloads = [f.read(directory / name) for name in
                    ('eval_manifest.json', 'completion.json', 'per_sample_yaw.json', 'metrics_yaw.json')]
        for payload in payloads:
            payload.get('meta', payload)['tier'] = 'L'
        f.replace(directory / 'completion.json', payloads[1])
        f.rebind(directory, manifest=payloads[0], sample=payloads[2], metrics=payloads[3])
        return original(*args, **kwargs)
    monkeypatch.setattr(pc, 'admit_runs', mutate)
    with pytest.raises(ValueError, match='contract input changed'):
        pc.admit(f.directories, f.profile, f.approved, f.producer)


def test_m_group_may_mix_individually_approved_evaluators(exp05_fixture):
    current, old = exp05_fixture(), exp05_fixture(m_exp04=True)
    source = Path(old.paths['M_simple'][0])
    target = Path(current.paths['M_simple'][0])
    fields = current.read(target / 'eval_manifest.json')
    old_fields = old.read(source / 'eval_manifest.json')
    fields['source_closures']['entrypoint'] = old_fields['source_closures']['entrypoint']
    # Identical fixture closure bytes in both roots produce the same approved digest.
    fields['mutable_inputs']['control_args'] = fields['mutable_inputs'].pop('train_args')
    current.rebind(target, manifest=fields)
    result = pc.admit(current.directories, current.profile, current.approved, current.producer)
    names = [v['evaluator'] for k, v in result['compatibility'].items() if '/M_simple_' in k]
    assert names.count('tools.exp04_eval') == 1 and names.count('tools.exp05_eval') == 4


@pytest.mark.parametrize('role', ['S_simple', 'S_cyl', 'L_simple', 'L_cyl'])
@pytest.mark.parametrize('kind', ['train_manifest', 'train_completion', 'launcher', 'epoch', 'digest', 'bytes'])
def test_training_evidence_refusals(exp05_fixture, role, kind):
    f = exp05_fixture()
    directory = Path(f.paths[role][0])
    fields = f.read(directory / 'eval_manifest.json')
    if kind in ('train_manifest', 'train_completion'):
        del fields['mutable_inputs'][kind]
        message = 'missing ' + kind
    else:
        binding = fields['mutable_inputs']['train_manifest' if kind == 'launcher' else 'train_completion']
        path = Path(binding['path'])
        value = f.read(path)
        if kind == 'launcher':
            value['source_closures']['launcher']['sha256'] = '0'*64
        elif kind == 'epoch':
            value['outputs']['epoch_011.pth'] = value['outputs'].pop('epoch_012.pth')
        else:
            value['outputs']['epoch_012.pth'] = '0'*64
        f.replace(path, value)
        if kind != 'bytes':
            binding['sha256'] = pc.provenance.sha256_file(path)
        message = 'training launcher closure' if kind == 'launcher' else 'train_completion'
    for path in f.paths[role]:
        other = f.read(Path(path) / 'eval_manifest.json')
        other['mutable_inputs'] = fields['mutable_inputs']
        f.rebind(Path(path), manifest=other)
    with pytest.raises(ValueError, match=message):
        pc.admit(f.directories, f.profile, f.approved, f.producer)


def test_equal_training_launchers_must_match_approved_pin(exp05_fixture):
    f = exp05_fixture()
    f.pins['closures']['training_launcher'] = '0'*64
    with pytest.raises(ValueError, match='training launcher closure'):
        pc.admit(f.directories, f.profile, f.approved, f.producer)


def rebind_args(f, directory, recorded):
    fields = f.read(directory / 'eval_manifest.json')
    binding = fields['mutable_inputs'].get('train_args', fields['mutable_inputs'].get('control_args'))
    path = Path(binding['path'])
    f.replace(path, recorded)
    binding['sha256'] = pc.provenance.sha256_file(path)
    for name in ('eval_manifest.json', 'completion.json', 'per_sample_yaw.json', 'metrics_yaw.json'):
        payload = fields if name == 'eval_manifest.json' else f.read(directory / name)
        meta = payload.get('meta', payload)
        if 'args_json_sha256' in meta:
            meta['args_json_sha256'] = binding['sha256']
        f.replace(directory / name, payload)
    f.rebind(directory)


@pytest.mark.parametrize('key', ['num_shot', 'max_len', 'lr', 'weight_decay', 'decay_epochs',
    'lr_gamma', 'epochs', 'batch_size', 'accum_steps', 'seed', 'tf32', 'yaw_aug', 'no_save',
    'vit_dim', 'vit_depth', 'vit_heads', 'vit_mlp_dim'])
@pytest.mark.parametrize('change', ['missing', 'value', 'type'])
def test_full_recipe_refusals(exp05_fixture, key, change):
    f = exp05_fixture()
    arm = f.profile['arms'][0]
    directory = Path(f.paths[arm['role']][0])
    recorded = f.read(Path(arm['checkpoint']).parent / 'args.json')
    value = recorded.pop(key)
    if change == 'value':
        recorded[key] = not value if type(value) is bool else value + 1
    elif change == 'type':
        recorded[key] = float(value) if type(value) is int else int(value)
    rebind_args(f, directory, recorded)
    with pytest.raises(ValueError, match='args'):
        pc.run_contract(directory, arm, f.profile, f.pins)


@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('index', [2, 3])
def test_historical_m_recipe_files_pass_unchanged(exp05_fixture, legacy, index):
    f = exp05_fixture(m_exp04=legacy)
    arm = f.profile['arms'][index]
    original = pc.REPO / pc.PROFILES['CURVE_K8']['arms'][index]['checkpoint']
    args_path = original.parent / 'args.json'
    before = args_path.read_bytes()
    directory = Path(f.paths[arm['role']][0])
    recorded = f.read(args_path)
    for path in f.paths[arm['role']]:
        rebind_args(f, Path(path), recorded)
    pc.admit(f.directories, f.profile, f.approved, f.producer)
    recorded['lr'] = 1
    rebind_args(f, directory, recorded)
    with pytest.raises(ValueError, match='args lr'):
        pc.run_contract(directory, arm, f.profile, f.pins)
    assert args_path.read_bytes() == before


@pytest.mark.parametrize('count', [1, 5])
def test_missing_m_runs_are_distinguished_from_invalid_runs(exp05_fixture, count):
    f = exp05_fixture()
    missing = f.paths['M_simple'][:count]
    with pytest.raises(ValueError) as error:
        pc.admit([d for d in f.directories if d not in missing], f.profile, f.approved, f.producer)
    assert 'M_simple missing' in str(error.value)
    assert 're-evaluate M' not in str(error.value)
