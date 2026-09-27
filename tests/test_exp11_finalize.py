"""exp_11's finalizer: its own run types, registry, recipe and heading rules.

Plan v3 item 7 and Codex round-2 change 4. What is checked here is what exp_11 owns:
the serialized run types (an exp_06 child is never certifiable here and vice versa),
the registry and state-dict identities of the new backbones, the adapter-heading
binding of arms J/K, and the job spec that carries either cue.
"""
import json
from pathlib import Path

import pytest
import torch

from tools import exp06_finalize as legacy
from tools import exp11_finalize as final
from tools import exp11_profiles, exp11_smoke

REPO = Path(__file__).resolve().parents[1]
HEADING_DIR = REPO / 'ckpt/exp06/heading'
HAA_ROOT = Path.home() / 'data_cache/HAA_xrir'
ROOMS = ('class_room', 'complex_room', 'dampened_room', 'hallway')


def heading_entry(room):
    record = json.loads((HEADING_DIR / (room + '.json')).read_text())
    from tools import provenance
    return {'phi_deg': record['phi_deg'], 'k': record['k'], 'decision': record['decision'],
            'sha256': provenance.sha256_file(HEADING_DIR / (room + '.json')),
            'path': str(HEADING_DIR / (room + '.json'))}


def heading_args(field, backbone, rooms=('class_room', 'hallway'), **extra):
    if not HEADING_DIR.is_dir() or not HAA_ROOT.is_dir():
        pytest.skip('the heading records or the HAA cache are not in this checkout')
    args = {'rooms': list(rooms), 'val_rooms': None, 'backbone': backbone,
            'haa_root': str(HAA_ROOT), 'heading': None, 'adapter_heading': None,
            'adapter_phi_deg': None}
    args[field] = {room: heading_entry(room) for room in rooms}
    if field == 'adapter_heading':
        args['adapter_phi_deg'] = args[field][rooms[0]]['phi_deg']
    args.update(extra)
    return args


def test_run_types_and_entry_modules_are_exp11s_own():
    assert final.RUN_TYPES == ('exp11_train', 'exp11_smoke', 'exp11_haa_finetune',
                               'exp11_haa_eval', 'exp11_haa_job')
    assert set(final.RUN_TYPES) & set(legacy.RUN_TYPES) == set()
    assert final.PROVENANCE_RUN_TYPE['exp11_haa_finetune'] == 'exp11_haa_train'
    assert final.PROVENANCE_RUN_TYPE['exp11_haa_eval'] == 'exp11_haa_eval'
    # An exp_06 child's provenance run types are not exp_11's HAA ones.
    assert 'haa_train' not in final.PROVENANCE_RUN_TYPE.values()
    for run_type, module in final.ENTRY_MODULES.items():
        assert module.startswith('tools.exp11_'), (run_type, module)
    assert final.registry_sha256() != legacy.registry_sha256()


def test_the_cli_refuses_an_exp06_run_type_and_vice_versa():
    assert final.main(['--run-dir', str(REPO), '--run-type', 'exp11_train',
                       '--log', '/nonexistent', '--child-exit', '0']) == 2
    with pytest.raises(SystemExit):
        final.main(['--run-dir', str(REPO), '--run-type', 'haa_train',
                    '--log', '/nonexistent', '--child-exit', '0'])
    with pytest.raises(SystemExit):
        legacy.main(['--run-dir', str(REPO), '--run-type', 'exp11_haa_finetune',
                     '--log', '/nonexistent', '--child-exit', '0'])
    with pytest.raises(SystemExit):
        final.passed_main(['--run-dir', str(REPO), '--run-type', 'haa_train'])


def test_child_roles_map_paths_to_exp11_run_types():
    assert final.child_role('stage1') == 'exp11_haa_finetune'
    assert final.child_role('stage2_hallway') == 'exp11_haa_finetune'
    assert final.child_role('eval/hallway') == 'exp11_haa_eval'
    assert final.child_role('zeroshot/eval/class_room') == 'exp11_haa_eval'
    with pytest.raises(ValueError, match='unexpected child path'):
        final.child_role('stage3')
    assert set(final.EVIDENCE_OF) == set(final.CHILD_EXTRA) == set(final.HAA_CHILD_TYPES)


def test_state_keys_come_from_the_exp11_factory():
    keys = final.expected_state_keys('simple_adapter', 8)
    assert 'source_network.heading_proj.weight' in keys
    assert 'source_network.vit.to_patch_embedding.2.weight' in keys
    assert final.expected_state_keys('simple_oriented', 8) != keys
    assert final.expected_state_keys('simple', 8) == frozenset(
        legacy.expected_state_keys('simple', 8))


def test_checkpoint_keys_refuse_another_arms_weights(tmp_path):
    from model.xrir_exp11_registry import build_xrir_exp11
    small = dict(dim=32, depth=1, heads=2, mlp_dim=32, intermediate_ch=32,
                 image_size=(32, 512), patch_size=(16, 32))
    path = tmp_path / 'best.pth'
    with torch.random.fork_rng(devices=[]):
        torch.save(build_xrir_exp11('simple', 2, **small).state_dict(), str(path))
    with pytest.raises(ValueError, match='build_xrir_exp11'):
        final._checkpoint_keys(path, 'best.pth', 'simple_adapter', 2)


def test_heading_frame_binding_is_exp06s_per_room_contract():
    args = heading_args('heading', 'simple_oriented')
    heading, adapter, phi = final.frame_binding(args, args['rooms'], 'heading', REPO)
    assert set(heading) == set(args['rooms']) and adapter is None and phi is None
    assert all(entry['k'] == 128 for entry in heading.values())
    with pytest.raises(ValueError, match='conditions on the frame'):
        final.frame_binding(dict(args, adapter_heading=args['heading']), args['rooms'],
                            'heading', REPO)


def test_room_frame_without_an_adapter_binds_no_cue():
    args = heading_args('heading', 'simple')
    plain = dict(args, heading=None)
    assert final.frame_binding(plain, args['rooms'], 'room', REPO) == (None, None, None)
    with pytest.raises(ValueError, match='must not record a heading'):
        final.frame_binding(args, args['rooms'], 'room', REPO)
    with pytest.raises(ValueError, match='simple_adapter'):
        final.frame_binding(dict(plain, adapter_heading=args['heading'], adapter_phi_deg=-90.0),
                            args['rooms'], 'room', REPO)


def test_the_adapter_binds_one_shared_heading_for_every_room():
    args = heading_args('adapter_heading', 'simple_adapter')
    heading, adapter, phi = final.frame_binding(args, args['rooms'], 'room', REPO)
    assert heading is None and set(adapter) == set(args['rooms'])
    assert phi == args['adapter_phi_deg'] == -90.0
    with pytest.raises(ValueError, match='installs the adapter heading'):
        final.frame_binding(dict(args, adapter_phi_deg=0.0), args['rooms'], 'room', REPO)
    with pytest.raises(ValueError, match='adapter_phi_deg'):
        final.frame_binding(dict(args, adapter_phi_deg=None), args['rooms'], 'room', REPO)


def test_mixed_adapter_headings_are_refused(monkeypatch):
    """All four HAA records declare -90, so the rule is exercised on a stub cohort.

    ``heading_records`` is exp_11's own function and is stubbed here (never an exp_06
    global): the point under test is that ``frame_binding`` refuses a cohort whose bound
    records disagree, which no real pair of records can currently produce.
    """
    bound = {'class_room': {'phi_deg': -90.0}, 'hallway': {'phi_deg': 0.0}}
    monkeypatch.setattr(final, 'heading_records', lambda *a, **k: bound)
    args = {'backbone': 'simple_adapter', 'heading': None, 'adapter_phi_deg': -90.0}
    with pytest.raises(ValueError, match='one cue for every room'):
        final.frame_binding(args, ['class_room', 'hallway'], 'room', REPO)
    bound['hallway'] = {'phi_deg': -90.0}
    assert final.frame_binding(args, ['class_room', 'hallway'], 'room', REPO)[2] == -90.0


def test_a_tampered_heading_record_is_refused(tmp_path):
    args = heading_args('adapter_heading', 'simple_adapter')
    tampered = json.loads(json.dumps(args))
    tampered['adapter_heading'][args['rooms'][0]]['sha256'] = 'a' * 64
    with pytest.raises(ValueError, match='does not hash to the recorded sha256'):
        final.frame_binding(tampered, args['rooms'], 'room', REPO)
    missing = json.loads(json.dumps(args))
    missing['adapter_heading'].pop(args['rooms'][0])
    with pytest.raises(ValueError, match='record for every room'):
        final.frame_binding(missing, args['rooms'], 'room', REPO)


def job_spec(tmp_path, **overrides):
    spec = {'init': 'simple_or', 'backbone': 'simple_oriented', 'frame': 'heading',
            'init_sha256': 'a' * 64, 'seed': 0, 'rooms': list(ROOMS), 'expect': 'finetune',
            'heading': {room: 128 for room in ROOMS}}
    spec.update(overrides)
    path = tmp_path / 'job.json'
    path.write_text(json.dumps(spec))
    return path


def test_job_spec_accepts_either_cue_and_refuses_the_other(tmp_path):
    loaded = final.load_job_spec(str(job_spec(tmp_path)), 'finetune')
    assert loaded['heading'] == {room: 128 for room in ROOMS}
    assert len(loaded['job_spec_sha256']) == 64
    adapter = job_spec(tmp_path, backbone='simple_adapter', frame='room', heading=None,
                       init='control_adapter', adapter_heading={room: 128 for room in ROOMS},
                       adapter_phi_deg=-90.0)
    loaded = final.load_job_spec(str(adapter), 'finetune')
    assert loaded['adapter_phi_deg'] == -90.0
    with pytest.raises(ValueError, match='declares no adapter_heading'):
        final.load_job_spec(str(job_spec(tmp_path, adapter_heading={'hallway': 128})),
                            'finetune')
    with pytest.raises(ValueError, match='records no adapter_heading roll'):
        final.load_job_spec(str(job_spec(tmp_path, backbone='simple_adapter', frame='room',
                                         heading=None)), 'finetune')
    with pytest.raises(ValueError, match='only an adapter job spec'):
        final.load_job_spec(str(job_spec(tmp_path, adapter_phi_deg=-90.0)), 'finetune')
    with pytest.raises(ValueError, match='job spec backbone'):
        final.load_job_spec(str(job_spec(tmp_path, backbone='nonesuch')), 'finetune')
    with pytest.raises(ValueError, match='not the --expect'):
        final.load_job_spec(str(job_spec(tmp_path)), 'zeroshot')


def test_a_plain_room_frame_job_declares_no_cue_at_all(tmp_path):
    plain = job_spec(tmp_path, backbone='simple', frame='room', heading=None,
                     init='control')
    loaded = final.load_job_spec(str(plain), 'finetune')
    assert not loaded.get('heading') and not loaded.get('adapter_heading')


def test_the_diagnostic_receipt_contract_is_the_enumerated_kinds(tmp_path):
    assert set(exp11_smoke.KIND_NAMES) == {'probe', 'haa_train_smoke', 'haa_eval_smoke'}
    receipt = tmp_path / 'receipt.json'
    base = dict(schema_version=1, diagnostic=True, admissible_arm=False,
                runner=exp11_smoke.RUNNER, kind='probe', entry='tools.exp11_train',
                run_type='exp11_probe', exit_status=0, outcome='ok', exploratory=False,
                argv=['--no-save'], started_at='2026-09-26T00:00:00+00:00',
                ended_at='2026-09-26T00:01:00+00:00', wall_s=60.0, peak_bytes=0,
                alarm_seconds=900.0, max_gb=40.0, runner_closure_sha256='a' * 64)
    receipt.write_text(json.dumps(base))
    record, path, spec = final.diagnostic_receipt(str(receipt))
    assert record['kind'] == 'probe' and spec['entry'] == 'tools.exp11_train'
    # exp_11's diagnostic run types are disjoint from exp_06's, so neither family's
    # receipt can be read as the other's.
    assert not set(exp11_smoke.RUN_TYPES) & {'smoke', 'probe', 'haa_smoke_train',
                                             'haa_smoke_eval'}
    for mutate, match in ((dict(admissible_arm=True), 'never an admissible arm'),
                          (dict(kind='nonesuch'), 'unknown exp_11 diagnostic kind'),
                          (dict(alarm_seconds=1e6), 'above the 900'),
                          (dict(max_gb=80.0), 'above the 40'),
                          (dict(argv=[]), 'must run with --no-save'),
                          (dict(run_type='probe'), 'must record the entry'),
                          (dict(outcome='great'), 'records outcome')):
        receipt.write_text(json.dumps(dict(base, **mutate)))
        with pytest.raises(ValueError, match=match):
            final.diagnostic_receipt(str(receipt))


def test_the_approval_gate_refuses_the_all_null_record():
    import subprocess
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(REPO),
                                   text=True).strip()
    with pytest.raises(ValueError, match='do not admit this run|not approved'):
        final.approval_gate('full', head, str(exp11_profiles.APPROVED_DIGESTS_PATH),
                            False, REPO)
    with pytest.raises(ValueError, match='exploratory launch is a diagnostic'):
        final.approval_gate('full', 'a' * 40, None, True, REPO)


def test_neither_family_certifies_the_others_child(tmp_path):
    """The serialized run type is the wall: a child record of one family is refused by
    the other's ``child_completion``, both ways. The completion records the *finalizer's*
    run type (``exp11_haa_finetune``); the child's own provenance records
    ``exp11_haa_train``, which is what ``load_provenance`` compares."""
    def record(run_type, extra):
        body = {key: 'x' for key in legacy.CHILD_COMPLETION}
        body.update(schema_version=1, run_type=run_type, run_dir=str(tmp_path.resolve()),
                    child_exit=0, diagnostic=False, admissible_arm=True, artifacts={},
                    rooms=['hallway'], init_sha256='a' * 64, best_epoch=1, seed=0,
                    adapter_heading=None, adapter_phi_deg=None, **extra)
        (tmp_path / 'completion.json').write_text(json.dumps(body))
    record('exp11_haa_finetune', {})
    assert final.child_completion(tmp_path, 'stage1', 'exp11_haa_finetune')['seed'] == 0
    with pytest.raises(ValueError, match="not the 'haa_train' its path requires"):
        legacy.child_completion(tmp_path, 'stage1', 'haa_train')
    record('haa_train', {})
    assert legacy.child_completion(tmp_path, 'stage1', 'haa_train')['seed'] == 0
    with pytest.raises(ValueError, match="not the 'exp11_haa_finetune' its path requires"):
        final.child_completion(tmp_path, 'stage1', 'exp11_haa_finetune')
    # exp_11's own completion schema additionally requires the adapter fields.
    body = json.loads((tmp_path / 'completion.json').read_text())
    body.update(run_type='exp11_haa_finetune')
    body.pop('adapter_heading')
    (tmp_path / 'completion.json').write_text(json.dumps(body))
    with pytest.raises(ValueError, match='missing adapter_heading'):
        final.child_completion(tmp_path, 'stage1', 'exp11_haa_finetune')


def test_a_backbone_may_not_be_admitted_in_the_frame_it_does_not_read():
    """The finalizer refuses from the other side what the entry point refuses at launch."""
    args = heading_args('heading', 'simple_adapter')
    with pytest.raises(ValueError, match='conditions on its adapter'):
        final.frame_binding(args, args['rooms'], 'heading', REPO)
    plain = dict(heading_args('heading', 'simple_oriented'), heading=None)
    with pytest.raises(ValueError, match='reads the heading frame'):
        final.frame_binding(plain, plain['rooms'], 'room', REPO)
    assert final.HEADING_BACKBONES == ('cylindrical_oriented', 'simple_oriented')
    from tools import exp11_haa_finetune as entry
    assert entry.HEADING_BACKBONES == final.HEADING_BACKBONES
    assert entry.ADAPTER_BACKBONE == final.ADAPTER_BACKBONE


# --- blocker 3: the diagnostic receipt is bound to its own provenance ----------------


def diagnostic_pair(tmp_path, **receipt_changes):
    """One receipt and the provenance record it must be bound to."""
    closure = 'd' * 64
    record = {'run_type': 'exp11_probe', 'repo': str(REPO), 'reviewed_commit': 'e' * 40,
              'entry': 'tools.exp11_train', 'kind': 'probe', 'diagnostic': True,
              'admissible_arm': False, 'exploratory': False,
              'source_closures': {'diagnostic': {'entry_module': exp11_smoke.RUNNER,
                                                 'files': [], 'sha256': closure}},
              'git_state': {'HEAD': 'a' * 40}, 'environment': {},
              'command': ['probe', '--no-save'], 'registry_sha256': 'b' * 64}
    receipt = dict(schema_version=1, diagnostic=True, admissible_arm=False,
                   runner=exp11_smoke.RUNNER, kind='probe', entry='tools.exp11_train',
                   run_type='exp11_probe', exit_status=0, outcome='ok', exploratory=False,
                   argv=['--no-save'], started_at='2026-09-26T00:00:00+00:00',
                   ended_at='2026-09-26T00:01:00+00:00', wall_s=60.0, peak_bytes=0,
                   alarm_seconds=900.0, max_gb=40.0, runner_closure_sha256=closure,
                   git_head='a' * 40)
    receipt.update(receipt_changes)
    path = tmp_path / 'receipt.json'
    path.write_text(json.dumps(receipt))
    return str(path), record


def test_a_diagnostic_receipt_is_bound_to_its_provenance(tmp_path):
    path, record = diagnostic_pair(tmp_path)
    fields, _, _ = final.diagnostic_receipt(path)
    final.check_receipt_identity(fields, record)


def test_a_receipt_whose_runner_closure_is_not_its_provenances_is_refused(tmp_path):
    path, record = diagnostic_pair(tmp_path, runner_closure_sha256='9' * 64)
    fields, _, _ = final.diagnostic_receipt(path)
    with pytest.raises(ValueError, match='runner closure'):
        final.check_receipt_identity(fields, record)


def test_a_receipt_without_a_git_head_is_refused(tmp_path):
    body = json.loads(Path(diagnostic_pair(tmp_path)[0]).read_text())
    body.pop('git_head')
    (tmp_path / 'receipt.json').write_text(json.dumps(body))
    with pytest.raises(ValueError, match='missing git_head'):
        final.diagnostic_receipt(str(tmp_path / 'receipt.json'))


def test_a_receipt_whose_git_head_is_not_its_provenances_is_refused(tmp_path):
    path, record = diagnostic_pair(tmp_path, git_head='c' * 40)
    fields, _, _ = final.diagnostic_receipt(path)
    with pytest.raises(ValueError, match='git_head'):
        final.check_receipt_identity(fields, record)


def test_a_receipt_that_contradicts_its_provenance_about_exploratory_is_refused(tmp_path):
    path, record = diagnostic_pair(tmp_path, exploratory=True)
    fields, _, _ = final.diagnostic_receipt(path)
    with pytest.raises(ValueError, match='exploratory'):
        final.check_receipt_identity(fields, record)
