"""exp_06 approvals: the null template, the code digests, and fail-closed admission."""
import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from tools import exp06_profiles as profiles

REPO = Path(__file__).resolve().parents[1]
HEAD = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()


@pytest.fixture(scope='module')
def template():
    return profiles.load_approved_digests()


def test_the_template_loads_and_round_trips(template):
    approved, identity = template
    raw = json.loads(profiles.TEMPLATE_PATH.read_bytes())
    assert profiles.json_value(approved) == raw
    assert identity['path'] == str(profiles.TEMPLATE_PATH) and len(identity['sha256']) == 64
    assert approved['schema_version'] == profiles.SCHEMA_VERSION
    assert set(approved) == {'schema_version', 'code', 'reused', 'artifacts'}
    assert set(approved['code']) == set(profiles.CODE_KEYS)
    assert set(approved['reused']) == set(profiles.REUSED_KEYS)
    assert set(approved['artifacts']) == set(profiles.ARTIFACT_KEYS)


def test_only_the_pinned_exp03_evaluator_is_filled(template):
    approved, _ = template
    filled = {key for key, value in approved['code'].items() if value is not None}
    assert filled == {'evaluator_exp03'}
    assert approved['code']['evaluator_exp03'] == profiles.EVALUATOR_EXP03
    assert all(approved['reused'][key] is None for key in profiles.REUSED_KEYS
               if key != 'legacy_receipt')
    assert all(value is None for value in approved['reused']['legacy_receipt'].values())
    assert approved['artifacts']['gate_g1'] is None
    assert all(value is None for value in approved['artifacts']['epoch_012'].values())


def test_the_training_keys_are_the_launch_critical_ones():
    assert profiles.TRAINING_KEYS == ('trainer', 'finalize', 'recipe', 'smoke', 'encoder',
                                      'factory', 'launch_sh', 'profiles')
    assert set(profiles.TRAINING_KEYS) <= set(profiles.CODE_KEYS)


@pytest.mark.parametrize('damage', ['unknown_key', 'missing_key', 'not_a_digest', 'bad_schema',
                                    'unknown_section', 'reused_shape'])
def test_a_malformed_approvals_file_is_refused(tmp_path, template, damage):
    approved, _ = template
    value = profiles.json_value(approved)
    if damage == 'unknown_key':
        value['code']['invented'] = None
    elif damage == 'missing_key':
        value['code'].pop('trainer')
    elif damage == 'not_a_digest':
        value['code']['trainer'] = 'not a digest'
    elif damage == 'bad_schema':
        value['schema_version'] = '1'
    elif damage == 'unknown_section':
        value['extra'] = {}
    else:
        value['reused']['legacy_receipt'] = 'ckpt/exp06/receipt.json'
    path = tmp_path / 'approved_digests.json'
    path.write_text(json.dumps(value, indent=2) + '\n')
    with pytest.raises(ValueError, match='approved digests'):
        profiles.load_approved_digests(path)


@pytest.fixture(scope='module')
def computed():
    notes = []
    return profiles.compute_code_digests(REPO, HEAD, notes=notes), notes


def key_files(key):
    """The files one approval key is taken over, named without importing anything."""
    module, extra = profiles.CODE_SPECS[key]
    return ([module.replace('.', '/') + '.py'] if module else []) + list(extra)


def test_every_present_key_gets_a_digest_and_absent_modules_are_noted(computed):
    digests, notes = computed
    assert set(digests) == set(profiles.PRESENT_KEYS_NOW)
    assert all(len(value) == 64 and set(value) <= set('0123456789abcdef')
               for value in digests.values())
    skipped = {note.split(':')[0] for note in notes}
    assert skipped == set(profiles.CODE_KEYS) - set(digests)
    # A key is skipped only because its files are really absent -- re-derived here rather
    # than read back from the module's own constant. Naming the keys of a round that has
    # not landed would go stale the moment it does, as round 2b's did on this merge, so
    # what is pinned instead is that the merged round's keys are digested now.
    absent = {key for key in profiles.CODE_KEYS
              if any(not (REPO / name).is_file() for name in key_files(key))}
    assert skipped == absent
    assert {'haa_finetune', 'haa_eval', 'haa_pipeline_sh'} <= set(digests)


def test_the_shell_launcher_is_bound_as_a_file(computed):
    digests, _ = computed
    assert digests['launch_sh'] == profiles.file_digest(['tools/exp06_launch.sh'], REPO, HEAD)
    assert digests['launch_sh'] != digests['finalize']


def test_the_pinned_exp03_evaluator_digest_reproduces(computed):
    """The pin is recomputable: exp_03's twelve files are byte-identical at HEAD."""
    digests, _ = computed
    assert digests['evaluator_exp03'] == profiles.EVALUATOR_EXP03


def test_require_refuses_null_approvals_unless_exploratory(template, computed):
    approved, _ = template
    digests, _ = computed
    with pytest.raises(ValueError, match='not approved'):
        profiles.require(approved, profiles.TRAINING_KEYS, repo=REPO, commit=HEAD,
                         current=digests)
    deviations = profiles.require(approved, profiles.TRAINING_KEYS, repo=REPO, commit=HEAD,
                                  exploratory=True, current=digests)
    assert deviations and all('not approved' in deviation for deviation in deviations)


def test_require_admits_matching_digests_and_names_every_drift(template, computed):
    approved, _ = template
    digests, _ = computed
    filled = copy.deepcopy(profiles.json_value(approved))
    filled['code'].update({key: digests[key] for key in profiles.TRAINING_KEYS})
    assert profiles.require(filled, profiles.TRAINING_KEYS, repo=REPO, commit=HEAD,
                            current=digests) == []
    filled['code']['launch_sh'] = 'a' * 64
    with pytest.raises(ValueError, match='launch_sh'):
        profiles.require(filled, profiles.TRAINING_KEYS, repo=REPO, commit=HEAD,
                         current=digests)
    with pytest.raises(ValueError, match='unknown approval key'):
        profiles.require(filled, ('invented',), repo=REPO, commit=HEAD, current=digests)


@pytest.fixture
def committed(tmp_path):
    """A tiny repository with the null template committed at HEAD."""
    root = tmp_path / 'repo'
    (root / 'assets').mkdir(parents=True)
    path = root / 'assets/approved_digests.json'
    path.write_bytes(profiles.TEMPLATE_PATH.read_bytes())
    for command in (['init', '-q'], ['add', '-A'], ['-c', 'user.email=a@b', '-c', 'user.name=t',
                                                    'commit', '-q', '-m', 'approvals']):
        subprocess.run(['git'] + command, cwd=root, check=True)
    return root, path, subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root,
                                               text=True).strip()


def test_approvals_can_be_bound_to_their_committed_blob(committed):
    """Finding 2: a confirmatory run's approvals must be bytes a reviewer committed."""
    root, path, head = committed
    value, identity = profiles.load_approved_digests(path, repo=root, commit=head)
    assert value['schema_version'] == profiles.SCHEMA_VERSION
    assert identity['repo_relative'] == 'assets/approved_digests.json'
    assert identity['committed_at'] == head
    assert identity['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    unbound = profiles.load_approved_digests(path)[1]
    assert unbound['sha256'] == identity['sha256'] and 'committed_at' not in unbound


@pytest.mark.parametrize('damage,cause', [('outside', 'outside'), ('untracked', 'not tracked'),
                                          ('edited', 'committed'), ('unknown_commit', 'not tracked')])
def test_approvals_that_no_reviewed_commit_carries_are_refused(committed, tmp_path, damage, cause):
    root, path, head = committed
    if damage == 'outside':
        path = tmp_path / 'approved_digests.json'
        path.write_bytes(profiles.TEMPLATE_PATH.read_bytes())
    elif damage == 'untracked':
        path = root / 'assets/second.json'
        path.write_bytes(profiles.TEMPLATE_PATH.read_bytes())
    elif damage == 'edited':
        path.write_bytes(path.read_bytes() + b'\n')
    else:
        head = 'b' * 40
    with pytest.raises(ValueError, match=cause):
        profiles.load_approved_digests(path, repo=root, commit=head)


def test_binding_needs_both_a_repository_and_a_commit(committed):
    root, path, head = committed
    for repo, commit in ((root, None), (None, head)):
        with pytest.raises(ValueError, match='repository and the commit'):
            profiles.load_approved_digests(path, repo=repo, commit=commit)


@pytest.fixture(scope='module')
def record():
    """The populated record copy: the approvals a confirmatory run is admitted against."""
    if not profiles.APPROVED_DIGESTS_PATH.is_file():
        pytest.skip('the record asset lives in the main tree, not on this branch')
    return profiles.load_approved_digests(profiles.APPROVED_DIGESTS_PATH)


def tracked_at(commit, relative):
    """Whether that commit's tree carries the path; False when git cannot answer at all."""
    try:
        return subprocess.run(['git', 'cat-file', '-e', '{}:{}'.format(commit, relative)],
                              cwd=REPO, capture_output=True).returncode == 0
    except OSError:
        return False


def test_the_record_copy_is_schema_valid_and_keyed_like_the_template(record, template):
    """Plan section 6.4 fills `code` in a second reviewed commit, so the record copy is no
    longer the null template the tools ship; what must still hold is its schema and its
    key sets -- an approval the template does not name is one no producer ever reads."""
    approved, identity = record
    raw = profiles.APPROVED_DIGESTS_PATH.read_bytes()
    assert profiles.json_value(approved) == json.loads(raw)
    assert identity['path'] == str(profiles.APPROVED_DIGESTS_PATH)
    assert identity['sha256'] == hashlib.sha256(raw).hexdigest()
    template_value = template[0]
    assert approved['schema_version'] == template_value['schema_version']
    assert set(approved) == set(template_value)
    for section in ('code', 'reused', 'artifacts'):
        assert set(approved[section]) == set(template_value[section])
    for name in profiles.NESTED:
        section, key = name.split('.')
        assert set(approved[section][key]) == set(template_value[section][key])


def test_the_record_copy_binds_to_the_bytes_committed_at_head(record):
    """Finding 2's binding on the real asset: the approvals a run cites are a reviewed
    commit's bytes, not a working-tree edit made after the review."""
    if not tracked_at(HEAD, profiles.APPROVED_RELATIVE):
        pytest.skip('the record copy is not tracked at HEAD in this checkout')
    bound, identity = profiles.load_approved_digests(profiles.APPROVED_DIGESTS_PATH,
                                                     repo=REPO, commit=HEAD)
    assert identity['repo_relative'] == profiles.APPROVED_RELATIVE
    assert identity['committed_at'] == HEAD
    assert identity['sha256'] == record[1]['sha256']
    assert profiles.json_value(bound) == profiles.json_value(record[0])


def test_every_filled_record_digest_is_the_one_this_checkout_computes(record, computed):
    """A filled key must equal what is here now, and a key this checkout cannot compute --
    a later round's module, absent rather than named in any list kept here -- must be null.
    The null `reused` and `artifacts` sections are filled after the runs and stay null."""
    approved, _ = record
    digests, _ = computed
    filled = {key: value for key, value in approved['code'].items() if value is not None}
    assert filled, 'the record copy carries no approved code digest at all'
    assert {key: digests.get(key) for key in filled} == filled
    assert set(profiles.TRAINING_KEYS) <= set(filled)
    assert profiles.require(approved, profiles.TRAINING_KEYS, repo=REPO, commit=HEAD,
                            current=digests) == []
