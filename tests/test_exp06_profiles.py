"""exp_06 approvals: the null template, the code digests, and fail-closed admission."""
import copy
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


def test_the_record_copy_is_byte_identical_when_it_is_present():
    """The committed record asset and the worktree copy the tools read must agree."""
    if not profiles.APPROVED_DIGESTS_PATH.is_file():
        pytest.skip('the record asset lives in the main tree, not on this branch')
    assert (profiles.APPROVED_DIGESTS_PATH.read_bytes()
            == profiles.TEMPLATE_PATH.read_bytes())


@pytest.fixture(scope='module')
def computed():
    notes = []
    return profiles.compute_code_digests(REPO, HEAD, notes=notes), notes


def test_every_present_key_gets_a_digest_and_absent_modules_are_noted(computed):
    digests, notes = computed
    assert set(digests) == set(profiles.PRESENT_KEYS_NOW)
    assert all(len(value) == 64 and set(value) <= set('0123456789abcdef')
               for value in digests.values())
    skipped = {note.split(':')[0] for note in notes}
    assert skipped == set(profiles.CODE_KEYS) - set(digests)
    assert 'haa_finetune' in skipped and 'haa_pipeline_sh' in skipped


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
