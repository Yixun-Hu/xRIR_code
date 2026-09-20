"""exp_09 may not move the bytes the completed exp_06 evaluations are verified against.

Code review round 1, finding 1: the ten simulated evaluations under ``ckpt/exp06/sim_eval``
record the closure digests they ran under, and ``exp06_compare.check_route`` re-checks
their ``writer_exp06`` digest against ``code.eval_launch`` of the approvals in force. Any
edit inside those closures -- ``tools/exp06_approvals_api.py`` is in three of them -- makes
ten completed runs unverifiable the moment this round's approvals are refilled, although
nothing those runs executed actually changed. exp_09's own code therefore lives outside
them, in ``tools/exp06_finalize.py``, which no evaluation writer imports.
"""
import json
import subprocess
from pathlib import Path

import pytest

from tools import exp06_profiles as profiles
from tools import provenance

REPO = Path(__file__).resolve().parents[1]
# The code keys whose closures the completed simulated evaluations are verified against.
SIM_EVAL_KEYS = ('eval', 'eval_launch', 'compare', 'mirror_probe')
WRITER_KEYS = ('eval_launch', 'compare', 'mirror_probe')     # the three that read approvals
SIM_EVAL_ROOT = REPO / 'ckpt' / 'exp06' / 'sim_eval'
SIM_EVAL_ARMS = ('cyl', 'cyl_or')
SIM_EVAL_SEEDS = tuple(range(42, 47))


def base_commit():
    """``main``: the tree this round branched from, and the one those runs were verified on."""
    probe = subprocess.run(['git', 'rev-parse', '--verify', '-q', 'main^{commit}'],
                           cwd=str(REPO), capture_output=True, text=True)
    if probe.returncode != 0:
        pytest.skip('no main ref in this checkout')
    return probe.stdout.strip()


def manifests():
    """The ten real evaluation manifests, or a skip on a checkout without the artefacts."""
    paths = [SIM_EVAL_ROOT / arm / 'seed{}'.format(seed) / 'eval_manifest.json'
             for arm in SIM_EVAL_ARMS for seed in SIM_EVAL_SEEDS]
    if any(not path.is_file() for path in paths):
        pytest.skip('the completed simulated evaluations are not in this checkout')
    return [(path, json.loads(path.read_text())) for path in paths]


def test_no_evaluation_writer_closure_contains_the_finalizer():
    """Where exp_09's resolver may live: a module none of these four keys imports."""
    for key in SIM_EVAL_KEYS:
        files = profiles._paths_of(key, str(REPO))
        assert 'tools/exp06_finalize.py' not in files, key
    for key in WRITER_KEYS:      # and where it may not: the shared approvals module
        assert 'tools/exp06_approvals_api.py' in profiles._paths_of(key, str(REPO)), key


def test_the_evaluation_writer_closures_are_still_the_bytes_main_carries():
    """Every closure a completed evaluation is checked against is unchanged by this round."""
    base = base_commit()
    head = provenance.git_state(str(REPO))['HEAD']
    assert (profiles.compute_code_digests(REPO, head, keys=SIM_EVAL_KEYS)
            == profiles.compute_code_digests(REPO, base, keys=SIM_EVAL_KEYS))


def test_the_completed_simulated_evaluations_still_check_out_at_this_tip():
    """The live regression: the recorded writer digest is the one a refill would publish,
    and every source file each run recorded still hashes to what it hashed then."""
    head = provenance.git_state(str(REPO))['HEAD']
    pinned = profiles.code_digest('eval_launch', str(REPO), head)
    for path, record in manifests():
        closures = record['source_closures']
        assert closures['writer_exp06']['sha256'] == pinned, path
        for role, closure in sorted(closures.items()):
            for entry in closure['files']:
                assert provenance.sha256_file(REPO / entry['path']) == \
                    entry['working_tree_sha256'], '{}: source.{}.{}'.format(
                        path, role, entry['path'])
