"""Blocker 1: the adapter arms' ZERO-SHOT evaluations must be finalizable.

The J/K queue evaluates the historical A/E checkpoint before any fine-tuning, so that
child's ``--checkpoint`` is the **base** xRIR state (266 tensors), not the adapter model's
(268). The loader already applies the contextual contract of plan v3 section 2.3
(``load_base_checkpoint`` for an A/E initialisation, strict adapter state afterwards);
admission has to apply the same one, or the required zero-shot jobs can never complete
and Phase 1b and the final publication are blocked.

The contract admission must enforce:

* an adapter **zero-shot** evaluation -- the child evaluates the very initialisation its
  job spec declares, i.e. ``checkpoint_sha256 == job_init_sha256`` -- validates the
  checkpoint strictly against the **pinned SimpleViT** parameter set and requires the
  adapter to be absent;
* every later adapter stage or evaluation checkpoint requires the complete adapter state
  strictly, so a fine-tuned J/K evaluation whose checkpoint lost its adapter is refused.
"""
import json
from pathlib import Path

import pytest
import torch

from model.xRIR_simple_adapter import ADAPTER_KEYS
from model.xrir_exp11_registry import build_xrir_exp11
from tools import exp11_finalize as final
from tools import provenance

SMALL = dict(dim=32, depth=1, heads=2, mlp_dim=32, intermediate_ch=32,
             image_size=(32, 512), patch_size=(16, 32))
NUM_SHOT = 2


def checkpoint(tmp_path, backbone, name):
    path = tmp_path / name
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        torch.save(build_xrir_exp11(backbone, NUM_SHOT, **SMALL).state_dict(), str(path))
    return path, provenance.sha256_file(path)


def args_for(checkpoint_path, init_digest, backbone='simple_adapter'):
    """The fields the evaluation admission reads when it decides the contract."""
    return {'backbone': backbone, 'num_shot': NUM_SHOT,
            'checkpoint': str(checkpoint_path), 'job_init': 'control_adapter',
            'job_init_sha256': init_digest, 'job_spec_sha256': 'a' * 64}


def test_the_pinned_base_and_adapter_parameter_sets_really_differ():
    base = final.expected_state_keys('simple', 8)
    adapter = final.expected_state_keys('simple_adapter', 8)
    assert len(adapter) == len(base) + len(ADAPTER_KEYS)
    assert set(ADAPTER_KEYS) <= set(adapter) and not set(ADAPTER_KEYS) & set(base)


@pytest.mark.parametrize('arm_init', ['control_adapter', 'yawaug_adapter'])
def test_an_adapter_zero_shot_evaluation_is_admitted(tmp_path, arm_init):
    """J and K both evaluate the A/E base checkpoint before any fine-tuning."""
    path, digest = checkpoint(tmp_path, 'simple', 'epoch_012.pth')
    args = dict(args_for(path, digest), job_init=arm_init)
    assert final.checkpoint_contract(args, digest) == 'base'
    final.verify_evaluation_checkpoint(path, 'checkpoint', args, digest)


def test_a_fine_tuned_adapter_evaluation_requires_the_whole_adapter(tmp_path):
    """A stage-2 checkpoint that lost its adapter is refused, not quietly admitted."""
    init, init_digest = checkpoint(tmp_path, 'simple', 'epoch_012.pth')
    stage2, stage2_digest = checkpoint(tmp_path, 'simple', 'best.pth')
    args = args_for(stage2, init_digest)
    assert final.checkpoint_contract(args, stage2_digest) == 'adapter'
    with pytest.raises(ValueError, match='build_xrir_exp11'):
        final.verify_evaluation_checkpoint(stage2, 'checkpoint', args, stage2_digest)


def test_a_fine_tuned_adapter_evaluation_with_its_adapter_is_admitted(tmp_path):
    init, init_digest = checkpoint(tmp_path, 'simple', 'epoch_012.pth')
    stage2, stage2_digest = checkpoint(tmp_path, 'simple_adapter', 'best.pth')
    args = args_for(stage2, init_digest)
    assert final.checkpoint_contract(args, stage2_digest) == 'adapter'
    final.verify_evaluation_checkpoint(stage2, 'checkpoint', args, stage2_digest)


def test_a_base_mode_checkpoint_that_carries_adapter_state_is_refused(tmp_path):
    """The zero-shot branch admits the base parameter set and nothing wider."""
    path, digest = checkpoint(tmp_path, 'simple_adapter', 'epoch_012.pth')
    args = args_for(path, digest)
    assert final.checkpoint_contract(args, digest) == 'base'
    with pytest.raises(ValueError, match='build_xrir_exp11'):
        final.verify_evaluation_checkpoint(path, 'checkpoint', args, digest)


def test_the_contract_is_the_adapter_arms_only(tmp_path):
    """A non-adapter arm evaluating its own initialisation stays on the strict path."""
    path, digest = checkpoint(tmp_path, 'simple_oriented', 'epoch_012.pth')
    args = args_for(path, digest, backbone='simple_oriented')
    assert final.checkpoint_contract(args, digest) == 'adapter'   # i.e. the arm's own model
    final.verify_evaluation_checkpoint(path, 'checkpoint', args, digest)
    base, base_digest = checkpoint(tmp_path, 'simple', 'other.pth')
    with pytest.raises(ValueError, match='build_xrir_exp11'):
        final.verify_evaluation_checkpoint(base, 'checkpoint',
                                           args_for(base, base_digest, 'simple_oriented'),
                                           base_digest)


def test_a_child_that_declares_no_job_initialisation_stays_strict(tmp_path):
    """Only the launch-time declaration can put a child on the base-mode branch."""
    path, digest = checkpoint(tmp_path, 'simple', 'epoch_012.pth')
    args = args_for(path, digest)
    args.pop('job_init_sha256')
    assert final.checkpoint_contract(args, digest) == 'adapter'
    with pytest.raises(ValueError, match='build_xrir_exp11'):
        final.verify_evaluation_checkpoint(path, 'checkpoint', args, digest)


def test_the_evidence_records_which_contract_was_applied(tmp_path):
    """The completion says how the checkpoint was validated, so a reader need not guess."""
    path, digest = checkpoint(tmp_path, 'simple', 'epoch_012.pth')
    args = args_for(path, digest)
    assert final.verify_evaluation_checkpoint(path, 'checkpoint', args, digest) == 'base'
    stage2, stage2_digest = checkpoint(tmp_path, 'simple_adapter', 'best.pth')
    assert final.verify_evaluation_checkpoint(
        stage2, 'checkpoint', args_for(stage2, digest), stage2_digest) == 'adapter'
    assert 'checkpoint_mode' in final.CHILD_EXTRA['exp11_haa_eval']


def test_the_zero_shot_lineage_still_ties_the_checkpoint_to_the_job(tmp_path):
    """``job_lineage`` is unchanged: a zero-shot child must evaluate the job's own init."""
    spec = {'backbone': 'simple_adapter', 'frame': 'room', 'seed': 0,
            'init': 'control_adapter', 'init_sha256': 'b' * 64,
            'adapter_heading': {'hallway': 128}, 'adapter_phi_deg': -90.0}
    records = {'eval/hallway': {'checkpoint_sha256': 'b' * 64}}
    fields = final.job_lineage(records, 'zeroshot', spec)
    assert fields['checkpoint_sha256'] == 'b' * 64
    with pytest.raises(ValueError, match='did not evaluate the job initialisation'):
        final.job_lineage({'eval/hallway': {'checkpoint_sha256': 'c' * 64}},
                          'zeroshot', spec)
