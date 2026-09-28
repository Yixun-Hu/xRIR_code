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
import pytest
import torch

from model.xRIR_simple_adapter import ADAPTER_KEYS
from tools import exp11_finalize as final
from tools import provenance

NUM_SHOT = 2
_SALT = [0]


def checkpoint(tmp_path, backbone, name):
    """A checkpoint carrying exactly one model's parameter names.

    The contract reads the parameter *set*, so the tensors are placeholders; the salt
    only makes two checkpoints of the same model hash differently, as a real stage-2
    checkpoint differs from the initialisation it started from. The names come from the
    factory at the production tier, which is what ``expected_state_keys`` builds.
    """
    _SALT[0] += 1
    path = tmp_path / name
    keys = sorted(final.expected_state_keys(backbone, NUM_SHOT))
    torch.save({key: torch.full((1,), float(_SALT[0])) for key in keys}, str(path))
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
    """A non-adapter arm evaluating its own initialisation stays on the strict path.

    Arms H and I have a zero-shot job too, but they load their own five-channel model
    both times, so nothing about the base-mode branch may apply to them. (The pinned
    ``_checkpoint_keys`` compares parameter *names*, and ``SimpleViTOriented`` differs
    from ``SimpleViT`` only in tensor shapes, so the case that must be refused here is
    an adapter checkpoint offered to an oriented arm.)
    """
    path, digest = checkpoint(tmp_path, 'simple_oriented', 'epoch_012.pth')
    args = args_for(path, digest, backbone='simple_oriented')
    assert final.checkpoint_contract(args, digest) == 'adapter'   # i.e. the arm's own model
    final.verify_evaluation_checkpoint(path, 'checkpoint', args, digest)
    other, other_digest = checkpoint(tmp_path, 'simple_adapter', 'other.pth')
    with pytest.raises(ValueError, match='build_xrir_exp11'):
        final.verify_evaluation_checkpoint(other, 'checkpoint',
                                           args_for(other, other_digest, 'simple_oriented'),
                                           other_digest)


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
