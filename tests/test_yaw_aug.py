"""Counter-seeded yaw augmentation: FLAC goldens, isolated draws, and batch geometry."""
import random

import numpy as np
import pytest
import torch

from tools.yaw_aug import (
    YawAug, apply_yaw_aug, counter, draw_offsets, fmix32,
    rotate_scene_yaw_batch, splitmix64, step_seed,
)
from tools.yaw_rotation import rotate_scene_yaw


_NEEDS_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _random_scene(seed=0, B=2, K=4, H=32, W=512):
    torch.manual_seed(seed)
    return (torch.randn(B, 3, H, W), torch.randn(B, 3), torch.randn(B, K, 3))


@pytest.mark.parametrize("x,sm,fm", [(0, 16294208416658607535, 0),
    (1, 10451216379200822465, 1364076727), (-1, 16490336266968443936, 2180083513)])
def test_primitive_goldens(x, sm, fm):
    assert splitmix64(x) == sm and fmix32(x) == fm
    assert splitmix64(x + 2**64) == sm and fmix32(x + 2**32) == fm


@pytest.mark.parametrize("t,seed,offset", [(0, 1254532375, 474), (1, 2068746652, 395),
    (9261, 2466316700, 358), (111131, 368013681, 401)])
def test_step_and_offset_goldens(t, seed, offset):
    assert step_seed(0, t, 0) == seed
    assert draw_offsets(1, 512, torch.Generator().manual_seed(seed)).tolist() == [offset]
    assert YawAug(True, 512, 0).offsets_for(t // 9261 + 1, t % 9261, 1).tolist() == [offset]
    assert YawAug(True, 512, 0, 9261).offsets_for(t // 9261 + 1, t % 9261, 1).tolist() == [offset]


def test_config_epoch_length_enters_counter():
    assert YawAug(True, 512, 0, 1).offsets_for(2, 0, 1).tolist() == [395]
    assert torch.equal(YawAug(True, 512, 0, 1).offsets_for(2, 0, 1),
                       YawAug(True, 512, 0).offsets_for(1, 1, 1))


def test_counter_and_complete_armed_domain():
    assert [counter(1, 0), counter(2, 0), counter(12, 9260), counter(3, 2, 7)] == [0, 9261, 111131, 16]
    seeds = [step_seed(0, t) for t in range(111132)]
    assert len(set(seeds)) == 111132 and all(0 <= s < 2**32 for s in seeds)
    assert step_seed(-1, 2**20 - 1, 4095) == 146380152


@pytest.mark.parametrize("function,args", [(step_seed, (0, 0, 0)), (counter, (1, 0, 9261)),
    (draw_offsets, (1, 512, torch.Generator()))])
def test_integer_arguments_are_strict(function, args):
    for index in range(2 if function is draw_offsets else 3):
        for bad in (True, False, 1.0, "1", None):
            invalid = list(args)
            invalid[index] = bad
            with pytest.raises(ValueError):
                function(*invalid)


@pytest.mark.parametrize("function,args", [(step_seed, (0, -1)), (step_seed, (0, 2**20)),
    (step_seed, (0, 0, -1)), (step_seed, (0, 0, 4096)), (counter, (0, 0)),
    (counter, (1, -1)), (counter, (1, 9261)), (counter, (1, 0, 0)), (counter, (1, 0, -1)),
    (draw_offsets, (-1, 512, torch.Generator())), (draw_offsets, (1, 0, torch.Generator()))])
def test_out_of_domain_arguments(function, args):
    with pytest.raises(ValueError):
        function(*args)


def test_offsets_deterministic_and_global_rngs_untouched():
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    py_before, np_before, cpu_before = random.getstate(), np.random.get_state(), torch.get_rng_state()
    aug = YawAug(True, 512, 0)
    first = aug.offsets_for(2, 3, 32)
    assert torch.equal(first, aug.offsets_for(2, 3, 32))
    assert first.shape == (32,) and first.dtype == torch.long and first.device.type == "cpu"
    assert aug.offsets_for(2, 3, 0).shape == (0,)
    assert random.getstate() == py_before
    assert all(np.array_equal(a, b) for a, b in zip(np_before, np.random.get_state()))
    assert torch.equal(cpu_before, torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(cuda_before, torch.cuda.get_rng_state_all()))


def test_million_draws_uniform_and_in_range():
    offsets = draw_offsets(10**6, 512, torch.Generator().manual_seed(0))
    assert offsets.min() >= 0 and offsets.max() < 512
    counts = torch.bincount(offsets, minlength=512).double()
    expected = 10**6 / 512
    chi2 = ((counts - expected).square() / expected).sum().item()
    # A1: sanity bound for 511 degrees of freedom, replacing the per-bin rule.
    assert chi2 < 700, f"chi-square statistic {chi2} exceeds the sanity bound"


@pytest.mark.parametrize("generator", [None, 0, object(), torch.default_generator])
def test_draw_requires_dedicated_generator(generator):
    with pytest.raises(ValueError):
        draw_offsets(1, 512, generator)


def test_apply_matches_batch_geometry_only():
    args = (torch.randn(2, 3, 2, 13), torch.randn(2, 3), torch.randn(2, 4, 3), torch.tensor([0, 7]))
    assert all(torch.equal(a, b) for a, b in zip(apply_yaw_aug(*args, W=13), rotate_scene_yaw_batch(*args, W=13)))


@pytest.mark.parametrize("kwargs", [{"enabled": 1}, {"enabled": None}, {"W": 0}, {"W": -1},
    {"W": True}, {"W": 1.0}, {"seed": 1.0}, {"seed": True},
    *[{"batches_per_epoch": value} for value in (0, -1, True, False, 1.0, "1", None)]])
def test_config_rejects_invalid_types(kwargs):
    with pytest.raises(ValueError):
        YawAug(**kwargs)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=_NEEDS_CUDA)])
@pytest.mark.parametrize("W", [13, 512])
def test_batch_yaw_matches_scalar_and_preserves_geometry(dtype, device, W):
    scene = tuple(t.to(device=device, dtype=dtype) for t in _random_scene(B=6, H=3, W=W))
    snapshots = tuple(t.clone() for t in scene)
    ks = torch.randint(-2 * W, 3 * W, (6,), device=device)
    ks_before = ks.clone()
    got = rotate_scene_yaw_batch(*scene, ks, W=W)
    for b, k in enumerate(ks.tolist()):
        want = rotate_scene_yaw(*(t[b:b + 1] for t in scene), k, W=W)
        for actual, expected in zip(got, want):
            assert torch.allclose(actual[b:b + 1], expected, atol=1e-6, rtol=0)
        # Yaw leaves z untouched: its full panorama exposes the exact column gather.
        assert torch.equal(got[0][b, 2], torch.roll(scene[0][b, 2], k, -1))
    for before, snapshot, after in zip(scene, snapshots, got):
        assert torch.equal(before, snapshot)
        assert after.data_ptr() != before.data_ptr()
        assert after.dtype == dtype and after.device == before.device
    assert torch.equal(ks, ks_before)
    for before, after in zip(scene[1:], got[1:]):
        assert torch.allclose(before.norm(dim=-1), after.norm(dim=-1), atol=1e-6, rtol=0)


@pytest.mark.parametrize("k", [0, 512, -512])
def test_batch_yaw_identity_is_bit_exact_and_new(k):
    scene = _random_scene(H=2)
    for tensor in scene:
        tensor.flatten()[0] = -0.0
    got = rotate_scene_yaw_batch(*scene, torch.full((2,), k, dtype=torch.long))
    for before, after in zip(scene, got):
        assert torch.equal(before.view(torch.uint8), after.contiguous().view(torch.uint8))
        assert before.data_ptr() != after.data_ptr()


def test_batch_yaw_empty_batch():
    scene = _random_scene(B=0, H=2)
    got = rotate_scene_yaw_batch(*scene, torch.empty(0, dtype=torch.long))
    for before, after in zip(scene, got):
        assert after is not before and after.shape == before.shape
        assert after.dtype == before.dtype and after.device == before.device


@pytest.mark.parametrize("index,shape", [
    (0, (3, 2, 512)), (0, (2, 4, 2, 512)), (0, (2, 3, 2, 511)),
    (1, (3,)), (1, (2, 1, 3)), (1, (2, 4)), (1, (1, 3)),
    (2, (2, 3)), (2, (2, 4, 4)), (2, (1, 4, 3)),
    (3, ()), (3, (2, 1)), (3, (1,)),
])
def test_batch_yaw_rejects_wrong_shapes(index, shape):
    args = list(_random_scene(H=2)) + [torch.zeros(2, dtype=torch.long)]
    args[index] = torch.zeros(shape, dtype=args[index].dtype)
    with pytest.raises(ValueError):
        rotate_scene_yaw_batch(*args)


@pytest.mark.parametrize("W", [0, -1, 511, 512.0, True, None])
def test_batch_yaw_rejects_wrong_width(W):
    with pytest.raises(ValueError):
        rotate_scene_yaw_batch(*_random_scene(H=2), torch.zeros(2, dtype=torch.long), W=W)


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("problem", ["integer", "mixed_float", "device", "not_tensor"])
def test_batch_yaw_rejects_wrong_dtype_or_device(index, problem):
    args = list(_random_scene(H=2)) + [torch.zeros(2, dtype=torch.long)]
    if problem == "device":
        args[index] = args[index].to("meta")
    elif problem == "not_tensor":
        args[index] = None
    else:
        dtype = torch.bool if index == 3 else torch.int64
        args[index] = args[index].to(torch.float64 if problem == "mixed_float" else dtype)
    with pytest.raises(ValueError):
        rotate_scene_yaw_batch(*args)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64])
def test_batch_yaw_accepts_integer_offset_dtypes(dtype):
    scene = _random_scene(H=2)
    got = rotate_scene_yaw_batch(*scene, torch.tensor([1, 2], dtype=dtype))
    want = rotate_scene_yaw_batch(*scene, torch.tensor([1, 2], dtype=torch.long))
    assert all(torch.equal(a, b) for a, b in zip(got, want))


def _audit_fixture():
    from tools.yaw_aug import AuditBatch
    src = torch.tensor([[0.007777777500450611, 0., 0.], [1., 0., 0.]])
    batch = (torch.zeros(2, 3), src, torch.zeros(2, 3, 1, 512),
             torch.zeros(2, 1, 8), torch.zeros(2, 1, 8), torch.zeros(2, 1, 3))
    return torch.nn.Linear(1, 1), AuditBatch(batch, ["room/query_a.wav", "room/query_b.wav"])


@pytest.mark.parametrize("offset,changed", [(0, 0), (1, 1)])
def test_alignment_audit_real_rounding_boundary_and_schema(offset, changed):
    from tools.yaw_aug import alignment_audit
    from tools.yaw_rotation import integer_delays
    model, batch = _audit_fixture()
    assert integer_delays(batch[1], batch[5]).tolist() == [[0], [64]]
    _, src, refs = apply_yaw_aug(batch[2], batch[1], batch[5], torch.tensor([offset, 0]))
    assert integer_delays(src, refs).tolist() == [[changed], [64]]
    before = [tensor.clone() for tensor in batch]
    result = alignment_audit(model, [batch], [torch.tensor([offset, 0])])
    assert set(result) == {"pairs", "changed_delays", "fraction", "cohort_sha256", "W"}
    assert (result["pairs"], result["changed_delays"], result["fraction"], result["W"]) == (2, changed, changed / 2, 512)
    assert len(result["cohort_sha256"]) == 64 and int(result["cohort_sha256"], 16) >= 0
    assert all(torch.equal(a, b) for a, b in zip(batch, before))


def test_alignment_cohort_hash_binds_paths_offsets_and_order():
    from tools.yaw_aug import AuditBatch, alignment_audit
    model, batch = _audit_fixture()
    def digest(paths, offsets):
        return alignment_audit(model, [AuditBatch(batch, paths)], [torch.tensor(offsets)])["cohort_sha256"]
    paths = batch.query_paths
    baseline = digest(paths, [0, 0])
    assert baseline == digest(paths, [0, 0])
    assert len({baseline, digest(paths, [1, 0]), digest(paths[::-1], [0, 0]), digest(["other", paths[1]], [0, 0])}) == 4


@pytest.mark.parametrize("problem", ["missing_paths", "short_paths", "short_offsets", "short_batches"])
def test_alignment_audit_refuses_incomplete_cohorts(problem):
    from tools.yaw_aug import AuditBatch, alignment_audit
    model, batch = _audit_fixture()
    batches, offsets = [batch], [torch.zeros(2, dtype=torch.long)]
    if problem == "missing_paths": batches = [tuple(batch)]
    if problem == "short_paths": batches = [AuditBatch(batch, ["one"])]
    if problem == "short_offsets": offsets = []
    if problem == "short_batches": batches = []
    with pytest.raises(ValueError):
        alignment_audit(model, batches, offsets)


@pytest.mark.parametrize("workers", [None, "0"])
def test_alignment_cli_matches_training_rng_and_loader(monkeypatch, tmp_path, workers):
    import json
    import train_xRIR_backbone as trainer
    import tools.yaw_aug as yaw
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, split="train", max_len=9600, num_shot=8):
            assert max_len == 9600 and num_shot == 8
            self.file_list = ["query_%d.wav" % i for i in range(64)]
        def __len__(self): return len(self.file_list)
        def __getitem__(self, i):
            return (torch.zeros(3), torch.tensor([float(i), np.random.rand(), 0.]),
                    torch.zeros(3, 1, 512), torch.zeros(1, 8), torch.zeros(8, 8), torch.ones(8, 3))
    def build(backbone, num_shot):
        assert (backbone, num_shot) == ("simple", 8)
        return torch.nn.Linear(3, 3)
    trainer.seed_everything(7)
    reference = torch.utils.data.DataLoader(Dataset(), batch_size=32, shuffle=True, num_workers=0)
    expected_model = build("simple", 8)
    expected = next(iter(reference))
    original_loader, original_audit = trainer.DataLoader, yaw.alignment_audit
    def loader(dataset, **kwargs):
        assert kwargs["worker_init_fn"] is trainer.seed_worker and kwargs["pin_memory"]
        assert kwargs["num_workers"] == (12 if workers is None else 0)
        assert kwargs["persistent_workers"] == (workers is None)
        return original_loader(dataset, **dict(kwargs, num_workers=0, persistent_workers=False))
    def audit(model, batches, offsets, **kwargs):
        batches, offsets = list(batches), list(offsets)
        assert all(torch.equal(a, b) for a, b in zip(expected, batches[0]))
        assert list(batches[0].query_paths) == ["query_%d.wav" % i for i in expected[1][:, 0]]
        assert torch.equal(model.weight, expected_model.weight)
        assert torch.equal(offsets[0], YawAug(seed=7).offsets_for(1, 0, 32))
        return original_audit(model, batches, offsets, **kwargs)
    for name, value in [("xRIR_Dataset", Dataset), ("build_xrir", build), ("DataLoader", loader)]:
        monkeypatch.setattr(trainer, name, value)
    monkeypatch.setattr(yaw, "alignment_audit", audit)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    out = tmp_path / "audit.json"
    yaw._audit_main(["--audit", "--n-batches", "1", "--seed", "7", "--out", str(out)] + ([] if workers is None else ["--num-workers", workers]))
    assert json.loads(out.read_text())["pairs"] == 32 * 8
