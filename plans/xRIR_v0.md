# xRIR baseline receiver-frame ablation

## Goal

Measure how much the pretrained xRIR baseline degrades when its reference
RIRs come from receivers different from the query receiver, and isolate the
part of that degradation caused by inconsistent coordinate frames.

This is a full unseen-test evaluation. It does not train a new model, create a
seen-room pipeline, compare K values, or use bounded train/test subsets.

## Controlled comparison

Use the same baseline checkpoint, query set, `K=8`, and deterministic reference
selection for all conditions.

1. **Original**: use `xRIR_Dataset`; every reference receiver equals the query
   receiver.
2. **Cross-receiver / mismatched**: use the cross-receiver dataset and pass
   `d_i = s_i - r_i` everywhere the baseline currently expects
   `ref_ir_locs`. This measures the baseline's actual behavior in the new query
   setting.
3. **Cross-receiver / aligned**: use the exact same reference RIRs as condition
   2. Use `s_i - r_q` for geometry combined with the query depth map, while
   retaining `d_i = s_i - r_i` for direct-path delay and amplitude alignment.

For every lower-is-better metric, report:

- `mismatched - original`: total cross-receiver degradation.
- `mismatched - aligned`: coordinate-frame inconsistency penalty.
- `aligned - original`: remaining cross-receiver distribution shift.

## Files

### Dataset

`treble_multi_room_dataset/treble_xRIR_v0_dataset.py` contains
`xRIRReceiverFrameDataset`, an evaluation-only dataset for the full unseen test
split. It strictly selects references with `r_i != r_q` and returns:

```text
(query_receiver_local, query_source_local, depth_coord, target_rir,
 reference_rirs, ref_pair_local, ref_query_local)
```

where:

```text
ref_pair_local  = s_i - r_i
ref_query_local = s_i - r_q
```

Reference selection is deterministic per query and seed. There is no dataset
subsampling logic.

### Baseline model adapter

Make a minimal backward-compatible change to `model/xRIR.py`:

```python
forward(depth_coord, x, src_loc, ref_ir_locs, tgt_wav,
        ref_query_locs=None)
```

- With `ref_query_locs=None`, behavior must remain identical to the current
  baseline.
- `shift_and_align` always uses `ref_ir_locs` (`s_i - r_i`).
- Reference geometry combined with `depth_coord` uses `ref_query_locs` when it
  is provided.
- Do not add or resize model parameters; the existing baseline checkpoint must
  load unchanged.

### Evaluation

Add `eval_xRIR_receiver_frame_ablation.py`. It loads the existing unseen
baseline checkpoint and evaluates all three conditions over the complete test
split with `shuffle=False`. Conditions 2 and 3 must be evaluated from the same
batch so their query and reference RIRs are paired exactly.

Report test loss, EDT error, C50 error, T60 percentage error, and STFT error,
plus the three deltas above. Generated checkpoints, logs, and result files are
not committed.

## Verification

- Every cross-receiver reference satisfies `r_i != r_q`.
- `ref_query_local - ref_pair_local` is nonzero for every reference.
- Passing `ref_query_locs=None` reproduces the original baseline output.
- Passing `ref_query_locs=ref_ir_locs` also reproduces the original output.
- The existing `checkpoints/xRIR_unseen.pth` loads without missing or resized
  parameters.

## Commit scope

Keep only the dataset, the small backward-compatible baseline change, the
ablation evaluator, and focused verification. Do not include v0 train/seen
scripts, a separate v0 model, bounded-experiment code, checkpoints, or logs.
