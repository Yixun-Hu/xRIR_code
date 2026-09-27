"""exp_11's orchestration loop is exp_06's, verified on a tiny synthetic run (CPU).

Codex round-2 change 4: ``tools/exp06_haa_finetune.py`` embeds optimiser construction,
epoch iteration, validation selection and checkpoint writing inside its ``main``, so
exp_11 owns a transcription of that loop. Two things guard the transcription:

1. ``reference_loop`` below is that ``main``'s epoch loop, line for line, with only the
   device moves and the record writing removed. Both loops are run on one synthetic
   problem and their **selection traces** (every ``history.jsonl`` row, the validation
   cadence, the epoch-0 selection, the strict-improvement rule) and the **bytes of the
   checkpoints they write** must agree.
2. The source digest of exp_06's ``main`` is pinned, so a change to the old loop fails
   this test and forces the transcription to be revisited rather than drifting.
"""
import hashlib
import inspect
import json
import os
import time

import torch
import torch.optim as optim
from torch import nn

from tools import exp06_haa_finetune as legacy
from tools import exp11_haa_finetune as exp11
from utils.lr_scheduler import ExponentialLR

# The bytes this transcription was taken from. A refusal here is not a bug in exp_11:
# it means exp_06's loop changed and the transcription below must be re-derived.
EXP06_MAIN_SHA256 = '88973e013b41cbfe257cf8f883e659adddaf7e8e3b2d0f5faaf988cc131a02d3'
# Scripted validation losses: improvement, a tie (strictly not an improvement), a rise,
# then a new best on the final epoch -- every branch of the selection rule.
VAL_SCRIPT = (5.0, 4.0, 4.0, 6.0, 3.0)


class Args:
    lr = 0.05
    weight_decay = 1e-4
    decay_epochs = 2
    lr_gamma = 0.1
    epochs = 4
    accum_steps = 2
    val_every = 1
    log_every = 10
    rooms = ['hallway']
    init = 'init.pth'
    backbone = 'simple'
    seed = 0


class Context:
    val_rooms = ['hallway']
    frame = 'room'
    adapter_phi_deg = None


def model_and_batches():
    """One deterministic model and three batches shaped like the pipeline's."""
    torch.manual_seed(11)
    model = nn.Linear(4, 1)
    batches = [(None, torch.full((2, 4), 0.1 * (i + 1)), None, None, None, None)
               for i in range(3)]
    return model, batches


def loss_fn(model, batch):
    """The signature of ``train_xRIR_backbone.compute_loss``: (loss, stft, decay)."""
    out = model(batch[1])
    loss = (out - 1.0).pow(2).mean()
    return loss, loss.detach(), loss.detach()


def scripted_evaluate():
    """``evaluate(model, loader)``'s signature, returning the pinned sequence."""
    calls = []

    def evaluate(model, loader):
        value = VAL_SCRIPT[min(len(calls), len(VAL_SCRIPT) - 1)]
        calls.append(value)
        return value
    return evaluate, calls


def reference_loop(model, train_loader, val_loader, args, save_dir, context,
                   compute_loss, evaluate):
    """``tools/exp06_haa_finetune.py::main``'s epoch loop, transcribed line for line."""
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ExponentialLR(optimizer, decay_epochs=args.decay_epochs, gamma=args.lr_gamma)

    history_path = os.path.join(save_dir, "history.jsonl")
    open(history_path, "w").close()
    val0 = evaluate(model, val_loader)
    with open(history_path, "a") as f:
        f.write(json.dumps({"epoch": 0, "train_loss": None, "val_loss": val0,
                            "lr": args.lr}) + "\n")
    best_val, best_epoch = val0, 0
    torch.save(model.state_dict(), os.path.join(save_dir, "best.pth"))

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tot, n = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for bi, batch in enumerate(train_loader):
            loss, _, _ = compute_loss(model, batch)
            (loss / args.accum_steps).backward()
            if (bi + 1) % args.accum_steps == 0 or bi + 1 == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            tot += loss.item() * batch[1].shape[0]
            n += batch[1].shape[0]
        scheduler.step()
        train_loss = tot / max(n, 1)
        rec = {"epoch": epoch, "train_loss": train_loss,
               "lr": optimizer.param_groups[0]["lr"]}
        if epoch % args.val_every == 0 or epoch == args.epochs:
            val = evaluate(model, val_loader)
            rec["val_loss"] = val
            if val < best_val:
                best_val, best_epoch = val, epoch
                torch.save(model.state_dict(), os.path.join(save_dir, "best.pth"))
                rec["is_best"] = True
        with open(history_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    torch.save(model.state_dict(), os.path.join(save_dir, "last.pth"))
    summary = {"best_epoch": best_epoch, "best_val_loss": best_val, "init_val_loss": val0,
               "epochs": args.epochs, "minutes": (time.time() - t0) / 60,
               "rooms": args.rooms, "val_rooms": context.val_rooms, "init": args.init,
               "backbone": args.backbone, "seed": args.seed, "frame": context.frame}
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def run(loop, directory):
    model, batches = model_and_batches()
    evaluate, calls = scripted_evaluate()
    os.makedirs(directory, exist_ok=True)
    if loop is reference_loop:
        summary = reference_loop(model, batches, batches, Args, str(directory), Context(),
                                 loss_fn, evaluate)
    else:
        summary = exp11.fine_tune(model, batches, batches, Args, str(directory), Context(),
                                  loss_fn=loss_fn, evaluate_fn=evaluate)
    rows = [json.loads(line) for line in
            open(os.path.join(directory, 'history.jsonl')) if line.strip()]
    return {'summary': {key: value for key, value in summary.items() if key != 'minutes'},
            'rows': rows, 'calls': calls,
            'best': hashlib.sha256(open(os.path.join(directory, 'best.pth'), 'rb').read()).hexdigest(),
            'last': hashlib.sha256(open(os.path.join(directory, 'last.pth'), 'rb').read()).hexdigest()}


def test_the_transcription_is_taken_from_the_current_exp06_loop():
    digest = hashlib.sha256(inspect.getsource(legacy.main).encode()).hexdigest()
    assert digest == EXP06_MAIN_SHA256, (
        "exp_06's fine-tuning main changed ({}); re-derive reference_loop and this "
        'pin before trusting the parity result'.format(digest))


def test_selection_traces_and_written_checkpoints_agree(tmp_path):
    reference = run(reference_loop, tmp_path / 'exp06')
    ours = run(exp11.fine_tune, tmp_path / 'exp11')
    assert ours['calls'] == reference['calls'] == list(VAL_SCRIPT)
    assert ours['rows'] == reference['rows']
    assert ours['best'] == reference['best'] and ours['last'] == reference['last']
    for key in ('best_epoch', 'best_val_loss', 'init_val_loss', 'epochs', 'rooms',
                'val_rooms', 'init', 'backbone', 'seed', 'frame'):
        assert ours['summary'][key] == reference['summary'][key], key


def test_the_trace_exercises_every_branch_of_the_selection_rule(tmp_path):
    """Epoch 0 selects; a tie does not; a rise does not; the final epoch validates."""
    rows = run(exp11.fine_tune, tmp_path / 'exp11')['rows']
    assert [row['epoch'] for row in rows] == [0, 1, 2, 3, 4]
    assert rows[0]['train_loss'] is None and rows[0]['val_loss'] == 5.0
    assert rows[1].get('is_best') is True                      # 4.0 < 5.0
    assert 'is_best' not in rows[2] and rows[2]['val_loss'] == 4.0   # a tie is not better
    assert 'is_best' not in rows[3] and rows[3]['val_loss'] == 6.0
    assert rows[4].get('is_best') is True and rows[4]['val_loss'] == 3.0
    summary = json.loads((tmp_path / 'exp11' / 'summary.json').read_text())
    assert summary['best_epoch'] == 4 and summary['best_val_loss'] == 3.0
    assert summary['init_val_loss'] == 5.0


def test_the_pinned_helpers_are_the_loops_defaults():
    """Production never substitutes anything: the defaults are the pinned functions."""
    import train_xRIR_backbone
    from sim_to_real import finetune_haa
    defaults = inspect.signature(exp11.fine_tune).parameters
    assert defaults['loss_fn'].default is train_xRIR_backbone.compute_loss
    assert defaults['evaluate_fn'].default is finetune_haa.evaluate
    assert defaults['save'].default is torch.save
    assert exp11.ExponentialLR is ExponentialLR


def test_validation_cadence_is_respected(tmp_path):
    class Every3(Args):
        epochs = 4
        val_every = 3
    saved, Args.val_every = Args.val_every, 3
    try:
        rows = run(exp11.fine_tune, tmp_path / 'cadence')['rows']
    finally:
        Args.val_every = saved
    assert [('val_loss' in row) for row in rows] == [True, False, False, True, True]
