"""exp_11 pretraining entry point: the pinned trainer's steps on the exp_11 registry.

Plan v3 item 6. Nothing of ``train_xRIR_backbone`` or of ``tools/exp06_train.py`` is
modified or monkey patched: the pinned ``seed_everything`` / ``seed_worker`` /
``train_epoch`` / ``test_epoch`` / ``save_checkpoint`` are called, and exp_06's data,
geometry and held-out inventories are imported as they stand. What this module owns is a
parser whose ``--backbone`` ranges over ``BACKBONES_EXP11`` and whose ``--yaw-aug``
accepts 0 **and** 1 under exp_04's constraints (``--save-every 0``, no ``--resume``,
``epochs * batches < 2**20``), the ``exp11_*`` provenance class, and a
``registry_sha256`` over exp_11's own registry.

It writes ``provenance.json`` once, before the epoch loop, and never writes
``completion.json`` -- completion is the external finalizer's decision
(``tools/exp11_finalize.py``).

    python tools/exp11_train.py --backbone simple_oriented \
        --save-dir ckpt/exp11/pretrain/xRIR_simpor_8_shot/attempt_<UTC> --num-shot 8 \
        --max-len 9600 --lr 1e-3 --weight-decay 1e-4 --decay-epochs 3 --lr-gamma 0.1 \
        --epochs 12 --batch-size 32 --accum-steps 2 --num-workers 12 --seed 0 --tf32 \
        --log-interval 50 --save-every 500 --epoch-ckpt-every 1 --run-type full
"""
import argparse
import json
import contextlib
import fcntl
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import train_xRIR_backbone as trainer
from model.xrir_exp11_registry import BACKBONES_EXP11, build_xrir_exp11, registry_sha256
from tools import (exp06_train, exp11_pathprobe, exp11_pidrecord, exp11_profiles,
                   exp11_recipe, provenance)
from tools.exp05_params import TIERS, count_parameters, tier_of
from treble_multi_room_dataset.treble_xRIR_dataset import xRIR_Dataset
from utils.lr_scheduler import ExponentialLR

REPO = Path(__file__).resolve().parents[1]
ENTRY_MODULE = 'tools.exp11_train'
RUN_TYPES = ('full', 'smoke', 'probe')
RESOLVED_EXIT = 3
DIAGNOSTIC_RUN_TYPES = ('smoke', 'probe')
ENV_KEYS = exp06_train.ENV_KEYS


def build_parser():
    """The trainer's flags and defaults, on the exp_11 registry, yaw augmentation allowed."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--backbone', choices=sorted(BACKBONES_EXP11), required=True)
    p.add_argument('--save-dir', required=True)
    p.add_argument('--num-shot', type=int, default=8)
    p.add_argument('--max-len', type=int, default=9600)
    for key, value in TIERS['M'].items():
        p.add_argument('--vit-' + key.replace('_', '-'), type=int, default=value)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--decay-epochs', type=int, default=50)
    p.add_argument('--lr-gamma', type=float, default=0.1)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--accum-steps', type=int, default=1)
    p.add_argument('--num-workers', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--yaw-aug', type=int, choices=(0, 1), default=0,
                   help='I_RECIPE trains with per-sample active yaw (exp_04 port)')
    p.add_argument('--yaw-aug-seed', type=int, default=None)
    p.add_argument('--yaw-aug-width', type=int, default=512)
    p.add_argument('--no-save', action='store_true', help='write nothing to save-dir')
    p.add_argument('--tf32', action='store_true')
    p.add_argument('--log-interval', type=int, default=20)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--epoch-ckpt-every', type=int, default=5)
    p.add_argument('--resume', default=None)
    p.add_argument('--max-train-batches', type=int, default=0)
    p.add_argument('--max-test-batches', type=int, default=0)
    p.add_argument('--test-subset', type=int, default=0)
    p.add_argument('--run-type', choices=RUN_TYPES, default='full')
    p.add_argument('--provenance-out', default=None)
    p.add_argument('--approved', default=None,
                   help='approved_digests.json this run is admitted under (required for full)')
    p.add_argument('--reviewed-commit', default=None)
    p.add_argument('--exploratory', action='store_true')
    return p


def parse_args(argv=None):
    """Parse and apply the trainer's post-conditions, including exp_04's saving rules."""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.yaw_aug_seed = args.seed if args.yaw_aug_seed is None else args.yaw_aug_seed
    if args.yaw_aug and (args.resume is not None or args.save_every != 0):
        parser.error('--yaw-aug 1 requires --resume to be None and --save-every exactly 0')
    if args.provenance_out is not None and not args.no_save:
        parser.error('--provenance-out applies to --no-save runs')
    if args.exploratory and args.run_type == 'full':
        parser.error('--exploratory is a diagnostic mode; a full run must match the approvals')
    return args


def check_admission(args):
    """A confirmatory run always names the approvals it is admitted under."""
    if args.approved is not None:
        return args
    if args.run_type not in DIAGNOSTIC_RUN_TYPES:
        raise ValueError('a full run must name the approvals it is admitted under (--approved)')
    if not args.no_save:
        raise ValueError('a {} run that writes to its save-dir must name its approvals '
                         '(--approved), or run with --no-save'.format(args.run_type))
    return args


def check_counter(args, batches_per_epoch):
    """exp_04's bound, refused here as the pinned trainer refuses it."""
    if args.yaw_aug and args.epochs * batches_per_epoch >= exp11_recipe.YAW_COUNTER_LIMIT:
        raise ValueError('--yaw-aug 1 requires epochs * train_batches_per_epoch < 2**20')


def build_model_exp11(args):
    """Construct exactly as the trainer's build_model, on the exp_11 registry."""
    return build_xrir_exp11(args.backbone, args.num_shot,
                            **{key: getattr(args, 'vit_' + key) for key in TIERS['M']})


def prepare_args(args, model, fields, destination):
    """Record the derived fields and the six ``exp11_*`` provenance keys."""
    args.param_counts = (count_parameters(model) if model is not None
                         else dict(exp11_recipe.expected_param_counts(args.backbone)))
    tier = tier_of(vars(args))
    if tier != exp11_recipe.TIER:
        raise ValueError('exp_11 pretrains at tier {}, not {}'.format(exp11_recipe.TIER, tier))
    args.tier = tier
    args.exp11_run_type = args.run_type
    args.exp11_profile = exp11_recipe.select_profile(vars(args))
    args.exp11_registry_sha256 = fields.get('registry_sha256')
    args.exp11_source_closure_sha256 = fields.get('source_closures', {}).get(
        'training', {}).get('sha256')
    args.exp11_git_head = fields.get('git_state', {}).get('HEAD')
    args.exp11_provenance_path = destination
    for flag in ('run_type', 'provenance_out', 'approved', 'reviewed_commit', 'exploratory'):
        if hasattr(args, flag):
            delattr(args, flag)
    return args


def approvals_binding(approved):
    """Bind the approvals file's own bytes, so a mid-run edit is detectable."""
    if approved is None:
        return None
    value, identity = exp11_profiles.load_approved_digests(approved)
    return {'path': str(approved), 'sha256': identity['sha256'],
            'schema_version': value['schema_version']}


def orchestration_closures(repo, commit):
    """The launcher shell and the finalizer that decide this run, bound as exp_06 binds them."""
    launcher, launcher_digest = exp11_profiles.closure_of('launch_sh', repo, commit)
    finalizer, finalizer_digest = exp11_profiles.closure_of('finalize', repo, commit)
    return {'launcher': {'files': launcher, 'sha256': launcher_digest},
            'finalizer': {'entry_module': 'tools.exp11_finalize',
                          'files': finalizer, 'sha256': finalizer_digest}}


def provenance_fields(argv, run_type, identity=None, repo=REPO, approved=None,
                      reviewed_commit=None, exploratory=False, geometry=None, heldout=None):
    """Bind the import closure, HEAD, environment, data inventory and argv of one run."""
    state = provenance.git_state(repo)
    commit = state['HEAD'] if reviewed_commit is None else reviewed_commit
    records, digest = provenance.closure_record(
        provenance.source_closure(ENTRY_MODULE, repo), commit, repo)
    return dict(repo=str(repo), reviewed_commit=commit, run_type=run_type,
                data_root=exp06_train.resolve_data_root(),
                source_closures={'training': {'entry_module': ENTRY_MODULE,
                                              'files': records, 'sha256': digest}},
                orchestration_closures=orchestration_closures(repo, commit),
                code_digests=exp11_profiles.compute_code_digests(
                    repo, commit, keys=exp11_profiles.TRAINING_KEYS),
                approvals=approvals_binding(approved), exploratory=bool(exploratory),
                geometry_identity=geometry, test_wav_identity=heldout,
                registry_sha256=registry_sha256(), git_state=state,
                environment=provenance.environment(), train_data_identity=identity,
                command=list(argv))


# The one reader, shared with the launcher, which executes the same module rather than
# reading pid files in Bash (close review 10, blocker 1).
pid_record = exp11_pidrecord.pid_record


# --- three states for every path this trainer asks about (close reviews 14, 15) ------
# ``Path.is_file()`` answers False both for a path that is not there and for one it
# could not inspect: a self-referential ``<attempt>.resolved`` raises ELOOP inside it and
# comes back "no tombstone", so a trainer whose launch had been retired registered
# anyway. Nothing here decides on ``is_file``. The classification is the launcher's --
# literally: the same tools/exp11_pathprobe.py, where the errno is available.
PRESENT = exp11_pathprobe.PRESENT
ABSENT = exp11_pathprobe.ABSENT
UNSURE = exp11_pathprobe.UNKNOWN
probe_path = exp11_pathprobe.probe
file_state = exp11_pathprobe.file_state
directory_state = exp11_pathprobe.directory_state


def demand_known(state, detail, path, what):
    """Refuse rather than act on a path this process could not inspect."""
    if state == UNSURE:
        refuse('cannot inspect {} ({}: {}); this trainer does not register on a launch '
               'it cannot verify'.format(path, what, detail))
    return state == PRESENT


def registration_complete(attempt):
    """``child.pid`` holds one pid, by that grammar (the launcher's own test)."""
    return pid_record(Path(attempt) / 'child.pid') is not None


REGISTRATION_LOCK = '.registration.lock'
REGISTRATION_WAIT_S = 60


def registration_lock_path(save_dir):
    """The arm root's registration lock -- one per arm, beside its attempts."""
    return Path(save_dir).parent / REGISTRATION_LOCK


@contextlib.contextmanager
def registration_lock(save_dir, wait_seconds=REGISTRATION_WAIT_S):
    """Hold the arm's REGISTRATION lock, or give up after ``wait_seconds``.

    This is not the publication lock: a full-mode launcher holds that one for its
    child's whole run, so a trainer could never take it. This one is held only across a
    registration and across a resolution, which is exactly the pair that must not
    interleave (close review 9, blocker 2). The wait is blocking but bounded, because a
    resolution is short and a trainer that waits forever is a trainer nothing can end.
    """
    path = registration_lock_path(save_dir)
    try:
        handle = open(str(path), 'a')
    except OSError as error:
        refuse('cannot open the registration lock {}: {}'.format(path, error))
    try:
        # A duration belongs on the clock that only goes forward: NTP corrections and
        # a resuming VM step time.time() in either direction, which would shorten this
        # bound to nothing or stretch it past any trainer's patience.
        deadline = time.monotonic() + wait_seconds
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    refuse('the registration lock {} was held for more than {}s; '
                           'another registration or a resolution is in progress'
                           .format(path, wait_seconds))
                time.sleep(0.05)
        yield
    finally:
        handle.close()          # closing the descriptor releases the lock


def refuse(reason):
    print('refusing: ' + reason, file=sys.stderr, flush=True)
    raise SystemExit(RESOLVED_EXIT)


def register_trainer(save_dir, no_save=False, arm_root=None,
                     wait_seconds=REGISTRATION_WAIT_S):
    """Name this process in its own run directory, after parsing and before admission.

    The launcher's ``child.pid`` is written by the frozen lifecycle only *after* it has
    forked this process, so a launcher that dies in between leaves a trainer nothing in
    the arm can see (close review 7, blocker 2). This file is the trainer's own answer.

    It is also where this trainer finds out whether its launch is still a launch. A
    delayed trainer can wake long after an operator resolved the attempt as unregistered
    -- the marker's age proved nothing about it -- so the answer has to be in the state
    of the directory, and this process **fails closed** (close review 8, blocker 2):

    * the save-dir is gone: the launcher creates it, the trainer never does, so its
      absence means the launch it belongs to is over;
    * a ``<attempt>.resolved`` tombstone stands beside it: that launch was answered and
      retired, permanently;
    * neither ``launching`` nor a complete ``child.pid`` is there: from the fork onward
      an ordinary launch always carries one of them.

    All of that, and the write itself, happen under the arm's REGISTRATION lock, so a
    resolution cannot rename the directory between the checks and the write (close
    review 9, blocker 2); the write is a temp file and one ``os.replace``, so no reader
    ever sees half a pid.

    ``--no-save`` runs are diagnostics that no scan reads; they write no sidecars, take
    no lock, and only the tombstone can refuse them.
    """
    # The ATTEMPT is canonicalised first, and its arm is its canonical parent: the
    # validated directory, the registration lock and the tombstone must all be of ONE
    # arm. Reached through `arm_alias/attempt_X -> arm_canonical/attempt_X`, the alias's
    # parent had no tombstone and its own lock, while every write landed in the
    # canonical attempt -- whose tombstone said it had been retired (close review 15,
    # blocker 3).
    given = Path(save_dir)
    # `realpath` resolves what it can and leaves the rest: a --no-save diagnostic has no
    # run directory at all, and asking for one would refuse every diagnostic.
    directory = Path(exp11_pathprobe.canonical(given))
    # The containment below is derived from this path, so it is checked the way the
    # launcher checks every path field it is handed (close review 16).
    if not exp11_pathprobe.valid_path(str(directory)):
        refuse('{} does not canonicalise to an absolute, normalized path ({}); this '
               'trainer registers against nothing it cannot name'.format(given, directory))
    arm = directory.parent
    tombstone = arm / (directory.name + '.resolved')
    if no_save:                       # a diagnostic no scan reads: only a tombstone speaks
        if demand_known(*file_state(tombstone), path=tombstone, what='the tombstone'):
            refuse('{} was resolved as an unregistered launch and retired; this trainer '
                   'has nothing to run in'.format(directory))
        return None
    if not demand_known(*directory_state(given), path=given, what='the run directory'):
        refuse('{} does not exist; the launcher creates the attempt and this trainer '
               'never does'.format(given))
    root = Path(arm_root) if arm_root is not None else given.parent
    if not demand_known(*directory_state(root), path=root, what='the arm root'):
        refuse('{} does not exist; this trainer registers only in the arm whose lock '
               'it holds'.format(root))
    canonical_root = exp11_pathprobe.canonical(root)
    if not exp11_pathprobe.valid_path(canonical_root):
        refuse('{} does not canonicalise to an absolute, normalized path ({})'
               .format(root, canonical_root))
    if str(arm) != canonical_root:
        refuse('{} is an attempt of {}, not of the {} this trainer was given; the '
               'directory, the lock and the tombstone must be of one arm'
               .format(given, arm, root))
    with registration_lock(directory, wait_seconds):
        # Everything below happens under the lock, so a resolution cannot rename this
        # directory between the checks and the write. The checks are re-made here: the
        # directory may have been retired between the canonicalisation and the lock.
        if not demand_known(*directory_state(directory), path=directory,
                            what='the run directory'):
            refuse('{} does not exist; the launcher creates the attempt and this trainer '
                   'never does'.format(directory))
        if demand_known(*file_state(tombstone), path=tombstone, what='the tombstone'):
            refuse('{} was resolved as an unregistered launch and retired; this trainer '
                   'has nothing to run in'.format(directory))
        marker = demand_known(*file_state(directory / 'launching'),
                              path=directory / 'launching', what='the launching marker')
        if not marker:
            # The other half of a launch record. Its reader has the same three states,
            # and "I could not read child.pid" is not "there is no launch here".
            verdict, detail = exp11_pidrecord.inspect_record(directory / 'child.pid')
            if verdict == exp11_pidrecord.UNKNOWN:
                refuse('cannot inspect {}/child.pid ({}); this trainer does not register '
                       'on a launch it cannot verify'.format(directory, detail))
            if verdict != exp11_pidrecord.RECORD:
                refuse('{} carries no launch record -- no launching marker and no '
                       'complete child.pid -- so no launch of it is in progress'
                       .format(directory))
        path = directory / 'train.pid'
        # Atomic: a reader sees the old file or the whole new one, never a partial pid.
        temporary = directory / ('train.pid.{}.tmp'.format(os.getpid()))
        temporary.write_text('{}\n'.format(os.getpid()))
        os.replace(str(temporary), str(path))
    return path


def exit_code(exit_request):
    """``raise SystemExit`` carries None, which is status 0, not 1."""
    code = exit_request.code
    if code is None:
        return 0
    return code if isinstance(code, int) else 1


def record_trainer_exit(path, code):
    """Say how this trainer left, on every path it controls."""
    if path is None:
        return
    try:
        with open(str(path.parent / 'train.exit'), 'a') as stream:
            stream.write('train.exit {}\n'.format(code))
    except OSError:                       # a vanished run directory is not the trainer's
        pass                              # problem to report; the scan reads the pid file


def main(argv=None):
    """Register, run, and record how it ended -- whatever ends it."""
    args = parse_args(argv)
    registration = register_trainer(args.save_dir, no_save=args.no_save)
    code = 0
    try:
        run(list(sys.argv[1:] if argv is None else argv), check_admission(args))
    except SystemExit as exit_request:
        code = exit_code(exit_request)
        raise
    except BaseException:
        code = 'exception'
        raise
    finally:
        record_trainer_exit(registration, code)


def run(command, args):
    """The trainer's main, step for step, with the exp_11 registry and provenance."""
    trainer.seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32

    train_dataset = xRIR_Dataset(split='train', max_len=args.max_len, num_shot=args.num_shot)
    test_dataset = xRIR_Dataset(split='test', max_len=args.max_len, num_shot=args.num_shot)
    if args.test_subset and args.test_subset < len(test_dataset):
        idx = np.random.RandomState(args.seed).permutation(len(test_dataset))[: args.test_subset]
        test_dataset = Subset(test_dataset, sorted(idx.tolist()))
    print(f'train samples: {len(train_dataset)}  test samples: {len(test_dataset)}', flush=True)

    loader_kwargs = dict(num_workers=args.num_workers, pin_memory=True,
                         worker_init_fn=trainer.seed_worker,
                         persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                              **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size,
                             **loader_kwargs)
    args.train_batches_per_epoch = len(train_loader)
    check_counter(args, args.train_batches_per_epoch)
    args.env = {key: os.environ.get(key) for key in ENV_KEYS}
    model = build_model_exp11(args).cuda()

    destination = args.provenance_out if args.no_save else os.path.join(args.save_dir,
                                                                        'provenance.json')
    root = exp06_train.resolve_data_root()
    fields = provenance_fields(
        command, args.run_type,
        identity=exp06_train.data_identity(root) if destination else None,
        geometry=exp06_train.geometry_identity(root) if destination else None,
        heldout=exp06_train.heldout_wav_identity(root) if destination else None,
        approved=args.approved, reviewed_commit=args.reviewed_commit,
        exploratory=args.exploratory)
    prepare_args(args, model, fields, destination)
    fields['effective_args'] = vars(args)
    if destination:
        # The attempt directory is the launcher's, and it is this run's proof that a
        # launch is in progress; recreating it would undo a resolution (close review 8).
        if not args.no_save and not demand_known(
                *directory_state(args.save_dir), path=args.save_dir,
                what='the run directory'):
            refuse('{} vanished while this trainer was starting'.format(args.save_dir))
        provenance.write_manifest(destination, fields)  # exclusive; an attempt writes once
    print('XRIR_RUNTIME_ARGS ' + json.dumps(vars(args), sort_keys=True, allow_nan=False),
          flush=True)
    if not args.no_save:
        if not demand_known(*directory_state(args.save_dir), path=args.save_dir,
                            what='the run directory'):
            refuse('{} vanished while this trainer was starting'.format(args.save_dir))
        with open(os.path.join(args.save_dir, 'args.json'), 'w') as stream:
            json.dump(vars(args), stream, indent=2)

    n_params = sum(p.numel() for p in model.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 2
    print(f'backbone: {args.backbone}  params: {n_params / 1e6:.2f}M  '
          f'model size: {size_mb:.1f}MB', flush=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ExponentialLR(optimizer, decay_epochs=args.decay_epochs, gamma=args.lr_gamma)
    start_epoch, best_test_loss = 1, float('inf')
    history_path = os.path.join(args.save_dir, 'history.jsonl')
    print(f'yaw_aug ENABLED W={args.yaw_aug_width} seed={args.yaw_aug_seed} '
          f'counter=(epoch-1)*{args.train_batches_per_epoch}+batch_idx'
          if args.yaw_aug else 'yaw_aug DISABLED', flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss = trainer.train_epoch(model, train_loader, optimizer, scheduler, epoch,
                                         args, best_test_loss)
        test_loss = trainer.test_epoch(model, test_loader, epoch, args)
        if args.no_save:
            continue
        is_best = test_loss < best_test_loss
        if is_best:
            best_test_loss = test_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best.pth'))
        if args.epoch_ckpt_every and epoch % args.epoch_ckpt_every == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'epoch_{epoch:03d}.pth'))
        trainer.save_checkpoint(os.path.join(args.save_dir, 'last.pth'), model, optimizer,
                                scheduler, epoch, 0, best_test_loss, args)
        with open(history_path, 'a') as stream:
            stream.write(json.dumps({'epoch': epoch, 'train_loss': train_loss,
                                     'test_loss': test_loss, 'best_test_loss': best_test_loss,
                                     'is_best': is_best,
                                     'lr': optimizer.param_groups[0]['lr'],
                                     'epoch_minutes': (time.time() - t0) / 60.0}) + '\n')
        print(f'epoch {epoch} done in {(time.time() - t0) / 60:.1f} min; best test loss '
              f'{best_test_loss:.5f}{" (new best)" if is_best else ""}', flush=True)


if __name__ == '__main__':
    main()
