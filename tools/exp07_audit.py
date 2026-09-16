"""exp_07 alignment audit: ``tools.yaw_aug``'s audit on the seen protocol's training loader.

Amendment A1: ``tools/yaw_aug.py`` keeps main's bytes, so the protocol-aware audit lives
here.  Everything measured is the reviewed function ``yaw_aug.alignment_audit`` applied to
``yaw_aug.AuditBatch``es of the loader that ``tools/exp07_train.py`` would build, with the
offsets of ``yaw_aug.YawAug`` seeded exactly as training seeds them (counter width = the
loader length of the selected protocol, so the seen cohort uses 9 265, not 9 261).

The audit PASSES when the rotation leaves every integer direct-path delay unchanged --
``changed_delays == 0`` over a non-empty cohort, which is what exp_04's artefact recorded.

    python tools/exp07_audit.py --protocol seen --n-batches 3 \
        --out ckpt/exp07/alignment_audit_seen.json

Run it from the repository root with the training environment (PYTHONHASHSEED=0): the seen
dataset module opens ``seen_test_split.pkl`` relative to the working directory.
"""
import argparse
import json
import os
import platform
import socket
import subprocess
from itertools import islice, tee
from pathlib import Path

import torch

import train_xRIR_backbone as trainer
from tools import exp07_train, yaw_aug
from treble_multi_room_dataset.treble_xRIR_dataset import BASE_DATA_PATH

REPO = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', choices=exp07_train.PROTOCOLS, default='seen')
    parser.add_argument('--n-batches', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=12,
                        help="approved training run's worker count")
    parser.add_argument('--out', required=True, help='artefact path; never defaulted')
    args = parser.parse_args(argv)
    if Path(args.out).exists():
        parser.error('refusing to overwrite ' + args.out)
    if args.n_batches <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        parser.error('batch counts/sizes must be positive and num-workers nonnegative')
    return parser, args


def build_loader(args):
    """The training loader of ``args.protocol``, with the trainer's seeding and workers."""
    trainer.seed_everything(args.seed)
    dataset = exp07_train.dataset_class(args.protocol)(split='train', max_len=9600, num_shot=8)
    return trainer.DataLoader(yaw_aug._AuditDataset(dataset), shuffle=True,
                              batch_size=args.batch_size, num_workers=args.num_workers,
                              pin_memory=True, worker_init_fn=trainer.seed_worker,
                              persistent_workers=args.num_workers > 0)


def audit(args):
    """Measure the cohort; the result is the reviewed audit plus protocol provenance."""
    loader = build_loader(args)
    if args.n_batches > len(loader) or args.n_batches > yaw_aug._YAW_AUG_MAX_STEP:
        raise ValueError('requested cohort exceeds the training loader or the counter domain')
    # Model initialization precedes iterator creation: it determines sampler/worker RNG seeds.
    model = trainer.build_xrir('simple', 8).to('cuda' if torch.cuda.is_available() else 'cpu')
    batches, for_offsets = tee(yaw_aug.AuditBatch(batch, paths)
                               for batch, paths in islice(loader, args.n_batches))
    aug = yaw_aug.YawAug(True, 512, args.seed, len(loader))
    offsets = (aug.offsets_for(1, int(index), int(batch[1].shape[0]))
               for index, batch in enumerate(for_offsets))
    result = yaw_aug.alignment_audit(model, batches, offsets)
    result['schema_version'] = SCHEMA_VERSION
    result['passed'] = bool(result['pairs'] > 0 and result['changed_delays'] == 0)
    result['args'] = {'seed': args.seed, 'n_batches': args.n_batches,
                      'batch_size': args.batch_size, 'protocol': args.protocol,
                      'num_workers': args.num_workers, 'loader_batches': aug.batches_per_epoch,
                      'W': aug.W, 'data_root': str(Path(BASE_DATA_PATH).resolve()),
                      'PYTHONHASHSEED': os.environ.get('PYTHONHASHSEED'),
                      'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                                          cwd=str(REPO), text=True).strip()}
    result['env'] = {'python': platform.python_version(), 'torch': torch.__version__,
                     'cuda': torch.version.cuda, 'hostname': socket.gethostname()}
    return result


def main(argv=None):
    args = parse_args(argv)[1]
    result = audit(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as handle:
        handle.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    if not result['passed']:
        raise SystemExit(1)
    return result


if __name__ == '__main__':
    main()
