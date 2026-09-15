"""exp_07 training entry point: the pinned trainer under the authors' split conventions.

Amendment A1 (plan section 10a): exp_07 edits no pre-existing module, so ``--protocol``
lives here instead of in ``train_xRIR_backbone``.  This entry point

* imports the trainer and BOTH dataset modules at module level, so that the
  import-derived training closure of ``tools.exp07_train`` always contains
  ``treble_multi_room_dataset/treble_xRIR_seen_dataset.py`` and everything the trainer
  itself imports, whichever protocol a run selects;
* takes ``--protocol {unseen,seen}``, removes it from the argv the trainer parses, and
  binds ``trainer.xRIR_Dataset`` to the selected class before calling ``trainer.main()``;
* records ``protocol`` in ``args.json`` and in the ``XRIR_RUNTIME_ARGS`` line.

The recording seam is ``trainer.parse_args``.  The trainer builds both records inline in
``main()`` from ``vars(args)`` -- there is no writer or printer function to wrap -- so the
one seam that reaches both, before the first training step, is the namespace they are
built from.  ``run_trainer`` therefore calls the trainer's own ``parse_args`` and sets
``protocol`` on the namespace it returns; nothing else about the call is changed, and no
randomness is consumed, so ``--protocol unseen`` is the trainer's run.

    python tools/exp07_train.py --backbone simple --save-dir ckpt/exp07/seen_simple/... \
        --num-shot 8 --max-len 9600 --lr 1e-3 --weight-decay 1e-4 --decay-epochs 3 \
        --lr-gamma 0.1 --epochs 12 --batch-size 32 --accum-steps 2 --num-workers 12 \
        --seed 0 --tf32 --log-interval 50 --save-every 0 --epoch-ckpt-every 1 \
        --protocol seen

Run it from the repository root: the seen dataset module opens ``seen_test_split.pkl``
relative to the process working directory.
"""
import sys
from unittest.mock import patch

import train_xRIR_backbone as trainer
from treble_multi_room_dataset.treble_xRIR_dataset import xRIR_Dataset as UnseenDataset
from treble_multi_room_dataset.treble_xRIR_seen_dataset import xRIR_Dataset as SeenDataset

PROTOCOLS = ('unseen', 'seen')
FLAG = '--protocol'
SEEN_MODULE = 'treble_multi_room_dataset/treble_xRIR_seen_dataset.py'


def dataset_class(protocol):
    """The dataset class of the authors' split convention for ``protocol``."""
    if protocol not in PROTOCOLS:
        raise ValueError('unknown protocol: ' + str(protocol))
    return SeenDataset if protocol == 'seen' else UnseenDataset


def split_protocol(argv):
    """Return ``(protocol, trainer argv)``; ``--protocol`` is matched exactly, never by prefix.

    argparse would accept ``--proto`` as an abbreviation and would consume the trainer's
    own flags while doing it, so the scan below is explicit: only the literal
    ``--protocol VALUE`` and ``--protocol=VALUE`` forms are removed, at most once.
    """
    rest, protocol, index, argv = [], None, 0, list(argv)
    while index < len(argv):
        token = argv[index]
        if token != FLAG and not token.startswith(FLAG + '='):
            rest.append(token)
            index += 1
            continue
        if protocol is not None:
            raise ValueError(FLAG + ' given more than once')
        if token == FLAG:
            if index + 1 >= len(argv):
                raise ValueError(FLAG + ' requires a value')
            protocol, index = argv[index + 1], index + 2
        else:
            protocol, index = token[len(FLAG) + 1:], index + 1
        if protocol not in PROTOCOLS:
            raise ValueError('unknown protocol: ' + str(protocol))
    return protocol or PROTOCOLS[0], rest


def parse_args(argv=None):
    """The trainer's namespace plus ``protocol``: the recorded schema of an exp_07 run."""
    protocol, rest = split_protocol(sys.argv[1:] if argv is None else argv)
    with patch.object(sys, 'argv', [sys.argv[0]] + rest):
        args = trainer.parse_args()
    args.protocol = protocol
    return args


def run_trainer(argv=None):
    """Run ``trainer.main()`` on the selected protocol; restore every binding afterwards."""
    protocol, rest = split_protocol(sys.argv[1:] if argv is None else argv)
    selected, original = dataset_class(protocol), trainer.parse_args

    def parse_args_with_protocol():
        args = original()
        args.protocol = protocol
        return args

    with patch.object(sys, 'argv', [sys.argv[0]] + rest), \
            patch.object(trainer, 'xRIR_Dataset', selected), \
            patch.object(trainer, 'parse_args', parse_args_with_protocol):
        trainer.main()
    return protocol


if __name__ == '__main__':
    run_trainer()
