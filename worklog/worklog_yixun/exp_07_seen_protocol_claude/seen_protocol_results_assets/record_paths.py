"""Path identity for the exp_07 record: the names it publishes, and the files behind them.

These two helpers live with the record's generators rather than in ``tools/`` because
every module the producers import is inside a producer closure that has ALREADY been
published: ``tools.exp07_calibration`` imports ``tools.exp07_table``, which imports
``tools.exp07_record``, and the released-checkpoint calibration receipt in
``ckpt/exp07/results`` records that closure's digest.  Changing one byte of any of them
invalidates that evidence, which no amount of re-binding repairs.  The record's own
assets are loaded by path when a generator calls for them, so they are in no closure and
this is where the record may still learn something about the disk it reads.
"""
import os
from pathlib import Path


def logical(path):
    """A path as the record spells it: absolute and normalised, never resolved.

    A certified attempt -- or any ancestor of one, the products directory included -- is
    archived behind a directory symlink once the record is published, so the name the
    approval, the manifests, the sidecars and the binding report all give a file stays
    its name here too, and the documents this evidence is written into keep saying what
    the report says.  The bytes are still read, and digest-checked, through it.
    """
    return Path(os.path.abspath(str(path)))


def identical(one, other):
    """Two names for one file: relocation, and `final`, keep the inode, not the spelling."""
    try:
        return Path(one).samefile(Path(other))
    except OSError:
        return False
