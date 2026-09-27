"""exp_11's backbone registry: exp_06's routes plus the two SimpleViT cue models.

``BACKBONES_EXP06`` is never mutated. Its digest is recorded in every completed exp_06
and exp_09 child's ``provenance.json`` and recomputed at finalisation, so a new key in
that mapping would invalidate evidence that is already certified. ``BACKBONES_EXP11`` is
a separate mapping built from it; the inherited routes are the very same class objects,
so a model built through either factory is the pinned model.

    build_xrir_exp11('simple_oriented', 8)     # arm H / I: the five-channel SimpleViT
    build_xrir_exp11('simple_adapter', 8)      # arm J / K: the zero-initialised adapter
"""
import hashlib
import json

from model.xRIR_cyl_oriented import BACKBONES_EXP06
from model.xRIR_simple_adapter import xRIR_SimpleAdapter
from model.xRIR_simple_oriented import xRIR_SimpleOriented

BACKBONES_EXP11 = {**BACKBONES_EXP06,
                   'simple_oriented': xRIR_SimpleOriented,
                   'simple_adapter': xRIR_SimpleAdapter}


def build_xrir_exp11(backbone, num_shot, **kwargs):
    """Build one exp_11 backbone; an unregistered name is refused, never guessed."""
    if backbone not in BACKBONES_EXP11:
        raise ValueError(f'unknown backbone {backbone!r}; choose from {sorted(BACKBONES_EXP11)}')
    return BACKBONES_EXP11[backbone](num_channels=num_shot, **kwargs)


def registry_sha256():
    """Digest of exp_11's registry, defined exactly as ``tools.exp06_train`` defines its own."""
    mapping = {name: cls.__module__ + '.' + cls.__qualname__
               for name, cls in BACKBONES_EXP11.items()}
    return hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()
