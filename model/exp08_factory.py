"""exp_08 factory; the pinned ``simple`` / ``cylindrical`` entries keep their exact classes.

Mirrors ``model/xRIR_cyl_oriented.py`` (exp_06): the new backbones are *added* to the
pinned registry, never substituted into it, so ``simple`` still builds ``model.xRIR.xRIR``
and ``cylindrical`` still builds ``model.xRIR_cyl.xRIR_Cyl`` -- object identity, not just
equal behaviour.  That keeps exp_01/exp_03's results attributable to the code they were
produced with (handoff section 3.1: "keep the old ``simple`` and ``cylindrical`` behaviour",
"do not overwrite other sessions' code, runs or checkpoints").

The two new entries are the invariant-readout arms of the handoff's 2x2 (section 5):

==================  =========================================  =======================
backbone            class                                      C16 guarantee
==================  =========================================  =======================
``simple``          ``model.xRIR.xRIR``                        no (pinned baseline)
``cylindrical``     ``model.xRIR_cyl.xRIR_Cyl``                no (encoder only)
``simple_invariant``  ``xRIR_SimpleInvariant``                 no (control arm)
``cylindrical_invariant``  ``xRIR_CylInvariant``               **yes** (Route 2)
==================  =========================================  =======================
"""
from model.xRIR import xRIR
from model.xRIR_cyl import BACKBONES
from model.xRIR_cyl_invariant import xRIR_CylInvariant, xRIR_SimpleInvariant

#: The pinned registry plus the exp_08 arms.  ``BACKBONES`` is copied, not mutated.
BACKBONES_EXP08 = {**BACKBONES,
                   "cylindrical_invariant": xRIR_CylInvariant,
                   "simple_invariant": xRIR_SimpleInvariant}

#: Backbones whose single forward pass is structurally invariant under the C16 scene yaw.
INVARIANT_BACKBONES = ("cylindrical_invariant",)

assert BACKBONES_EXP08["simple"] is xRIR, "the pinned simple entry must not be rebound"


def build_xrir_exp08(backbone, num_shot, **kwargs):
    """Build a pinned backbone or one of the exp_08 invariant-readout arms.

    Args:
        backbone: a key of :data:`BACKBONES_EXP08`.
        num_shot: number of reference RIRs, i.e. the model's ``num_channels``.
        **kwargs: forwarded to the class (``dim``, ``depth``, ``heads``, ``mlp_dim``,
            ``image_size``, ``patch_size``, and ``basis_eps`` for the invariant arms).

    Returns:
        The constructed model.

    Raises:
        ValueError: if ``backbone`` is not registered.
    """
    if backbone not in BACKBONES_EXP08:
        raise ValueError("unknown backbone {!r}; choose from {}".format(
            backbone, sorted(BACKBONES_EXP08)))
    return BACKBONES_EXP08[backbone](num_channels=num_shot, **kwargs)
