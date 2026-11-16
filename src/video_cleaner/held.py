"""The Task 10b hold in one place: the reason text and how a held stage is recognised.

Encoding, CRF search, VMAF scoring and calibration stay signature stubs until Ari lifts the hold
(full 64 GB of memory back). Every stub raises ``NotImplementedError(HELD_REASON)``; the runner
and the calibration refuse to start while one is wired in. Lifting the hold means deleting this
module and its imports.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

HELD_REASON = "held: Task 10b"


def held_stages(stages: Mapping[str, object], held: Sequence[object]) -> list[str]:
    """Names in *stages* whose callable is one of the *held* stubs or carries ``held = True``."""
    return [
        name
        for name, stage in stages.items()
        if any(stage is stub for stub in held) or getattr(stage, "held", False)
    ]
