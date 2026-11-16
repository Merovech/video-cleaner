"""The one HELD_REASON constant and the shared held-stage detection."""

from __future__ import annotations

from pathlib import Path

from video_cleaner import encode, quality
from video_cleaner import held as held_module
from video_cleaner.held import HELD_REASON, held_stages


def _stub() -> None:
    raise NotImplementedError(HELD_REASON)


def _real() -> None:
    pass


def _flagged() -> None:
    pass


_flagged.held = True  # type: ignore[attr-defined]


def test_held_stages_by_identity_and_by_attribute() -> None:
    stages = {"a": _stub, "b": _real, "c": _flagged}
    assert held_stages(stages, (_stub,)) == ["a", "c"]
    assert held_stages({"b": _real}, (_stub,)) == []


def test_every_module_uses_the_one_constant() -> None:
    assert HELD_REASON == "held: Task 10b"
    package = Path(held_module.__file__).parent
    spelled = [p.name for p in package.glob("*.py") if HELD_REASON in p.read_text("utf-8")]
    assert spelled == ["held.py"]  # every other module imports the constant
    for stub, args in (
        (encode.search_crf, (None,) * 5),
        (encode.encode_full, (None,) * 6),
        (quality.vmaf_mean, (None,) * 5),
    ):
        try:
            stub(*args)
        except NotImplementedError as exc:
            assert exc.args == (HELD_REASON,)
        else:  # pragma: no cover - a held stub always raises
            raise AssertionError(f"{stub} did not raise")
