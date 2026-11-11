from __future__ import annotations

from vylor_estimator.parser import Turn

MAX_TURN_SECONDS = 600.0


def turn_seconds(turn: Turn) -> float:
    """Duration of a turn in seconds: None counts as 0 and the value is clamped."""
    if turn.duration_seconds is None:
        return 0.0
    return min(max(turn.duration_seconds, 0.0), MAX_TURN_SECONDS)

