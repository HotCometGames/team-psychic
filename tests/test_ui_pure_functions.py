"""Tests for pure UI helper methods and the game-over button-disable regression.

All tests here run headlessly — no Tkinter window is created.
Pure static methods are called directly on the class; the button regression
uses object.__new__(UI) to bypass __init__.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from gameplay.constants import SHIFT_LENGTH
from gameplay.enums import ActionState
from ui.ui import UI

# ---------------------------------------------------------------------------
# UI._scram_fallback — never surface an unactionable "scram" suggestion
# ---------------------------------------------------------------------------

_PROBS_ZOMBIE = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
_PROBS_HEALTHY = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
_PROBS_CORPSE = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def test_scram_fallback_squishes_predicted_zombie():
    """A scram recommendation with a predicted zombie falls back to squish."""
    assert UI._scram_fallback(_PROBS_ZOMBIE) == ActionState.SQUISH


def test_scram_fallback_squishes_predicted_corpse():
    """A scram recommendation with a predicted corpse falls back to squish."""
    assert UI._scram_fallback(_PROBS_CORPSE) == ActionState.SQUISH


def test_scram_fallback_skips_predicted_healthy():
    """A scram recommendation with a predicted healthy/injured falls back to skip (can't save)."""
    assert UI._scram_fallback(_PROBS_HEALTHY) == ActionState.SKIP


def test_scram_fallback_skips_when_no_probs():
    """With no classifier (probs=None), the fallback defaults to skip."""
    assert UI._scram_fallback(None) == ActionState.SKIP

# ---------------------------------------------------------------------------
# UI._should_blur_images — blur starts in round 2 for every mode
# ---------------------------------------------------------------------------


def test_should_blur_images_from_round_two_for_all_modes():
    """Round 2 should blur images even when no AI start round is configured."""
    ui = object.__new__(UI)
    ui.round_num = 2
    ui._ai_starts_round = None

    assert ui._should_blur_images() is True


def test_should_blur_images_before_round_two_is_off():
    """Images stay sharp before round 2."""
    ui = object.__new__(UI)
    ui.round_num = 1
    ui._ai_starts_round = None

    assert ui._should_blur_images() is False


# ---------------------------------------------------------------------------
# Regression: game-over buttons stay disabled even when log=True causes reset()
#
# Bug: get_next() called disable_buttons() AFTER scorekeeper.reset(), which
# restored remaining_time to SHIFT_LENGTH, fooling disable_buttons() into
# re-enabling all buttons at game end.
#
# Fix: disable_all() is now called BEFORE reset(). disable_buttons() is only
# called in the continue-play else branch.
# ---------------------------------------------------------------------------


def _make_headless_ui(log: bool = True) -> UI:
    """Build a UI instance without Tkinter, with just enough attributes for get_next()."""
    ui = object.__new__(UI)
    ui.log = log
    ui.humanoid = MagicMock()
    ui._last_probs = None
    ui._last_recommended_action = None
    ui.classifier = None
    ui.recommender = None
    ui._info_label = None
    ui.button_menu = MagicMock()
    ui.game_viewer = MagicMock()
    ui.capacity_meter = MagicMock()
    ui.horde_meter = MagicMock()
    ui.value_label = MagicMock()
    ui.round_num = 1
    ui.total_saved = 0
    ui.total_killed = 0
    ui.total_saved_value = 0.0
    ui.total_cured = 0
    ui.round_scores = []
    ui.engineer_bonus_ft = 0.0
    ui.saved_passengers = []
    return ui


def _make_game_over_scorekeeper(remaining_time: int = 0) -> MagicMock:
    """Return a mock ScoreKeeper in a game-over state.

    reset() side-effect restores remaining_time to SHIFT_LENGTH, reproducing
    the exact condition that triggered the original bug.
    """
    sk = MagicMock()
    sk.remaining_time = remaining_time
    sk.get_score.return_value = {
        "saved": 0,
        "killed": 0,
        "saved_value": 0.0,
        "cured": 0,
    }
    sk.get_current_capacity.return_value = 0

    def _reset_side_effect():
        sk.remaining_time = SHIFT_LENGTH  # this is what caused the bug

    sk.reset.side_effect = _reset_side_effect
    return sk


def test_game_over_disable_all_called_before_reset():
    """disable_all() is called before scorekeeper.reset() at game end."""
    ui = _make_headless_ui(log=True)
    sk = _make_game_over_scorekeeper(remaining_time=0)
    dp = MagicMock()
    dp.unvisited = set()  # no humanoids left

    call_order = []
    ui.button_menu.disable_all.side_effect = lambda: call_order.append("disable_all")
    sk.reset.side_effect = lambda: call_order.append("reset")  # type: ignore[method-assign]

    ui.get_next(data_fp="data", data_parser=dp, scorekeeper=sk)

    assert "disable_all" in call_order, "disable_all() was never called at game end"
    assert "reset" in call_order, "reset() was never called despite log=True"
    assert call_order.index("disable_all") < call_order.index("reset"), (
        "disable_all() must be called BEFORE reset() — "
        "reset() restores remaining_time which would fool disable_buttons()"
    )


def test_game_over_disable_buttons_not_called():
    """disable_buttons() is never called in the game-over path (only in continue-play path)."""
    ui = _make_headless_ui(log=True)
    sk = _make_game_over_scorekeeper(remaining_time=0)
    dp = MagicMock()
    dp.unvisited = set()

    ui.get_next(data_fp="data", data_parser=dp, scorekeeper=sk)

    ui.button_menu.disable_buttons.assert_not_called()


def test_game_over_no_log_buttons_still_disabled():
    """disable_all() is called even when log=False (no reset() called in that case)."""
    ui = _make_headless_ui(log=False)
    sk = _make_game_over_scorekeeper(remaining_time=0)
    dp = MagicMock()
    dp.unvisited = set()

    ui.get_next(data_fp="data", data_parser=dp, scorekeeper=sk)

    ui.button_menu.disable_all.assert_called_once()
    sk.reset.assert_not_called()
