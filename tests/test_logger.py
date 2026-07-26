import csv
import importlib
from pathlib import Path

import logger


def setup_function() -> None:
    for path in [Path("data/sessions.csv"), Path("data/moves.csv")]:
        path.unlink(missing_ok=True)
    importlib.reload(logger)


def test_logger_creates_csv_files_and_writes_rows() -> None:
    game_logger = logger.GameLogger("game-123")
    game_logger.log_move(
        round=1,
        num_people_in_view=3,
        default_cnn_prediction="save",
        default_cnn_confidence=0.91,
        role_cnn_prediction="save",
        role_cnn_confidence=0.87,
        humanoid_state_truth="injured",
        role_truth="support",
        horde_distance=4,
        action_taken="save",
    )
    game_logger.log_session_end(
        round=1,
        timestamp_start="2026-07-24T00:00:00Z",
        total_saved=2,
        total_cured=1,
        total_injured=0,
        total_killed=1,
    )

    moves_path = Path("data/moves.csv")
    sessions_path = Path("data/sessions.csv")

    assert moves_path.exists()
    assert sessions_path.exists()

    with moves_path.open(newline="", encoding="utf-8") as handle:
        moves_rows = list(csv.DictReader(handle))
    assert len(moves_rows) == 1
    assert moves_rows[0]["move_number"] == "1"
    assert moves_rows[0]["game_id"] == "game-123"
    assert moves_rows[0]["action_taken"] == "save"
    assert moves_rows[0]["agreed_with_cnn"] in {"True", "False"}

    with sessions_path.open(newline="", encoding="utf-8") as handle:
        sessions_rows = list(csv.DictReader(handle))
    assert len(sessions_rows) == 1
    assert sessions_rows[0]["game_id"] == "game-123"
    assert sessions_rows[0]["total_saved"] == "2"
