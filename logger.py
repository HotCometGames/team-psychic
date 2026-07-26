"""Simple CSV-backed game logging utilities."""

from __future__ import annotations

import csv
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SESSIONS_CSV = DATA_DIR / "sessions.csv"
MOVES_CSV = DATA_DIR / "moves.csv"

_SESSIONS_COLUMNS = [
    "timestamp",
    "game_id",
    "round",
    "timestamp_start",
    "total_saved",
    "total_cured",
    "total_injured",
    "total_killed",
]

_MOVES_COLUMNS = [
    "timestamp",
    "move_number",
    "game_id",
    "round",
    "num_people_in_view",
    "default_cnn_prediction",
    "default_cnn_confidence",
    "role_cnn_prediction",
    "role_cnn_confidence",
    "humanoid_state_truth",
    "role_truth",
    "horde_distance",
    "action_taken",
    "agreed_with_cnn",
]

_LOCK = threading.Lock()


def _ensure_csv(path: Path, columns: list[str]) -> None:
    """Create the CSV file and header row if it does not already exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()


def _initialize_files() -> None:
    """Create the data directory and both log files on import."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_csv(SESSIONS_CSV, _SESSIONS_COLUMNS)
    _ensure_csv(MOVES_CSV, _MOVES_COLUMNS)


_initialize_files()


class GameLogger:
    """Append game move and session summaries to CSV files."""

    def __init__(self, game_id: str | None = None) -> None:
        self.game_id = str(game_id) if game_id is not None else str(uuid.uuid4())

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _next_move_number(self) -> int:
        if not MOVES_CSV.exists():
            return 1
        with MOVES_CSV.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return 1
        return max(int(row["move_number"]) for row in rows if row.get("move_number")) + 1

    def _append_row(self, path: Path, columns: list[str], row: dict[str, Any]) -> None:
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            if path.stat().st_size == 0:
                writer.writeheader()
            writer.writerow(row)

    def log_move(
        self,
        round: int,
        num_people_in_view: int,
        default_cnn_prediction: Any,
        default_cnn_confidence: float,
        role_cnn_prediction: Any,
        role_cnn_confidence: float,
        humanoid_state_truth: Any,
        role_truth: Any,
        horde_distance: int,
        action_taken: Any,
    ) -> None:
        """Append one move row to the moves log with a monotonic move number."""
        agreed_with_cnn = (
            str(default_cnn_prediction).lower() == str(action_taken).lower()
            or str(role_cnn_prediction).lower() == str(action_taken).lower()
        )

        with _LOCK:
            _initialize_files()
            move_number = self._next_move_number()
            row = {
                "timestamp": self._timestamp(),
                "move_number": move_number,
                "game_id": self.game_id,
                "round": round,
                "num_people_in_view": num_people_in_view,
                "default_cnn_prediction": default_cnn_prediction,
                "default_cnn_confidence": default_cnn_confidence,
                "role_cnn_prediction": role_cnn_prediction,
                "role_cnn_confidence": role_cnn_confidence,
                "humanoid_state_truth": humanoid_state_truth,
                "role_truth": role_truth,
                "horde_distance": horde_distance,
                "action_taken": action_taken,
                "agreed_with_cnn": str(agreed_with_cnn),
            }
            self._append_row(MOVES_CSV, _MOVES_COLUMNS, row)

    def log_session_end(
        self,
        round: int,
        timestamp_start: str,
        total_saved: int,
        total_cured: int,
        total_injured: int,
        total_killed: int,
    ) -> None:
        """Append a session summary row to the sessions log."""
        with _LOCK:
            _initialize_files()
            row = {
                "timestamp": self._timestamp(),
                "game_id": self.game_id,
                "round": round,
                "timestamp_start": timestamp_start,
                "total_saved": total_saved,
                "total_cured": total_cured,
                "total_injured": total_injured,
                "total_killed": total_killed,
            }
            self._append_row(SESSIONS_CSV, _SESSIONS_COLUMNS, row)


__all__ = ["GameLogger"]
