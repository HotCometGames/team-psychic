import json
import os
import uuid
from typing import Any
from datetime import datetime
import pandas as pd

import gspread
from gspread_dataframe import set_with_dataframe

from gameplay.constants import HEADCOUNT_WEIGHT, VALUE_WEIGHT
from gameplay.enums import ActionCost, ActionState
from gameplay.humanoid import Humanoid

CNN_CLASS_MAP = {
    0: "zombie",
    1: "healthy",
    2: "injured",
    3: "corpse"
}

CNN_ROLE_MAP = {
    0: "none",
    1: "medic",
    2: "engineer"
}

# Convenience mappings to make converting these throughout the code less error prone
MAP_ACTION_STR_TO_INT: dict[str, int] = {s.value: i for i, s in enumerate(ActionState)}
MAP_ACTION_INT_TO_STR: list[str] = [s.value for s in ActionState]


class ScoreKeeper:
    """Tracks game state, scoring, and action history.

    Maintains the ambulance manifest, kill/save tallies, remaining time, and
    a per-episode action log. Also exposes the action helpers used by the RL
    environment (``map_do_action``) and the UI (``save``, ``squish``, etc.).

    Attributes:
        shift_length: Total game duration in minutes.
        capacity: Maximum number of humanoids the ambulance can hold at once.
        seed: Random seed recorded in ``summary.csv``. ``None`` if no seed was set.
        num_actions: Number of distinct actions (length of :class:`~gameplay.enums.ActionState`).
        current_run_id: UUID4 string identifying the current episode. Generated
            at construction and after each ``reset()``. Written to both
            ``log.csv`` and ``summary.csv`` so the two files can be joined.
        logger: Action log for the current episode.
        all_logs: Completed episode records. Each entry is a dict with keys
            ``run_id`` (str), ``log`` (list of action dicts), and ``score``
            (final score dict captured before reset).
        ambulance: Count of each humanoid type currently on board.
        ambulance_medics: Number of medics currently on board (alive — injured
            or healthy — subset of ``ambulance["injured"] + ambulance["healthy"]``).
        ambulance_engineers: Number of engineers currently on board (same
            alive-subset rule as ``ambulance_medics``).
        score: Cumulative kill/save tallies for the current episode; keys are
            ``"killed"``, ``"saved"``, ``"saved_value"`` (float — cumulative
            value delivered to hospital, post-scram, zombie-free),
            ``"injured_saved"`` (int — saved this episode who are still injured
            and not yet cured), and ``"cured"`` (int — injured survivors healed
            by a doctor at the camp screen via :meth:`cure_injured`).
        squished: Total number of squish actions taken this episode.
        remaining_time: Minutes left in the current shift.
        ambulance_value: Running sum of the values of humanoids currently on the ambulance.
        medics_saved: Number of medics successfully delivered to the hospital
            this episode. Each one grants 2 cure charges at the camp screen.
    """

    shift_length: int
    capacity: int
    seed: int | None
    num_actions: int
    current_run_id: str
    logger: list[dict[str, Any]]
    all_logs: list[dict[str, Any]]
    ambulance: dict[str, int]
    ambulance_medics: int
    ambulance_engineers: int
    score: dict[str, int | float]
    squished: int
    remaining_time: int
    ambulance_value: float
    medics_saved: int

    def __init__(
        self, shift_length: int, capacity: int, seed: int | None = None
    ) -> None:
        """Initialize a ScoreKeeper.

        Args:
            shift_length: Total game duration in minutes.
            capacity: Maximum number of humanoids the ambulance can hold at once.
            seed: Random seed used for this run, recorded in ``summary.csv``.
                Pass ``None`` (default) if no seed was set.
        """
        self.shift_length = int(shift_length)  # minutes
        self.capacity = capacity
        self.seed = seed
        self.num_actions = len(ActionState)
        self.logger = []
        self.all_logs = []
        self._all_logs_written = (
            0  # cursor: how many all_logs entries have been written
        )
        self.current_run_id = str(uuid.uuid4())
        self.reset()

    def reset(self) -> None:
        """Reset game state for a new episode.

        Captures the completed episode's score and logger into ``all_logs``
        (if any actions were taken), generates a fresh UUID for the next
        episode, then clears per-episode state.

        If the ambulance has passengers at reset time (e.g. the RL training
        loop called reset() without calling get_score() first), a final scram
        is issued so in-flight passengers are correctly reflected in all_logs.
        """
        if sum(getattr(self, "ambulance", {}).values()) > 0:
            self.get_score()  # finalize in-flight passengers before capturing score
        prev_score = dict(getattr(self, "score", {}))
        if self.logger:
            self.all_logs.append(
                {
                    "run_id": self.current_run_id,
                    "log": self.logger,
                    "score": prev_score,  # captured before reset
                }
            )
            self.current_run_id = str(uuid.uuid4())

        self.ambulance = {"zombie": 0, "injured": 0, "healthy": 0, "corpse": 0}
        self.ambulance_medics = 0
        self.ambulance_engineers = 0
        self.ambulance_value = 0.0
        self.score = {
            "killed": 0,
            "saved": 0,
            "saved_value": 0.0,
            "injured_saved": 0,
            "cured": 0,
        }
        self.squished = 0
        self.remaining_time = int(self.shift_length)  # minutes
        self.logger = []
        self.medics_saved = 0

    def log_turn(
        self,
        humanoid: "Humanoid | None",
        action: str,
        predicted_class: int | None = None,
        predicted_role: int | None = None,
        recommended_action: str | None = None,
        confidence: float | str | None = None,
        obs: list[float] | None = None,
        zombie_distance: float | None = None,
        ambulance_distance: float | None = None,
        blur: bool = False,
        is_split_screen: bool = False,
        people_in_view: int | None = None,
    ) -> None:
        """Append a streamlined, highly readable action entry to the current log."""
        
        # 1. Horde Distance
        distance_to_horde = (
            ambulance_distance - zombie_distance
            if ambulance_distance is not None and zombie_distance is not None
            else None
        )

        # 2. Format CNN prediction into a single clean string (e.g. "healthy (medic)")
        cnn_class_str = CNN_CLASS_MAP.get(predicted_class, "") if predicted_class is not None else ""
        cnn_role_str = CNN_ROLE_MAP.get(predicted_role, "") if predicted_role is not None else ""
        
        if cnn_class_str and cnn_role_str and cnn_role_str != "none":
            cnn_prediction = f"{cnn_class_str} ({cnn_role_str})"
        else:
            cnn_prediction = cnn_class_str or "N/A"

        # 3. Formatted pure numerical confidence percentage (e.g., 94.5)
        numeric_confidence = None
        if isinstance(confidence, (int, float)):
            numeric_confidence = round(confidence * 100, 2) if confidence <= 1.0 else round(confidence, 2)

        # 4. Humanoid ground truth summary
        truth_summary = "None"
        if humanoid is not None:
            state = getattr(humanoid, "img_class", getattr(humanoid, "state", "Humanoid"))
            role = getattr(humanoid, "role", "none")
            truth_summary = f"{state} ({role})" if role and role != "none" else str(state)

        # 5. Agreement check
        agreed_with_cnn = None
        if predicted_class is not None:
            agreed_with_cnn = (
                (action == "save" and predicted_class in [1, 2]) or 
                (action == "squish" and predicted_class == 0)
            )

        # Streamlined log dictionary
        self.logger.append(
            {
                "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "is_split": is_split_screen,
                "people_in_view": people_in_view if people_in_view is not None else 1,
                "blur": blur,
                "cnn_prediction": cnn_prediction,
                "confidence_pct": numeric_confidence,  # Clean number for Google Sheets calculations
                "capacity": self.get_current_capacity(),
                "humanoid_truth": truth_summary,
                "zombie_dist": zombie_distance,
                "horde_dist": distance_to_horde,
                "agreed_with_cnn": agreed_with_cnn,
                "action_taken": action,
            }
        )

    def save_log(
        self,
        spreadsheet_name: str = "GameLogs",
        sync_gsheets: bool = True,
        **kwargs: Any,
    ) -> None:
        """Save active logs directly to Google Sheets with session headers and summaries."""
        # 1. Gather all logged turns
        logs_to_save = []
        for entry in self.all_logs:
            if isinstance(entry, dict) and "log" in entry:
                logs_to_save.extend(entry["log"])
            elif isinstance(entry, dict):
                logs_to_save.append(entry)

        if self.logger:
            logs_to_save.extend(self.logger)

        if not logs_to_save:
            print("[Sheets Sync] No new log entries to save.")
            return

        df = pd.DataFrame(logs_to_save)
        df_clean = df.fillna("")
        data_rows = df_clean.values.tolist()

        # 2. Build Session Header and Summary Rows
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        headers = df_clean.columns.tolist()

        # Start Header Row (spans columns nicely)
        session_start_row = [f"--- SESSION START: {timestamp} ---"] + [""] * (len(headers) - 1)

        # Summary Metrics
        saved = self.score.get("saved", 0)
        cured = self.score.get("cured", 0)
        injured = self.score.get("injured_saved", 0)
        killed = self.score.get("killed", 0)

        # Session End Row with stats
        summary_text = (
            f"--- SESSION END: {timestamp} | "
            f"TOTAL SAVED: {saved} | CURED: {cured} | INJURED: {injured} | KILLED: {killed} ---"
        )
        session_end_row = [summary_text] + [""] * (len(headers) - 1)

        # 3. Sync and STACK in Google Sheets
        if sync_gsheets:
            try:
                gc = gspread.service_account(filename="credentials.json")
                sh = gc.open(spreadsheet_name)
                worksheet = sh.sheet1

                # Check if worksheet is completely empty
                existing_headers = worksheet.row_values(1)

                rows_to_append = []

                if not existing_headers:
                    # Write main column headers if it's a blank sheet
                    rows_to_append.append(headers)

                # Assemble: Start Divider -> Data Rows -> Summary End Divider
                rows_to_append.append(session_start_row)
                rows_to_append.extend(data_rows)
                rows_to_append.append(session_end_row)

                # Append everything to Google Sheets
                worksheet.append_rows(rows_to_append, value_input_option="USER_ENTERED")
                print(
                    f"Google Sheets synced! Added session log ({len(data_rows)} actions) "
                    f"with start/end markers to '{spreadsheet_name}'."
                )

            except gspread.exceptions.SpreadsheetNotFound:
                print(
                    f"[Google Sheets Error] Could not find spreadsheet named '{spreadsheet_name}'. "
                    "Make sure the sheet exists and is shared with your credentials.json client_email!"
                )
            except Exception as e:
                err_str = str(e)
                status_code = getattr(e, "status_code", None)
                if "200" in err_str or status_code == 200:
                    print(f"Google Sheets synced successfully! Added session log.")
                else:
                    print(f"[Google Sheets Error] Failed to sync logs: {e}")

        # Clear buffers AFTER saving
        self.logger.clear()
        self.all_logs.clear()
        
    def sync_to_google_sheets(
        self, 
        spreadsheet_name: str, 
        credentials_path: str = "credentials.json",
        log_path: str = "log.csv"
    ) -> None:
        """Appends formatted log data from log.csv to Google Sheets without overwriting."""
        if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
            print(f"Sync skipped: '{log_path}' is empty or missing.")
            return

        df = pd.read_csv(log_path)
        if df.empty:
            print("Sync skipped: 'log.csv' contains no data rows.")
            return

        try:
            gc = gspread.service_account(filename=credentials_path)
            sh = gc.open(spreadsheet_name)
            worksheet = sh.get_worksheet(0)

            # Convert Dataframe to clean list of rows, converting NaN to empty strings
            df_clean = df.fillna("")
            data_rows = df_clean.values.tolist()

            # Check if headers exist on row 1
            existing_headers = worksheet.row_values(1)

            if not existing_headers:
                # Sheet is empty: add headers + data
                headers = df_clean.columns.tolist()
                rows_to_append = [headers] + data_rows
            else:
                # Sheet already has headers: append ONLY new data rows
                rows_to_append = data_rows

            worksheet.append_rows(rows_to_append, value_input_option="USER_ENTERED")
            print(f"Successfully stacked {len(data_rows)} new rows to Google Sheets!")

        except Exception as e:
            print(f"[Google Sheets Error] Failed to sync logs: {e}")

    def save(self, humanoid: Humanoid) -> None:
        """Load a humanoid onto the ambulance.

        Deducts :attr:`~gameplay.enums.ActionCost.SAVE` minutes from
        ``remaining_time`` and increments the appropriate ambulance slot.

        Args:
            humanoid: The humanoid to save.
        """
        self.remaining_time -= ActionCost.SAVE.value
        if humanoid.is_zombie():
            self.ambulance["zombie"] += 1
        elif humanoid.is_corpse():
            # Corpse takes a slot but is harmless — discarded at scram, no kill-rule trigger.
            # Extension point: consider penalizing wasted capacity here.
            self.ambulance["corpse"] += 1
        elif humanoid.is_injured():
            self.ambulance["injured"] += 1
        else:
            self.ambulance["healthy"] += 1
        if humanoid.is_injured() or humanoid.is_healthy():
            self.ambulance_value += humanoid.value
            if humanoid.is_medic():
                self.ambulance_medics += 1
            if humanoid.is_engineer():
                self.ambulance_engineers += 1

    def squish(self, humanoid: Humanoid) -> None:
        """Squish a humanoid (destroy in place).

        Deducts :attr:`~gameplay.enums.ActionCost.SQUISH` minutes. Counts as a
        kill if the humanoid is not a zombie or corpse.

        Args:
            humanoid: The humanoid to squish.
        """
        self.remaining_time -= ActionCost.SQUISH.value
        self.squished += 1
        # Extension point: reward squishing zombies or penalize squishing humans.
        # This is a two-step change: (1) add a counter here (e.g. self.zombie_squished += 1
        # if humanoid.is_zombie()), then (2) use that counter in get_cumulative_reward()
        # to adjust the saved/killed totals. Changes here alone have no effect on reward.
        if not (humanoid.is_zombie() or humanoid.is_corpse()):
            self.score["killed"] += 1

    def skip(self, humanoid: Humanoid) -> None:
        """Skip a humanoid (leave them behind).

        Deducts :attr:`~gameplay.enums.ActionCost.SKIP` minutes. Counts as a
        kill if the humanoid is injured (they won't survive without rescue).
        Skipping a healthy, zombie, or corpse humanoid deducts time but has no
        effect on the score.

        Args:
            humanoid: The humanoid to skip.
        """
        self.remaining_time -= ActionCost.SKIP.value
        if humanoid.is_injured():
            self.score["killed"] += 1

    def scram(self, humanoid: Humanoid | None = None, deduct_time: bool = True) -> None:
        """Drive the ambulance to the hospital and unload all passengers.

        ``humanoid`` is accepted for API consistency but unused — scram applies
        to all passengers currently on board, not a single humanoid.

        Time is deducted even when the ambulance is empty and even when
        ``remaining_time`` is already 0 or negative (no floor).

        Deducts :attr:`~gameplay.enums.ActionCost.SCRAM` minutes. If any zombie
        is on board, all injured and healthy passengers are killed. Corpses are
        silently discarded (already dead, no kill-rule trigger). All ambulance
        slots are cleared after unloading.

        Args:
            humanoid: The humanoid present when scram was triggered, or ``None``
                if called internally (e.g. from ``get_score``).
            deduct_time: If ``False``, skip the time deduction. Used by
                ``get_score()`` when scramming at game-end to prevent driving
                ``remaining_time`` further below zero after the shift has ended.
        """
        if deduct_time:
            self.remaining_time -= ActionCost.SCRAM.value
        if self.ambulance["zombie"] > 0:
            # Any zombie on board kills all injured and healthy passengers.
            # Corpses are unaffected — they are already dead and are simply discarded.
            self.score["killed"] += (
                self.ambulance["injured"] + self.ambulance["healthy"]
            )
        else:
            self.score["saved"] += self.ambulance["injured"] + self.ambulance["healthy"]
            self.score["saved_value"] += self.ambulance_value
            self.score["injured_saved"] += self.ambulance["injured"]
            self.medics_saved += self.ambulance_medics

        self.ambulance["zombie"] = 0
        self.ambulance["injured"] = 0
        self.ambulance["healthy"] = 0
        self.ambulance["corpse"] = 0
        self.ambulance_medics = 0
        self.ambulance_engineers = 0
        self.ambulance_value = 0.0

    def available_action_space(self) -> list[bool]:
        """Return which actions are currently available.

        Returns:
            A list of bools in :class:`~gameplay.enums.ActionState` enum order.
            ``True`` means the action is available; ``False`` means it is
            disabled (e.g. Save is disabled when the ambulance is full).
        """
        action_dict = {s.value: True for s in ActionState}
        if self.remaining_time <= 0:
            action_dict["save"] = False
            action_dict["squish"] = False
            action_dict["skip"] = False
            # scram is always available — never disabled per design doc §8
        if self.at_capacity():
            action_dict["save"] = False
        return [action_dict[s.value] for s in ActionState]

    def map_do_action(self, action: ActionState, humanoid: Humanoid) -> bool:
        """Execute an action. Intended for RL use.

        Validates preconditions before acting (time remaining, capacity). Does
        nothing and returns ``False`` if the action is not currently available.

        Note:
            SCRAM is never blocked — it has no precondition guard. Calling SCRAM
            repeatedly continues to deduct ``ActionCost.SCRAM`` minutes on each
            call, with no floor on ``remaining_time``.

        Args:
            action: An :class:`~gameplay.enums.ActionState` member.
            humanoid: The humanoid to act on.

        Returns:
            ``True`` if the action was executed, ``False`` if it was blocked.
        """
        if action == ActionState.SAVE:
            if self.remaining_time <= 0 or self.at_capacity():
                return False
            self.save(humanoid)
        elif action == ActionState.SQUISH:
            if self.remaining_time <= 0:
                return False
            self.squish(humanoid)
        elif action == ActionState.SKIP:
            if self.remaining_time <= 0:
                return False
            self.skip(humanoid)
        elif action == ActionState.SCRAM:
            self.scram(humanoid)
        return True

    def get_cumulative_reward(self) -> float:
        """Return the current weighted reward combining headcount and value signals.

        Accounts for humanoids currently on the ambulance as if scram were
        called now, without modifying state.

        The step reward in ``train_env.step()`` is the *delta* between two
        calls: ``reward = get_cumulative_reward() - previous``. This means the
        reward for a single action can be a large swing, not just ±1.

        Example — zombie-contamination swing:
            Assumptions: no previously scrammed humanoids (score starts at 0);
            each injured humanoid has ``value = 1.0``.
            Ambulance has 3 injured (hypothetical saved = +3, saved_value = +3.0).
            Cumulative reward before save: 0.5 * (3 - 0) + 0.5 * 3.0 = 3.0.
            Agent saves a zombie: ambulance now has 1 zombie + 3 injured.
            Hypothetical saved drops to 0; killed jumps to +3; ambulance_value is lost.
            Cumulative reward after save: 0.5 * (0 - 3) + 0.5 * 0.0 = -1.5.
            Step delta = -1.5 - 3.0 = -4.5 for that one save action.

        Returns:
            Weighted reward: ``α × (saved − killed) + β × saved_value``, where
            α = ``HEADCOUNT_WEIGHT`` and β = ``VALUE_WEIGHT`` (both ``0.5`` by
            default — adjust in ``gameplay/constants.py``). Includes undelivered
            ambulance passengers hypothetically.
            Zombie contamination sets ``saved_value`` contribution to zero for
            in-flight passengers (their value is lost).
        """
        killed = self.score["killed"]
        saved = self.score["saved"]
        saved_value = self.score["saved_value"]
        if self.ambulance["zombie"] > 0:
            killed += self.ambulance["injured"] + self.ambulance["healthy"]
            # ambulance_value is lost — zombie contamination wipes all value
        else:
            saved += self.ambulance["injured"] + self.ambulance["healthy"]
            saved_value += self.ambulance_value

        # Reward = HEADCOUNT_WEIGHT × (saved − killed) + VALUE_WEIGHT × saved_value
        # Both weights are defined in gameplay/constants.py and must sum to 1.
        #
        # Notice that (saved - killed) can swing by up to
        # 2 × (capacity − 1) = 18 on a zombie contamination event (worst case:
        # ambulance full with capacity−1 injured plus 1 zombie), while
        # saved_value changes by at most 1.0
        # per step. Try adjusting HEADCOUNT_WEIGHT in constants.py — what happens
        # to agent behavior at the extremes (1.0 vs 0.0)?
        return HEADCOUNT_WEIGHT * (saved - killed) + VALUE_WEIGHT * saved_value

    def get_current_capacity(self) -> int:
        """Return the number of humanoids currently on the ambulance.

        Returns:
            Total occupied ambulance slots.
        """
        return sum(self.ambulance.values())

    def at_capacity(self) -> bool:
        """Return True if the ambulance is full.

        Returns:
            ``True`` if occupied slots >= ``capacity``.
        """
        return sum(self.ambulance.values()) >= self.capacity

    def cure_injured(self) -> int:
        """Apply doctor cures to this episode's injured survivors — call at camp.

        Each medic delivered to the hospital this episode (``medics_saved``)
        grants 2 cure charges. Cures are capped by both that capacity and the
        number of injured survivors still uncured (``score["injured_saved"]``).
        Cured survivors move from ``"injured_saved"`` to ``"cured"`` — they were
        already counted in ``"saved"``/``"saved_value"`` when the ambulance
        reached the hospital, so this only affects the injured/cured breakdown
        shown on the camp screen, not the total saved count.

        Call once per round, after :meth:`get_score` finalizes any in-flight
        passengers and before :meth:`reset` clears ``medics_saved`` for the
        next round.

        Returns:
            Number of injured survivors cured by this call.
        """
        cure_capacity = 2 * self.medics_saved
        cured_now = min(cure_capacity, self.score["injured_saved"])
        self.score["injured_saved"] -= cured_now
        self.score["cured"] += cured_now
        return cured_now

    def get_score(self) -> dict[str, int | float]:
        """Return the final score, scramming first if passengers remain.

        If the ambulance is non-empty, logs a synthetic scram entry
        (``humanoid=None``) and then calls ``scram()``. The log entry captures
        the final game-end state — it's a real game event with real consequences
        (passengers on board are saved or killed depending on zombie presence).
        Humanoid fields in this entry are ``None``, which appear as ``NaN``
        after a CSV round-trip via pandas.

        Returns:
            Dict with ``"killed"`` (int), ``"saved"`` (int), and
            ``"saved_value"`` (float) totals.
        """
        if sum(self.ambulance.values()) > 0:
            self.log_turn(None, "scram")
            self.scram(
                deduct_time=False
            )  # game is already over — don't push remaining_time further negative
        return self.score

    @staticmethod
    def get_action_idx(class_string: str) -> int:
        """Return the integer index for an action string.

        Args:
            class_string: An :class:`~gameplay.enums.ActionState` value string
                (e.g. ``"save"``).

        Returns:
            Integer index corresponding to the action in enum order.
        """
        return MAP_ACTION_STR_TO_INT[class_string]

    @staticmethod
    def get_action_string(class_idx: int) -> str:
        """Return the action string for an integer index.

        Args:
            class_idx: Integer index of the action in enum order.

        Returns:
            Action string (e.g. ``"save"``).
        """
        return MAP_ACTION_INT_TO_STR[class_idx]

    @staticmethod
    def get_all_actions() -> list[str]:
        """Return all action strings in enum order.

        Returns:
            List of action strings (e.g. ``["save", "squish", "skip", "scram"]``).

        Note:
            Returns a copy — safe to mutate; the original mapping is not affected.
        """
        return list(MAP_ACTION_INT_TO_STR)
