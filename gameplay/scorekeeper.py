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
    0: "civilian",
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

    HEADERS = [
        "datetime",
        "blur",
        "is_split",
        "people_in_view",
        "cnn_prediction",
        "confidence_pct",
        "capacity",
        "dist_of horde",
        "agreed_with_cnn",
        "action_taken",
    ]

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
        obs: list[float] | None = None,
        confidence: float | str | None = None,
        zombie_distance: float | None = None,
        ambulance_distance: float | None = None,
        image_blurred: bool = False,
        is_split_screen: bool = False,
        people_in_view: int | None = None,
    ) -> None:
        """Append a streamlined, highly readable action entry to the current log."""
        
        if humanoid is None:
            active_humanoids = []
        elif isinstance(humanoid, list):
            active_humanoids = [h for h in humanoid if h is not None]
        else:
            active_humanoids = [humanoid]

        # 2. Extract detailed types/roles for people in view
        # 2. Extract ground-truth types/roles
        people_names = []
        for h in active_humanoids:
            # Check helper methods first, fallback to state/img_class attributes
            if getattr(h, "is_zombie", lambda: False)() or str(getattr(h, "state", getattr(h, "img_class", ""))).lower() in ["zombie", "0"]:
                name_item = "zombie"
            elif getattr(h, "is_corpse", lambda: False)() or str(getattr(h, "state", getattr(h, "img_class", ""))).lower() in ["corpse", "dead", "3"]:
                name_item = "corpse"
            elif getattr(h, "is_injured", lambda: False)() or str(getattr(h, "state", getattr(h, "img_class", ""))).lower() in ["injured", "2"]:
                name_item = "injured"
            else:
                role_attr = getattr(h, "role", "civilian")
                role_val = str(getattr(role_attr, "value", role_attr) or "").lower()
                name_item = role_val if role_val not in ["none", "", "civilian"] else "civilian"

            people_names.append(name_item)

        people_names_str = " | ".join(people_names) if people_names else "none"

        # 3. Standardize predictions into lists
        pred_classes = predicted_class if isinstance(predicted_class, list) else ([predicted_class] if predicted_class is not None else [])
        pred_roles = predicted_role if isinstance(predicted_role, list) else ([predicted_role] if predicted_role is not None else [])

        cnn_preds = []
        for idx in range(len(active_humanoids)):
            p_cls = pred_classes[idx] if idx < len(pred_classes) else None
            p_role = pred_roles[idx] if idx < len(pred_roles) else None

            cls_str = CNN_CLASS_MAP.get(p_cls, "").lower() if p_cls is not None else ""
            role_str = CNN_ROLE_MAP.get(p_role, "").lower() if p_role is not None else ""

            if cls_str in ["healthy", "default", "humanoid"]:
                cls_str = "civilian"

            if "zombie" in cls_str:
                cnn_preds.append("zombie")
            elif cls_str == "civilian":
                if role_str and role_str not in ["none", "civilian"]:
                    cnn_preds.append(role_str)
                else:
                    cnn_preds.append("civilian")
            elif cls_str:
                if role_str and role_str not in ["none", "civilian"]:
                    cnn_preds.append(f"{cls_str} ({role_str})")
                else:
                    cnn_preds.append(cls_str)

        cnn_prediction_str = " | ".join(cnn_preds) if cnn_preds else "n/a"

        # 4. Confidence Percentage formatting
        conf_list = confidence if isinstance(confidence, list) else ([confidence] if confidence is not None else [])
        conf_formatted = []
        for c in conf_list:
            if isinstance(c, (int, float)):
                val = c * 100 if c <= 1.0 else c
                conf_formatted.append(f"{val:.1f}")
            elif c is not None and str(c).strip():
                conf_formatted.append(str(c))

        numeric_confidence = " | ".join(conf_formatted) if conf_formatted else "n/a"
        # 5. Horde distance calculation
        distance_to_horde = (
            ambulance_distance - zombie_distance
            if ambulance_distance is not None and zombie_distance is not None
            else None
        )

        # 6. Agreement check
        agreed_with_cnn = None
        if pred_classes and pred_classes[0] is not None:
            agreed_with_cnn = (
                (action.lower() == "save" and pred_classes[0] in [1, 2]) or
                (action.lower() == "squish" and pred_classes[0] == 0)
            )

        # Streamlined log dictionary
        entry_dict = {
            "datetime": datetime.now().strftime("%m-%d %H:%M:%S"),
            "image_blurred": bool(image_blurred),
            "is_split": bool(is_split_screen or len(active_humanoids) > 1),
            "people_in_view": people_names_str,
            "cnn_prediction": cnn_prediction_str,
            "confidence_pct": numeric_confidence,
            "capacity": self.get_current_capacity(),
            "dist_of horde": distance_to_horde,
            "agreed_with_cnn": agreed_with_cnn,
            "action_taken": action.lower(),
        }
        self.logger.append(entry_dict)

        # Append directly to local log.csv
        file_exists = os.path.exists("log.csv") and os.path.getsize("log.csv") > 0
        df_row = pd.DataFrame([entry_dict])
        df_row.to_csv("log.csv", mode="a", index=False, header=not file_exists)

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

        # 2. SYNC TO GOOGLE SHEETS
        if sync_gsheets:
            try:
                gc = gspread.service_account(filename="credentials.json")
                sh = gc.open(spreadsheet_name)
                worksheet = sh.sheet1

                existing_headers = worksheet.row_values(1)
                data_rows = df_clean.values.tolist() 
                # rows_to_append = []

                if not existing_headers:
                    rows_to_append = [df_clean.columns.tolist()] + data_rows
                else:
                    rows_to_append = data_rows

                worksheet.append_rows(rows_to_append, value_input_option="USER_ENTERED")
                print(f"Google Sheets synced! Added {len(data_rows)} rows to '{spreadsheet_name}'.")
            except Exception as e:
                print(f"[Google Sheets Sync Error]: {e}")

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
    
