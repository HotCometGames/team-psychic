"""Main game UI — Tkinter window with injectable classifier and recommender.

The ``classifier`` and ``recommender`` parameters accept any object that
subclasses :class:`~models.HumanoidClassifier.HumanoidClassifier` or
:class:`~models.ActionRecommender.ActionRecommender` respectively.
The examples below use the production defaults (``CNNClassifier`` and
``RLRecommender``), but you can substitute any compatible implementation —
for example ``OracleClassifier`` (perfect ground-truth labels) or
``HeuristicRecommender`` (rule-based, no RL weights needed).

Usage
-----
    # Pure human play — no machine assistance (mode: human)
    UI(data_parser, scorekeeper, data_fp, log=False)

    # Any HumanoidClassifier labels the humanoid; human decides the action (mode: human_cnn)
    UI(data_parser, scorekeeper, data_fp, log=False, classifier=CNNClassifier())
    UI(data_parser, scorekeeper, data_fp, log=False, classifier=OracleClassifier())

    # Any ActionRecommender suggests an action; human decides (mode: human_rl)
    # Classifier probs are computed internally and passed to the recommender;
    # the classification label is suppressed from the display.
    UI(data_parser, scorekeeper, data_fp, log=False,
       classifier=CNNClassifier(), recommender=RLRecommender(),
       show_classifier_label=False)

    # Classifier labels + recommender suggests; human decides (mode: human_cnn_rl)
    UI(data_parser, scorekeeper, data_fp, log=False,
       classifier=CNNClassifier(), recommender=RLRecommender())

    # Step-by-step machine play; advance-turn button auto-executes (mode: machine)
    UI(data_parser, scorekeeper, data_fp, log=False,
       classifier=CNNClassifier(), recommender=RLRecommender(), auto_execute=True)

-----------------------------------------------------------------------
NOTE: the auto game mode runs headlessly and does not use the UI
-----------------------------------------------------------------------
"""

import logging
import math
import tkinter as tk
from os.path import abspath, dirname, join


import numpy as np
from PIL import Image, ImageTk

from game_interfaces.data_parser import DataParser
from gameplay.constants import (
    AMBULANCE_CAPACITY,
    CNN_CONFIDENCE_HIGH,
    CNN_CONFIDENCE_MID,
    HORDE_INITIAL_DISTANCE,
    HORDE_MAX_DISTANCE,
    MAX_ROUNDS,
    UI_COLORS,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from gameplay.enums import HordeDistance
from gameplay.humanoid import Humanoid
from gameplay.scorekeeper import ScoreKeeper
from models.ActionRecommender import ActionRecommender
from models.HumanoidClassifier import HumanoidClassifier, probs_to_class
from ui.button_menu import ButtonMenu
from ui.capacity_meter import CapacityMeter
from ui.game_viewer import GameViewer

from random import random
from ui.horde_meter import HordeMeter


class UI:
    """Tkinter game window.

    Args:
        data_parser: DataParser supplying humanoid images.
        scorekeeper: ScoreKeeper tracking game state.
        data_fp: Root path for image files.
        log: If True, save a log.csv after the game ends.
        classifier: Optional HumanoidClassifier — probs are computed from
            this and passed to the recommender each turn. When
            ``show_classifier_label`` is True, the predicted class is also
            displayed on screen.
        recommender: Optional ActionRecommender — receives classifier probs
            each turn and returns a suggested action. In auto_execute mode
            the suggestion is also executed automatically. None = no
            suggestion display.
        auto_execute: If True, user action buttons are replaced by a single
            "Advance turn" button that fires the recommender and executes its
            action automatically. Requires recommender to be set. Defaults to
            False.
        show_classifier_label: If True (default), render the classifier
            prediction label on screen. Set to False to run the classifier
            internally (for the recommender) without showing the result to
            the player.

    Attributes:
        root: Tkinter root window.
        log: Whether to write a log.csv on game end.
        classifier: Injected classifier, or ``None``.
        recommender: Injected recommender, or ``None``.
        humanoid: The humanoid currently on screen.
        button_menu: Action button row widget.
        game_viewer: Central photo panel widget.
        capacity_meter: Ambulance capacity meter widget.
        horde_meter: Ambulance-vs-horde distance meter — the round ends when
            the horde catches the ambulance (caught) or the ambulance reaches
            camp (round complete), replacing the old shift-clock end condition.
        value_label: Label displaying the current humanoid's value and category.
    """

    root: tk.Tk
    log: bool
    classifier: HumanoidClassifier | None
    recommender: ActionRecommender | None
    _auto_execute: bool
    _show_classifier_label: bool
    round_num: int
    total_saved: int
    total_killed: int
    total_saved_value: float
    total_cured: int
    humanoid: Humanoid
    button_menu: ButtonMenu
    game_viewer: GameViewer
    capacity_meter: CapacityMeter
    horde_meter: HordeMeter
    value_label: tk.Label
    _info_label: tk.Label | None
    _last_probs: "np.ndarray | None"
    _last_recommended_action: "str | None"
    _game_label: tk.Label | None  # keeps game-logo PhotoImage alive
    _bwsi_label: tk.Label | None  # keeps BWSI-logo PhotoImage alive


    def __init__(
        self,
        data_parser: DataParser,
        scorekeeper: ScoreKeeper,
        data_fp: str,
        log: bool = False,
        classifier: HumanoidClassifier | None = None,
        recommender: ActionRecommender | None = None,
        auto_execute: bool = False,
        show_classifier_label: bool = True,
    ) -> None:
        """Build the game window and enter the Tkinter main loop.

        Args:
            data_parser: DataParser supplying humanoid images.
            scorekeeper: ScoreKeeper tracking game state.
            data_fp: Root path for image files.
            log: If True, save a log.csv after the game ends.
            classifier: Optional classifier — probs computed from this are
                passed to the recommender each turn. Label is displayed only
                when ``show_classifier_label`` is True.
            recommender: Optional recommender — displays action suggestions.
                In auto_execute mode the suggestion is also auto-executed.
            auto_execute: If True, replaces the four action buttons with a
                single "Advance turn" button. Requires ``recommender``.
            show_classifier_label: If True (default), the classifier's
                predicted class and confidence are shown on screen. If False,
                the classifier still runs (to supply probs to the recommender)
                but the label is hidden.
        """
        w, h = WINDOW_WIDTH, WINDOW_HEIGHT
        self.root = tk.Tk()
        self.root.title("Beaverworks SGAI 2026 - Dead or Alive")
        self.root.geometry(f"{w}x{h}")
        self.root.resizable(False, False)
        self.root.configure(bg=UI_COLORS["bg"])
        self._game_label = None
        self._bwsi_label = None
        self._create_header(w)

        self.log = log
        self.classifier = classifier
        self.recommender = recommender
        if recommender is not None and classifier is None:
            logging.warning(
                "UI: recommender is set but classifier is None — "
                "recommender.recommend() will never be called (probs will be None). "
                "Pass a classifier to enable recommendations."
            )
        self._auto_execute = auto_execute and recommender is not None
        self._show_classifier_label = show_classifier_label
        self._last_probs = (
            None  # cached per-turn classifier output; set by _update_info_display
        )
        self._last_recommended_action = None  # cached alongside _last_probs

        if len(data_parser.unvisited) == 0:
            raise ValueError(
                "Dataset is empty — no humanoids to draw. "
                "Check that the data directory contains a valid consolidated_metadata.csv."
            )
        self.round_num = 1
        self.total_saved = 0
        self.total_killed = 0
        self.total_saved_value = 0.0
        self.total_cured = 0
        self.round_scores = []  # per-round score dicts, appended at each round-end
        self.saved_passengers = []
        self.humanoid = data_parser.get_random()
        self.humanoids = [self.humanoid]


        # --- Action buttons ---
        if self._auto_execute:
            # Machine mode: single advance-turn button replaces user buttons
            def _on_advance() -> None:
                self._machine_advance(data_fp, data_parser, scorekeeper)

            machine_buttons = [("Advance turn", _on_advance)]
            self.button_menu = ButtonMenu(self.root, machine_buttons)
        else:
            # Human mode: standard four action buttons
            def _on_skip() -> None:
                self.button_menu.disable_all()
                scorekeeper.log_turn(
                    self.humanoid,
                    "skip",
                    **self._get_log_context(scorekeeper),
                )
                scorekeeper.skip(self.humanoid)
                self.horde_meter.advance(
                    zombie_delta=HordeDistance.ZOMBIE.value,
                    ambulance_delta=HordeDistance.SKIP.value,
                )
                self.update_ui(scorekeeper)
                self.get_next(data_fp, data_parser, scorekeeper)

            def _on_squish() -> None:
                self.button_menu.disable_all()

                if len(self.humanoids) > 1:
                    # Prompt the user to select an image by clicking it
                    self.value_label.config(text="SELECT: Click the image you want to SQUISH")

                    def handle_choice(chosen_index: int) -> None:
                        # Grab the targeted humanoid from the active list
                        selected_humanoid = self.humanoids[chosen_index]

                        # Temporarily swap self.humanoid context for accurate logging calculations
                        self.humanoid = selected_humanoid

                        scorekeeper.log_turn(
                            selected_humanoid,
                            "squish",
                            **self._get_log_context(scorekeeper),
                        )
                        scorekeeper.squish(selected_humanoid)

                        # Log any unchosen image(s) as skipped
                        for idx, h in enumerate(self.humanoids):
                            if idx != chosen_index:
                                self.humanoid = h
                                scorekeeper.log_turn(
                                    h,
                                    "skip",
                                    **self._get_log_context(scorekeeper),
                                )
                                scorekeeper.skip(h)

                        self.horde_meter.advance(
                            zombie_delta=HordeDistance.ZOMBIE.value,
                            ambulance_delta=HordeDistance.SQUISH.value,
                        )

                        # Clear the selection listener callback so it doesn't fire unexpectedly later
                        self.humanoid = selected_humanoid
                        self.game_viewer.on_image_selected = None

                        self.update_ui(scorekeeper)
                        if hasattr(self.button_menu, 'enable_all'):
                            self.button_menu.enable_all()
                        self.get_next(data_fp, data_parser, scorekeeper)

                    # Inject our selection handler closure into the game viewer
                    self.game_viewer.on_image_selected = handle_choice
                else:
                    scorekeeper.log_turn(
                        self.humanoid,
                        "squish",
                        **self._get_log_context(scorekeeper),
                    )
                    scorekeeper.squish(self.humanoid)
                    self.horde_meter.advance(
                        zombie_delta=HordeDistance.ZOMBIE.value,
                        ambulance_delta=HordeDistance.SQUISH.value,
                    )
                    self.update_ui(scorekeeper)
                    self.get_next(data_fp, data_parser, scorekeeper)

            def _on_save() -> None:
                self.button_menu.disable_all()

                if len(self.humanoids) > 1:
                    self.value_label.config(text="SELECT: Click the image you want to SAVE")

                    def handle_choice(chosen_index: int) -> None:
                        # Grab the targeted humanoid from the active list
                        selected_humanoid = self.humanoids[chosen_index]

                        # Temporarily swap self.humanoid context for accurate logging calculations
                        self.humanoid = selected_humanoid

                        scorekeeper.log_turn(
                            selected_humanoid,
                            "save",
                            **self._get_log_context(scorekeeper),
                        )
                        scorekeeper.save(selected_humanoid)
                        self.saved_passengers.append(selected_humanoid)

                        # Log any unchosen image(s) as skipped
                        for idx, h in enumerate(self.humanoids):
                            if idx != chosen_index:
                                self.humanoid = h
                                scorekeeper.log_turn(
                                    h,
                                    "skip",
                                    **self._get_log_context(scorekeeper),
                                )
                                scorekeeper.skip(h)

                        self.horde_meter.advance(
                            zombie_delta=HordeDistance.ZOMBIE.value,
                            ambulance_delta=HordeDistance.SAVE.value,
                        )

                        # Clear the selection listener callback so it doesn't fire unexpectedly later
                        self.humanoid = selected_humanoid
                        self.game_viewer.on_image_selected = None

                        self.update_ui(scorekeeper)
                        if hasattr(self.button_menu, 'enable_all'):
                            self.button_menu.enable_all()
                        self.get_next(data_fp, data_parser, scorekeeper)

                    # Inject our selection handler closure into the game viewer
                    self.game_viewer.on_image_selected = handle_choice
                else:
                    scorekeeper.log_turn(
                        self.humanoid,
                        "save",
                        **self._get_log_context(scorekeeper),
                    )
                    scorekeeper.save(self.humanoid)
                    self.saved_passengers.append(self.humanoid)
                    self.horde_meter.advance(
                        zombie_delta=HordeDistance.ZOMBIE.value,
                        ambulance_delta=HordeDistance.SAVE.value,
                    )
                    self.update_ui(scorekeeper)
                    self.get_next(data_fp, data_parser, scorekeeper)

            def _on_scram() -> None:
                self.button_menu.disable_all()
                scorekeeper.log_turn(
                    self.humanoid,
                    "scram",
                    **self._get_log_context(scorekeeper),
                )
                scorekeeper.scram(self.humanoid)
                # No HordeDistance member for scram yet (not currently a button) —
                # a big ambulance jump matches what scram represents (driving away).
                self.horde_meter.advance(
                    zombie_delta=HordeDistance.ZOMBIE.value, ambulance_delta=40
                )
                self.update_ui(scorekeeper)
                self.get_next(data_fp, data_parser, scorekeeper)

            user_buttons = [
                (f"Skip [+{HordeDistance.SKIP.value} ft]", _on_skip),
                (f"Squish [+{HordeDistance.SQUISH.value} ft]", _on_squish),
                (f"Save [+{HordeDistance.SAVE.value} ft]", _on_save),
            ]
            self.button_menu = ButtonMenu(self.root, user_buttons)

        # Validate initial button state against live game conditions.
        self.button_menu.disable_buttons(
            len(data_parser.unvisited),
            scorekeeper.at_capacity(),
        )

        # --- Central photo ---
        self.game_viewer = GameViewer(self.root, w, h, data_fp, self.humanoid)
        self.root.bind("<Delete>", self.game_viewer.delete_photo)

        # --- Ambulance capacity meter ---
        self.capacity_meter = CapacityMeter(self.root, w, h, AMBULANCE_CAPACITY)

        # --- Horde distance meter — replaces the shift clock as the round's
        # --- end condition (see get_next() and _start_next_round()).
        self.horde_meter = HordeMeter(
            self.root, w, h,
            max_distance=HORDE_MAX_DISTANCE,
            initial_distance=HORDE_INITIAL_DISTANCE,
        )

        # --- Value display ---
        self.value_label = tk.Label(
            self.root,
            text="Value: — / —",
            font=("Arial", 14),
            bg=UI_COLORS["bg"],
            fg=UI_COLORS["fg"],
        )
        self.value_label.place(x=math.floor(0.05 * w), y=math.floor(0.88 * h))
        self.value_label.config(text=self._value_label_text(self.humanoid.value))

        # --- Machine output display (classifier label + recommender label) ---
        self._info_label = None
        if classifier is not None or recommender is not None:
            self._info_label = tk.Label(
                self.root,
                text="",
                font=("Arial", 14),
                justify=tk.LEFT,
                anchor="w",
                bg=UI_COLORS["bg"],
                fg=UI_COLORS["fg"],
            )
            self._info_label.place(
                x=math.floor(0.75 * w),
                y=math.floor(0.1 * h),
                width=math.floor(0.5 * w),
                height=math.floor(0.125 * h),
            )
            # Show initial classification/recommendation for the first humanoid
            self._update_info_display(scorekeeper)

        def _on_close() -> None:
            # Only save if there is unsaved data — guards against double-writing
            # when the player closes the window after a game that already called
            # save_log() at game-end. Also catches mid-game window closes where
            # the current episode was never committed.
            # if self.log and (
            #     scorekeeper.logger
            #     or scorekeeper._all_logs_written < len(scorekeeper.all_logs)
            # ):
            #     scorekeeper.save_log()
            self.root.destroy()

        self.root.protocol("WM_DELETE_WINDOW", _on_close)

        # Rules screen — shown once, on top of everything else, before the
        # player can interact with the game underneath.
        self.game_viewer.display_rules(on_start=lambda: None)

        self.root.mainloop()

    # ------------------------------------------------------------------
    # Machine-mode advance turn
    # ------------------------------------------------------------------

    def _machine_advance(
        self,
        data_fp: str,
        data_parser: DataParser,
        scorekeeper: ScoreKeeper,
    ) -> None:
        """Fire the recommender, execute its action, and advance the turn.

        Called by the "Advance turn" button in machine mode. Does nothing
        if the game is already over.

        Args:
            data_fp: Root path for image files.
            data_parser: DataParser supplying the remaining humanoids.
            scorekeeper: ScoreKeeper tracking game state.
        """
        if scorekeeper.remaining_time <= 0 or len(data_parser.unvisited) == 0:
            return
        if self.classifier is None:
            raise ValueError(
                "Attempted to advance machine turn but classifier is None — cannot compute probs for the recommender. Pass a classifier to enable machine play."
            )
        if self.recommender is None:
            raise ValueError(
                "Attempted to advance machine turn but recommender is None — cannot compute action for the humanoid. Pass a recommender to enable machine play."
            )
        probs = self.classifier.classify(self.humanoid)
        action = self.recommender.recommend(self.humanoid, probs, scorekeeper)
        predicted_class = int(np.argmax(probs))
        max_prob = float(np.max(probs))
        confidence = (
            "high"
            if max_prob >= CNN_CONFIDENCE_HIGH
            else "mid" if max_prob >= CNN_CONFIDENCE_MID else "low"
        )
        action_executed = scorekeeper.map_do_action(action, self.humanoid)
        if action_executed:
            scorekeeper.log_turn(
                self.humanoid,
                action.value,
                predicted_class=predicted_class,
                recommended_action=action.value,
                confidence=confidence,
                obs=None,
                zombie_distance=self.horde_meter.zombie_distance,
                ambulance_distance=self.horde_meter.ambulance_distance,
            )
            self.update_ui(scorekeeper)
            self.get_next(data_fp, data_parser, scorekeeper)
        else:
            raise RuntimeError(
                f"_machine_advance: recommender returned {action!r} but the game engine "  # nosec B608
                f"blocked it (map_do_action returned False). "
                f"This should never happen — a recommender must only return actions that "
                f"are currently available. Before returning an action, call "
                f"scorekeeper.available_action_space() and select from the True entries. "
                f"If you wrote a custom ActionRecommender, fix recommend() to respect "
                f"the available action mask."
            )

    # ------------------------------------------------------------------
    # UI update helpers
    # ------------------------------------------------------------------

    def _get_log_context(self, scorekeeper: ScoreKeeper, target_humanoid: Humanoid | None = None) -> dict:
        """Return keyword args for ``scorekeeper.log_turn()`` with classifier context.

        Reads from ``self._last_probs`` (cached by ``_update_info_display``) to
        avoid a redundant forward pass. Falls back to recomputing if the cache
        is empty (e.g. when called before the first display update).

        Args:
            scorekeeper: ScoreKeeper passed to the recommender for context.

        Returns:
            Dict with ``predicted_class`` (``int | None``),
            ``recommended_action`` (``str | None``), ``confidence``, ``obs``,
            and the horde meter's current ``zombie_distance``/
            ``ambulance_distance`` — the position of each at the moment this
            decision was made, before this turn's ``horde_meter.advance()``
            call moves them.
        """
        h = target_humanoid if target_humanoid is not None else self.humanoid

        predicted_class = None
        predicted_role = None
        recommended_action = None
        confidence = None

        probs = None # use cached output from _update_info_display
        if probs is None and self.classifier is not None:
            probs = self.classifier.classify(h)

        if probs is not None:
            predicted_class = int(np.argmax(probs))
            max_prob = float(np.max(probs))
            confidence = max_prob  # Pass as raw float (0.0 - 1.0) for numeric percentage formatting

            # Fetch predicted role if the classifier supports role classification
            if hasattr(self.classifier, "classify_role"):
                role_probs = self.classifier.classify_role(h, state_probs=probs)
                if role_probs is not None:
                    predicted_role = int(np.argmax(role_probs))

        if self.recommender is not None and probs is not None:
            # Use cached action from _update_info_display — avoids a second recommend()
            # call that could return a different result for stochastic recommenders.
            recommended_action = self.recommender.recommend(
                h, probs, scorekeeper
            ).value

        is_split_screen = len(getattr(self, "humanoids", [])) > 1
        people_in_view = len(getattr(self, "humanoids", [h]))

        return {
            "predicted_class": predicted_class,
            "predicted_role": predicted_role,
            "recommended_action": recommended_action,
            "confidence": confidence,
            "obs": None,
            "zombie_distance": self.horde_meter.zombie_distance,
            "ambulance_distance": self.horde_meter.ambulance_distance,
            "is_split_screen": is_split_screen,
            "people_in_view": people_in_view,
        }

    def update_ui(self, scorekeeper: ScoreKeeper) -> None:
        """Update the capacity meter to reflect current game state.

        Args:
            scorekeeper: ScoreKeeper with updated ambulance state.
        """
        self.capacity_meter.update_fill(self.saved_passengers)

    def _update_info_display(self, scorekeeper: ScoreKeeper) -> None:
        """Refresh the classifier/recommender output label for the current humanoid.

        Computes probs once from the classifier, caches them in
        ``self._last_probs`` for reuse by ``_get_log_context``, and passes them
        to the recommender — a single forward pass per turn regardless of mode.

        Args:
            scorekeeper: ScoreKeeper passed to the recommender for context.
        """
        if self._info_label is None:
            return
        lines = []
        self._last_probs = []
        for idx, h in enumerate(self.humanoids):
            prefix = f"Img {idx + 1}: " if len(self.humanoids) > 1 else ""
            probs = None

            if self.classifier is not None:
                probs = self.classifier.classify(h)
                self._last_probs.append(probs)  # cache for _get_log_context — single forward pass
                
                if self._show_classifier_label:
                    label, confidence = probs_to_class(probs)
                    lines.append(f"{prefix} Classified as: {label.value} ({confidence:.0%})")
                    if label.value == "healthy" and hasattr(self.classifier, "classify_role"):
                        role_probs = self.classifier.classify_role(h, state_probs=probs)
                        if role_probs is not None:
                            role_names = ("civilian", "medic", "engineer")
                            role_idx = int(np.argmax(role_probs))
                            lines.append(
                                f"{prefix} Role: {role_names[role_idx]} "
                                f"({float(role_probs[role_idx]):.0%})"
                            )
            
            if self.recommender is not None and probs is not None:
                action = self.recommender.recommend(h, probs, scorekeeper)
                self._last_recommended_action = action.value  # cache for _get_log_context
                lines.append(f"Suggested: {action.value}")
        self._info_label.config(text="\n".join(lines))

    def get_next(
        self,
        data_fp: str,
        data_parser: DataParser,
        scorekeeper: ScoreKeeper,
    ) -> None:
        """Advance to the next humanoid, or end the game when exhausted.

        Args:
            data_fp: Root path for image files.
            data_parser: DataParser supplying the remaining humanoids.
            scorekeeper: ScoreKeeper tracking game state.
        """
        if (
            len(data_parser.unvisited) == 0
            or self.horde_meter.is_caught()
            or self.horde_meter.reached_camp()
        ):
            self.game_viewer.delete_photo(None)
            # Binary framing per round, regardless of how it ended: caught by
            # the horde is a death; reaching camp or running out of humanoids
            # to process both count as living. Either way, play continues on
            # to the next round (until MAX_ROUNDS ends the run below).
            end_reason = "You Died" if self.horde_meter.is_caught() else "You Lived"
            score = (
                scorekeeper.get_score()
            )  # scrams remaining passengers; ambulance now empty
            # Doctors cure at camp, before reset() clears medics_saved for the next round.
            scorekeeper.cure_injured()
            self.round_scores.append(score)  # keep this round's own numbers for later display
            self.total_saved += score["saved"]
            self.total_killed += score["killed"]
            self.total_saved_value += score["saved_value"]
            self.total_cured += score["cured"]
            self.capacity_meter.update_fill(self.saved_passengers)
            # Disable buttons before reset() — reset() restores remaining_time to the
            # full shift length, which would fool disable_buttons() into re-enabling them.
            self.button_menu.disable_all()
            if self.log:
                scorekeeper.reset()  # commit episode to all_logs before writing
                scorekeeper.save_log()

            # No more rounds left in this run — omit the continue button so
            # display_score() shows a run-complete screen instead.
            is_final_round = self.round_num >= MAX_ROUNDS
            on_continue = None
            if not is_final_round:

                def _on_continue() -> None:
                    self._start_next_round(data_fp, data_parser, scorekeeper)

                on_continue = _on_continue

            self.game_viewer.display_score(
                score,
                end_reason=end_reason,
                round_num=getattr(self, "round_num", 1),
                totals={
                    "saved": self.total_saved,
                    "killed": self.total_killed,
                    "saved_value": self.total_saved_value,
                    "cured": self.total_cured,
                },
                on_continue=on_continue,
                # Once round 1 is in the books, show its own numbers alongside
                # this round's and the running totals (3 boards on the final screen).
                round1_score=self.round_scores[0] if len(self.round_scores) >= 2 else None,
            )
        else:
            show_two = random() < 0.20

            self.humanoids = [data_parser.get_random()]
            if show_two and data_parser.unvisited:
                 self.humanoids.append(data_parser.get_random())

            self.humanoid = self.humanoids[0]
            self._last_probs = (
                None  # invalidate cache — new humanoid, old probs are stale
            )
            self._last_recommended_action = None

            img_paths = [join(data_fp, h.fp) for h in self.humanoids]
            self.game_viewer.create_photos(img_paths)

            # Refresh classifier/recommender output and value label for the new humanoid
            self._update_info_display(scorekeeper)
            self.value_label.config(text=self._value_label_text(self.humanoid.value))
            self.button_menu.disable_buttons(
                len(data_parser.unvisited),
                scorekeeper.at_capacity(),
            )

    def _start_next_round(
        self,
        data_fp: str,
        data_parser: DataParser,
        scorekeeper: ScoreKeeper,
    ) -> None:
        """Reset shift state and draw the first humanoid of the next round.

        Fired by the camp screen's "Continue to Next Round" button.
        ``scorekeeper.reset()`` is idempotent — safe to call again even when
        ``get_next()`` already called it for logging purposes — and
        ``data_parser.reset()`` refills the unvisited pool so the next round
        has the full dataset to draw from.

        Args:
            data_fp: Root path for image files.
            data_parser: DataParser to refill for the new round.
            scorekeeper: ScoreKeeper to reset for the new round.
        """
        self.round_num += 1
        self.saved_passengers.clear()
        scorekeeper.reset()
        data_parser.reset()
        self.horde_meter.reset()
        self.game_viewer.clear_score_display()

        self.humanoids = [data_parser.get_random()]
        self.humanoid = self.humanoids[0]
        self._last_probs = None
        self._last_recommended_action = None
        img_paths = [join(data_fp, h.fp) for h in self.humanoids]
        self.game_viewer.create_photos(img_paths)
        self._update_info_display(scorekeeper)
        self.value_label.config(text=self._value_label_text(self.humanoid.value))
        self.update_ui(scorekeeper)
        self.button_menu.disable_buttons(
            len(data_parser.unvisited),
            scorekeeper.at_capacity(),
        )

    @staticmethod
    def _value_label_text(value: float) -> str:
        """Format a humanoid value for display as 'Value: {float} / {category}'.

        Args:
            value: Float in [0.0, 1.0]. Values outside this range are clamped.

        Returns:
            Display string, e.g. ``'Value: 0.73 / HIGH'``.
        """
        categories = ["VERY LOW", "LOW", "NEUTRAL", "HIGH", "VERY HIGH"]
        # Clamp to [0.0, 1.0] before binning — guards against negative values
        # (which would produce bin_idx=-1, returning categories[-1]="VERY HIGH" by mistake)
        # and values > 1.0 (which would produce bin_idx > 4, raising IndexError).
        clamped = max(0.0, min(1.0, value))
        bin_idx = min(int(clamped * 5), 4)
        return f"Value: {value:.2f} / {categories[bin_idx]}"

    def _create_header(self, w: int) -> None:
        """Place game and BWSI logos across the top of the window.

        The game logo (logo2.png) is placed on the left; the BWSI logo
        (logo.png) is centered horizontally. Both occupy the y=0–90 band
        that sits above all other widgets.

        Stores label widgets as ``self._game_label`` and ``self._bwsi_label``
        so the ``PhotoImage`` objects they hold are not garbage-collected
        while the window is alive.

        Args:
            w: Window width in pixels, used to center the BWSI logo.
        """
        graphics = dirname(abspath(__file__))

        # Game logo — left side
        game_logo_path = join(graphics, "graphics", "logo2.png")
        try:
            with Image.open(game_logo_path) as img:
                game_logo = ImageTk.PhotoImage(img.resize((128, 80), Image.LANCZOS))  # type: ignore[attr-defined]
            self._game_label = tk.Label(self.root, image=game_logo, bg=UI_COLORS["bg"])
            self._game_label.image = (  # type: ignore[attr-defined]
                game_logo  # belt-and-suspenders: also stored on self
            )
            self._game_label.place(x=20, y=5)
        except FileNotFoundError:
            print(
                f"Game logo missing — expected ui/graphics/logo2.png (looked at: {game_logo_path})"
            )

        # BWSI logo — right side
        bwsi_logo_path = join(graphics, "graphics", "logo.png")
        bwsi_w, bwsi_h = 376, 60
        try:
            with Image.open(bwsi_logo_path) as img:
                bwsi_logo = ImageTk.PhotoImage(
                    img.resize((bwsi_w, bwsi_h), Image.LANCZOS)  # type: ignore[attr-defined]
                )
            self._bwsi_label = tk.Label(self.root, image=bwsi_logo, bg=UI_COLORS["bg"])
            self._bwsi_label.image = (  # type: ignore[attr-defined]
                bwsi_logo  # belt-and-suspenders: also stored on self
            )
            self._bwsi_label.place(x=w // 2 - bwsi_w // 2, y=15)
        except FileNotFoundError:
            print(
                f"BWSI logo missing — expected ui/graphics/logo.png (looked at: {bwsi_logo_path})"
            )
