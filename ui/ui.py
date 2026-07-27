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
    ENGINEER_SPEED_BONUS,
    HORDE_INITIAL_DISTANCE,
    HORDE_MAX_DISTANCE,
    IMAGE_BLUR_START_ROUND,
    MAX_ROUNDS,
    UI_COLORS,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from gameplay.enums import ActionState, HordeDistance, State
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
        value_label: Label used for on-screen prompts during save/squish selection.
    """

    root: tk.Tk
    log: bool
    classifier: HumanoidClassifier | None
    recommender: ActionRecommender | None
    _auto_execute: bool
    _show_classifier_label: bool
    _ai_starts_round: int | None
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
        ai_starts_round: int | None = None,
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
            ai_starts_round: If set, the classifier/recommender output stays
                hidden and images stay sharp until ``round_num`` reaches this
                value — round 1 plays as pure human triage, and AI assistance
                plus blurred images both start together from this round on.
                ``None`` (default) means AI assistance (if any classifier/
                recommender was passed) is visible from round 1, as in the
                human_cnn/human_rl/human_cnn_rl demo modes.
        """
        w, h = WINDOW_WIDTH, WINDOW_HEIGHT
        self.root = tk.Tk()
        self.root.title("Beaverworks SGAI 2026 - Dead or Alive")
        self.root.geometry(f"{w}x{h}")
        self.root.resizable(False, False)
        self.root.configure(bg=UI_COLORS["bg"])

        bg_path = join(dirname(abspath(__file__)), "graphics", "background.png")
        bg_raw = Image.open(bg_path)
        self.bg_image = ImageTk.PhotoImage(bg_raw.resize((w, h), Image.LANCZOS))

        self.bg_canvas = tk.Canvas(self.root, width=w, height=h, highlightthickness=0)
        self.bg_canvas.pack(fill="both", expand=True)

        # 2. Bottom Layer: Background Image
        try:
            bg_path = join(dirname(abspath(__file__)), "graphics", "background.png")
            bg_raw = Image.open(bg_path)
            self.bg_image = ImageTk.PhotoImage(bg_raw.resize((w, h), Image.LANCZOS))
            self.bg_canvas.create_image(0, 0, image=self.bg_image, anchor="nw")
        except FileNotFoundError:
            print("backgroundGame.jpg not found in ui/graphics/")

        # 3. Top Layer Pre-load: Frame Image
        try:
            frame_path = join(dirname(abspath(__file__)), "graphics", "frame.png")
            # Resize frame to surround your photo (adjust 450x450 to match photo size)
            frame_raw = Image.open(frame_path).convert("RGBA")
            self.frame_photo = ImageTk.PhotoImage(frame_raw.resize((450, 450), Image.LANCZOS))
        except FileNotFoundError:
            self.frame_photo = None
            print("frame.png not found in ui/graphics/")

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
        self._ai_starts_round = ai_starts_round
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
        self.engineer_bonus_ft = 0.0
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
                    self.humanoids,
                    "skip",
                    **self._get_log_context(scorekeeper, target_humanoids=self.humanoids),
                )
                for h in self.humanoids:
                    scorekeeper.skip(h)

                self.horde_meter.advance(
                    zombie_delta=HordeDistance.ZOMBIE.value,
                    ambulance_delta=HordeDistance.SKIP.value + self.engineer_bonus_ft,
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
                            self.humanoids,
                            "squish",
                            **self._get_log_context(scorekeeper, target_humanoids=self.humanoids),
                        )
                        scorekeeper.squish(selected_humanoid)

                        # Log any unchosen image(s) as skipped
                        for idx, h in enumerate(self.humanoids):
                            if idx != chosen_index:
                                scorekeeper.skip(h)

                        self.horde_meter.advance(
                            zombie_delta=HordeDistance.ZOMBIE.value,
                            ambulance_delta=HordeDistance.SQUISH.value + self.engineer_bonus_ft,
                        )

                        self.game_viewer.on_image_selected = None
                        self.update_ui(scorekeeper)
                        if hasattr(self.button_menu, 'enable_all'):
                            self.button_menu.enable_all()
                        self.get_next(data_fp, data_parser, scorekeeper)

                    self.game_viewer.on_image_selected = handle_choice
                else:
                    scorekeeper.log_turn(
                        self.humanoids,
                        "squish",
                        **self._get_log_context(scorekeeper, target_humanoids=self.humanoids),
                    )
                    scorekeeper.squish(self.humanoid)
                    self.horde_meter.advance(
                        zombie_delta=HordeDistance.ZOMBIE.value,
                        ambulance_delta=HordeDistance.SQUISH.value
                        + self.engineer_bonus_ft,
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
                            self.humanoids,
                            "save",
                            **self._get_log_context(scorekeeper, target_humanoids=self.humanoids),
                        )
                        scorekeeper.save(selected_humanoid)
                        self.saved_passengers.append(selected_humanoid)

                        # Log any unchosen image(s) as skipped
                        for idx, h in enumerate(self.humanoids):
                            if idx != chosen_index:
                                scorekeeper.skip(h)

                        self.horde_meter.advance(
                            zombie_delta=HordeDistance.ZOMBIE.value,
                            ambulance_delta=HordeDistance.SAVE.value
                            + self.engineer_bonus_ft,
                        )
                        # Applied after this turn's advance — the bonus starts
                        # with the next action, not this Save.
                        self._apply_engineer_bonus(selected_humanoid)

                        # Clear the selection listener callback so it doesn't fire unexpectedly later
                        self.game_viewer.on_image_selected = None
                        self.update_ui(scorekeeper)
                        if hasattr(self.button_menu, 'enable_all'):
                            self.button_menu.enable_all()
                        self.get_next(data_fp, data_parser, scorekeeper)

                    self.game_viewer.on_image_selected = handle_choice
                else:
                    scorekeeper.log_turn(
                        self.humanoids,
                        "save",
                        **self._get_log_context(scorekeeper, target_humanoids=self.humanoids),
                    )
                    scorekeeper.save(self.humanoid)
                    self.saved_passengers.append(self.humanoid)
                    self.horde_meter.advance(
                        zombie_delta=HordeDistance.ZOMBIE.value,
                        ambulance_delta=HordeDistance.SAVE.value + self.engineer_bonus_ft,
                    )
                    # Applied after this turn's advance — the bonus starts
                    # with the next action, not this Save.
                    self._apply_engineer_bonus(self.humanoid)
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
           self.root,
            w,
            h,
            max_distance=HORDE_MAX_DISTANCE,
            initial_distance=HORDE_INITIAL_DISTANCE,
            zombie_icon_path="ui/graphics/zombie.png",
            ambulance_icon_path="ui/graphics/ambulance.png",
        )

        # --- Value display ---
        self.value_label = tk.Label(
            self.root,
            text="",
            font=("Arial", 14),
            bg=UI_COLORS["bg"],
            fg=UI_COLORS["fg"],
        )
        self.value_label.place(x=math.floor(0.05 * w), y=math.floor(0.88 * h))

        # --- Machine output display (classifier label + recommender label) ---
        self._info_label = None
        if classifier is not None or recommender is not None:
            panel_x = math.floor(0.75 * w)  # Same X alignment as Capacity box
            panel_w = math.floor(0.22 * w)  # Same width as Capacity box
            panel_h = math.floor(0.24 * h)  # Compact height to avoid covering Capacity

            self._info_container = tk.Frame(
                self.root,
                bg="#051405",                   # Dark terminal background
                highlightbackground="#33ff33", # Phosphor green border
                highlightthickness=2,
                padx=14,
                pady=12,
            )
            self._info_container.place(
                x=panel_x,
                y=math.floor(0.10 * h),      # Starts top-right
                width=panel_w,
                height=panel_h,
            )

            # Retro CRT Scanner Header
            scanner_box = tk.Frame(
                self._info_container,
                bg="#051405",
                highlightbackground="#33ff33",
                highlightthickness=1,
                padx=6,
                pady=4,
            )
            scanner_box.pack(fill=tk.X, pady=(0, 10))

            tk.Label(
                scanner_box,
                text="▶ AI SCANNER ACTIVE",
                font=("Courier New", 11, "bold"),
                fg="#55ff55",
                bg="#051405",
            ).pack(anchor=tk.CENTER)

            # self._info_label
            self._info_label = tk.Label(
                self._info_container,
                text="",
                font=("Courier New", 12, "bold"),
                fg="#55ff55",
                bg="#051405",
                justify=tk.LEFT,
                anchor="nw",
                wraplength=panel_w - 28,
            )
            self._info_label.pack(fill=tk.BOTH, expand=True)

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
        action = self.recommender.recommend(self.humanoids, probs, scorekeeper)
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
                self.humanoids,
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

    def _apply_engineer_bonus(self, humanoid: Humanoid) -> None:
        """Bump the ambulance's permanent speed bonus if ``humanoid`` is a live engineer.

        Call this AFTER this turn's ``horde_meter.advance()``, so the bonus
        takes effect starting with the *next* action, not the Save action
        that brought the engineer aboard. Stacks additively — each engineer
        saved adds another ``ENGINEER_SPEED_BONUS`` ft on top of the running
        total. Only counts a *living* engineer (injured or healthy), matching
        the same alive-subset rule ``ScoreKeeper`` uses for ``ambulance_engineers``.

        Args:
            humanoid: The humanoid just passed to ``scorekeeper.save()``.
        """
        if humanoid.is_engineer() and (humanoid.is_injured() or humanoid.is_healthy()):
            self.engineer_bonus_ft += ENGINEER_SPEED_BONUS

    @staticmethod
    def _scram_fallback(probs: "np.ndarray | None") -> ActionState:
        """Substitute an actionable equivalent for a SCRAM recommendation.

        Human play only has Skip/Squish/Save buttons — there is no way to
        act on "scram", so it must never reach the on-screen suggestion or
        the log. Falls back to the same classifier-driven rule
        ``HeuristicRecommender`` uses for its non-capacity branch: squish for
        a predicted zombie/corpse, skip otherwise (can't save — scram only
        ever comes from an at-capacity or time-exhausted state).

        Args:
            probs: Classifier probability vector, or ``None`` if no
                classifier is set (falls back to skip).

        Returns:
            ``ActionState.SQUISH`` or ``ActionState.SKIP`` — never SCRAM.
        """
        if probs is not None:
            predicted_state = list(State)[int(np.argmax(probs))]
            if predicted_state in (State.ZOMBIE, State.CORPSE):
                return ActionState.SQUISH
        return ActionState.SKIP

    def _should_blur_images(self) -> bool:
        """Return True when humanoid images should be blurred for the current round.

        Blur now starts in round 2 for every mode, regardless of whether the
        AI assistance display is active yet.
        """
        return self.round_num >= IMAGE_BLUR_START_ROUND

    def _ai_display_active(self) -> bool:
        """Return True if classifier/recommender output should be shown right now.

        True immediately when ``ai_starts_round`` is ``None`` (human_cnn/
        human_rl/human_cnn_rl demo modes). For plain human mode
        (``ai_starts_round`` set), stays False until ``round_num`` reaches it.
        """
        return self._ai_starts_round is None or self.round_num >= self._ai_starts_round

    def _get_log_context(self, scorekeeper: ScoreKeeper, target_humanoids: Humanoid | None = None) -> dict:
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

        humanoids_to_log = target_humanoids if target_humanoids is not None else getattr(self, "humanoids", [self.humanoid])
        humanoids_to_log = [item for item in humanoids_to_log if item is not None]

        predicted_classes = []
        predicted_roles = []
        confidences = []

        if self.classifier is not None:
            for h in humanoids_to_log:
                probs = self.classifier.classify(h)
                if probs is not None:
                    predicted_classes.append(int(np.argmax(probs)))
                    confidences.append(float(np.max(probs)))

                    # Fetch predicted role if supported
                    if hasattr(self.classifier, "classify_role"):
                        role_probs = self.classifier.classify_role(h, state_probs=probs)
                        predicted_roles.append(int(np.argmax(role_probs)) if role_probs is not None else None)
                    else:
                        predicted_roles.append(None)

        is_split_screen = len(humanoids_to_log) > 1


        return {
            "predicted_class": predicted_classes,
            "predicted_role": predicted_roles,
            "recommended_action": self._last_recommended_action,
            "confidence": confidences,
            "obs": None,
            "zombie_distance": self.horde_meter.zombie_distance,
            "ambulance_distance": self.horde_meter.ambulance_distance,
            "is_split_screen": is_split_screen,
            "people_in_view": len(humanoids_to_log),
            "image_blurred": self._should_blur_images(),
        }

    def update_ui(self, scorekeeper: ScoreKeeper) -> None:
        """Update the capacity meter and action button labels for the current state.

        Args:
            scorekeeper: ScoreKeeper with updated ambulance state.
        """
        self.capacity_meter.update_fill(self.saved_passengers)
        self._refresh_button_labels()

    def _refresh_button_labels(self) -> None:
        """Update Skip/Squish/Save button text to reflect the engineer speed bonus.

        Button labels are static text at construction time otherwise — once an
        engineer's bonus starts stacking, the buttons would silently move the
        ambulance farther than their printed ``[+N ft]`` without this. No-op in
        machine mode, which has no Skip/Squish/Save keys (only "Advance turn").
        """
        if self._auto_execute:
            return
        self.button_menu.set_text(
            "Skip", f"Skip [+{HordeDistance.SKIP.value + self.engineer_bonus_ft:.0f} ft]"
        )
        self.button_menu.set_text(
            "Squish",
            f"Squish [+{HordeDistance.SQUISH.value + self.engineer_bonus_ft:.0f} ft]",
        )
        self.button_menu.set_text(
            "Save", f"Save [+{HordeDistance.SAVE.value + self.engineer_bonus_ft:.0f} ft]"
        )

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
                    lines.append(f"{prefix}Classified as: {label.value} ({confidence:.0%})")
                    if label.value == "healthy" and hasattr(self.classifier, "classify_role"):
                        role_probs = self.classifier.classify_role(h, state_probs=probs)
                        if role_probs is not None:
                            role_names = ("civilian", "medic", "engineer")
                            role_idx = int(np.argmax(role_probs))
                            lines.append(
                                f"{prefix}Role: {role_names[role_idx]} "
                                f"({float(role_probs[role_idx]):.0%})"
                            )
            
            if self.recommender is not None and probs is not None:
                action = self.recommender.recommend(h, probs, scorekeeper)
                if action == ActionState.SCRAM:
                    action = self._scram_fallback(probs)
                self._last_recommended_action = action.value  # cache for _get_log_context
                lines.append(f"Suggested: {action.value}")

        self._info_label.config(text="\n\n".join(lines) if self._ai_display_active() else "")
        
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
                scorekeeper.save_log()
            scorekeeper.reset()  # commit episode to all_logs before writing

            # No more rounds left in this run — omit the continue button so
            # display_score() shows a run-complete screen with a restart
            # button instead.
            is_final_round = self.round_num >= MAX_ROUNDS
            on_continue = None
            on_restart = None
            if not is_final_round:

                def _on_continue() -> None:
                    # Round 2 is the only round with the blur mechanic, so
                    # the briefing only shows ahead of it, not before every
                    # round transition in general.
                    if self.round_num == 1:
                        self.game_viewer.display_round2_briefing(
                            on_start=lambda: self._start_next_round(
                                data_fp, data_parser, scorekeeper
                            )
                        )
                    else:
                        self._start_next_round(data_fp, data_parser, scorekeeper)

                on_continue = _on_continue
            else:

                def _on_restart() -> None:
                    self._restart_game(data_fp, data_parser, scorekeeper)

                on_restart = _on_restart

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
                on_restart=on_restart,
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
            self.game_viewer.create_photos(img_paths, blur=self._should_blur_images())

            # Refresh classifier/recommender output and value label for the new humanoid
            self._update_info_display(scorekeeper)
            self.value_label.config(text="")
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
        self.engineer_bonus_ft = 0.0
        self._begin_round(data_fp, data_parser, scorekeeper)

    def _restart_game(
        self,
        data_fp: str,
        data_parser: DataParser,
        scorekeeper: ScoreKeeper,
    ) -> None:
        """Reset the whole run back to round 1 — fired by the final run-complete screen.

        Unlike ``_start_next_round`` (which keeps career totals running), this
        zeroes everything: round number, career totals, saved passengers, and
        the engineer speed bonus — a genuine fresh start, not just another round.

        Args:
            data_fp: Root path for image files.
            data_parser: DataParser to refill for the new run.
            scorekeeper: ScoreKeeper to reset for the new run.
        """
        self.round_num = 1
        self.total_saved = 0
        self.total_killed = 0
        self.total_saved_value = 0.0
        self.total_cured = 0
        self.round_scores = []
        self.saved_passengers = []
        self.engineer_bonus_ft = 0.0
        self._begin_round(data_fp, data_parser, scorekeeper)

    def _begin_round(
        self,
        data_fp: str,
        data_parser: DataParser,
        scorekeeper: ScoreKeeper,
    ) -> None:
        """Shared first-turn setup for both a new round and a full restart.

        ``scorekeeper.reset()`` is idempotent — safe to call again even when
        ``get_next()`` already called it for logging purposes — and
        ``data_parser.reset()`` refills the unvisited pool so the round has
        the full dataset to draw from.

        Args:
            data_fp: Root path for image files.
            data_parser: DataParser to refill.
            scorekeeper: ScoreKeeper to reset.
        """
        scorekeeper.reset()
        data_parser.reset()
        self.horde_meter.reset()
        self.game_viewer.clear_score_display()

        self.humanoids = [data_parser.get_random()]
        self.humanoid = self.humanoids[0]
        self._last_probs = None
        self._last_recommended_action = None
        img_paths = [join(data_fp, h.fp) for h in self.humanoids]
        self.game_viewer.create_photos(img_paths, blur=self._should_blur_images())
        self._update_info_display(scorekeeper)
        self.value_label.config(text="")
        self.update_ui(scorekeeper)
        self.button_menu.disable_buttons(
            len(data_parser.unvisited),
            scorekeeper.at_capacity(),
        )

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
