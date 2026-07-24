import math
import tkinter as tk
from collections.abc import Callable
from os.path import join
from typing import Any
from PIL import ImageTk, Image, ImageFilter

from gameplay.constants import IMAGE_BLUR_RADIUS, UI_COLORS

# Emoji glyphs used as lightweight icons on the camp screen — no image assets
# to manage, and Tk resolves them via the system emoji font automatically.
_ICON_CAMP = "\U0001F3D5"  # camping tent
_ICON_KILLED = "\U0001F480"  # skull
_ICON_SAVED = "❤"  # heart
_ICON_VALUE = "⭐"  # star
_ICON_CURED = "\U0001FA79"  # bandage — injured survivors healed by a doctor
_ICON_CONTINUE = "➡"  # arrow
_ICON_RULES = "\U0001F4CB"  # clipboard — game-rules screen shown once at start

_ICON_FOG = "\U0001F32B"  # fog — round-2 briefing screen (blur mechanic)

_ICON_RESTART = "↺"  # restart arrow



class GameViewer:
    """Central photo panel — displays the current humanoid image and end-game score.

    Attributes:
        canvas: Tkinter canvas that holds the humanoid photo.
        photo: Currently displayed :class:`~PIL.ImageTk.PhotoImage`, or
            ``None`` if no photo is loaded. Held as an instance attribute to
            prevent garbage collection.
    """

    canvas: tk.Canvas
    photo: list[Any] | None  # ImageTk.PhotoImage | None
    current_paths: list[str] # Add an attribute to remember current paths
    on_image_selected: Any

    _score_widgets: list[tk.Widget]


    def __init__(self, root: Any, w: int, h: int, data_fp: str, humanoid: Any) -> None:
        """Create the game viewer canvas and display the first humanoid.

        Args:
            root: Tkinter parent widget.
            w: Window width in pixels (used to size and position the canvas).
            h: Window height in pixels (used to size the canvas).
            data_fp: Root path for image files.
            humanoid: Initial :class:`~gameplay.humanoid.Humanoid` whose image
                is displayed on creation.
        """
        self._root = root
        self._w = w
        self._h = h

        self.canvas = tk.Canvas(
            root,
            width=math.floor(0.5 * w),
            height=math.floor(0.75 * h),
            bg=UI_COLORS["panel"],
            highlightthickness=0,
        )
        self.canvas.place(x=300, y=100)
        self.canvas.update()

        self.photo = None
        self.current_paths = []
        self.on_image_selected = None

        self._score_widgets = []
        self.create_photo(join(data_fp, humanoid.fp))

        # Bind events for clicking and hover-outline indications
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Motion>", self._on_canvas_hover)
        self.canvas.bind("<Leave>", self._on_canvas_leave)

    def delete_photo(self, event: Any = None) -> None:
        """Remove the current photo from the canvas.

        Args:
            event: Tkinter event passed when called as a key binding, or
                ``None`` when called directly.
        """
        self.canvas.delete("photo")
        self.photos = []

    def create_photos(self, file_paths: list[str], blur: bool = False) -> None:
        """Load a list of images from disk and display them side-by-side.

        Replaces any existing photos. Automatically divides the canvas width
        equally among all images provided.

        Args:
            file_paths: Image file paths to display, one per canvas slice.
            blur: If True, apply a Gaussian blur to every image — used from
                round ``IMAGE_BLUR_START_ROUND`` onward to push the player
                toward the classifier/recommender's suggestion instead of
                the raw image.
        """
        self.canvas.delete("photo")
        self.canvas.delete("indicator")

        # Re-arm mouse listeners for new choices
        self.canvas.bind("<Motion>", self._on_canvas_hover)
        self.canvas.bind("<Leave>", self._on_canvas_leave)

        self.photos = []
        self.current_paths = file_paths

        num_images = len(file_paths)
        if num_images == 0:
            return

        # Calculate layout constraints
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        target_w = math.floor(canvas_w / num_images)

        target_w = math.floor(canvas_w / num_images)

        blur_radius = IMAGE_BLUR_RADIUS if blur else 0.0
        for i, fp in enumerate(file_paths):
            # Calculate the horizontal offset for each image
            x_offset = i * target_w

            # Load and resize image to fit its fractional slice
            photo = self._get_photo(fp, target_w, canvas_h, blur_radius=blur_radius)
            self.photos.append(photo)

            img_h = photo.height()
            y_offset = max(0, math.floor((canvas_h - img_h) / 2))

            # Place onto the canvas canvas
            self.canvas.create_image(
                x_offset, y_offset,
                anchor=tk.NW,
                image=photo,
                tags=("photo", f"idx_{i}")
            )

    def create_photo(self, fp: str, blur: bool = False) -> None:
        """Load an image from disk and display it on the canvas.

        Replaces any existing photo. Holds a reference in ``self.photo``
        to prevent garbage collection.

        Args:
            fp: Path to the image file.
            blur: If True, apply a Gaussian blur (see ``create_photos``).
        """
        self.create_photos([fp], blur=blur)

    def _on_canvas_click(self, event: Any) -> None:
            """Determines which image slice was clicked and fires the selection callback."""
            num_images = len(self.current_paths)
            if num_images <= 1 or not self.on_image_selected:
                return

            canvas_w = self.canvas.winfo_width()
            slice_w = canvas_w / num_images
            clicked_idx = int(event.x // slice_w)

            if 0 <= clicked_idx < num_images:
                # Draw a permanent red outline around the final selected choice
                self.canvas.delete("indicator")
                canvas_h = self.canvas.winfo_height()
                self.canvas.create_rectangle(
                    clicked_idx * slice_w, 0,
                    (clicked_idx + 1) * slice_w, canvas_h,
                    outline="red", width=4, tags="photo"
                )

                # Unbind temporary hover indicators so they stop moving
                self.canvas.unbind("<Motion>")
                self.canvas.unbind("<Leave>")

                # Execute the UI action callback switch
                self.on_image_selected(clicked_idx)

    def _on_canvas_hover(self, event: Any) -> None:
        """Draws a real-time red selection box outline following the cursor layout."""
        if not self.on_image_selected:
            return

        num_images = len(self.current_paths)
        if num_images <= 1:
            return

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        slice_w = canvas_w / num_images
        hovered_idx = int(event.x // slice_w)

        self.canvas.delete("indicator")
        if 0 <= hovered_idx < num_images:
            self.canvas.create_rectangle(
                hovered_idx * slice_w, 0,
                (hovered_idx + 1) * slice_w, canvas_h,
                outline="red", width=3, tags="indicator"
            )

    def _on_canvas_leave(self, event: Any) -> None:
        """Cleans up outline tracking marks if the cursor exits the canvas."""
        self.canvas.delete("indicator")


    def clear_score_display(self) -> None:
        """Destroy any camp-screen widgets left over from the previous round.

        Call before starting a new round — ``display_score``'s labels and
        button are real child widgets (not canvas items), so they persist
        across ``create_photo`` calls unless explicitly destroyed.
        """
        for widget in self._score_widgets:
            widget.destroy()
        self._score_widgets = []

    def display_score(
        self,
        score: dict[str, int | float],
        end_reason: str = "",
        round_num: int | None = None,
        totals: "dict[str, int | float] | None" = None,
        on_continue: "Callable[[], None] | None" = None,
        round1_score: "dict[str, int | float] | None" = None,
        on_restart: "Callable[[], None] | None" = None,
    ) -> None:
        """Render the camp screen — round results — over the canvas.

        Args:
            score: Dict with ``"killed"`` and ``"saved"`` integer counts and
                ``"saved_value"`` float (total value of saved humanoids) for
                *this round only*, as returned by
                :meth:`~gameplay.scorekeeper.ScoreKeeper.get_score`.
            end_reason: Optional end-of-round reason string (``"You Died"`` if
                the horde caught the ambulance, ``"You Lived"`` otherwise).
                Displayed as a bold header if provided.
            round_num: Optional round number to display (e.g. ``"ROUND 1 COMPLETE"``).
                Omit for a generic ``"CAMP"`` header.
            totals: Optional dict with the same shape as ``score`` but summed
                across every round played so far. Omit to hide the career-totals
                section (e.g. before round 1 has any history to show).
            on_continue: Optional callback fired by a "Continue to Next Round"
                button. Omit (with ``on_restart`` set instead) to render the
                run-complete screen instead — no more rounds left.
            round1_score: Optional dict with the same shape as ``score``,
                holding round 1's own numbers. Only pass this once round 1 is
                complete and a later round is what's currently being shown —
                renders a third "ROUND 1" card alongside this round's and the
                career totals.
            on_restart: Optional callback fired by a "Restart Game" button,
                shown only when ``on_continue`` is omitted (i.e. the run just
                ended). Resets the whole run back to round 1.
        """
        self.clear_score_display()  # guard against stacking widgets on a stale screen

        # A single outer Frame holds the whole screen — destroying it in
        # clear_score_display() takes every child down with it, so nothing
        # below needs to be tracked individually.
        #
        # Placed directly on root (not self.canvas) and sized to nearly fill
        # the whole window — this is a full takeover screen, not just a
        # panel inside the humanoid-photo canvas.
        margin = 20
        outer = tk.Frame(
            self._root,
            bg=UI_COLORS["panel"],
            highlightbackground=UI_COLORS["card_border"],
            highlightthickness=1,
        )
        self._score_widgets.append(outer)
        outer.place(
            x=margin,
            y=margin,
            width=self._w - 2 * margin,
            height=self._h - 2 * margin,
        )
        outer.lift()

        if end_reason:
            tk.Label(
                outer,
                text=end_reason,
                font=("Arial", 15, "bold"),
                bg=UI_COLORS["panel"],
                fg=UI_COLORS["accent"],
            ).pack(anchor=tk.W)

        header_text = (
            f"{_ICON_CAMP}  ROUND {round_num} COMPLETE"
            if round_num is not None
            else f"{_ICON_CAMP}  CAMP"
        )
        tk.Label(
            outer,
            text=header_text,
            font=("Arial", 28, "bold"),
            bg=UI_COLORS["panel"],
            fg=UI_COLORS["fg"],
        ).pack(anchor=tk.W, pady=(6, 20))

        cards_row = tk.Frame(outer, bg=UI_COLORS["panel"])
        cards_row.pack(anchor=tk.W, fill=tk.X)

        if round1_score is not None:
            self._build_stat_card(cards_row, "ROUND 1", round1_score)
            self._build_stat_card(
                cards_row,
                f"ROUND {round_num}" if round_num is not None else "THIS ROUND",
                score,
                padx=(20, 0),
            )
        else:
            self._build_stat_card(cards_row, "THIS ROUND", score)
        if totals is not None:
            self._build_stat_card(cards_row, "CAREER TOTALS", totals, padx=(20, 0))

        if on_continue is not None:
            tk.Button(
                outer,
                text=f"Continue to Next Round  {_ICON_CONTINUE}",
                font=("Arial", 15, "bold"),
                bg=UI_COLORS["btn_save"],
                fg="black",
                activebackground="black",
                activeforeground=UI_COLORS["fg"],
                relief=tk.FLAT,
                bd=0,
                padx=18,
                pady=10,
                command=on_continue,
            ).pack(anchor=tk.W, pady=(24, 0))
        else:
            # No on_continue means no rounds remain (MAX_ROUNDS reached) —
            # say so explicitly rather than leaving the screen looking stuck,
            # and offer a way to play the whole thing again.
            tk.Label(
                outer,
                text="RUN COMPLETE — thanks for playing!",
                font=("Arial", 15, "bold"),
                bg=UI_COLORS["panel"],
                fg=UI_COLORS["accent"],
            ).pack(anchor=tk.W, pady=(24, 0))
            if on_restart is not None:
                tk.Button(
                    outer,
                    text=f"Restart Game  {_ICON_RESTART}",
                    font=("Arial", 15, "bold"),
                    bg=UI_COLORS["btn_save"],
                    fg="black",
                    activebackground="black",
                    activeforeground=UI_COLORS["fg"],
                    relief=tk.FLAT,
                    bd=0,
                    padx=18,
                    pady=10,
                    command=on_restart,
                ).pack(anchor=tk.W, pady=(10, 0))

    def display_rules(self, on_start: "Callable[[], None]") -> None:
        """Render the one-time rules screen shown when the game first launches.

        Follows the same full-window overlay treatment as display_score(),
        but isn't tracked in _score_widgets since it's dismissed by its own
        button rather than by clear_score_display().

        Args:
            on_start: Callback fired by the "Start Game" button, right after
                this screen destroys itself.
        """
        margin = 20
        outer = tk.Frame(
            self._root,
            bg=UI_COLORS["panel"],
            highlightbackground=UI_COLORS["card_border"],
            highlightthickness=1,
        )
        outer.place(
            x=margin,
            y=margin,
            width=self._w - 2 * margin,
            height=self._h - 2 * margin,
        )
        outer.lift()

        tk.Label(
            outer,
            text=f"{_ICON_RULES}  GAME RULES",
            font=("Arial", 28, "bold"),
            bg=UI_COLORS["panel"],
            fg=UI_COLORS["fg"],
        ).pack(anchor=tk.W, padx=28, pady=(24, 16))

        sections = [
            (
                "THE MISSION",
                "Get your ambulance from the outbreak zone to Camp — 300 ft "
                "away — before the zombie horde catches up.",
            ),
            (
                "EACH TURN",
                "A person appears in your path (sometimes two — click one "
                "to choose). Decide their fate:\n"
                "•  Skip — leave them behind. Ambulance advances 10 ft.\n"
                "•  Squish — destroy them where they stand. Ambulance "
                "advances 15 ft, your fastest option — but squishing a "
                "healthy or injured human counts as a kill against you. "
                "Zombies and corpses are always safe to squish.\n"
                "•  Save — load them aboard. The ambulance doesn't move at "
                "all — you're trading distance for a life.\n"
                "No matter what you choose, the horde always advances 10 ft.",
            ),
            (
                "WIN / LOSE",
                "The horde catches you → round over, you died.\n"
                "You reach Camp first → round over, you lived.",
            ),
            (
                "AMBULANCE CAPACITY",
                "15 seats. Once full, you can't Save anyone else until you "
                "reach Camp.",
            ),
            (
                "SPECIAL ROLES",
                "•  Engineers — stack a +2 ft bonus onto every Skip/Squish "
                "for as long as they're aboard.\n"
                "•  Medics — every medic delivered to Camp can cure 2 "
                "injured survivors.",
            ),
            (
                "THE RUN",
                "You'll play 2 rounds. Your stats carry over — the final "
                "screen shows Round 1, Round 2, and your combined career "
                "totals.",
            ),
        ]

        body = tk.Frame(outer, bg=UI_COLORS["panel"])
        body.pack(anchor=tk.W, fill=tk.BOTH, expand=True, padx=28)

        for header, text in sections:
            tk.Label(
                body,
                text=header,
                font=("Arial", 16, "bold"),
                bg=UI_COLORS["panel"],
                fg=UI_COLORS["accent"],
            ).pack(anchor=tk.W, pady=(10, 2))
            tk.Label(
                body,
                text=text,
                font=("Arial", 14),
                bg=UI_COLORS["panel"],
                fg=UI_COLORS["fg"],
                justify=tk.LEFT,
                anchor="w",
                wraplength=self._w - 2 * margin - 56,
            ).pack(anchor=tk.W)

        def _dismiss() -> None:
            outer.destroy()
            on_start()

        tk.Button(
            outer,
            text=f"Start Game  {_ICON_CONTINUE}",
            font=("Arial", 16, "bold"),
            bg=UI_COLORS["btn_save"],
            fg="black",
            activebackground="black",
            activeforeground=UI_COLORS["fg"],
            relief=tk.FLAT,
            bd=0,
            padx=18,
            pady=10,
            command=_dismiss,
        ).pack(anchor=tk.W, padx=28, pady=(16, 20))

    def display_round2_briefing(self, on_start: "Callable[[], None]") -> None:
        """Render the one-time briefing shown between Round 1's end screen and Round 2.

        Same full-window overlay treatment as display_rules(). Explains the
        Round 2 blur mechanic before gameplay resumes; the caller is
        responsible for only invoking this ahead of Round 2 specifically
        (see the round_num check around _on_continue in ui.py).

        Args:
            on_start: Callback fired by the "Enter Round 2" button, right
                after this screen destroys itself.
        """
        margin = 20
        outer = tk.Frame(
            self._root,
            bg=UI_COLORS["panel"],
            highlightbackground=UI_COLORS["card_border"],
            highlightthickness=1,
        )
        outer.place(
            x=margin,
            y=margin,
            width=self._w - 2 * margin,
            height=self._h - 2 * margin,
        )
        outer.lift()

        tk.Label(
            outer,
            text=f"{_ICON_FOG}  ROUND 2",
            font=("Arial", 28, "bold"),
            bg=UI_COLORS["panel"],
            fg=UI_COLORS["fg"],
        ).pack(anchor=tk.W, padx=28, pady=(24, 16))

        sections = [
            (
                "WHAT'S CHANGED",
                "The horde's gotten close enough that your headlights are "
                "the only thing cutting through the dark — and it's not "
                "enough. You won't be able to see who's standing in the "
                "road anymore. Every image this round is blurred.",
            ),
            (
                "TRUST THE CNN",
                "Since you can't judge them by sight, you'll have to rely "
                "on the CNN classifier's prediction instead. Check its "
                "label and confidence before deciding — a low-confidence "
                "call is a bigger gamble than a high-confidence one.",
            ),
            (
                "EVERYTHING ELSE IS THE SAME",
                "Skip, Squish, and Save still cost the same distance. "
                "Capacity, Engineers, and Medics all carry over exactly "
                "like Round 1. Only what you can see has changed.",
            ),
        ]

        body = tk.Frame(outer, bg=UI_COLORS["panel"])
        body.pack(anchor=tk.W, fill=tk.BOTH, expand=True, padx=28)

        for header, text in sections:
            tk.Label(
                body,
                text=header,
                font=("Arial", 16, "bold"),
                bg=UI_COLORS["panel"],
                fg=UI_COLORS["accent"],
            ).pack(anchor=tk.W, pady=(10, 2))
            tk.Label(
                body,
                text=text,
                font=("Arial", 14),
                bg=UI_COLORS["panel"],
                fg=UI_COLORS["fg"],
                justify=tk.LEFT,
                anchor="w",
                wraplength=self._w - 2 * margin - 56,
            ).pack(anchor=tk.W)

        def _dismiss() -> None:
            outer.destroy()
            on_start()

        tk.Button(
            outer,
            text=f"Enter Round 2  {_ICON_CONTINUE}",
            font=("Arial", 16, "bold"),
            bg=UI_COLORS["btn_save"],
            fg="black",
            activebackground="black",
            activeforeground=UI_COLORS["fg"],
            relief=tk.FLAT,
            bd=0,
            padx=18,
            pady=10,
            command=_dismiss,
        ).pack(anchor=tk.W, padx=28, pady=(16, 20))

    @staticmethod
    def _build_stat_card(
        parent: tk.Widget,
        title: str,
        stats: dict[str, int | float],
        padx: int | tuple[int, int] = 0,
    ) -> None:
        """Build one bordered stat card (title + killed/saved/value rows).

        Args:
            parent: Frame the card is packed into.
            title: Card heading, e.g. ``"THIS ROUND"`` or ``"CAREER TOTALS"``.
            stats: Dict with ``"killed"``, ``"saved"``, ``"saved_value"`` keys.
            padx: Horizontal padding passed to the card's own ``pack()`` call —
                used to space multiple cards apart in the same row.
        """
        card = tk.Frame(
            parent,
            bg=UI_COLORS["panel"],
            highlightbackground=UI_COLORS["card_border"],
            highlightthickness=1,
        )
        card.pack(side=tk.LEFT, anchor=tk.N, padx=padx)

        inner = tk.Frame(card, bg=UI_COLORS["panel"], padx=16, pady=12)
        inner.pack()

        tk.Label(
            inner,
            text=title,
            font=("Arial", 13, "bold"),
            bg=UI_COLORS["panel"],
            fg=UI_COLORS["fg"],
        ).pack(anchor=tk.W, pady=(0, 8))

        rows = [
            (_ICON_KILLED, "Killed {}".format(stats["killed"]), UI_COLORS["accent"]),
            (_ICON_SAVED, "Saved {}".format(stats["saved"]), UI_COLORS["fg"]),
            (
                _ICON_VALUE,
                "Value {0:.4f}".format(stats["saved_value"]),
                UI_COLORS["fg"],
            ),
            (_ICON_CURED, "Cured {}".format(stats["cured"]), UI_COLORS["fg"]),
        ]
        for icon, text, color in rows:
            tk.Label(
                inner,
                text=f"{icon}  {text}",
                font=("Arial", 15),
                bg=UI_COLORS["panel"],
                fg=color,
            ).pack(anchor=tk.W, pady=2)

    @staticmethod
    def _get_photo(
        img_path: str, max_w: int, max_h: int, blur_radius: float = 0.0
    ) -> Any:
        """Load, resize, and return a Tkinter-compatible photo image.

        Args:
            img_path: Path to the image file.
            w: Target width in pixels.
            h: Target height in pixels.
            blur_radius: Gaussian blur radius in pixels. ``0`` (default) skips
                blurring entirely. Applied after resizing so the radius is
                consistent with what's actually displayed, regardless of the
                source image's native resolution.

        Returns:
            An :class:`~PIL.ImageTk.PhotoImage` resized to ``(w, h)``.
        """
        with Image.open(img_path) as img:
            orig_w, orig_h = img.size

            # Calculate aspect ratios
            ratio_w = max_w / orig_w
            ratio_h = max_h / orig_h

            # Use the smaller scaling factor to ensure it fits without stretching
            scale = min(ratio_w, ratio_h)

            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)

            resized = img.resize((new_w, new_h), Image.LANCZOS)
            if blur_radius > 0:
                resized = resized.filter(ImageFilter.GaussianBlur(blur_radius))
            tk_img = ImageTk.PhotoImage(resized)
        return tk_img
