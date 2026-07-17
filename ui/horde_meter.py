"""Horde distance meter — shows how far the ambulance is from the zombie horde.

Follows the same widget pattern as CapacityMeter / Clock: a small class that
owns its own tk.Canvas, is constructed once in UI.__init__, and exposes an
advance() method that action handlers in ui.py call each turn to move the
zombie horde and the ambulance.

Usage (inside ui/ui.py, alongside your other widget construction):

    from ui.horde_meter import HordeMeter

    # in UI.__init__, near where capacity_meter is created:
    self.horde_meter = HordeMeter(self.root, w, h, max_distance=100, initial_distance=10)

    # in an action handler, e.g. _on_squish:
    self.horde_meter.advance(zombie_delta=10, ambulance_delta=10)
"""

import math
from os.path import abspath, dirname, join
import tkinter as tk

from PIL import Image, ImageTk

from gameplay.constants import UI_COLORS


class HordeMeter:
    """A horizontal progress bar showing ambulance-to-horde distance.

    The bar spans from 0 ft (the horde's starting point, left) to
    max_distance ft (camp, right). The zombie horde and the ambulance each
    have their own position on the bar and move independently as advance()
    is called; the true gap between them is shown in a separate label off
    to the side, since the moving zombie icon's own position no longer
    represents 0 ft once it starts advancing.

    Attributes:
        root: Parent Tkinter window.
        max_distance: Total length of the bar in feet (e.g. 100).
        canvas: The Canvas widget everything is drawn on.
    """

    root: tk.Tk
    max_distance: float
    canvas: tk.Canvas

    def __init__(
        self,
        root: tk.Tk,
        w: int,
        h: int,
        max_distance: float = 100.0,
        initial_distance: float = 10.0,
        zombie_img_path: str | None = None,
        ambulance_img_path: str | None = None,
    ) -> None:
        """Build the meter and place it near the bottom of the window.

        Args:
            root: Parent Tkinter window.
            w: Window width in pixels (used to size/position the bar).
            h: Window height in pixels (used to position the bar vertically).
            max_distance: Total bar length in feet. Default 100.
            initial_distance: Starting ambulance position in feet. The
                horde always starts at 0 ft. Default 10 (close call at the
                start, per your design).
            zombie_img_path: Optional path to a zombie icon image. If None
                or missing, a placeholder shape is drawn instead.
            ambulance_img_path: Optional path to an ambulance icon image.
                If None or missing, a placeholder shape is drawn instead.
        """
        self.root = root
        self.max_distance = max_distance

        # --- Layout: bar sits near the bottom of the window ---
        self._bar_x0 = math.floor(0.08 * w)
        self._bar_x1 = math.floor(0.92 * w)
        self._bar_y = math.floor(0.95 * h)
        self._bar_height = 18
        self._icon_size = 40
        self._canvas_height = self._icon_size + 40

        # Canvas coordinates are LOCAL to the canvas, not window-global. The
        # labels are drawn above the icons, so the icon row sits low in the
        # canvas (near the bottom), leaving headroom above it for text —
        # otherwise the text would be clipped outside the canvas's visible
        # area.
        self._bar_y_local = self._canvas_height - (self._icon_size // 2 + 10)

        self.canvas = tk.Canvas(
            root,
            width=w,
            height=self._canvas_height,
            bg=root["bg"],
            highlightthickness=0,
        )
        self.canvas.place(x=0, y=self._bar_y - self._bar_y_local)

        # Cache PhotoImage refs so Tkinter doesn't garbage-collect them
        self._zombie_img = self._load_icon(zombie_img_path, fallback_color="#4a7a3a")
        self._ambulance_img = self._load_icon(
            ambulance_img_path, fallback_color="#d94f4f"
        )

        # Positions along the bar, in feet from the bar's 0-ft end. The
        # horde always starts at 0; the ambulance starts initial_distance
        # feet ahead of it. Both move independently via advance().
        self._zombie_distance = 0.0
        self._ambulance_distance = initial_distance

        # Track canvas item ids so we can move/redraw things in place
        self._zombie_icon_id = None
        self._ambulance_icon_id = None
        self._ambulance_label_id = None

        self._draw_static_bar()

        # Separate label off to the side showing the true gap between the
        # horde and the ambulance (ambulance_distance - zombie_distance) —
        # once the horde starts moving, its bar position is no longer 0 ft,
        # so this is the only place the actual remaining gap is visible.
        self._gap_label = tk.Label(
            root,
            text="",
            font=("Arial", 11),
            bg=root["bg"],
            fg=UI_COLORS["fg"],
            justify=tk.LEFT,
            anchor="w",
        )
        self._gap_label.place(
            x=self._bar_x0,
            y=self._bar_y - self._bar_y_local - 26,
        )

        self._redraw()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _load_icon(
        self, path: str | None, fallback_color: str
    ) -> "ImageTk.PhotoImage | None":
        """Load and resize an icon image, or return None if unavailable.

        Args:
            path: Path to the image file, or None to skip loading.
            fallback_color: Unused here directly — kept for callers that
                want to draw a placeholder shape instead when this returns
                None (see _draw_icon_at).

        Returns:
            A PhotoImage sized to _icon_size, or None if path is None/missing.
        """
        if path is None:
            return None
        try:
            with Image.open(path) as img:
                resized = img.resize(
                    (self._icon_size, self._icon_size), Image.LANCZOS  # type: ignore[attr-defined]
                )
                return ImageTk.PhotoImage(resized)
        except FileNotFoundError:
            print(f"HordeMeter: icon not found at {path!r} — using placeholder shape.")
            return None

    def _draw_static_bar(self) -> None:
        """Draw the background bar and the fixed end labels."""
        y0 = self._bar_y_local - self._bar_height // 2
        y1 = self._bar_y_local + self._bar_height // 2
        self.canvas.create_rectangle(
            self._bar_x0,
            y0,
            self._bar_x1,
            y1,
            fill="#e0e0e0",
            outline="#888888",
            width=1,
        )
        self.canvas.create_text(
            self._bar_x0,
            self._bar_y_local - self._icon_size // 2 - 14,
            text="Horde",
            font=("Arial", 10),
        )
        self.canvas.create_text(
            self._bar_x1,
            self._bar_y_local - self._icon_size // 2 - 14,
            text="Camp",
            font=("Arial", 10),
        )

    def _draw_icon_at(
        self,
        x: int,
        y: int,
        img: "ImageTk.PhotoImage | None",
        fallback_color: str,
        tag: str,
    ) -> int:
        """Draw an icon (image if available, else a colored circle) at (x, y).

        Args:
            x: Canvas x-coordinate (center of the icon).
            y: Canvas y-coordinate (center of the icon).
            img: Preloaded PhotoImage, or None to draw a placeholder.
            fallback_color: Fill color used when img is None.
            tag: Canvas tag so this item can be found/moved later.

        Returns:
            The canvas item id created.
        """
        if img is not None:
            return self.canvas.create_image(x, y, image=img, tags=(tag,))
        r = self._icon_size // 2
        return self.canvas.create_oval(
            x - r, y - r, x + r, y + r, fill=fallback_color, outline="", tags=(tag,)
        )

    def _distance_to_x(self, distance: float) -> float:
        """Convert a feet-from-0 distance to a canvas x-coordinate on the bar.

        Args:
            distance: Position in feet, clamped to [0, max_distance].

        Returns:
            The corresponding x-coordinate between _bar_x0 and _bar_x1.
        """
        clamped = max(0.0, min(self.max_distance, distance))
        frac = clamped / self.max_distance if self.max_distance > 0 else 0.0
        return self._bar_x0 + frac * (self._bar_x1 - self._bar_x0)

    # ------------------------------------------------------------------
    # Live updates
    # ------------------------------------------------------------------

    def advance(self, zombie_delta: float = 0.0, ambulance_delta: float = 0.0) -> None:
        """Move the zombie horde and/or the ambulance along the bar.

        Args:
            zombie_delta: Feet the horde advances this turn (e.g. 10 ft for
                every action taken).
            ambulance_delta: Feet the ambulance moves this turn — varies by
                which action the player took (e.g. 10 ft for squish, 5 ft
                for skip).
        """
        self._zombie_distance = max(
            0.0, min(self.max_distance, self._zombie_distance + zombie_delta)
        )
        self._ambulance_distance = max(
            0.0, min(self.max_distance, self._ambulance_distance + ambulance_delta)
        )
        self._redraw()

    def _redraw(self) -> None:
        """Redraw the zombie icon, the ambulance icon + label, and the side gap label."""
        if self._zombie_icon_id is not None:
            self.canvas.delete(self._zombie_icon_id)
        zx = self._distance_to_x(self._zombie_distance)
        self._zombie_icon_id = self._draw_icon_at(
            zx, self._bar_y_local, self._zombie_img, "#4a7a3a", tag="zombie"
        )

        if self._ambulance_icon_id is not None:
            self.canvas.delete(self._ambulance_icon_id)
        if self._ambulance_label_id is not None:
            self.canvas.delete(self._ambulance_label_id)
        ax = self._distance_to_x(self._ambulance_distance)
        self._ambulance_icon_id = self._draw_icon_at(
            ax, self._bar_y_local, self._ambulance_img, "#d94f4f", tag="ambulance"
        )
        self._ambulance_label_id = self.canvas.create_text(
            ax,
            self._bar_y_local - self._icon_size // 2 - 10,
            text=f"{max(0.0, min(self.max_distance, self._ambulance_distance)):.0f} ft",
            font=("Arial", 11, "bold"),
        )

        gap = self._ambulance_distance - self._zombie_distance
        self._gap_label.config(
            text=f"Distance between zombies and ambulance: {gap:.0f} ft"
        )
