"""Horde distance meter — shows how far the ambulance is from the zombie horde.
 
Follows the same widget pattern as CapacityMeter / Clock: a small class that
owns its own tk.Canvas, is constructed once in UI.__init__, and exposes an
update_*() method that UI.update_ui() calls each turn to refresh the display.
 
Usage (inside ui/ui.py, alongside your other widget construction):
 
    from ui.horde_meter import HordeMeter
 
    # in UI.__init__, near where capacity_meter is created:
    self.horde_meter = HordeMeter(self.root, w, h, max_distance=100, initial_distance=10)
 
    # in UI.update_ui(), alongside clock/capacity_meter updates:
    self.horde_meter.update_distance(new_distance)
"""
 
import math
from os.path import abspath, dirname, join
import tkinter as tk
 
from PIL import Image, ImageTk
 
 
class HordeMeter:
    """A horizontal progress bar showing ambulance-to-horde distance.
 
    The bar spans from 0 ft (horde, left) to max_distance ft (ambulance
    starts near this end, right). As distance shrinks, the ambulance icon
    moves left, toward the zombie horde icon.
 
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
        """Build the meter and place it near the top of the window.
 
        Args:
            root: Parent Tkinter window.
            w: Window width in pixels (used to size/position the bar).
            h: Window height in pixels (used to position the bar vertically).
            max_distance: Total bar length in feet. Default 100.
            initial_distance: Starting ambulance-to-horde distance in feet.
                Default 10 (close call at the start, per your design).
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
 
        # Track distance label + moving icon ids so we can update them in place
        self._label_id = None
        self._icon_id = None
        self._icon_shape_id = None  # used only when no image is provided
 
        self._draw_static_bar()
        self.update_distance(initial_distance)
 
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
        """Draw the background bar and the zombie horde icon at the left end."""
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
        # Zombie horde stays fixed at the left end (0 ft)
        self._draw_icon_at(
            self._bar_x0, self._bar_y_local, self._zombie_img, "#4a7a3a", tag="zombie"
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
 
    # ------------------------------------------------------------------
    # Live updates
    # ------------------------------------------------------------------
 
    def update_distance(self, distance: float) -> None:
        """Move the ambulance icon and refresh the distance label.
 
        Args:
            distance: Current ambulance-to-horde distance in feet. Clamped
                to [0, max_distance] before rendering.
        """
        clamped = max(0.0, min(self.max_distance, distance))
        frac = clamped / self.max_distance if self.max_distance > 0 else 0.0
        x = self._bar_x0 + frac * (self._bar_x1 - self._bar_x0)
 
        # Remove the previous ambulance icon + label before redrawing
        if self._icon_id is not None:
            self.canvas.delete(self._icon_id)
        if self._label_id is not None:
            self.canvas.delete(self._label_id)
 
        self._icon_id = self._draw_icon_at(
            x, self._bar_y_local, self._ambulance_img, "#d94f4f", tag="ambulance"
        )
        self._label_id = self.canvas.create_text(
            x,
            self._bar_y_local - self._icon_size // 2 - 10,
            text=f"{clamped:.0f} ft",
            font=("Arial", 11, "bold"),
        )
 