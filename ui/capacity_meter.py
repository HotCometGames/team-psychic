import math
import tkinter as tk
import os
from typing import Any
from PIL import Image, ImageTk

from gameplay.constants import UI_COLORS


class CapacityMeter:
    """Displays the ambulance capacity as a grid of colored unit squares.

    Each square represents one ambulance slot. Filled slots are shown in the
    accent (danger red) color; empty slots are shown in the panel background.

    Attributes:
        canvas: Tkinter canvas containing the unit squares and label.
        unit_size: Side length of each capacity square in pixels.
    """

    canvas: tk.Canvas
    unit_size: int
    _units: list[int]
    CATEGORY_IMAGES = {
        "medic": "ui/graphics/icons/doctor.png",        # Cyan (Healthy Specialist)
        "engineer": "ui/graphics/icons/engineer.png",        # Cyan (Healthy Specialist)
        "injured_medic": "ui/graphics/icons/injured.png",    # Deep Blue (Injured Specialist)
        "injured_engineer": "ui/graphics/icons/injured.png",# Deep Blue (Injured Specialist)
        "civilian": "ui/graphics/icons/civilian.png",       # Green (Healthy Civilian)
        "injured": "ui/graphics/icons/injured.png",       # Orange (Injured Civilian)
        "zombie": "ui/graphics/icons/zombie.png",         # Red (Threat)
        "corpse": "ui/graphics/icons/zombie.png",        # Dark Red (Corpse)
    }

    CATEGORY_PRIORITY = {
        "medic": 1,
        "engineer": 2,
        "injured_medic": 3,
        "injured_engineer": 4,
        "civilian": 5,
        "injured": 6,
        "zombie": 7,
        "corpse": 8,
    }

    def __init__(self, root: Any, w: int, h: int, max_cap: int) -> None:
        """Create the capacity meter canvas and render all unit squares.

        Args:
            root: Tkinter parent widget.
            w: Window width in pixels (used to position the canvas).
            h: Window height in pixels (used to position the canvas).
            max_cap: Total number of ambulance slots to display.
        """
        self._canvas_w = math.floor(0.2 * w)
        self.canvas = tk.Canvas(
            root,
            width=self._canvas_w,
            height=math.floor(0.45 * h),
            bg=UI_COLORS["panel"],
            highlightthickness=0,
        )
        self.canvas.place(x=math.floor(0.75 * w), y=math.floor(0.4 * h))
        self._units = []
        self._unit_icons = []
        self.unit_size = 40

        self.icon_cache: dict[str, ImageTk.PhotoImage] = {}
        
        # Preload and resize icons to fit the square unit size
        self._load_icons(self.unit_size)

        self.canvas.update()
        self.render(max_cap, self.unit_size)

    def _load_icons(self, size: int) -> None:
        """Pre-load and resize custom icons into Tkinter PhotoImages."""
        for category, path in self.CATEGORY_IMAGES.items():
            if os.path.exists(path):
                with Image.open(path) as img:
                    resized = img.resize((size, size), Image.LANCZOS)
                    self.icon_cache[category] = ImageTk.PhotoImage(resized)
            else:
                # Optional: silence or log missing file warnings
                pass

    def render(self, max_cap: int, size: int) -> None:
        """Draw the capacity label and all unit squares on the canvas.

        Args:
            max_cap: Number of unit squares to create.
            size: Side length of each square in pixels.
        """
        tk.Label(
            self.canvas,
            text="Capacity",
            font=("Arial", 15),
            bg=UI_COLORS["panel"],
            fg=UI_COLORS["fg"],
        ).place(x=100, y=0)

        x = 3
        y = 50
        for _ in range(max_cap):
            # 1. Unpack BOTH IDs from create_unit
            rect_id, img_id = self.create_unit(self.canvas, x, y, size)
            
            # 2. Store rectangle ID in _units and image ID in _unit_icons
            self._units.append(rect_id)
            self._unit_icons.append(img_id)

            x += size * 3 // 2
            if (x + size * 3 // 2) > self._canvas_w:
                x = 3
                y += size * 3 // 2

    def update_fill(self, saved_passengers: Any) -> None:
        """Update the filled/empty state of the capacity squares."""
        # 1. Reset all canvas squares to empty background
        for i in range(len(self._units)):
            self.canvas.itemconfig(self._units[i], fill=UI_COLORS["panel"], outline="#555555")
            self.canvas.itemconfig(self._unit_icons[i], image="")

        # Handle integer inputs (e.g. update_fill(0))
        if isinstance(saved_passengers, int):
            for i in range(min(saved_passengers, len(self._units))):
                self.canvas.itemconfig(
                    self._units[i],
                    fill=UI_COLORS["meter_filled"],
                    outline=UI_COLORS["meter_filled"],
                )
            return

        # If passed ScoreKeeper.ambulance dictionary, expand it into a list of strings
        if isinstance(saved_passengers, dict):
            expanded = []
            for category, count in saved_passengers.items():
                expanded.extend([category] * count)
            saved_passengers = expanded

        if not isinstance(saved_passengers, (list, tuple)):
            return

        # 2. Extract sorting/color key from Humanoid objects, strings, or dicts
        def get_key(h):
            if isinstance(h, str):
                return h.lower()
            if isinstance(h, dict):
                state = h.get("state", "").lower()
                role = h.get("role", "civilian").lower()
            else:
                state = getattr(h, "state", "").lower()
                role = getattr(h, "role", "civilian").lower()

            if state in ("zombie", "corpse"):
                return state

            if state == "injured":
                if role in ("medic", "engineer"):
                    return f"injured_{role}"
                return "injured"

            return role if role in ("medic", "engineer") else "civilian"

        # 3. Sort passengers by CATEGORY_PRIORITY
        sorted_passengers = sorted(
            saved_passengers,
            key=lambda h: self.CATEGORY_PRIORITY.get(get_key(h), 99)
        )

        # 4. Display icons on grid slots
        for i, humanoid in enumerate(sorted_passengers):
            if i >= len(self._units):
                break  # Prevent capacity overflow
            
            key = get_key(humanoid)
            
            if key in self.icon_cache:
                self.canvas.itemconfig(self._unit_icons[i], image=self.icon_cache[key])
    @staticmethod
    def create_unit(canvas: tk.Canvas, x: float, y: float, size: int) -> int:
        """Draw a single capacity square on the canvas and return its item ID.

        Args:
            canvas: Canvas to draw on.
            x: Left edge x-coordinate.
            y: Top edge y-coordinate.
            size: Side length of the square in pixels.

        Returns:
            Tkinter canvas item ID for the created rectangle.
        """
        rect_id = canvas.create_rectangle(
            x, y, x + size, y + size, fill=UI_COLORS["panel"], outline="#555555"
        )
        # Added an image placeholder in the middle of the slot
        img_id = canvas.create_image(
            x + size // 2, y + size // 2, anchor=tk.CENTER, image=""
        )
        return rect_id, img_id
