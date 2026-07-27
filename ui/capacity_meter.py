import math
import tkinter as tk
import os
from PIL import Image, ImageTk
from typing import Any

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
        "medic": "ui/graphics/icons/doctor.png",
        "engineer": "ui/graphics/icons/engineer.png",
        "injured_medic": "ui/graphics/icons/injured.png",
        "injured_engineer": "ui/graphics/icons/injured.png",
        "civilian": "ui/graphics/icons/civilian.png",
        "injured": "ui/graphics/icons/injured.png",
        "zombie": "ui/graphics/icons/zombie.png",
        "corpse": "ui/graphics/icons/zombie.png",
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
        self.canvas = tk.Canvas(
            root,
            width=250,
            height=280,
            bg=UI_COLORS["panel"],
            highlightbackground=UI_COLORS["card_border"],
            highlightthickness=3,
            relief=tk.GROOVE,
            bd=3,
        )
        self.canvas.place(x=math.floor(0.76 * w), y=350)
        
        self._units = []
        self.unit_size = 42  # Slightly larger slots to fill the expanded box
        self._unit_icons = []

        self.icon_cache: dict[str, ImageTk.PhotoImage] = {}
        self._load_icons(self.unit_size)

        self.canvas.update()
        self.render(max_cap, self.unit_size)

    def _load_icons(self, size: int) -> None:
        """Load and resize icon images to fit the capacity unit square."""
        for category, path in self.CATEGORY_IMAGES.items():
            if os.path.exists(path):
                with Image.open(path) as img:
                    resized = img.resize((size, size), Image.LANCZOS)
                    self.icon_cache[category] = ImageTk.PhotoImage(resized)

    def render(self, max_cap: int, size: int) -> None:
        """Draw the capacity label and all unit squares on the canvas.

        Args:
            max_cap: Number of unit squares to create.
            size: Side length of each square in pixels.
        """
        tk.Label(
            self.canvas,
            text="[ CAPACITY ]",
            font=("Courier New", 11, "bold"),
            bg=UI_COLORS["panel"],
            fg=UI_COLORS["fg"],
        ).pack(side=tk.TOP, pady=(15, 15))

        grid_frame = tk.Frame(self.canvas, bg=UI_COLORS["panel"])
        grid_frame.pack(side=tk.TOP, padx=15, pady=5)

        cols = 4
        for i in range(max_cap):
            row = i // cols
            col = i % cols
            
            # Canvas slot with matching mechy border color instead of white/default outline
            slot_canvas = tk.Canvas(
                grid_frame,
                width=size,
                height=size,
                bg=UI_COLORS["slot_bg"],
                highlightbackground=UI_COLORS["slot_border"],
                highlightcolor=UI_COLORS["slot_border"],
                highlightthickness=2,
                relief=tk.SUNKEN,
                bd=1,
            )
            slot_canvas.grid(row=row, column=col, padx=5, pady=5)

            img_id = slot_canvas.create_image(
                size // 2, size // 2, anchor=tk.CENTER, image=""
            )
            
            self._units.append(slot_canvas)
            self._unit_icons.append((slot_canvas, img_id))

    def update_fill(self, saved_passengers: Any) -> None:
        """Update the filled/empty state of the capacity squares."""
        # 1. Reset all canvas squares to empty background
        for slot_canvas, img_id in self._unit_icons:
            slot_canvas.config(bg=UI_COLORS["slot_bg"])
            slot_canvas.itemconfig(img_id, image="")

        # Handle integer inputs (e.g. update_fill(0))
        if isinstance(saved_passengers, int):
            for i in range(min(saved_passengers, len(self._units))):
                slot_canvas, _ = self._unit_icons[i]
                slot_canvas.config(bg=UI_COLORS["meter_filled"])
            return

        # If passed ScoreKeeper.ambulance dictionary, expand it into a list
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

        # 4. Apply colors to grid slots
        for i, humanoid in enumerate(sorted_passengers):
            if i >= len(self._units):
                break  # Prevent capacity overflow

            key = get_key(humanoid)
            slot_canvas, img_id = self._unit_icons[i]
            slot_canvas.config(bg=UI_COLORS["meter_filled"])
            
            if key in self.icon_cache:
                slot_canvas.itemconfig(img_id, image=self.icon_cache[key])
            

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
        img_id = canvas.create_image(
            x + size // 2, y + size // 2, anchor=tk.CENTER, image=""
        )
        return rect_id, img_id
