import math
import tkinter as tk
from os.path import join
from typing import Any
from PIL import ImageTk, Image

from gameplay.constants import UI_COLORS


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

        self.canvas.bind("<Button-1>", self._on_canvas_click)
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

    def create_photos(self, file_paths: list[str]) -> None:
        """Load a list of images from disk and display them side-by-side.

        Replaces any existing photos. Automatically divides the canvas width 
        equally among all images provided.
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
        
        for i, fp in enumerate(file_paths):
            # Calculate the horizontal offset for each image
            x_offset = i * target_w
            
            # Load and resize image to fit its fractional slice
            photo = self._get_photo(fp, target_w, canvas_h)
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

    def create_photo(self, fp: str) -> None:
        """Load an image from disk and display it on the canvas.

        Replaces any existing photo. Holds a reference in ``self.photo``
        to prevent garbage collection.

        Args:
            fp: Path to the image file.
        """
        self.create_photos([fp])

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
        
    def display_score(
        self, score: dict[str, int | float], end_reason: str = ""
    ) -> None:
        """Render the final score over the canvas.

        Args:
            score: Dict with ``"killed"`` and ``"saved"`` integer counts and
                ``"saved_value"`` float (total value of saved humanoids),
                as returned by :meth:`~gameplay.scorekeeper.ScoreKeeper.get_score`.
            end_reason: Optional end-of-game reason string (e.g. ``"Time's Up"``
                or ``"The City Is Empty"``). Displayed as a bold header if provided.
        """
        if end_reason:
            tk.Label(
                self.canvas,
                text=end_reason,
                font=("Arial", 20, "bold"),
                bg=UI_COLORS["panel"],
                fg=UI_COLORS["accent"],
            ).pack(anchor=tk.NW)
        tk.Label(
            self.canvas,
            text="FINAL SCORE",
            font=("Arial", 30, "bold"),
            bg=UI_COLORS["panel"],
            fg=UI_COLORS["fg"],
        ).pack(anchor=tk.NW)
        tk.Label(
            self.canvas,
            text="Killed {}".format(score["killed"]),
            font=("Arial", 18),
            bg=UI_COLORS["panel"],
            fg=UI_COLORS["accent"],
        ).pack(anchor=tk.NW)
        tk.Label(
            self.canvas,
            text="Saved {}".format(score["saved"]),
            font=("Arial", 18),
            bg=UI_COLORS["panel"],
            fg=UI_COLORS["fg"],
        ).pack(anchor=tk.NW)
        tk.Label(
            self.canvas,
            text="Saved Value {0:.4f}".format(score["saved_value"]),
            font=("Arial", 18),
            bg=UI_COLORS["panel"],
            fg=UI_COLORS["fg"],
        ).pack(anchor=tk.NW)

    @staticmethod
    def _get_photo(img_path: str, max_w: int, max_h: int) -> Any:
        """Load, resize, and return a Tkinter-compatible photo image.

        Args:
            img_path: Path to the image file.
            w: Target width in pixels.
            h: Target height in pixels.

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
            
            tk_img = ImageTk.PhotoImage(img.resize((new_w, new_h), Image.LANCZOS))
        return tk_img
