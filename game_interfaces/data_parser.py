import os
import random
import pandas as pd

from gameplay.enums import State
from gameplay.humanoid import Humanoid


class DataParser:
    """Parses the image dataset and supplies humanoids for the game.

    Reads the metadata CSV once at construction time and maintains a split of
    unvisited / visited row indices. Each call to `get_random` draws one
    humanoid without replacement.

    Attributes:
        fp: Root directory of the image dataset.
        df: Full metadata DataFrame loaded from the CSV.
        unvisited: Row indices not yet drawn this episode. Consumed by
            ``get_random()``; checked by ``TrainEnv.step()`` to detect
            episode end (``len(unvisited) == 0``).
        visited: Row indices already drawn this episode. Maintained for
            audit / debugging purposes; not used by any game logic.
    """

    fp: str
    df: pd.DataFrame
    unvisited: set[int]
    visited: list[int]

    def __init__(
        self,
        data_dir: str,
        metadata_filename: str = "consolidated_metadata.csv",
        seed: int | None = None,
    ) -> None:
        """Load the metadata CSV and initialize the unvisited pool.

        Args:
            data_dir: Path to the folder containing the metadata CSV and images.
            metadata_filename: Name of the metadata CSV file.
            seed: Optional random seed for reproducible humanoid ordering and
                value assignment. Pass ``None`` (default) to use the global
                random state.

        Raises:
            FileNotFoundError: If the metadata CSV does not exist at the
                expected path.
        """
        metadata_file_path = os.path.join(data_dir, metadata_filename)
        self.fp = data_dir
        self.rng = random.Random(seed)
        if not os.path.exists(metadata_file_path):
            raise FileNotFoundError(
                f"Metadata CSV not found at '{metadata_file_path}'. "
                f"Pass the correct directory via '-d/--data' or update DATA_DIR "
                f"in gameplay/constants.py."
            )
        self.df = pd.read_csv(metadata_file_path)
        self.unvisited = set(self.df.index)
        self.visited = []

    def reset(self) -> None:
        """Reset the unvisited pool for a new episode."""
        self.unvisited = set(self.df.index)
        self.visited = []

    def get_random(self) -> Humanoid:
        """Draw and return a random humanoid without replacement.

        Returns:
            A :class:`~gameplay.humanoid.Humanoid` sampled uniformly from the
            remaining unvisited rows.

        Raises:
            ValueError: If no humanoids remain in the unvisited pool.
        """
        if len(self.unvisited) == 0:
            raise ValueError("No humanoids remain")
        # sorted() for reproducibility: set iteration order is undefined in Python.
        # Invariant: unvisited contains integer DataFrame indices — sorted() produces
        # a deterministic list, so rng.choice() yields the same draw for a given seed.
        index = self.rng.choice(sorted(self.unvisited))
        self.unvisited.discard(index)
        self.visited.append(index)

        data_row = self.df.iloc[index]

        state = datarow_to_state(data_row)

        role = data_row["Role"]
        if not isinstance(role, str) or not role:
            raise ValueError(
                f"Invalid 'Role' value at row index {data_row.name!r}: {role!r}. "
                "Check the metadata CSV for missing or non-string roles."
            )

        filename = data_row["Filename"]
        if not isinstance(filename, str) or not filename:
            raise ValueError(
                f"Invalid 'Filename' value at row index {data_row.name!r}: {filename!r}. "
                "Check the metadata CSV for missing or non-string filenames."
            )
        value = datarow_to_value(data_row, rng=self.rng)

        # Raw CSV fields, kept on the Humanoid alongside the derived `state`
        # so per-turn logging can record the row as-is (class/injured/gender/
        # item), not just the game's 4-way state classification. Gender/Item
        # use .get() (not direct indexing) — unlike Class/Injured/Role/Filename
        # they're descriptive, not load-bearing, so a dataset or test fixture
        # missing those columns shouldn't hard-fail the draw.
        img_class = str(data_row["Class"]).strip()
        injured = _parse_injured_flag(data_row)
        gender = data_row.get("Gender", "")
        item = data_row.get("Item", "")

        humanoid = Humanoid(
            fp=filename,
            state=state,
            role=role,
            value=value,
            img_class=img_class,
            injured=injured,
            gender=str(gender) if pd.notna(gender) else "",
            item=str(item) if pd.notna(item) else "",
        )
        return humanoid


def _parse_injured_flag(data_row: pd.Series) -> bool:
    """Normalize the ``Injured`` CSV column to a reliable bool.

    pandas may read this column as a real bool, an int (1/0), or a string
    ("True"/"False") depending on how the CSV was produced. A bare
    ``if img_injured:`` is wrong for string values — ``"False"`` is truthy.
    NaN is treated as not-injured (all three conditions evaluate to False).

    Args:
        data_row: A pandas Series (one row of the metadata dataframe) with
            an ``Injured`` column.

    Returns:
        True if the row is flagged as injured.
    """
    img_injured = data_row["Injured"]
    return (
        img_injured is True
        or str(img_injured).strip().lower() == "true"
        or img_injured == 1
    )


# can be customized
def datarow_to_state(data_row: pd.Series) -> str:
    """Map a metadata CSV row to a humanoid state string.

    Takes in a row of the consolidated_metadata.csv dataframe and returns
    the corresponding State value for that humanoid. This is an extension
    point (see docs/game-design.md §14) — replace this function to customize
    how CSV rows map to humanoid states.

    Args:
        data_row: A pandas Series (one row of the metadata dataframe).
            Required columns:

            - ``Class``   — ``"Default"`` for humans, anything else for zombies
            - ``Injured`` — boolean; ``True`` means the humanoid is injured

    Returns:
        A State string value: one of ``"healthy"``, ``"injured"``,
        ``"zombie"``, or ``"corpse"``.
    """
    img_class = str(data_row["Class"]).strip()
    injured = _parse_injured_flag(data_row)
    # Extension point: uncomment and use these columns to customize state mapping.
    # img_gender = data_row['Gender']
    # img_item = data_row['Item']
    if img_class == "Default":
        state = State.HEALTHY.value
        if injured:
            state = State.INJURED.value
    else:
        state = State.ZOMBIE.value
        if injured:
            state = State.CORPSE.value
    return state


# can be customized
def datarow_to_value(data_row: pd.Series, rng: random.Random | None = None) -> float:
    """Assign a numeric priority value to a humanoid from its metadata row.

    Returns a float in [0.0, 1.0] representing the triage priority of this
    humanoid. Higher values indicate higher rescue priority. This is an
    extension point — replace this function to assign values based on CSV
    columns (e.g. item rarity, injury severity, demographic factors) rather
    than uniformly at random.

    Args:
        data_row: A pandas Series (one row of the metadata dataframe).
            Available columns: ``Class``, ``Injured``, ``Gender``, ``Item``,
            ``Filename``.
        rng: Optional seeded :class:`random.Random` instance for reproducible
            value assignment. Falls back to the module-level ``random`` state
            when ``None`` (default).

    Returns:
        A float in [0.0, 1.0]. The default implementation returns a value in
        [0.0, 1.0) (exclusive upper bound from ``random.random()``); custom
        implementations may return any value in the closed interval [0.0, 1.0].
    """
    # Default: uniform random — every humanoid gets an independent random value.
    # To customize: inspect data_row columns and return a computed float.
    _rng = rng if rng is not None else random
    return _rng.random()
