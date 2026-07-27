"""
Game configuration constants.

Edit these values to design experiments and test different game configurations.
- SHIFT_LENGTH controls how much time you have per game
- AMBULANCE_CAPACITY controls how many humanoids can ride in the ambulance
- CLASSIFIER_MODEL_PATH / RL_MODEL_PATH point to your trained model weights
"""

import os

# Game rules
SHIFT_LENGTH: int = 720  # total game time in minutes
AMBULANCE_CAPACITY: int = 15  # max humanoids the ambulance can hold at once

# Total rounds in a run, win or die each round — placeholder while the
# roguelike loop is still being built out; raise as more rounds are added.
MAX_ROUNDS: int = 2

# UI window dimensions
WINDOW_WIDTH: int = 1280
WINDOW_HEIGHT: int = 800

# Data directory — path to the folder containing consolidated_metadata.csv
DATA_DIR: str = "data"

# Model weight paths — update these to point to your trained models
CLASSIFIER_MODEL_PATH: str = os.path.join("models", "weights", "baselineCNN.pth")
ROLE_CLASSIFIER_MODEL_PATH: str = os.path.join("models", "weights", "baselineRoleCNN.pth")
RL_MODEL_PATH: str = os.path.join("models", "weights", "baselineRL.pth")

# Clock display style.
# True  = analog clock (counts up from midnight — shows elapsed time)
# False = digital countdown clock (shows remaining minutes, e.g. "342 min remaining")
#
# This is an example of the feature-flag pattern. To add your own swappable
# component: (1) define a flag here, (2) add a conditional import in the UI
# file that uses it, (3) make sure both implementations share the same public
# interface (same __init__ signature and methods).
LEGACY_CLOCK: bool = True

# Observation space size — total number of slots in the flat obs vector.
# Update this if you add or remove slots; all consumers import from here.
OBS_SIZE: int = 15

# Reward weights — must sum to 1. Adjust to shift agent priorities:
#   HEADCOUNT_WEIGHT=1.0 → pure headcount (saved − killed), ignores value
#   HEADCOUNT_WEIGHT=0.0 → pure value signal, ignores headcount
HEADCOUNT_WEIGHT: float = 0.5  # weight on (saved − killed)
VALUE_WEIGHT: float = 1.0 - HEADCOUNT_WEIGHT  # weight on saved_value

# Penalty applied when the agent attempts a blocked action (e.g. save when full).
# Negative nudge to discourage no-ops without distorting the reward scale too much.
BLOCKED_ACTION_PENALTY: float = -0.5

# CNN confidence thresholds — used to bin the max softmax probability into
# low/medium/high confidence. Shared between train_env and rl_recommender;
# change here and both inference and training stay in sync.
CNN_CONFIDENCE_MID: float = 0.5  # below this → low confidence (bin 0)
CNN_CONFIDENCE_HIGH: float = 0.75  # above this → high confidence (bin 2)

# Horde-distance mechanic (human GUI modes only) — replaces the shift clock
# as the round's end condition. The horde starts at 0 ft; the ambulance
# starts HORDE_INITIAL_DISTANCE ft ahead of it and must reach
# HORDE_MAX_DISTANCE ft (camp) before the horde closes the gap. Action time
# costs (ActionCost) are unchanged — they still deduct remaining_time for
# logging/RL purposes, they just no longer end the round.
# Per-action feet moved each turn live in gameplay/enums.py — HordeDistance.
HORDE_MAX_DISTANCE: float = 300.0  # ft from horde start (0) to camp
HORDE_INITIAL_DISTANCE: float = 20.0  # ambulance's starting lead over the horde

# Feet added to EVERY future ambulance movement this round, per engineer
# saved (alive) — stacks additively, applies to Skip/Squish/Save/Scram alike,
# and takes effect starting with the very action that saved the engineer.
# Kept small so a few engineers speed things up without trivializing the
# horde chase.
ENGINEER_SPEED_BONUS: float = 2.0

# From this round onward, humanoid images are blurred — pushes the player to
# rely on the classifier/recommender's on-screen suggestion instead of just
# eyeballing the raw image. Only takes effect when a classifier is active
# (human_cnn/human_rl/human_cnn_rl/machine) — plain human mode has no AI
# assist to fall back on, so images stay sharp there.
IMAGE_BLUR_START_ROUND: int = 2
IMAGE_BLUR_RADIUS: float = 10.0

# UI color palette — adjust these to restyle the interface.
UI_COLORS: dict[str, str] = {
    "bg": "#1a1a1a",             # Root canvas background
    "panel": "#242424",          # Panel enclosure background
    "fg": "#ecf4e4",             # Default text color
    "accent": "#55ff55",         # Phosphor green highlight
    "card_border": "#445544",    # Heavy stamped-steel border color
    "slot_border": "#335533",    # Inset slot border color
    "slot_bg": "#111811",        # Dark cargo bay slot background
    
    # Action button face colors (industrial palette)
    "btn_skip": "#363e25",       # Slate Blue
    "btn_squish": "#3e3525",     # Hazard Warning Red
    "btn_save": "#3e3525",       # Armored Recovery Green
    "btn_scram": "#3b331a",      # Caution Amber
    "btn_default": "#2a2a2a",    # Fallback button background
    "btn_active": "#29b729",     # Hover/Active state
    "meter_filled": "#1e4d35",   # Slot filled color
}
