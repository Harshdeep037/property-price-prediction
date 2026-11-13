"""Central place for paths, column names and settings.

Every other file imports from here, so a name or path is only ever defined once.
"""
from pathlib import Path

# ---- Paths -----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
IMAGES_DIR = BASE_DIR / "static" / "images"

DATA_PATH = DATA_DIR / "housing.csv"
DATA_URL = (
    "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
    "datasets/housing/housing.csv"
)

# ---- Columns ---------------------------------------------------------------
TARGET = "median_house_value"

NUMERIC_FEATURES = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",   # the only column with missing values in this dataset
    "population",
    "households",
    "median_income",
]

CATEGORICAL_FEATURES = ["ocean_proximity"]
OCEAN_CATEGORIES = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]

# Columns the raw dataset must contain (features + target)
REQUIRED_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]

# ---- Training settings -----------------------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ---- Input validation (website / API) ---------------------------------------
# Allowed ranges, based on the values that exist in the dataset.
# Predictions far outside these ranges would be unreliable guesses.
INPUT_LIMITS = {
    "longitude": (-124.5, -114.0),
    "latitude": (32.5, 42.0),
    "housing_median_age": (1, 52),
    "total_rooms": (1, 40000),
    "total_bedrooms": (1, 7000),
    "population": (1, 36000),
    "households": (1, 6500),
    "median_income": (0.5, 15.0),
}

MODEL_PATH = MODELS_DIR / "model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

# ---- Website form: (label, hint, step) for each numeric input ---------------
FIELD_INFO = {
    "longitude": ("Longitude", "West is negative, e.g. -122.23", "0.01"),
    "latitude": ("Latitude", "e.g. 37.88", "0.01"),
    "housing_median_age": ("Median house age (years)", "1 to 52", "1"),
    "total_rooms": ("Total rooms", "All rooms in the district", "1"),
    "total_bedrooms": ("Total bedrooms", "Cannot exceed total rooms", "1"),
    "population": ("Population", "People in the district", "1"),
    "households": ("Households", "Number of households", "1"),
    "median_income": ("Median income", "In tens of thousands of $ (8.3 = $83,000)", "0.01"),
}
EXAMPLE_INPUT = {
    "longitude": -122.23, "latitude": 37.88, "housing_median_age": 41,
    "total_rooms": 880, "total_bedrooms": 129, "population": 322,
    "households": 126, "median_income": 8.3252, "ocean_proximity": "NEAR BAY",
}

# ---- Login system -----------------------------------------------------------
DB_PATH = DATA_DIR / "users.db"
