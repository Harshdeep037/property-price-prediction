"""Download and load the California Housing dataset (housing.csv)."""
import urllib.request

import pandas as pd

from src import config


def download_dataset(url: str = config.DATA_URL, path=config.DATA_PATH) -> None:
    """Download housing.csv into the data/ folder (needs internet)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from {url} ...")
    urllib.request.urlretrieve(url, path)
    print(f"Saved to {path}")


def load_dataset(path=config.DATA_PATH) -> pd.DataFrame:
    """Read housing.csv and check that it has the columns we expect."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Run:  python -m src.data_loader   (downloads it automatically)\n"
            "or download housing.csv manually and put it in the data/ folder."
        )
    df = pd.read_csv(path)
    missing = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"housing.csv is missing expected columns: {missing}")
    return df


if __name__ == "__main__":
    download_dataset()
    data = load_dataset()
    print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns.")
    print("Missing values per column:")
    print(data.isna().sum()[data.isna().sum() > 0])
