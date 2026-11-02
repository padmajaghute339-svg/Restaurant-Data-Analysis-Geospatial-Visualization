"""
Shared helpers for all Cognifyz Data Science internship levels.
Put your CSV (e.g. Dataset.csv) in the SAME folder as these scripts.
"""
from pathlib import Path
import warnings
import matplotlib

# ---------------------------------------------------------------
# Set to True if you want plot windows to pop up while running.
# (Plots are ALWAYS saved as PNG files inside the "outputs" folder.)
# ---------------------------------------------------------------
SHOW_PLOTS = False
if not SHOW_PLOTS:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# Zomato country codes -> country names
COUNTRY_MAP = {
    1: "India", 14: "Australia", 30: "Brazil", 37: "Canada", 94: "Indonesia",
    148: "New Zealand", 162: "Philippines", 166: "Qatar", 184: "Singapore",
    189: "South Africa", 191: "Sri Lanka", 208: "Turkey", 214: "UAE",
    215: "United Kingdom", 216: "United States",
}

# Colours used by the dataset's "Rating color" column
RATING_COLORS = {
    "Dark Green": "#1b7a3d", "Green": "#4caf50", "Yellow": "#f2c811",
    "Orange": "#f28c28", "Red": "#d32f2f", "White": "#d9d9d9",
}


def find_data_file():
    csvs = [p for p in BASE_DIR.glob("*.csv")]
    if not csvs:
        raise FileNotFoundError(
            f"No CSV file found in {BASE_DIR}. Copy your dataset next to the scripts.")
    for p in csvs:  # prefer a file with 'dataset' in its name
        if "dataset" in p.name.lower():
            return p
    return csvs[0]


def load_raw():
    """Read the CSV exactly as it is (no cleaning)."""
    path = find_data_file()
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    df.columns = [c.strip().replace("\ufeff", "").replace("ï»¿", "") for c in df.columns]
    return df


def clean_data(df=None):
    """Light cleaning shared by every level."""
    df = load_raw() if df is None else df.copy()
    df["Cuisines"] = df["Cuisines"].fillna("Not Available")      # 9 missing values
    df["Country"] = df["Country Code"].map(COUNTRY_MAP).fillna("Other")
    for col in ["Has Table booking", "Has Online delivery",
                "Is delivering now", "Switch to order menu"]:
        df[col] = df[col].astype("category")
    return df


def valid_coords(df):
    """Rows with usable latitude/longitude (drops 0,0 placeholders)."""
    ok = (df["Latitude"].between(-90, 90) & df["Longitude"].between(-180, 180)
          & ~((df["Latitude"] == 0) & (df["Longitude"] == 0)))
    return df[ok]


def explode_cuisines(df):
    """One row per (restaurant, single cuisine)."""
    d = df[df["Cuisines"] != "Not Available"].copy()
    d["Cuisine"] = d["Cuisines"].str.split(",")
    d = d.explode("Cuisine")
    d["Cuisine"] = d["Cuisine"].str.strip()
    return d


def save_fig(name, subfolder=""):
    folder = OUT_DIR / subfolder
    folder.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(folder / f"{name}.png", dpi=130, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def header(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)
