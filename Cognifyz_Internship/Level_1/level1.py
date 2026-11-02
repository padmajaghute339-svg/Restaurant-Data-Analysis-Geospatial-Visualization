"""
LEVEL 1
  Task 1 - Data Exploration and Preprocessing
  Task 2 - Descriptive Analysis
  Task 3 - Geospatial Analysis
Run:  python level1.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from common import (load_raw, clean_data, valid_coords, explode_cuisines,
                    save_fig, header, OUT_DIR)

SUB = "level1"


# ------------------------------------------------------------------
# TASK 1: Data Exploration and Preprocessing
# ------------------------------------------------------------------
def task1():
    header("LEVEL 1 - TASK 1: DATA EXPLORATION & PREPROCESSING")
    raw = load_raw()

    # 1. rows & columns
    print(f"Rows: {raw.shape[0]}  |  Columns: {raw.shape[1]}")
    print("\nColumn names & data types:")
    print(raw.dtypes)

    # 2. missing values
    print("\nMissing values per column:")
    miss = raw.isna().sum()
    print(miss[miss > 0] if miss.sum() else "None")
    print(f"\nColumns with missing values: {list(miss[miss > 0].index)}")
    print("-> 'Cuisines' has 9 missing values. They are filled with 'Not Available' "
          "(only 9 of 9551 rows, so dropping or labelling are both fine).")
    print(f"Duplicate rows: {raw.duplicated().sum()}")

    df = clean_data(raw)
    print("\nMissing values AFTER handling:", int(df.isna().sum().sum()))

    # 3. data type conversion
    print("\nData type conversion:")
    print(" - Yes/No columns converted to 'category' dtype")
    print(" - Country Code mapped to a readable 'Country' column")
    print(df.dtypes)

    # 4. target distribution
    header("Target variable: Aggregate rating")
    print(df["Aggregate rating"].describe())
    counts = df["Aggregate rating"].value_counts().sort_index()
    print("\nValue counts per rating:\n", counts)

    print("\nClass distribution (Rating text):")
    cls = df["Rating text"].value_counts()
    print(pd.DataFrame({"count": cls, "percent": (cls / len(df) * 100).round(2)}))
    ratio = cls.max() / cls.min()
    print(f"\nImbalance ratio (largest / smallest class) = {ratio:.1f} : 1")
    print("-> Classes are IMBALANCED: 'Average' (~39%) and 'Not rated' (~22%) dominate, "
          "while 'Excellent' (~3%) and 'Poor' (~2%) are rare.")
    print(f"-> {int((df['Aggregate rating'] == 0).sum())} restaurants have rating 0 "
          "(= not rated yet, not a true 0 score).")

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df["Aggregate rating"], bins=25, kde=True, ax=ax[0], color="#4c72b0")
    ax[0].set_title("Distribution of Aggregate rating")
    order = ["Not rated", "Poor", "Average", "Good", "Very Good", "Excellent"]
    sns.countplot(data=df, x="Rating text", order=order, ax=ax[1], palette="viridis")
    ax[1].set_title("Class balance (Rating text)")
    ax[1].tick_params(axis="x", rotation=30)
    save_fig("task1_target_distribution", SUB)
    return df


# ------------------------------------------------------------------
# TASK 2: Descriptive Analysis
# ------------------------------------------------------------------
def task2(df):
    header("LEVEL 1 - TASK 2: DESCRIPTIVE ANALYSIS")

    num_cols = ["Longitude", "Latitude", "Average Cost for two", "Price range",
                "Aggregate rating", "Votes"]
    stats = df[num_cols].agg(["mean", "median", "std", "min", "max", "skew"]).T
    stats["mode"] = [df[c].mode().iloc[0] for c in num_cols]
    print(stats.round(3))
    print("\nFull describe():\n", df[num_cols].describe().round(3))

    # Categorical distributions
    header("Categorical variable distributions")
    cc = df["Country Code"].value_counts()
    print("Country Code (top 10):\n",
          pd.DataFrame({"country": [df.loc[df["Country Code"] == c, "Country"].iloc[0] for c in cc.index],
                        "restaurants": cc.values}, index=cc.index).head(10))
    city = df["City"].value_counts()
    print("\nCity (top 10):\n", city.head(10))

    cu = explode_cuisines(df)["Cuisine"].value_counts()
    print("\nCuisines (top 10 individual cuisines):\n", cu.head(10))
    print("\nMost common cuisine COMBINATIONS (top 10):\n", df["Cuisines"].value_counts().head(10))

    print(f"\nTop cuisine : {cu.index[0]} ({cu.iloc[0]} restaurants)")
    print(f"Top city    : {city.index[0]} ({city.iloc[0]} restaurants)")
    print(f"Unique cities: {df['City'].nunique()} | unique cuisines: {len(cu)}")

    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    cc_named = df["Country"].value_counts()
    sns.barplot(x=cc_named.values, y=cc_named.index, ax=ax[0], palette="Blues_r")
    ax[0].set_title("Restaurants per Country")
    sns.barplot(x=city.head(10).values, y=city.head(10).index, ax=ax[1], palette="Greens_r")
    ax[1].set_title("Top 10 Cities")
    sns.barplot(x=cu.head(10).values, y=cu.head(10).index, ax=ax[2], palette="Oranges_r")
    ax[2].set_title("Top 10 Cuisines")
    save_fig("task2_categorical_distributions", SUB)

    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    for a, c in zip(ax.ravel(), num_cols):
        sns.histplot(df[c], bins=30, kde=True, ax=a, color="#6a5acd")
        a.set_title(c)
    save_fig("task2_numeric_distributions", SUB)


# ------------------------------------------------------------------
# TASK 3: Geospatial Analysis
# ------------------------------------------------------------------
def task3(df):
    header("LEVEL 1 - TASK 3: GEOSPATIAL ANALYSIS")
    geo = valid_coords(df)
    print(f"Restaurants with valid coordinates: {len(geo)} of {len(df)} "
          f"({len(df) - len(geo)} dropped as invalid 0,0 placeholders)")

    # 1. map of all restaurants
    plt.figure(figsize=(14, 7))
    sc = plt.scatter(geo["Longitude"], geo["Latitude"], c=geo["Aggregate rating"],
                     cmap="RdYlGn", s=6, alpha=0.7)
    plt.colorbar(sc, label="Aggregate rating")
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.title("Restaurant locations worldwide (coloured by rating)")
    save_fig("task3_world_map", SUB)

    # zoom into the biggest country cluster(s)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    for a, country in zip(ax, ["India", "United States"]):
        sub = geo[geo["Country"] == country]
        s = a.scatter(sub["Longitude"], sub["Latitude"], c=sub["Aggregate rating"],
                      cmap="RdYlGn", s=8, alpha=0.7)
        a.set_title(f"{country} ({len(sub)} restaurants)")
        a.set_xlabel("Longitude"); a.set_ylabel("Latitude")
    plt.colorbar(s, ax=ax, label="Aggregate rating")
    save_fig("task3_zoom_maps", SUB)

    # optional interactive map (needs: pip install folium)
    try:
        import folium
        from folium.plugins import MarkerCluster
        m = folium.Map(location=[geo["Latitude"].mean(), geo["Longitude"].mean()], zoom_start=2)
        cluster = MarkerCluster().add_to(m)
        for _, r in geo.sample(min(2000, len(geo)), random_state=1).iterrows():
            folium.Marker([r["Latitude"], r["Longitude"]],
                          popup=f"{r['Restaurant Name']} - {r['Aggregate rating']}").add_to(cluster)
        (OUT_DIR / SUB).mkdir(parents=True, exist_ok=True)
        m.save(str(OUT_DIR / SUB / "task3_interactive_map.html"))
        print("Interactive map saved: outputs/level1/task3_interactive_map.html")
    except ImportError:
        print("(folium not installed - skipping interactive map. Run: pip install folium)")

    # 2. distribution across cities / countries
    by_country = df.groupby("Country").agg(restaurants=("Restaurant ID", "count"),
                                           avg_rating=("Aggregate rating", "mean")).round(2)
    print("\nRestaurants & average rating per country:\n",
          by_country.sort_values("restaurants", ascending=False))
    by_city = df.groupby("City").agg(restaurants=("Restaurant ID", "count"),
                                     avg_rating=("Aggregate rating", "mean")).round(2)
    print("\nTop 10 cities by restaurant count:\n",
          by_city.sort_values("restaurants", ascending=False).head(10))
    share = df["Country"].value_counts(normalize=True).iloc[0] * 100
    print(f"\n{df['Country'].value_counts().index[0]} holds {share:.1f}% of all restaurants.")

    # 3. correlation between location & rating
    print("\nCorrelation of location with Aggregate rating:")
    for col in ["Latitude", "Longitude"]:
        print(f"  {col:9s} Pearson={geo[col].corr(geo['Aggregate rating']):+.3f}  "
              f"Spearman={geo[col].corr(geo['Aggregate rating'], method='spearman'):+.3f}")
    print("-> Raw lat/long correlation is weak, because location effects are by COUNTRY/CITY "
          "(clusters), not a straight line across the globe.")

    # rated restaurants only + country effect
    rated = geo[geo["Aggregate rating"] > 0]
    print("\nAverage rating per country (rated restaurants only):")
    print(rated.groupby("Country")["Aggregate rating"].mean().round(2).sort_values(ascending=False))

    plt.figure(figsize=(11, 6))
    order = rated.groupby("Country")["Aggregate rating"].mean().sort_values(ascending=False).index
    sns.boxplot(data=rated, y="Country", x="Aggregate rating", order=order, palette="coolwarm")
    plt.title("Rating by country (rated restaurants)")
    save_fig("task3_rating_by_country", SUB)

    cities = by_city[by_city["restaurants"] >= 30].sort_values("avg_rating", ascending=False)
    plt.figure(figsize=(10, 8))
    sns.barplot(x=cities["avg_rating"].head(20), y=cities.head(20).index, palette="viridis")
    plt.title("Top 20 cities by average rating (cities with >= 30 restaurants)")
    save_fig("task3_top_cities_rating", SUB)


if __name__ == "__main__":
    data = task1()
    task2(data)
    task3(data)
    print(f"\nDone. Plots saved in: {OUT_DIR / SUB}")
