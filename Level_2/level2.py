"""
LEVEL 2
  Task 1 - Table Booking and Online Delivery
  Task 2 - Price Range Analysis
  Task 3 - Feature Engineering
Run:  python level2.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from common import clean_data, save_fig, header, OUT_DIR, RATING_COLORS

SUB = "level2"


# ------------------------------------------------------------------
# TASK 1: Table Booking and Online Delivery
# ------------------------------------------------------------------
def task1(df):
    header("LEVEL 2 - TASK 1: TABLE BOOKING & ONLINE DELIVERY")

    # 1. percentages
    tb = df["Has Table booking"].value_counts(normalize=True).mul(100).round(2)
    od = df["Has Online delivery"].value_counts(normalize=True).mul(100).round(2)
    print("Table booking (%):\n", tb)
    print("\nOnline delivery (%):\n", od)

    # 2. average rating with / without table booking
    avg_tb = df.groupby("Has Table booking", observed=True)["Aggregate rating"].mean().round(3)
    print("\nAverage rating by table booking:\n", avg_tb)
    rated = df[df["Aggregate rating"] > 0]
    avg_tb_rated = rated.groupby("Has Table booking", observed=True)["Aggregate rating"].mean().round(3)
    print("\nAverage rating by table booking (rated restaurants only):\n", avg_tb_rated)
    diff = avg_tb["Yes"] - avg_tb["No"]
    print(f"\n-> Restaurants WITH table booking rate {diff:+.2f} points higher on average.")

    avg_od = df.groupby("Has Online delivery", observed=True)["Aggregate rating"].mean().round(3)
    print("\n(Extra) Average rating by online delivery:\n", avg_od)

    # 3. online delivery by price range
    ct = pd.crosstab(df["Price range"], df["Has Online delivery"])
    ct_pct = pd.crosstab(df["Price range"], df["Has Online delivery"], normalize="index").mul(100).round(2)
    print("\nOnline delivery availability by price range (counts):\n", ct)
    print("\nOnline delivery availability by price range (% within each range):\n", ct_pct)
    best = ct_pct["Yes"].idxmax()
    print(f"\n-> Price range {best} has the highest share of online delivery "
          f"({ct_pct['Yes'].max():.1f}%); price range 4 has the lowest "
          f"({ct_pct['Yes'].loc[4]:.1f}%).")

    fig, ax = plt.subplots(1, 4, figsize=(21, 5))
    ax[0].pie(tb.values, labels=tb.index, autopct="%1.1f%%", colors=["#9ecae1", "#3182bd"])
    ax[0].set_title("Has Table booking")
    ax[1].pie(od.values, labels=od.index, autopct="%1.1f%%", colors=["#a1d99b", "#31a354"])
    ax[1].set_title("Has Online delivery")
    sns.barplot(x=avg_tb.index.astype(str), y=avg_tb.values, ax=ax[2], palette="Blues")
    ax[2].set_title("Avg rating: table booking"); ax[2].set_ylabel("Avg rating")
    ct_pct["Yes"].plot(kind="bar", ax=ax[3], color="#31a354")
    ax[3].set_title("% with online delivery by price range"); ax[3].set_ylabel("%")
    ax[3].tick_params(axis="x", rotation=0)
    save_fig("task1_booking_delivery", SUB)


# ------------------------------------------------------------------
# TASK 2: Price Range Analysis
# ------------------------------------------------------------------
def task2(df):
    header("LEVEL 2 - TASK 2: PRICE RANGE ANALYSIS")

    pr = df["Price range"].value_counts().sort_index()
    print("Restaurants per price range:\n",
          pd.DataFrame({"count": pr, "percent": (pr / len(df) * 100).round(2)}))
    print(f"\n-> Most common price range: {pr.idxmax()} ({pr.max()} restaurants)")

    avg = df.groupby("Price range")["Aggregate rating"].mean().round(3)
    print("\nAverage rating per price range:\n", avg)
    avg_rated = df[df["Aggregate rating"] > 0].groupby("Price range")["Aggregate rating"].mean().round(3)
    print("\nAverage rating per price range (rated restaurants only):\n", avg_rated)

    # colour representing the highest average rating
    by_color = df.groupby("Rating color")["Aggregate rating"].mean().sort_values(ascending=False).round(3)
    print("\nAverage rating per 'Rating color':\n", by_color)
    print(f"\n-> Colour with the highest average rating: {by_color.index[0]} "
          f"({by_color.iloc[0]})")

    top_pr = avg.idxmax()
    dom_color = df[df["Price range"] == top_pr]["Rating color"].mode().iloc[0]
    print(f"-> Price range with highest average rating: {top_pr} ({avg.max()}), "
          f"its most common rating colour is '{dom_color}'.")
    print("\nRating colour share inside each price range (%):")
    print(pd.crosstab(df["Price range"], df["Rating color"], normalize="index").mul(100).round(1))

    fig, ax = plt.subplots(1, 3, figsize=(19, 5))
    sns.barplot(x=pr.index, y=pr.values, ax=ax[0], palette="Blues_d")
    ax[0].set_title("Restaurants per price range"); ax[0].set_ylabel("count")
    sns.barplot(x=avg.index, y=avg.values, ax=ax[1], palette="Greens_d")
    ax[1].set_title("Average rating per price range"); ax[1].set_ylabel("avg rating")
    cols = [RATING_COLORS[c] for c in by_color.index]
    ax[2].bar(by_color.index, by_color.values, color=cols, edgecolor="black")
    ax[2].set_title("Average rating by rating colour"); ax[2].tick_params(axis="x", rotation=30)
    save_fig("task2_price_range", SUB)

    pd.crosstab(df["Price range"], df["Rating color"], normalize="index").mul(100)[
        list(RATING_COLORS.keys())].plot(
        kind="bar", stacked=True, color=list(RATING_COLORS.values()), figsize=(9, 6), edgecolor="black")
    plt.title("Rating colour mix by price range (%)"); plt.ylabel("%"); plt.xticks(rotation=0)
    save_fig("task2_color_by_price", SUB)


# ------------------------------------------------------------------
# TASK 3: Feature Engineering
# ------------------------------------------------------------------
def engineer_features(df):
    """Reusable - level3.py imports this too."""
    d = df.copy()
    # length-based features
    d["Name Length"] = d["Restaurant Name"].str.len()
    d["Name Word Count"] = d["Restaurant Name"].str.split().str.len()
    d["Address Length"] = d["Address"].str.len()
    d["Address Word Count"] = d["Address"].str.split().str.len()
    d["Locality Length"] = d["Locality"].str.len()
    # cuisine features
    d["Cuisine Count"] = d["Cuisines"].apply(
        lambda x: 0 if x == "Not Available" else len(x.split(",")))
    d["Primary Cuisine"] = d["Cuisines"].str.split(",").str[0].str.strip()
    # binary encodings (Yes/No -> 1/0)
    d["Has Table Booking"] = (d["Has Table booking"].astype(str) == "Yes").astype(int)
    d["Has Online Delivery"] = (d["Has Online delivery"].astype(str) == "Yes").astype(int)
    d["Is Delivering Now"] = (d["Is delivering now"].astype(str) == "Yes").astype(int)
    # extra useful ones
    d["Cost per Person"] = d["Average Cost for two"] / 2
    d["Is Rated"] = (d["Aggregate rating"] > 0).astype(int)
    d["Votes per Rating Point"] = d["Votes"] / d["Aggregate rating"].replace(0, float("nan"))
    return d


def task3(df):
    header("LEVEL 2 - TASK 3: FEATURE ENGINEERING")
    fe = engineer_features(df)
    new_cols = ["Name Length", "Name Word Count", "Address Length", "Address Word Count",
                "Locality Length", "Cuisine Count", "Primary Cuisine", "Has Table Booking",
                "Has Online Delivery", "Is Delivering Now", "Cost per Person", "Is Rated"]
    print("New features created:", new_cols)
    print("\nPreview:\n", fe[["Restaurant Name", "Name Length", "Address Length", "Cuisine Count",
                              "Has Table Booking", "Has Online Delivery", "Aggregate rating"]].head(10))
    print("\nSummary of new numeric features:\n",
          fe[[c for c in new_cols if c != "Primary Cuisine"]].describe().round(2).T)

    corr = fe[["Name Length", "Address Length", "Cuisine Count", "Has Table Booking",
               "Has Online Delivery", "Cost per Person", "Votes", "Aggregate rating"]].corr()["Aggregate rating"]
    print("\nCorrelation of new features with Aggregate rating:\n", corr.drop("Aggregate rating").round(3))

    out = OUT_DIR / SUB
    out.mkdir(parents=True, exist_ok=True)
    fe.to_csv(out / "restaurants_with_new_features.csv", index=False)
    print(f"\nSaved engineered dataset -> {out / 'restaurants_with_new_features.csv'}")

    plt.figure(figsize=(9, 7))
    sns.heatmap(fe[["Name Length", "Address Length", "Cuisine Count", "Has Table Booking",
                    "Has Online Delivery", "Cost per Person", "Votes", "Price range",
                    "Aggregate rating"]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation: engineered features vs rating")
    save_fig("task3_feature_correlation", SUB)
    return fe


if __name__ == "__main__":
    data = clean_data()
    task1(data)
    task2(data)
    task3(data)
    print(f"\nDone. Outputs saved in: {OUT_DIR / SUB}")
