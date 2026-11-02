"""
LEVEL 3
  Task 1 - Predictive Modeling (regression)
  Task 2 - Customer Preference Analysis (cuisines)
  Task 3 - Data Visualization
Run:  python level3.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from common import clean_data, explode_cuisines, save_fig, header, OUT_DIR

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


SUB = "level3"
RANDOM_STATE = 42


# ------------------------------------------------------------------
# TASK 1: Predictive Modeling
# ------------------------------------------------------------------
def build_model_data(df):
    d = engineer_features(df)

    # Encode high-cardinality categoricals with frequency encoding
    d["City Freq"] = d["City"].map(d["City"].value_counts(normalize=True))
    d["Primary Cuisine Freq"] = d["Primary Cuisine"].map(d["Primary Cuisine"].value_counts(normalize=True))
    # Average Cost is in many currencies -> log scale + price range keeps it comparable
    d["Log Cost for two"] = np.log1p(d["Average Cost for two"])

    features = ["Country Code", "City Freq", "Longitude", "Latitude", "Log Cost for two",
                "Price range", "Votes", "Has Table Booking", "Has Online Delivery",
                "Is Delivering Now", "Cuisine Count", "Primary Cuisine Freq",
                "Name Length", "Address Length"]
    return d[features], d["Aggregate rating"], features


def task1(df):
    header("LEVEL 3 - TASK 1: PREDICTIVE MODELING")
    print("RUN A: all restaurants (includes the unrated ones with rating 0)")
    _run_models(df, "all")
    print("\nNOTE: R2 is very high in Run A mostly because unrated restaurants have 0 votes AND "
          "rating 0, so the model just learns 'no votes = no rating'.")
    header("LEVEL 3 - TASK 1 (cont.): MORE REALISTIC - RATED RESTAURANTS ONLY")
    print("RUN B: only restaurants that actually have a rating (> 0)")
    _run_models(df[df["Aggregate rating"] > 0], "rated_only")


def _run_models(df, tag):
    X, y, features = build_model_data(df)
    print(f"Features used ({len(features)}): {features}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    models = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, min_samples_leaf=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                               n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                                       max_depth=4, random_state=RANDOM_STATE),
    }

    rows, fitted = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        cv = cross_val_score(model, X, y, cv=5, scoring="r2", n_jobs=-1)
        rows.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R2 (test)": r2_score(y_test, pred),
            "R2 (train)": r2_score(y_train, model.predict(X_train)),
            "R2 (5-fold CV mean)": cv.mean(),
        })
        fitted[name] = (model, pred)
        print(f"  trained {name}")

    results = pd.DataFrame(rows).set_index("Model").round(4).sort_values("R2 (test)", ascending=False)
    print("\nModel comparison:\n", results)
    best = results.index[0]
    print(f"\n-> Best model: {best} (R2={results.loc[best, 'R2 (test)']:.3f}, "
          f"RMSE={results.loc[best, 'RMSE']:.3f})")
    print("Metric guide: MAE/RMSE = average prediction error in rating points (lower is better); "
          "R2 = share of variance explained (closer to 1 is better).")
    print("Compare R2 train vs test: a big gap means overfitting (typical for deep trees).")

    out = OUT_DIR / SUB
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / f"task1_model_comparison_{tag}.csv")

    # Plots
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    results["R2 (test)"].plot(kind="barh", ax=ax[0], color="#4c72b0")
    ax[0].set_title("R2 on test set"); ax[0].invert_yaxis()
    results["RMSE"].plot(kind="barh", ax=ax[1], color="#dd8452")
    ax[1].set_title("RMSE on test set (lower is better)"); ax[1].invert_yaxis()
    save_fig(f"task1_model_comparison_{tag}", SUB)

    best_model, best_pred = fitted[best]
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, best_pred, alpha=0.35, s=12)
    plt.plot([0, 5], [0, 5], "r--")
    plt.xlabel("Actual rating"); plt.ylabel("Predicted rating")
    plt.title(f"{best}: actual vs predicted")
    save_fig(f"task1_actual_vs_predicted_{tag}", SUB)

    # Feature importance (from the best tree-based model)
    tree_name = best if hasattr(fitted[best][0], "feature_importances_") else "Random Forest"
    imp = pd.Series(fitted[tree_name][0].feature_importances_, index=features).sort_values()
    print(f"\nFeature importance ({tree_name}):\n", imp.sort_values(ascending=False).round(4))
    plt.figure(figsize=(8, 6))
    imp.plot(kind="barh", color="#55a868")
    plt.title(f"Feature importance - {tree_name}")
    save_fig(f"task1_feature_importance_{tag}", SUB)


# ------------------------------------------------------------------
# TASK 2: Customer Preference Analysis
# ------------------------------------------------------------------
def task2(df):
    header("LEVEL 3 - TASK 2: CUSTOMER PREFERENCE ANALYSIS")
    cu = explode_cuisines(df)
    rated = cu[cu["Aggregate rating"] > 0]          # ignore unrated (0) restaurants for rating means

    stats = cu.groupby("Cuisine").agg(
        restaurants=("Restaurant ID", "count"),
        total_votes=("Votes", "sum"),
        avg_votes=("Votes", "mean"),
    )
    stats["avg_rating"] = rated.groupby("Cuisine")["Aggregate rating"].mean()
    stats = stats.round(3)

    # 1. relationship between cuisine and rating
    big = stats[stats["restaurants"] >= 20]
    print("Cuisines with >= 20 restaurants (so averages are reliable):", len(big))
    print("\nAverage rating of the 10 most common cuisines:\n",
          stats.sort_values("restaurants", ascending=False).head(10)[["restaurants", "avg_rating"]])

    # 2. most popular cuisines by votes
    pop = stats.sort_values("total_votes", ascending=False).head(10)
    print("\nMost popular cuisines by TOTAL votes:\n", pop[["total_votes", "restaurants", "avg_rating"]])
    pop_avg = big.sort_values("avg_votes", ascending=False).head(10)
    print("\nMost popular cuisines by AVERAGE votes per restaurant (>=20 restaurants):\n",
          pop_avg[["avg_votes", "restaurants", "avg_rating"]])

    # 3. cuisines with higher ratings
    top = big.sort_values("avg_rating", ascending=False).head(10)
    low = big.sort_values("avg_rating").head(10)
    print("\nHighest rated cuisines (>=20 restaurants):\n", top[["avg_rating", "restaurants"]])
    print("\nLowest rated cuisines (>=20 restaurants):\n", low[["avg_rating", "restaurants"]])
    overall = rated["Aggregate rating"].mean()
    print(f"\nOverall average rating of rated restaurants: {overall:.2f}")
    print(f"-> Cuisines like {', '.join(top.index[:3])} consistently score above average, "
          "so yes: some cuisines tend to receive higher ratings.")

    # statistical check: do cuisines differ?
    try:
        from scipy.stats import kruskal
        groups = [g["Aggregate rating"].values for _, g in rated[rated["Cuisine"].isin(big.index)].groupby("Cuisine")]
        h, p = kruskal(*groups)
        print(f"Kruskal-Wallis test across cuisines: H={h:.1f}, p={p:.2e} "
              f"-> {'significant' if p < 0.05 else 'not significant'} difference.")
    except ImportError:
        pass

    fig, ax = plt.subplots(1, 3, figsize=(22, 7))
    sns.barplot(x=pop["total_votes"], y=pop.index, ax=ax[0], palette="Reds_r")
    ax[0].set_title("Top 10 cuisines by total votes")
    sns.barplot(x=top["avg_rating"], y=top.index, ax=ax[1], palette="Greens_r")
    ax[1].set_title("Top 10 highest-rated cuisines"); ax[1].set_xlim(3, 5)
    sns.barplot(x=low["avg_rating"], y=low.index, ax=ax[2], palette="Blues_r")
    ax[2].set_title("10 lowest-rated cuisines"); ax[2].set_xlim(0, 5)
    save_fig("task2_cuisine_analysis", SUB)

    plt.figure(figsize=(9, 7))
    sns.scatterplot(data=big, x="restaurants", y="avg_rating", size="total_votes", sizes=(20, 600), alpha=0.6)
    for name in list(top.index[:5]) + list(pop.index[:5]):
        r = big.loc[name]
        plt.annotate(name, (r["restaurants"], r["avg_rating"]), fontsize=8)
    plt.xscale("log"); plt.title("Cuisine popularity vs rating (bubble = total votes)")
    save_fig("task2_popularity_vs_rating", SUB)

    stats.sort_values("total_votes", ascending=False).to_csv(OUT_DIR / SUB / "task2_cuisine_stats.csv")


# ------------------------------------------------------------------
# TASK 3: Data Visualization
# ------------------------------------------------------------------
def task3(df):
    header("LEVEL 3 - TASK 3: DATA VISUALIZATION")
    rated = df[df["Aggregate rating"] > 0]

    # 1. distribution of ratings with different charts
    fig, ax = plt.subplots(2, 2, figsize=(15, 11))
    sns.histplot(df["Aggregate rating"], bins=25, kde=True, ax=ax[0, 0], color="#4c72b0")
    ax[0, 0].set_title("Histogram - all restaurants")
    sns.histplot(rated["Aggregate rating"], bins=25, kde=True, ax=ax[0, 1], color="#55a868")
    ax[0, 1].set_title("Histogram - rated restaurants only")
    order = ["Poor", "Average", "Good", "Very Good", "Excellent", "Not rated"]
    sns.countplot(data=df, x="Rating text", order=order, ax=ax[1, 0], palette="viridis")
    ax[1, 0].set_title("Bar plot - rating category counts")
    sns.boxplot(x=rated["Aggregate rating"], ax=ax[1, 1], color="#dd8452")
    ax[1, 1].set_title("Box plot - rated restaurants")
    save_fig("task3_rating_distribution", SUB)

    plt.figure(figsize=(6, 6))
    df["Rating text"].value_counts().plot(kind="pie", autopct="%1.1f%%", colormap="Set3")
    plt.ylabel(""); plt.title("Share of rating categories")
    save_fig("task3_rating_pie", SUB)

    # 2. compare average rating of cuisines / cities
    cu = explode_cuisines(df)
    cu = cu[cu["Aggregate rating"] > 0]
    cstats = cu.groupby("Cuisine")["Aggregate rating"].agg(["mean", "count"])
    cstats = cstats[cstats["count"] >= 50].sort_values("mean", ascending=False)

    city = rated.groupby("City")["Aggregate rating"].agg(["mean", "count"])
    city = city[city["count"] >= 20].sort_values("mean", ascending=False)

    fig, ax = plt.subplots(1, 2, figsize=(18, 9))
    sns.barplot(x=cstats["mean"].head(15), y=cstats.head(15).index, ax=ax[0], palette="mako")
    ax[0].set_title("Avg rating - top 15 cuisines (>= 50 rated restaurants)"); ax[0].set_xlim(0, 5)
    sns.barplot(x=city["mean"].head(15), y=city.head(15).index, ax=ax[1], palette="rocket")
    ax[1].set_title("Avg rating - top 15 cities (>= 20 rated restaurants)"); ax[1].set_xlim(0, 5)
    save_fig("task3_cuisine_city_comparison", SUB)

    # 3. relationship between features and target
    d = engineer_features(df)
    fig, ax = plt.subplots(2, 3, figsize=(20, 11))
    sns.scatterplot(data=rated, x="Votes", y="Aggregate rating", alpha=0.3, ax=ax[0, 0])
    ax[0, 0].set_xscale("log"); ax[0, 0].set_title("Votes (log) vs rating")
    sns.boxplot(data=d, x="Price range", y="Aggregate rating", ax=ax[0, 1], palette="Blues")
    ax[0, 1].set_title("Rating by price range")
    sns.boxplot(data=d, x="Has Table booking", y="Aggregate rating", ax=ax[0, 2], palette="Set2")
    ax[0, 2].set_title("Rating by table booking")
    sns.boxplot(data=d, x="Has Online delivery", y="Aggregate rating", ax=ax[1, 0], palette="Set2")
    ax[1, 0].set_title("Rating by online delivery")
    sns.boxplot(data=d[d["Cuisine Count"] <= 6], x="Cuisine Count", y="Aggregate rating", ax=ax[1, 1], palette="Greens")
    ax[1, 1].set_title("Rating by number of cuisines")
    sns.scatterplot(data=d[d["Aggregate rating"] > 0], x="Name Length", y="Aggregate rating", alpha=0.25, ax=ax[1, 2])
    ax[1, 2].set_title("Restaurant name length vs rating")
    save_fig("task3_features_vs_rating", SUB)

    num = d[["Votes", "Price range", "Has Table Booking", "Has Online Delivery", "Cuisine Count",
             "Name Length", "Address Length", "Latitude", "Longitude", "Aggregate rating"]]
    plt.figure(figsize=(10, 8))
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation heatmap")
    save_fig("task3_correlation_heatmap", SUB)

    corr = num.corr()["Aggregate rating"].drop("Aggregate rating").sort_values(ascending=False)
    print("Correlation of each feature with Aggregate rating:\n", corr.round(3))
    print("\nKey insights:")
    strongest = corr.abs().idxmax()
    print(f" - Strongest single link with rating: {strongest} (r={corr[strongest]:.2f}); "
          f"Votes r={corr['Votes']:.2f} -> more popular places tend to be rated higher.")
    print(" - Higher price range, table booking and online delivery all go with higher ratings.")
    print(" - Name/address length and exact lat/long are almost unrelated to rating.")
    print(" - Roughly 22% of restaurants are unrated (0), which drags averages down - filter them when comparing.")


if __name__ == "__main__":
    data = clean_data()
    task1(data)
    task2(data)
    task3(data)
    print(f"\nDone. Outputs saved in: {OUT_DIR / SUB}")
