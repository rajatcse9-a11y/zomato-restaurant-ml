"""
Cognifyz ML Internship - Zomato restaurants
Flask UI for 4 tasks:
  1. Predict restaurant ratings (regression)
  2. Restaurant recommendation (content-based)
  3. Cuisine classification
  4. Location-based analysis

Dataset: Kaggle "shrutimehta/zomato-restaurants-data" (zomato.csv)
Put the csv inside the data/ folder (same folder as this file).
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from flask import Flask, redirect, render_template, request, jsonify, url_for
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             mean_squared_error, r2_score)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)


# ----------------------------------------------------------------- helpers
def records(frame):
    """DataFrame -> list of plain python dicts (safe for tojson)."""
    return json.loads(frame.to_json(orient="records"))


def load_df():
    files = glob.glob(os.path.join(BASE, "data", "*.csv"))
    if not files:
        raise SystemExit("data/ folder mein zomato.csv rakho.")
    for f in files:
        d = pd.read_csv(f, encoding="latin-1")
        # BOM wale column naam theek karo (ï»¿Restaurant ID -> Restaurant ID)
        d.columns = d.columns.str.replace(r"^[^A-Za-z]+", "", regex=True).str.strip()
        if "Aggregate rating" in d.columns:
            return d
    raise SystemExit("data/ mein Zomato wali csv nahi mili.")


df = load_df()
df["Cuisines"] = df["Cuisines"].fillna("")
df["Main"] = df["Cuisines"].str.split(",").str[0].str.strip()

# lat/long clean data + city centre (used to auto-fill coordinates in forms)
g = df[(df["Latitude"] != 0) & (df["Longitude"] != 0)].copy()
CITY_GEO = g.groupby("City")[["Latitude", "Longitude"]].median().to_dict("index")
CITY_COUNTRY = df.groupby("City")["Country Code"].agg(lambda x: x.mode()[0]).to_dict()


def num(form, name, default=0.0):
    try:
        v = form.get(name, "")
        return float(v) if str(v).strip() != "" else default
    except ValueError:
        return default


def geo_for(city, form):
    lat = num(form, "latitude", None)
    lon = num(form, "longitude", None)
    c = CITY_GEO.get(city, {"Latitude": 0.0, "Longitude": 0.0})
    return (lat if lat is not None else c["Latitude"],
            lon if lon is not None else c["Longitude"])


# ------------------------------------------------------- Task 1: ratings
def train_task1():
    d = df.drop(columns=["Rating color", "Rating text", "Restaurant ID",
                         "Restaurant Name", "Address", "Locality Verbose",
                         "Switch to order menu", "Currency", "Locality", "Main"],
                errors="ignore").copy()
    d["Cuisines"] = d["Cuisines"].replace("", "Unknown").str.split(",").str[0].str.strip()
    for c in ["Has Table booking", "Has Online delivery", "Is delivering now"]:
        d[c] = d[c].map({"Yes": 1, "No": 0})
    d = pd.get_dummies(d, columns=["City", "Cuisines"], drop_first=True)

    X = d.drop(columns="Aggregate rating")
    y = d["Aggregate rating"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=12, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42),
    }
    metrics = []
    for name, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        metrics.append({"model": name,
                        "mse": round(float(mean_squared_error(yte, p)), 3),
                        "r2": round(float(r2_score(yte, p)), 3)})
    rf = models["Random Forest"]
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
    return {"model": rf, "cols": list(X.columns), "metrics": metrics,
            "imp_labels": [str(i) for i in imp.index],
            "imp_values": [round(float(v), 4) for v in imp.values]}


T1 = train_task1()

main_counts = df.loc[df["Main"] != "", "Main"].value_counts()
CUISINES = sorted(main_counts[main_counts >= 5].index.tolist())
CITIES = sorted(df["City"].unique().tolist())


def predict_rating(form):
    city, cuisine = form.get("city"), form.get("cuisine")
    row = pd.DataFrame(0.0, index=[0], columns=T1["cols"])
    lat, lon = geo_for(city, form)
    vals = {
        "Country Code": CITY_COUNTRY.get(city, 1),
        "Longitude": lon, "Latitude": lat,
        "Average Cost for two": num(form, "avg_cost", 500),
        "Price range": num(form, "price_range", 2),
        "Votes": num(form, "votes", 100),
        "Has Table booking": 1 if form.get("table_booking") else 0,
        "Has Online delivery": 1 if form.get("online_delivery") else 0,
        "Is delivering now": 0,
    }
    for k, v in vals.items():
        if k in row.columns:
            row[k] = v
    for col in (f"City_{city}", f"Cuisines_{cuisine}"):
        if col in row.columns:
            row[col] = 1
    pred = float(T1["model"].predict(row)[0])
    return round(min(max(pred, 0), 5), 2)


# ------------------------------------------------- Task 2: recommendation
rec = df[["Restaurant Name", "City", "Cuisines", "Price range",
          "Average Cost for two", "Aggregate rating"]].copy()
rec = rec.drop_duplicates(subset=["Restaurant Name", "City"]).reset_index(drop=True)


def recommend(cuisine, price_range=None, city=None, top_n=8):
    pool = rec
    if city:
        pool = pool[pool["City"] == city]
    if price_range:
        pool = pool[pool["Price range"] == int(price_range)]
    pool = pool[pool["Cuisines"] != ""]
    if pool.empty:
        return []
    tfidf = TfidfVectorizer(
        tokenizer=lambda s: [x.strip().lower() for x in s.split(",")],
        token_pattern=None)
    mat = tfidf.fit_transform(pool["Cuisines"])
    q = tfidf.transform([cuisine])
    pool = pool.assign(score=cosine_similarity(q, mat).ravel())
    pool = pool.sort_values(["score", "Aggregate rating"], ascending=False).head(top_n)
    pool["score"] = (pool["score"] * 100).round(0)
    return records(pool)


# ------------------------------------------------ Task 3: classification
def train_task3():
    c = df[df["Main"] != ""].copy()
    top = c["Main"].value_counts().head(10).index
    c = c[c["Main"].isin(top)]
    feat = c[["City", "Price range", "Average Cost for two", "Aggregate rating",
              "Votes", "Has Table booking", "Has Online delivery",
              "Latitude", "Longitude"]].copy()
    for col in ["Has Table booking", "Has Online delivery"]:
        feat[col] = feat[col].map({"Yes": 1, "No": 0})
    cities = sorted(c["City"].unique().tolist())
    feat = pd.get_dummies(feat, columns=["City"], drop_first=True)
    y = c["Main"]
    Xtr, Xte, ytr, yte = train_test_split(feat, y, test_size=0.2,
                                          random_state=42, stratify=y)
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, n_jobs=-1, class_weight="balanced", random_state=42),
    }
    acc = []
    for name, m in models.items():
        m.fit(Xtr, ytr)
        acc.append({"model": name,
                    "accuracy": round(float(accuracy_score(yte, m.predict(Xte))), 3)})
    rf = models["Random Forest"]
    rep = classification_report(yte, rf.predict(Xte), output_dict=True, zero_division=0)
    report = [{"cuisine": k, "precision": round(v["precision"], 2),
               "recall": round(v["recall"], 2), "f1": round(v["f1-score"], 2),
               "support": int(v["support"])}
              for k, v in rep.items() if k in set(top)]
    report.sort(key=lambda r: -r["support"])
    return {"model": rf, "cols": list(feat.columns), "cities": cities,
            "acc": acc, "report": report}


T3 = train_task3()


def predict_cuisine(form, top_k=3):
    city = form.get("city")
    row = pd.DataFrame(0.0, index=[0], columns=T3["cols"])
    lat, lon = geo_for(city, form)
    vals = {"Price range": num(form, "price_range", 2),
            "Average Cost for two": num(form, "avg_cost", 500),
            "Aggregate rating": num(form, "rating", 3.5),
            "Votes": num(form, "votes", 100),
            "Has Table booking": 1 if form.get("table_booking") else 0,
            "Has Online delivery": 1 if form.get("online_delivery") else 0,
            "Latitude": lat, "Longitude": lon}
    for k, v in vals.items():
        row[k] = v
    if f"City_{city}" in row.columns:
        row[f"City_{city}"] = 1
    m = T3["model"]
    probs = m.predict_proba(row)[0]
    idx = np.argsort(probs)[::-1][:top_k]
    return [{"cuisine": str(m.classes_[i]), "prob": round(float(probs[i]) * 100, 1)} for i in idx]


# ----------------------------------------------- Task 4: location analysis
def build_task4():
    city_cnt = df["City"].value_counts()
    stats = df.groupby("City").agg(
        restaurants=("Restaurant ID", "count"),
        avg_rating=("Aggregate rating", "mean"),
        avg_cost=("Average Cost for two", "mean"),
        avg_price_range=("Price range", "mean"),
    ).round(2)
    top_cuisine = (df[df["Main"] != ""].groupby("City")["Main"]
                   .agg(lambda x: x.value_counts().index[0]))
    stats["top_cuisine"] = top_cuisine
    stats = stats.loc[city_cnt.head(15).index].reset_index()
    pts = g.sample(min(1500, len(g)), random_state=42)[
        ["Restaurant Name", "City", "Latitude", "Longitude", "Aggregate rating"]]
    return {"stats": records(stats),
            "city_labels": [str(c) for c in city_cnt.head(10).index],
            "city_values": [int(v) for v in city_cnt.head(10).values],
            "points": records(pts),
            "total": int(len(df)), "n_cities": int(df["City"].nunique()),
            "n_countries": int(df["Country Code"].nunique())}


T4 = build_task4()


# ----------------------------------------------------------------- routes
@app.route("/")
def home():
    return redirect(url_for("task1"))


@app.route("/task1", methods=["GET", "POST"])
def task1():
    form = request.form if request.method == "POST" else {
        "city": "New Delhi" if "New Delhi" in CITIES else CITIES[0],
        "cuisine": "North Indian" if "North Indian" in CUISINES else CUISINES[0],
        "votes": 200, "price_range": 2, "avg_cost": 600}
    result = predict_rating(form) if request.method == "POST" else None
    return render_template("task1.html", page="task1", form=form, result=result,
                           cities=CITIES, cuisines=CUISINES, t1=T1)


@app.route("/task2", methods=["GET", "POST"])
def task2():
    form = request.form if request.method == "POST" else {
        "cuisine": "North Indian, Chinese", "price_range": "2", "city": "New Delhi"}
    results = None
    if request.method == "POST":
        results = recommend(form.get("cuisine", ""), form.get("price_range"),
                            form.get("city"))
    return render_template("task2.html", page="task2", form=form,
                           results=results, cities=CITIES)


@app.route("/task3", methods=["GET", "POST"])
def task3():
    form = request.form if request.method == "POST" else {
        "city": "New Delhi" if "New Delhi" in T3["cities"] else T3["cities"][0],
        "price_range": 2, "avg_cost": 600, "rating": 3.8, "votes": 250}
    result = predict_cuisine(form) if request.method == "POST" else None
    return render_template("task3.html", page="task3", form=form,
                           result=result, t3=T3)


@app.route("/task4")
def task4():
    return render_template("task4.html", page="task4", t4=T4, cities=CITIES)


@app.route("/api/city")
def api_city():
    name = request.args.get("name", "")
    x = df[df["City"] == name]
    if x.empty:
        return jsonify({"error": "City nahi mili"}), 404
    loc = (x[x["Locality"].notna()].groupby("Locality")
           .agg(restaurants=("Restaurant ID", "count"),
                avg_rating=("Aggregate rating", "mean"))
           .query("restaurants >= 5").sort_values("avg_rating", ascending=False)
           .head(8).round(2).reset_index())
    cuis = x.loc[x["Main"] != "", "Main"].value_counts().head(3)
    return jsonify({
        "city": name, "restaurants": int(len(x)),
        "avg_rating": round(float(x["Aggregate rating"].mean()), 2),
        "avg_cost": round(float(x["Average Cost for two"].mean()), 1),
        "top_cuisines": {str(k): int(v) for k, v in cuis.items()},
        "localities": records(loc)})


@app.route("/api/nearby")
def api_nearby():
    try:
        lat, lon = float(request.args["lat"]), float(request.args["lon"])
    except (KeyError, ValueError):
        return jsonify({"error": "lat/lon chahiye"}), 400
    d = g.assign(dist=np.sqrt((g["Latitude"] - lat) ** 2 + (g["Longitude"] - lon) ** 2))
    near = d.sort_values("dist").head(6)[
        ["Restaurant Name", "City", "Cuisines", "Aggregate rating", "Price range"]]
    return jsonify(records(near))

@app.route("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    app.run(debug=False, port=5000)