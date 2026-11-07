# Restaurant Lab: Zomato ML Tasks with a Flask Web App

A machine learning project built on the Zomato restaurants dataset. It covers four tasks from the Cognifyz Machine Learning internship: rating prediction, restaurant recommendation, cuisine classification and location-based analysis. Each task is explored in a Jupyter notebook and also available as an interactive page in a Flask web app.

## Tasks

| # | Task | Type | What it does |
|---|------|------|--------------|
| 1 | Predict restaurant ratings | Regression | Estimates a restaurant's aggregate rating (0 to 5) from its location, price, votes and services |
| 2 | Restaurant recommendation | Content-based filtering | Suggests restaurants that match the cuisines, price range and city a user picks |
| 3 | Cuisine classification | Multi-class classification | Predicts the three most likely main cuisines of a restaurant, with probabilities |
| 4 | Location-based analysis | Exploratory analysis | Maps restaurants, compares cities and localities, and finds the nearest restaurants to any point |

## Web app features

- **Predict a rating:** fill in a form and get a predicted rating, a comparison of three models (MSE and R²) and a chart of the most influential features.
- **Find restaurants:** type cuisines such as `North Indian, Chinese`, optionally filter by price range and city, and get ranked matches with a similarity score.
- **Classify a cuisine:** get the top three cuisines with probabilities, plus precision, recall and F1 for each cuisine so the model's weak spots are visible.
- **Explore locations:** an interactive map coloured by rating, a top-cities chart, a city summary with the best-rated localities, and a click-to-find-nearest-restaurants tool.

<!-- Add screenshots here, for example:
![Predict a rating](screenshots/task1.png)
![Explore locations](screenshots/task4.png)
-->

## Tech stack

- **Python 3.10+**
- **pandas, NumPy** for data handling
- **scikit-learn** for models and evaluation
- **Flask** for the web app
- **Chart.js and Leaflet** for charts and maps in the browser
- **Jupyter / Google Colab** for the notebooks

## Dataset

[Zomato Restaurants Data](https://www.kaggle.com/datasets/shrutimehta/zomato-restaurants-data) on Kaggle (`zomato.csv`, about 9,500 restaurants across many cities and countries). Check the dataset's page for its licence before reusing it. The CSV is not included in this repository, so download it yourself.

## Getting started

1. **Clone the repository** and move into the web app folder:
   ```bash
   cd zomato_flask
   ```
2. **Add the dataset.** Download `zomato.csv` from Kaggle and place it in `zomato_flask/data/`.
3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the app:**
   ```bash
   python app.py
   ```
5. Open **http://127.0.0.1:5000** in your browser.

The models are trained once when the app starts, which takes a few seconds to under a minute. After that every page responds instantly.

To run the notebooks, open them in Jupyter or Google Colab. They read the same `zomato.csv`.

## How each task works

**Task 1: Rating prediction**
- Removes columns that would leak the answer (`Rating color`, `Rating text`) and identifier or free-text columns.
- Fills missing cuisines, keeps each restaurant's main cuisine, and one-hot encodes city and cuisine.
- Compares Linear Regression, Decision Tree and Random Forest on an 80/20 split using MSE and R². The Random Forest powers the predictions and the feature-importance chart.

**Task 2: Recommendation**
- Turns each restaurant's cuisines into TF-IDF vectors.
- Ranks restaurants by cosine similarity to the cuisines the user typed, then by rating, after applying the price range and city filters.

**Task 3: Cuisine classification**
- Uses the ten most common main cuisines as classes.
- Features: city, price range, average cost, rating, votes, table booking, online delivery and coordinates.
- Compares Logistic Regression and Random Forest, both with balanced class weights because North Indian dominates the data. Reports accuracy and per-cuisine precision, recall and F1.

**Task 4: Location analysis**
- Cleans zero coordinates, plots restaurants on a map and groups them by city and locality.
- Calculates average rating, cost, price range and the most common cuisine per city.

## Project structure

```
.
├── Task-1.ipynb          # Rating prediction notebook
├── Task-2.ipynb          # Recommendation notebook
├── Task-3.ipynb          # Cuisine classification notebook
├── Task-4.ipynb          # Location analysis notebook
└── zomato_flask/
    ├── app.py            # Flask app and model training
    ├── requirements.txt
    ├── data/             # Put zomato.csv here (not committed)
    ├── static/
    │   └── style.css
    └── templates/
        ├── base.html
        ├── task1.html
        ├── task2.html
        ├── task3.html
        └── task4.html
```

## Limitations

- Task 3 only sees location, price, rating and votes, not the menu, so cuisines that look alike (for example North Indian and Mughlai) are often confused. Cuisines with few restaurants have lower recall.
- Average costs are in each country's local currency, so compare them only within one country.
- Most restaurants in the dataset are in India, mainly Delhi NCR, so results for other countries rest on far less data.
- The models are trained on a single snapshot of the data and are meant for learning and demonstration, not for real business decisions.

## Possible next steps

- Save the trained models with `joblib` and load them at startup for faster deployment.
- Tune hyperparameters with cross-validation.
- Add a hybrid recommender that mixes cuisine similarity with ratings and votes.
- Deploy the app on a free host such as Hugging Face Spaces or Render.

## Author

Built by Rajat as part of the Cognifyz Machine Learning internship.