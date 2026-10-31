"""
Iris Flower Classification Using Machine Learning
=================================================
Predicts the species of an Iris flower (Setosa, Versicolor, Virginica)
from four measurements: sepal length/width and petal length/width.

Run:   python iris_classification.py
Extra: python iris_classification.py --interactive   (type your own measurements)
"""

# ----------------------------------------------------------------------
# 0. IMPORTS AND SETTINGS
# ----------------------------------------------------------------------
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score,
)

RANDOM_STATE = 42            # makes results identical on every run
# The script looks for the CSV in these places (first match wins):
DATA_CANDIDATES = [Path("data/Iris.csv"), Path("Iris.csv")]
IMAGE_DIR = Path("images")
IMAGE_DIR.mkdir(exist_ok=True)

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
sns.set_theme(style="whitegrid")


def section(title):
    """Print a clear heading in the console."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def save_and_show(filename):
    """Save the current figure to /images and display it."""
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / filename, dpi=150)
    plt.show()
    plt.close()


# ----------------------------------------------------------------------
# STEP 1: LOAD AND INSPECT THE DATASET
# ----------------------------------------------------------------------
def load_data():
    """Load Iris.csv. If it is missing, fall back to scikit-learn's built-in copy."""
    data_path = next((p for p in DATA_CANDIDATES if p.exists()), None)
    if data_path is not None:
        df = pd.read_csv(data_path)
        print(f"Loaded dataset from: {data_path}")
        df = df.rename(columns={
            "SepalLengthCm": "sepal_length", "SepalWidthCm": "sepal_width",
            "PetalLengthCm": "petal_length", "PetalWidthCm": "petal_width",
            "Species": "species",
        })
        df = df.drop(columns=["Id"], errors="ignore")   # Id is just a row number
        df["species"] = df["species"].str.replace("Iris-", "", regex=False)  # 'Iris-setosa' -> 'setosa'
    else:
        print("Iris.csv not found -> using scikit-learn's built-in Iris dataset.")
        iris = load_iris(as_frame=True)
        df = iris.frame.rename(columns=dict(zip(iris.feature_names, FEATURES)))
        df["species"] = iris.target.map(dict(enumerate(iris.target_names)))
        df = df.drop(columns=["target"], errors="ignore")
    return df


def inspect_data(df):
    section("STEP 1: DATASET OVERVIEW")
    print(f"Samples (rows): {df.shape[0]}   Columns: {df.shape[1]}")
    print(f"Features: {FEATURES}")
    print(f"Target classes: {sorted(str(s) for s in df['species'].unique())}")
    print("\nFirst five rows:")
    print(df.head().to_string())
    print("\nData types / non-null counts:")
    df.info()
    print("\nMissing values per column:")
    print(df.isnull().sum().to_string())
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    print("\nSamples per species:")
    print(df["species"].value_counts().to_string())


# ----------------------------------------------------------------------
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ----------------------------------------------------------------------
def explore_data(df):
    section("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("Summary statistics (all flowers):")
    print(df[FEATURES].describe().round(2).to_string())
    print("\nAverage measurements per species:")
    print(df.groupby("species")[FEATURES].mean().round(2).to_string())

    # Plot 1: histograms of each feature, coloured by species
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, feat in zip(axes.ravel(), FEATURES):
        sns.histplot(data=df, x=feat, hue="species", kde=True, ax=ax)
        ax.set_title(f"Distribution of {feat.replace('_', ' ')}")
        ax.set_xlabel(f"{feat.replace('_', ' ')} (cm)")
        ax.set_ylabel("Number of flowers")
    save_and_show("01_feature_histograms.png")

    # Plot 2: box plots to compare species and spot outliers
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, feat in zip(axes.ravel(), FEATURES):
        sns.boxplot(data=df, x="species", y=feat, hue="species", legend=False, ax=ax)
        ax.set_title(f"{feat.replace('_', ' ')} by species")
        ax.set_xlabel("Species")
        ax.set_ylabel(f"{feat.replace('_', ' ')} (cm)")
    save_and_show("02_boxplots.png")

    # Plot 3: pairplot = every feature against every other feature
    pair = sns.pairplot(df, hue="species", corner=True)
    pair.figure.suptitle("Pairplot of Iris Measurements", y=1.02)
    pair.figure.savefig(IMAGE_DIR / "03_pairplot.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close("all")

    # Plot 4: correlation heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(df[FEATURES].corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Between Features")
    save_and_show("04_correlation_heatmap.png")

    print("\nWhat the plots tell us (plain English):")
    print("- Setosa has much smaller petals than the other two species and is easy to separate.")
    print("- Versicolor and Virginica overlap a lot in sepal size and only slightly in petal size.")
    print("- Petal length and petal width separate the species best.")
    print("- Petal length and petal width are strongly correlated (about 0.96): they grow together.")


# ----------------------------------------------------------------------
# STEP 3: PREPROCESSING (X / y SPLIT AND TRAIN / TEST SPLIT)
# ----------------------------------------------------------------------
def prepare_data(df):
    section("STEP 3: DATA PREPROCESSING")
    X = df[FEATURES]        # independent variables (inputs)
    y = df["species"]       # dependent variable (what we predict)

    # Why split? A model that is tested on data it already saw can just
    # memorise it. The test set is "unseen" data that shows how the model
    # behaves on new flowers.
    # stratify=y keeps the same species proportions in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training samples: {len(X_train)} ({len(X_train) / len(df):.0%})")
    print(f"Testing samples : {len(X_test)} ({len(X_test) / len(df):.0%})")
    print("\nSpecies counts in training set:\n", y_train.value_counts().to_string())
    print("\nSpecies counts in testing set:\n", y_test.value_counts().to_string())
    return X_train, X_test, y_train, y_test


# ----------------------------------------------------------------------
# STEP 4: MODEL TRAINING
# ----------------------------------------------------------------------
def build_models():
    """Each model is a Pipeline: scale the numbers first, then classify.
    Putting the scaler inside the pipeline means it learns its mean/std
    from the training data only (prevents data leakage)."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=200, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("model", DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)),
        ]),
        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5)),
        ]),
    }


def train_and_compare(models, X_train, X_test, y_train, y_test):
    section("STEP 4: MODEL TRAINING AND COMPARISON")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)                                   # learn from training data
        test_acc = accuracy_score(y_test, pipe.predict(X_test))      # score on unseen data
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv)   # 5-fold check on training data
        rows.append({"Model": name,
                     "CV accuracy (train, mean)": round(cv_scores.mean(), 4),
                     "CV std": round(cv_scores.std(), 4),
                     "Test accuracy": round(test_acc, 4)})
    results = pd.DataFrame(rows)
    print(results.to_string(index=False))
    print("\nLogistic Regression is the main model required by the project;")
    print("the other two are included only for comparison.")
    return results


# ----------------------------------------------------------------------
# STEP 5: MODEL EVALUATION
# ----------------------------------------------------------------------
def evaluate_model(model, X_test, y_test):
    section("STEP 5: EVALUATION OF LOGISTIC REGRESSION")
    y_pred = model.predict(X_test)
    labels = list(model.classes_)       # class order used everywhere below

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}  ({int(round(acc * len(y_test)))} of {len(y_test)} correct)")

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(pd.DataFrame(cm, index=[f"actual {l}" for l in labels],
                       columns=[f"pred {l}" for l in labels]).to_string())

    print("\nPrecision / Recall / F1 (macro = simple average over the 3 classes):")
    print(f"Precision: {precision_score(y_test, y_pred, average='macro'):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred, average='macro'):.4f}")
    print(f"F1-score : {f1_score(y_test, y_pred, average='macro'):.4f}")

    print("\nFull classification report:")
    print(classification_report(y_test, y_pred, labels=labels, digits=3))

    print("How to read these metrics (simple example):")
    print("- Accuracy : share of all flowers named correctly. 9 right out of 10 -> 0.90.")
    print("- Precision: of flowers PREDICTED as 'virginica', how many really are virginica?")
    print("- Recall   : of flowers that ARE 'virginica', how many did we find?")
    print("- F1-score : one number balancing precision and recall (their harmonic mean).")
    print("- Confusion matrix: the diagonal = correct; anything off the diagonal = a mistake.")
    return y_pred, cm, labels


# ----------------------------------------------------------------------
# STEP 6: VISUALIZATIONS
# ----------------------------------------------------------------------
def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix - Logistic Regression (Test Set)")
    plt.xlabel("Predicted species")
    plt.ylabel("Actual species")
    save_and_show("05_confusion_matrix.png")


def plot_scatter(df):
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x="petal_length", y="petal_width",
                    hue="species", style="species", s=70)
    plt.title("Petal Length vs Petal Width")
    plt.xlabel("Petal length (cm)")
    plt.ylabel("Petal width (cm)")
    plt.legend(title="Species")
    save_and_show("06_scatter_petal.png")

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x="sepal_length", y="sepal_width",
                    hue="species", style="species", s=70)
    plt.title("Sepal Length vs Sepal Width")
    plt.xlabel("Sepal length (cm)")
    plt.ylabel("Sepal width (cm)")
    plt.legend(title="Species")
    save_and_show("07_scatter_sepal.png")


def plot_model_comparison(results):
    plt.figure(figsize=(7, 4.5))
    ax = sns.barplot(data=results, x="Model", y="Test accuracy", hue="Model", legend=False)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f")
    plt.ylim(0, 1.1)
    plt.title("Test Accuracy by Model")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    save_and_show("08_model_comparison.png")


def plot_decision_regions(df, X_train, y_train):
    """Extra visual: where the model draws boundaries between species.
    Uses only the 2 petal features so the result can be drawn in 2D
    (this small model is for illustration only, not the main model)."""
    cols = ["petal_length", "petal_width"]
    model2d = Pipeline([("scaler", StandardScaler()),
                        ("model", LogisticRegression(max_iter=200, random_state=RANDOM_STATE))])
    model2d.fit(X_train[cols], y_train)

    xx, yy = np.meshgrid(
        np.linspace(df[cols[0]].min() - 0.5, df[cols[0]].max() + 0.5, 300),
        np.linspace(df[cols[1]].min() - 0.5, df[cols[1]].max() + 0.5, 300))
    grid = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=cols)
    class_idx = {c: i for i, c in enumerate(model2d.classes_)}
    zz = pd.Series(model2d.predict(grid)).map(class_idx).to_numpy().reshape(xx.shape)

    plt.figure(figsize=(7, 5))
    palette = sns.color_palette()[:3]   # same colours for background regions and points
    plt.contourf(xx, yy, zz, alpha=0.2, cmap=ListedColormap(palette), levels=np.arange(-0.5, 3, 1))
    sns.scatterplot(data=df, x=cols[0], y=cols[1], hue="species", style="species",
                    hue_order=list(model2d.classes_), palette=palette, s=60)
    plt.title("Decision Regions (Logistic Regression on Petal Features)")
    plt.xlabel("Petal length (cm)")
    plt.ylabel("Petal width (cm)")
    plt.legend(title="Species")
    save_and_show("09_decision_regions.png")


# ----------------------------------------------------------------------
# STEP 7: PREDICTION SYSTEM
# ----------------------------------------------------------------------
def predict_species(model, sepal_length, sepal_width, petal_length, petal_width):
    """Validate four measurements (cm) and return the predicted species name."""
    values = {"sepal_length": sepal_length, "sepal_width": sepal_width,
              "petal_length": petal_length, "petal_width": petal_width}
    cleaned = []
    for name, value in values.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a number, got {value!r}.")
        if not np.isfinite(number):
            raise ValueError(f"{name} must be a finite number.")
        if not 0 < number <= 15:     # generous limit; real Iris values are roughly 0.1-8 cm
            raise ValueError(f"{name} must be between 0 and 15 cm, got {number}.")
        cleaned.append(number)

    row = pd.DataFrame([cleaned], columns=FEATURES)   # same column names as training
    species = model.predict(row)[0]
    proba = model.predict_proba(row)[0]
    confidence = proba.max()
    return species, confidence


def show_prediction(model, *measurements):
    try:
        species, conf = predict_species(model, *measurements)
        print(f"Input {measurements} -> Predicted: Iris {species}  (confidence {conf:.1%})")
    except ValueError as err:
        print(f"Input {measurements} -> Invalid input: {err}")


def interactive_loop(model):
    print("\nType 4 measurements in cm, e.g. 5.1 3.5 1.4 0.2   (or 'q' to quit)")
    while True:
        text = input("sepal_length sepal_width petal_length petal_width > ").strip()
        if text.lower() in {"q", "quit", "exit"}:
            break
        parts = text.replace(",", " ").split()
        if len(parts) != 4:
            print("Please enter exactly 4 numbers.")
            continue
        show_prediction(model, *parts)


# ----------------------------------------------------------------------
# MAIN PROGRAM
# ----------------------------------------------------------------------
def main():
    df = load_data()
    inspect_data(df)
    explore_data(df)

    X_train, X_test, y_train, y_test = prepare_data(df)
    models = build_models()
    results = train_and_compare(models, X_train, X_test, y_train, y_test)

    final_model = models["Logistic Regression"]       # already fitted above
    _, cm, labels = evaluate_model(final_model, X_test, y_test)

    section("STEP 6: VISUALIZATIONS")
    plot_confusion_matrix(cm, labels)
    plot_scatter(df)
    plot_model_comparison(results)
    plot_decision_regions(df, X_train, y_train)
    print(f"Figures saved in: {IMAGE_DIR.resolve()}")

    section("STEP 7: PREDICTION SYSTEM DEMO")
    show_prediction(final_model, 5.1, 3.5, 1.4, 0.2)   # looks like setosa
    show_prediction(final_model, 6.0, 2.9, 4.5, 1.5)   # looks like versicolor
    show_prediction(final_model, 6.9, 3.1, 5.8, 2.3)   # looks like virginica
    show_prediction(final_model, -1, 3.0, 4.0, 1.0)    # invalid: negative
    show_prediction(final_model, "abc", 3.0, 4.0, 1.0) # invalid: not a number

    if "--interactive" in sys.argv:
        interactive_loop(final_model)

    print("\nNote: This is a learning project on a small, clean dataset. "
          "It is not production-ready.")


if __name__ == "__main__":
    main()
