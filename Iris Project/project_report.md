# Iris Flower Classification Using Machine Learning
### Project Report

## Abstract
This project builds a supervised machine learning classifier that predicts the species of an Iris flower
(Setosa, Versicolor or Virginica) from four measurements: sepal length, sepal width, petal length and petal width.
A 150-sample dataset was explored, split 80/20 into training and testing sets with a fixed random seed and
stratification, and used to train a Logistic Regression model inside a scikit-learn pipeline with feature scaling.
Decision Tree and K-Nearest Neighbors models were trained for comparison. On the 30-flower held-out test set,
Logistic Regression reached an accuracy of 0.933 (28 of 30 correct) with macro precision, recall and F1-score of 0.933.
Setosa was classified perfectly; the two errors were between Versicolor and Virginica. The project also includes
visualizations and a validated prediction function. Because the dataset and test set are small, the results are
indicative rather than conclusive, and the model is a learning exercise, not a production system.

## 1. Introduction
Classification is a core machine learning task: assigning an item to one of several known categories.
The Iris dataset, introduced by Ronald Fisher in 1936, is a classic beginner benchmark. Its measurements are
numeric, its classes are balanced, and it contains no missing values, which lets a learner focus on the
workflow itself: exploration, splitting, training, evaluation and prediction.

## 2. Problem Statement
Given the sepal length, sepal width, petal length and petal width (in centimetres) of an Iris flower, predict
whether it is Iris setosa, Iris versicolor or Iris virginica.

## 3. Objectives
1. Load, inspect and describe the dataset and check data quality.
2. Explore the data visually and identify which measurements separate the species.
3. Split the data into 80% training and 20% testing sets (`random_state=42`, `stratify=y`).
4. Train a Logistic Regression classifier and compare it with Decision Tree and K-Nearest Neighbors.
5. Evaluate with accuracy, confusion matrix, precision, recall and F1-score, computed by code.
6. Visualize the results and build a prediction function with input validation.

## 4. Methodology
**Data.** `Iris.csv`: 150 rows, 50 per species, four numeric features, no missing values, 3 duplicate rows (kept).
The `Id` column was dropped because it is only a row counter, and the `Iris-` prefix was removed from labels.

**Exploration.** Summary statistics, histograms, box plots, a pairplot and a correlation heatmap.

**Preprocessing.** Features `X` and target `y` were separated. `train_test_split(test_size=0.20, random_state=42, stratify=y)`
produced 120 training and 30 test samples (40/40/40 and 10/10/10 per species). Test data is kept unseen so the evaluation
reflects performance on new flowers. Standardization (`StandardScaler`) is included in each pipeline so that scaling
parameters are learned from training data only, which avoids data leakage.

**Models.** Logistic Regression (main model, `max_iter=200`), Decision Tree (`max_depth=3`) and K-Nearest Neighbors (`k=5`).
5-fold stratified cross-validation on the training set was used as an additional stability check.

**Evaluation.** Accuracy, confusion matrix, per-class and macro-averaged precision, recall and F1-score on the test set.

## 5. Implementation
The code is in `iris_classification.py` (and the equivalent `iris_classification.ipynb`), organised as functions:
`load_data`, `inspect_data`, `explore_data`, `prepare_data`, `build_models`, `train_and_compare`, `evaluate_model`,
plotting functions, and `predict_species`. Libraries: Python, pandas, NumPy, Matplotlib, Seaborn and scikit-learn.
The prediction function checks that each input is a finite number between 0 and 15 cm before predicting, and
returns the species name together with the model's predicted probability for it.

## 6. Results
Exploration findings (from the data):
* Mean petal length is 1.46 cm (setosa), 4.26 cm (versicolor) and 5.55 cm (virginica); setosa petals (length 1.0-1.9 cm) do not overlap with the other two species (3.0 cm and above).
* Versicolor and Virginica overlap heavily in sepal measurements and slightly in petal measurements.
* Petal length and petal width are strongly correlated (r = 0.96).

Model comparison (scikit-learn 1.8.0, `random_state=42`):

| Model | 5-fold CV accuracy (train) | Test accuracy |
|---|---|---|
| Logistic Regression | 0.9583 | 0.9333 |
| Decision Tree (max_depth=3) | 0.9500 | 0.9667 |
| K-Nearest Neighbors (k=5) | 0.9583 | 0.9333 |

Logistic Regression, test set (30 flowers):

| Species | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Setosa | 1.000 | 1.000 | 1.000 | 10 |
| Versicolor | 0.900 | 0.900 | 0.900 | 10 |
| Virginica | 0.900 | 0.900 | 0.900 | 10 |
| Macro average | 0.933 | 0.933 | 0.933 | 30 |

Confusion matrix (rows actual, columns predicted):

| | Setosa | Versicolor | Virginica |
|---|---|---|---|
| Setosa | 10 | 0 | 0 |
| Versicolor | 0 | 9 | 1 |
| Virginica | 0 | 1 | 9 |

The Decision Tree scored one flower higher on this particular test split, but a single flower equals 3.3 percentage
points and cross-validation scores are essentially the same for all three models, so no model can be called better from this evidence.

## 7. Limitations
* Small dataset (150 samples) and small test set (30), so metrics have high variance and depend on the split.
* A single train/test split was used for the final evaluation; repeated or nested cross-validation would be more reliable.
* No hyperparameter tuning was performed.
* The dataset is clean and well separated; real-world data usually has noise, missing values and imbalance.
* The prediction function validates numeric ranges but cannot tell whether inputs describe a realistic Iris flower.
* The model is not production-ready.

## 8. Conclusion
Using only four flower measurements, a scaled Logistic Regression model identified the correct species for 28 of 30
unseen flowers (accuracy 0.933), classifying Setosa perfectly and confusing only Versicolor with Virginica.
The project demonstrates a complete, reproducible beginner machine learning workflow: data inspection, exploration,
stratified splitting, pipeline-based training, metric-based evaluation, visualization and a validated prediction
interface. Future work could include repeated cross-validation, hyperparameter tuning and testing on a larger dataset.
