
import pandas as pd

# train_test_split separates rows into training and testing sets.
from sklearn.model_selection import train_test_split
# StandardScaler standardizes numeric features by fitting mean and standard deviation.
from sklearn.preprocessing import StandardScaler
# LogisticRegression is the binary or multiclass linear classifier used for prediction.
from sklearn.linear_model import LogisticRegression
# accuracy_score compares predicted labels with the actual test labels.
from sklearn.metrics import accuracy_score


def train_model(df):
    """Trains a logistic-regression classifier and returns it with the scaler and accuracy."""

    # DataFrame is a pandas table; drop() removes diagnosis to form the feature columns.
    # df["diagnosis"] selects the target column as a pandas Series of labels.
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    # BUG: Data leakage occurs here.
    # fit_transform() learns the scaler's mean and standard deviation from every row in X,
    # including rows that should remain in the test set. The split happens only afterward,
    # so test-set statistics have already influenced preprocessing.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the already-scaled arrays and target Series into reproducible train/test subsets.
    # test_size=0.20 uses 20% of the rows for testing.
    # random_state=42 makes the random split reproducible.
    # stratify=y preserves the target class proportions in both subsets.
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Configure LogisticRegression and train it with only the training features and labels.
    # max_iter=2000 allows enough iterations for the optimizer to converge.
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    # predict() returns test labels, and accuracy_score() reports the fraction predicted correctly.
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    # Return the fitted model, fitted scaler, and test accuracy for later use.
    return model, scaler, accuracy
