
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


def train_model(df):

    # 1. Separate features and target
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    # 2. Split BEFORE preprocessing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # 3. Create preprocessing and training pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000))
    ])

    # 4. Fit pipeline using training data ONLY
    pipeline.fit(X_train, y_train)

    # 5. Predict on unseen test data
    predictions = pipeline.predict(X_test)

    # 6. Evaluate model
    accuracy = accuracy_score(y_test, predictions)

    return pipeline, accuracy
