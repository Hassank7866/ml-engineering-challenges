
import numpy as np
import pandas as pd
import pytest

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from task_01_preprocessing.solution import train_model

@pytest.fixture
def sample_data():
    """Create a reproducible binary classification dataset."""

    rng = np.random.RandomState(42)

    X1 = rng.normal(50, 10, 100)
    X2 = rng.normal(100, 20, 100)

    y = (X1 + rng.normal(0, 5, 100) > 50).astype(int)

    return pd.DataFrame({
        "feature_1": X1,
        "feature_2": X2,
        "diagnosis": y
    })


def test_returns_pipeline_and_accuracy(sample_data):

    pipeline, accuracy = train_model(sample_data)

    assert isinstance(pipeline, Pipeline)
    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 1


def test_pipeline_contains_correct_steps(sample_data):

    pipeline, accuracy = train_model(sample_data)

    assert isinstance(
        pipeline.named_steps["scaler"],
        StandardScaler
    )

    assert isinstance(
        pipeline.named_steps["classifier"],
        LogisticRegression
    )


def test_scaler_is_fitted_only_on_training_data(sample_data):

    X = sample_data.drop(columns=["diagnosis"])
    y = sample_data["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    pipeline, accuracy = train_model(sample_data)

    scaler = pipeline.named_steps["scaler"]

    # The scaler's learned mean must match training data only.
    np.testing.assert_allclose(
        scaler.mean_,
        X_train.mean().values,
        rtol=1e-7
    )

    # It must have seen only training rows.
    assert scaler.n_samples_seen_ == len(X_train)


def test_model_can_predict_test_data(sample_data):

    X = sample_data.drop(columns=["diagnosis"])
    y = sample_data["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    pipeline, accuracy = train_model(sample_data)

    predictions = pipeline.predict(X_test)

    assert len(predictions) == len(y_test)
    assert set(predictions).issubset({0, 1})


def test_training_is_reproducible(sample_data):

    pipeline1, accuracy1 = train_model(sample_data)
    pipeline2, accuracy2 = train_model(sample_data)

    assert accuracy1 == pytest.approx(accuracy2)

    np.testing.assert_allclose(
        pipeline1.named_steps["scaler"].mean_,
        pipeline2.named_steps["scaler"].mean_
    )
