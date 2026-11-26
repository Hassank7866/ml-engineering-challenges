
import pytest

from sklearn.metrics import f1_score as sklearn_f1_score

from task_02_f1_score.solution import calculate_f1


def test_perfect_predictions():

    y_true = [1, 1, 0, 0]
    y_pred = [1, 1, 0, 0]

    result = calculate_f1(y_true, y_pred)

    assert result == 1.0


def test_incorrect_predictions():

    y_true = [1, 1, 1, 1, 0, 0, 0]
    y_pred = [1, 1, 0, 0, 1, 0, 0]

    result = calculate_f1(y_true, y_pred)

    expected = 0.5714285714

    assert result == pytest.approx(expected)


def test_zero_positive_predictions():

    y_true = [1, 1, 0, 0]
    y_pred = [0, 0, 0, 0]

    result = calculate_f1(y_true, y_pred)

    assert result == 0.0


def test_no_actual_positive_samples():

    y_true = [0, 0, 0, 0]
    y_pred = [0, 0, 0, 0]

    result = calculate_f1(y_true, y_pred)

    assert result == 0.0


def test_invalid_inputs():

    with pytest.raises(ValueError):
        calculate_f1([1, 0, 1], [1, 0])

    with pytest.raises(ValueError):
        calculate_f1([1, 2, 0], [1, 0, 0])

    with pytest.raises(ValueError):
        calculate_f1([], [])


def test_matches_sklearn():

    y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]

    our_result = calculate_f1(y_true, y_pred)

    reference_result = sklearn_f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    assert our_result == pytest.approx(reference_result)


def test_f1_score_is_always_between_zero_and_one():

    y_true = [1, 0, 1, 0, 1, 0]
    y_pred = [1, 1, 0, 0, 1, 0]

    result = calculate_f1(y_true, y_pred)

    assert 0.0 <= result <= 1.0
