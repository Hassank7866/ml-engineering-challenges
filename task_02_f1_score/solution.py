
def calculate_f1(y_true, y_pred):

    # 1. Validate and convert inputs into lists
    try:
        y_true = list(y_true)
        y_pred = list(y_pred)
    except TypeError:
        raise ValueError("Inputs must be iterable sequences.")

    # 2. Check for empty inputs
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Input sequences cannot be empty.")

    # 3. Check equal lengths
    if len(y_true) != len(y_pred):
        raise ValueError("Both sequences must have equal lengths.")

    # 4. Validate binary labels
    if any(value not in (0, 1) for value in y_true + y_pred):
        raise ValueError("Labels must contain only 0 and 1.")

    # 5. Calculate confusion counts
    tp = sum(
        1 for actual, predicted in zip(y_true, y_pred)
        if actual == 1 and predicted == 1
    )

    fp = sum(
        1 for actual, predicted in zip(y_true, y_pred)
        if actual == 0 and predicted == 1
    )

    fn = sum(
        1 for actual, predicted in zip(y_true, y_pred)
        if actual == 1 and predicted == 0
    )

    # 6. Calculate precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # 7. Calculate recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # 8. Calculate harmonic mean (F1-score)
    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)

    return float(f1)
