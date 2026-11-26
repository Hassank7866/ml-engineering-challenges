
def calculate_f1(y_true, y_pred):

    # Count True Positives
    tp = sum(
        1 for actual, predicted in zip(y_true, y_pred)
        if actual == 1 and predicted == 1
    )

    # Count False Positives
    fp = sum(
        1 for actual, predicted in zip(y_true, y_pred)
        if actual == 0 and predicted == 1
    )

    # Count False Negatives
    fn = sum(
        1 for actual, predicted in zip(y_true, y_pred)
        if actual == 1 and predicted == 0
    )

    # BUG 1: Incorrect precision calculation
    precision = tp / (tp + fn) if (tp + fn) > 0 else 0

    # BUG 2: Incorrect recall calculation
    recall = tp / (tp + fp) if (tp + fp) > 0 else 0

    # BUG 3: Arithmetic mean instead of harmonic mean
    f1 = (precision + recall) / 2

    return f1
