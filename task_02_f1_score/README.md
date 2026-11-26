# Challenge 02: Correct F1-Score Calculation

## Problem Description

You are given a binary classification model's predictions and actual labels.

Your task is to implement a reliable F1-score calculation function that correctly evaluates the model's classification performance.

The existing implementation contains mathematical errors that produce incorrect evaluation metrics.

This is particularly problematic when working with imbalanced datasets, where accuracy alone can hide poor performance on the minority class.

## Your Task

Implement an F1-score calculation function that:

1. Accepts actual labels (`y_true`) and predicted labels (`y_pred`).
2. Calculates True Positives (TP), False Positives (FP), and False Negatives (FN).
3. Calculates precision correctly.
4. Calculates recall correctly.
5. Computes the F1-score using the harmonic mean of precision and recall.
6. Handles zero-division cases safely.
7. Returns a valid F1-score between 0 and 1.

## Expected Outcome

The implementation must produce mathematically correct F1-scores for binary classification.

It must handle perfect predictions, no positive predictions, no actual positive samples, imbalanced datasets, and incorrect classifications.

## Acceptance Criteria

### AC1 — Input Validation
- Accept two sequences: `y_true` and `y_pred`.
- Both sequences must have equal lengths.
- Both sequences must contain only binary labels (0 and 1).
- Empty sequences must raise a ValueError.
- Invalid inputs must raise a clear ValueError.

### AC2 — Confusion Count Calculation
- Correctly calculate True Positives (TP).
- Correctly calculate False Positives (FP).
- Correctly calculate False Negatives (FN).
- True Negatives must not contribute to the F1 calculation.

### AC3 — Precision
- Calculate precision using TP / (TP + FP).
- If TP + FP equals zero, precision must safely become zero.

### AC4 — Recall
- Calculate recall using TP / (TP + FN).
- If TP + FN equals zero, recall must safely become zero.

### AC5 — F1-Score
- Calculate the harmonic mean using 2 × (precision × recall) / (precision + recall).
- If precision + recall equals zero, return zero.
- The result must be between 0 and 1.

### AC6 — Correctness
- Perfect predictions must produce an F1-score of 1.
- Incorrect predictions must produce the mathematically correct result.
- The implementation must agree with Scikit-learn's `f1_score` for supported binary inputs, using `zero_division=0`.

## Evaluation

The implementation will be evaluated using automated pytest tests covering mathematical correctness, input validation, edge cases, and comparison against Scikit-learn.