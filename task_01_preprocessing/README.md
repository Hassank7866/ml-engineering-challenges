# Challenge 01: Prevent Data Leakage in a Machine Learning Preprocessing Pipeline

## Problem Description

You are given a binary classification dataset containing patient information and a target column named `diagnosis`.

The existing implementation preprocesses the entire dataset using `StandardScaler` before splitting it into training and testing sets.

This introduces **data leakage** because the scaler calculates statistical parameters (mean and standard deviation) using both training and testing data.

Consequently, information from the test dataset influences the preprocessing applied during model training. This can produce overly optimistic evaluation results and reduce confidence in the model's ability to generalize to unseen data.

## Your Task

Refactor the existing implementation to prevent data leakage.

You must:

1. Separate input features (`X`) and target labels (`y`).
2. Split the dataset into training and testing sets before fitting any preprocessing transformations.
3. Use Scikit-learn's `Pipeline` to combine `StandardScaler` and `LogisticRegression`.
4. Ensure that preprocessing parameters are learned exclusively from training data.
5. Train the model using the training dataset.
6. Evaluate the trained model on the test dataset.

## Expected Outcome

The final implementation must ensure that the test dataset remains completely unseen during model fitting and preprocessing.

The solution should follow Scikit-learn best practices, be reproducible, and include automated tests verifying that no data leakage occurs.

