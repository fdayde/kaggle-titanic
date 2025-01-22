from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from .ml_preprocessing import create_preprocessor


def create_logistic_regression_pipeline(
    numeric_features,
    categorical_features,
    C=1.0,
    random_state=42
):
    """
    Create a pipeline for Logistic Regression on the Titanic dataset.

    Args:
        numeric_features (list): List of numeric column names.
        categorical_features (list): List of categorical column names.
        C (float, optional): Inverse of regularization strength for LogisticRegression.
        random_state (int, optional): Random seed.

    Returns:
        Pipeline: A scikit-learn Pipeline object.
    """
    preprocessor = create_preprocessor(numeric_features, categorical_features)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            C=C, 
            random_state=random_state, 
            max_iter=500
        ))
    ])

    return pipeline