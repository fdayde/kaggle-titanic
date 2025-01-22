from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from .ml_preprocessing import create_preprocessor


def create_random_forest_pipeline(
    numeric_features,
    categorical_features,
    n_estimators=100,
    max_depth=None,
    random_state=42
):
    """
    Create a pipeline for Random Forest on the Titanic dataset.

    Args:
        numeric_features (list): Numeric column names.
        categorical_features (list): Categorical column names.
        n_estimators (int): Number of trees in the forest.
        max_depth (int or None): The maximum depth of the tree.
        random_state (int): Random seed.

    Returns:
        Pipeline: A scikit-learn Pipeline object.
    """
    preprocessor = create_preprocessor(numeric_features, categorical_features)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        ))
    ])

    return pipeline