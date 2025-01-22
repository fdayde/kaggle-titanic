from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from .ml_preprocessing import create_preprocessor


def create_knn_pipeline(
    numeric_features,
    categorical_features,
    n_neighbors=5
):
    """
    Create a pipeline for k-Nearest Neighbors on the Titanic dataset.

    Args:
        numeric_features (list): Numeric column names.
        categorical_features (list): Categorical column names.
        n_neighbors (int): Number of neighbors to use.

    Returns:
        Pipeline: A scikit-learn Pipeline object.
    """
    preprocessor = create_preprocessor(numeric_features, categorical_features)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", KNeighborsClassifier(
            n_neighbors=n_neighbors
        ))
    ])

    return pipeline
