from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from .ml_preprocessing import create_preprocessor


def create_xgboost_pipeline(
    numeric_features,
    categorical_features,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
):
    """
    Create a pipeline for XGBoost on the Titanic dataset.

    Args:
        numeric_features (list): Numeric column names.
        categorical_features (list): Categorical column names.
        n_estimators (int): Number of gradient boosted trees.
        learning_rate (float): Boosting learning rate.
        max_depth (int): Maximum tree depth for base learners.
        random_state (int): Random seed.

    Returns:
        Pipeline: A scikit-learn Pipeline object.
    """
    preprocessor = create_preprocessor(numeric_features, categorical_features)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=random_state
        ))
    ])

    return pipeline