from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def create_preprocessor(numeric_features, categorical_features):
    """
    Create a ColumnTransformer that processes numeric and 
    categorical features for the Titanic dataset.

    Args:
        numeric_features (list): List of numeric column names.
        categorical_features (list): List of categorical column names.

    Returns:
        ColumnTransformer: A scikit-learn ColumnTransformer pipeline.
    """

    # Pipeline for numeric features
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Pipeline for categorical features using OneHotEncoder:
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])

    # Combine numeric and categorical pipelines
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    return preprocessor