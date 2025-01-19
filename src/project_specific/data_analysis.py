import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency
from itertools import combinations
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm


def plot_categorical_features_vs_target(
    data, categorical_features, target, max_cols=2, figsize=(14, 12)
):
    """
    Plots countplots for categorical features against the target variable.

    Args:
        data (pd.DataFrame): The dataset containing the features and target.
        categorical_features (list): List of categorical feature names to plot.
        target (str): The name of the target variable.
        max_cols (int): Maximum number of columns in the plot grid.
        figsize (tuple): Figure size for the plot.
    """
    num_features = len(categorical_features)
    ncols = min(max_cols, num_features)
    nrows = math.ceil(num_features / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    axes = axes.ravel()

    for i, feature in enumerate(categorical_features):
        sns.countplot(
            data=data,
            x=feature,
            hue=target,
            order=data[feature].value_counts().index,
            ax=axes[i],
        )
        axes[i].set_title(f"{feature} vs {target}")
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel("Count")
        axes[i].legend(title=target)

    # Remove unused axes if the grid is larger than the number of features
    for j in range(len(categorical_features), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_numerical_features_vs_target(
    data, numerical_features, target, max_cols=2, figsize=(14, 12)
):
    """
    Plots boxplots for numerical features against the target variable.

    Args:
        data (pd.DataFrame): The dataset containing the features and target.
        numerical_features (list): List of numerical feature names to plot.
        target (str): The name of the target variable.
        max_cols (int): Maximum number of columns in the plot grid.
        figsize (tuple): Figure size for the plot.
    """
    num_features = len(numerical_features)
    ncols = min(max_cols, num_features)
    nrows = math.ceil(num_features / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # Flatten axes to easily iterate over them, regardless of grid size
    axes = axes.ravel()

    for i, feature in enumerate(numerical_features):
        if i < len(axes):
            sns.boxplot(x=target, y=feature, data=data, ax=axes[i])
            axes[i].set_title(f"{feature} vs {target}")
            axes[i].set_xlabel(target)
            axes[i].set_ylabel(feature)

    # Remove unused axes if the grid is larger than the number of features
    for j in range(len(numerical_features), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def chi2_categorical_features(df, categorical_features, target_col):
    """
    Perform Chi-Square tests between each categorical feature and the target variable.

    Args:
        df (pd.DataFrame): The dataset containing the features and target.
        categorical_features (list): List of categorical feature column names.
        target_col (str): The name of the target column.

    Returns:
        pd.DataFrame: A summary DataFrame with feature names, chi2 statistics, p-values,
                      and whether the relationship is statistically significant.
    """
    results = []
    for feature in categorical_features:
        contingency_table = pd.crosstab(df[feature], df[target_col])

        chi2, p, dof, _ = chi2_contingency(contingency_table)

        significant = p < 0.05

        results.append(
            {
                "Feature": feature,
                "Chi2_Statistic": chi2,
                "P_Value": p,
                "Significant": significant,
            }
        )

    results_df = pd.DataFrame(results)
    return results_df


def chi2_all_categorical(df, categorical_features):
    """
    Perform pairwise Chi-Square tests between all categorical variables.

    Args:
        df (pd.DataFrame): The dataset containing the categorical variables.
        categorical_features (list): List of column names representing categorical features.

    Returns:
        pd.DataFrame: A DataFrame summarizing the Chi-Square statistics, p-values, and significance
                      for all variable pairs.
    """
    results = []

    for var1, var2 in combinations(categorical_features, 2):
        contingency_table = pd.crosstab(df[var1], df[var2])

        chi2, p, dof, _ = chi2_contingency(contingency_table)

        significant = p < 0.05

        results.append(
            {
                "Variable_1": var1,
                "Variable_2": var2,
                "Chi2_Statistic": chi2,
                "P_Value": p,
                "Significant": significant,
            }
        )

    results_df = pd.DataFrame(results)
    return results_df


def analyze_correlation(df, var1, var2):
    """
    Analyze correlations between levels of two categorical variables using their contingency table.

    Args:
        df (pd.DataFrame): The input DataFrame.
        var1 (str): The first categorical variable.
        var2 (str): The second categorical variable.

    Returns:
        dict: A dictionary containing:
            - 'contingency_table': The observed frequency table.
            - 'expected_frequencies': The expected frequency table.
    """
    contingency_table = pd.crosstab(df[var1], df[var2])

    chi2, p, dof, expected = chi2_contingency(contingency_table)

    return {
        "contingency_table": contingency_table,
        "expected_frequencies": pd.DataFrame(
            expected, index=contingency_table.index, columns=contingency_table.columns
        ),
    }


def cramers_v(x, y):
    """Calculate Cramér's V for two categorical variables."""
    contingency_table = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    return np.sqrt(phi2 / min(r - 1, k - 1))


def chi_square_test_train(train, test, feature):
    train_counts = train[feature].value_counts()
    test_counts = test[feature].value_counts()

    contingency_table = pd.DataFrame(
        [train_counts, test_counts], index=["Train", "Test"]
    ).fillna(0)

    # Perform the Chi-Square test
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    return chi2, p


def extract_most_represented_levels(df, categorical_features):
    """
    Extract the most represented (most frequent) level for each categorical feature.

    Args:
        df (pd.DataFrame): The input DataFrame.
        categorical_features (list): List of categorical feature names.

    Returns:
        dict: A dictionary where keys are feature names and values are the most represented levels.
    """
    most_represented_levels = {}
    for feature in categorical_features:
        most_represented_level = df[feature].value_counts().idxmax()
        most_represented_levels[feature] = most_represented_level
    return most_represented_levels


def one_hot_encode_drop_most_populated(
    df, categorical_features, most_represented_levels
):
    """
    One-hot encode categorical features and drop the most represented level for each.

    Args:
        df (pd.DataFrame): The input DataFrame.
        categorical_features (list): List of categorical feature names.
        most_represented_levels (dict): Dictionary of most frequent levels for each feature.

    Returns:
        pd.DataFrame: One-hot encoded DataFrame.
    """
    for feature in categorical_features:
        most_frequent = most_represented_levels[feature]
        # One-hot encode and drop the most frequent level
        df = pd.get_dummies(df, columns=[feature], drop_first=False).drop(
            f"{feature}_{most_frequent}", axis=1
        )
    return df


def preprocessing_log_reg(data, categorical_features, numerical_features, target):
    """
    Preprocesses the data for logistic regression.

    Args:
        data (pd.DataFrame): The input dataset.
        categorical_features (list): List of categorical feature names.
        numerical_features (list): List of numerical feature names.
        target (str): The target column name.

    Returns:
        tuple: Preprocessed feature matrix (X_train) and target vector (y_train).
    """
    # Filter relevant columns
    data = data[categorical_features + numerical_features + [target]]

    # Extract the most represented levels
    most_represented_levels = extract_most_represented_levels(
        data, categorical_features
    )
    print("Most represented levels:", most_represented_levels)

    # One-hot encode categorical features and drop most populated levels
    encoded_data = one_hot_encode_drop_most_populated(
        data, categorical_features, most_represented_levels
    )

    # Scale numerical features
    scaler = StandardScaler()
    encoded_data[numerical_features] = scaler.fit_transform(
        encoded_data[numerical_features]
    )

    # Features-Target split
    X_train = encoded_data.drop(target, axis=1)
    y_train = encoded_data[target]

    # Convert boolean columns to integers
    X_train = X_train.astype(int)
    y_train = y_train.astype(int)

    # Add a constant for the intercept
    X_train = sm.add_constant(X_train)

    return X_train, y_train
