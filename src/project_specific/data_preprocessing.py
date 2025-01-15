import numpy as np
import pandas as pd


def type_conversion(df: pd.DataFrame, columns_to_convert: list):
    """
    Convert the type of specified columns from int to str.

    Parameters:
    - df: pd.DataFrame
        The DataFrame to process.
    - columns_to_convert: list
        A list of column names to convert from int to str.

    Returns:
    - df: pd.DataFrame
        The updated DataFrame.
    """
    for column in columns_to_convert:
        if column in df.columns:
            df[column] = df[column].astype(str)
        else:
            print(f"Warning: Column '{column}' not found in DataFrame.")

    return df


def get_data_types(df: pd.DataFrame, df_name: str = None, target_variable="Survived"):
    """
    Identify numerical and categorical features.
    Returns 2 lists: numerical_features and categorical_features,
    without the target in the categorical features list.

    Parameters:
    - df: pandas DataFrame
        The DataFrame to process.
    - df_name: str, optional
        The “name” of the DataFrame (e.g., 'train', 'test'). Defaults to None.
    - target_variable: str, default 'Survived'
        The name of the target variable (if present).

    Returns:
    - numerical_features: list
        List of numerical feature names.
    - categorical_features: list
        List of categorical feature names.
    """
    if target_variable in df.columns:
        target_in_df = True
    else:
        target_in_df = False

    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Remove the target variable from numerical features if present
    if target_in_df and target_variable in numerical_features:
        numerical_features.remove(target_variable)

    # Print info about the target variable
    if target_in_df:
        print(f"Target Variable: {target_variable}")
    else:
        print(f"Target Variable '{target_variable}' not found in DataFrame {df_name}.")

    return numerical_features, categorical_features


def fare_missing_values_imputation(df_test: pd.DataFrame, df_train: pd.DataFrame):
    """
    Impute missing 'Fare' values in the test dataset based on median fares calculated from the test and train datasets combined.

    This function handles missing 'Fare' values by computing the median fare grouped by 'Pclass' and 'Embarked'
    from the combined dataset. It imputes missing 'Fare' values in the test dataset using these medians.
    If 'Embarked' is missing for a passenger, the function uses the median fare for the corresponding 'Pclass'.
    After imputation, it verifies that there are no remaining missing 'Fare' values in the test dataset.

    Parameters:
    - df_test (pd.DataFrame): The test DataFrame containing passenger data with potential missing 'Fare' values.
    - df_train (pd.DataFrame): The training DataFrame used to compute median fares.

    Returns:
    - pd.DataFrame: The test DataFrame with missing 'Fare' values imputed.
    """
    combined = pd.concat(
        [df_train.drop(columns=["Survived"], errors="ignore"), df_test], sort=False
    )

    # Compute median fare by 'Pclass' and 'Embarked'
    fare_median = combined.groupby(["Pclass", "Embarked"])["Fare"].median()
    fare_median_pclass = combined.groupby("Pclass")["Fare"].median()

    # Identify passenger(s) with missing 'Fare' in the test dataset
    missing_fare_indices = df_test[df_test["Fare"].isnull()].index

    # Impute missing 'Fare' values
    for idx in missing_fare_indices:
        pclass = df_test.loc[idx, "Pclass"]
        embarked = df_test.loc[idx, "Embarked"]

        if pd.isnull(embarked):
            # 'Embarked' is missing; use median fare for 'Pclass'
            imputed_fare = fare_median_pclass.loc[pclass]
        else:
            # 'Embarked' is not missing; use median fare for 'Pclass' and 'Embarked'
            imputed_fare = fare_median.loc[pclass, embarked]
            print(
                f"'Embarked' is {embarked} for passenger at index {idx}. Imputing 'Fare' with median fare {imputed_fare} for 'Pclass' {pclass} and 'Embarked' {embarked}."
            )

        # Impute the missing 'Fare'
        df_test.loc[idx, "Fare"] = imputed_fare

    # Verify that all missing 'Fare' values have been imputed
    print(
        "Remaining missing 'Fare' values in test dataset:",
        df_test["Fare"].isnull().sum(),
    )

    return df_test


def embarked_missing_value_imputation(df_train: pd.DataFrame):
    """
    Impute missing 'Embarked' value in the train dataset based on similar passengers according to Fare and Pclass on the combined train and test datasets.

    If 'Embarked' is missing for a passenger in the train dataset, the function repalces it with the value "C".
    After imputation, it verifies that there are no remaining missing 'Embarked' values in the test dataset.

    Parameters:
    - df_train (pd.DataFrame): The training DataFrame with a missing value for Embarked.

    Returns:
    - pd.DataFrame: The train DataFrame with the missing 'Emarked' value reaplced with "C".
    """
    df_train.loc[:, "Embarked"] = df_train["Embarked"].fillna("C")

    # Verify that all missing 'Embarked' values have been imputed
    print(
        "Remaining missing 'Embarked' values in train dataset:",
        df_train["Embarked"].isnull().sum(),
    )

    return df_train


def delete_features(var_list, var_to_delete):
    """
    Removes the features in 'var_to_delete' list from a list of fariables if they exist.

    Parameters:
    - var_list (list): A list of fariables.
    - var_to_delete (list): A list of features to remove from var_list.

    Returns:
    - list: var_list without the features in var_to_delete.
    """
    filtered_var_list = [
        feature for feature in var_list if feature not in var_to_delete
    ]
    # print("Features:", filtered_var_list)

    return filtered_var_list


def add_features(var_list, features_to_add):
    """
    Adds multiple features to a list of variables if they do not already exist.

    Parameters:
    - var_list (list): A list of variables.
    - features_to_add (list): A list of features to add to var_list.

    Returns:
    - list: var_list with the features_to_add included if they were not already present.
    """
    for feature in features_to_add:
        if feature not in var_list:
            var_list.append(feature)

    # print("Features:", var_list)

    return var_list


def extract_title_lastname(df):
    """Extracts 'Title' and 'Last_Name' from 'Name' column."""
    df["Last_Name"] = df["Name"].apply(lambda x: x.split(",")[0])
    df["Title"] = df["Name"].apply(lambda x: x.split(",")[1].split(".")[0].strip())
    return df


def manage_titles(df):
    """Manages 'Title' column by mapping it to a new 'Simplified_Title' column."""
    title_mapping = {
        "Master": "Child",
        "Miss": "Commoner",
        "Mrs": "Commoner",
        "Mr": "Commoner",
        "Dr": "Title",
        "Rev": "Title",
        "Sir": "Title",
        "Major": "Title",
        "Col": "Title",
        "Capt": "Title",
        "Don": "Title",
        "Jonkheer": "Title",
        "Lady": "Title",
        "Countess": "Title",
        "Other": "Commoner",
    }

    df["Simplified_Title"] = df["Title"].map(title_mapping)
    df.loc[:, "Simplified_Title"] = df["Simplified_Title"].fillna("Commoner")
    return df


def extract_deck(df):
    """Extract Deck from 'Cabin' column."""
    df["Deck"] = df["Cabin"].apply(lambda x: str(x)[0] if pd.notnull(x) else "Unknown")
    return df


def extract_family_alone(df):
    """Extracts 'Family_Size' and 'Is_Alone' from 'SibSp' and 'Parch' columns."""
    df["Family_Size"] = df["SibSp"] + df["Parch"] + 1
    df["Is_Alone"] = df["Family_Size"].apply(lambda x: 1 if x == 1 else 0)
    return df


def create_combined_dataset(train_df, test_df):
    """
    Create combined dataste from train and test without target variable
    """
    train_df["dataset_type"] = "train"
    test_df["dataset_type"] = "test"
    combined_df = pd.concat([train_df, test_df], sort=False)

    return combined_df


def replace_missing_age_with_median(combined_df):
    """
    Replace missing 'Age' values in the combined DataFrame with the median age
    of the corresponding group defined by 'Sex', 'Pclass', and 'Simplified_Title'.
    Then, separate the combined DataFrame back into train and test DataFrames.

    Parameters:
    - combined_df (pd.DataFrame): The combined DataFrame containing both train and test data,
                                with a 'dataset_type' column to distinguish between them.

    Returns:
    - tuple: A tuple containing two DataFrames:
           - train_data (pd.DataFrame): The train DataFrame with imputed 'Age' values and the original 'Survived' column.
           - test_data (pd.DataFrame): The test DataFrame with imputed 'Age' values.
    """
    combined_df["Age"] = combined_df["Age"].fillna(
        combined_df.groupby(["Sex", "Pclass", "Simplified_Title"])["Age"].transform(
            "median"
        )
    )

    train_data = combined_df[combined_df["dataset_type"] == "train"].drop(
        columns=["dataset_type"]
    )
    test_data = combined_df[combined_df["dataset_type"] == "test"].drop(
        columns=["dataset_type", "Survived"]
    )

    # Verify that all missing 'Age' values have been imputed
    print(
        "Remaining missing 'Age' values in the combined dataset:",
        combined_df["Age"].isnull().sum(),
    )
    return train_data, test_data


def run_dm_pipeline(test, train):
    """
    Run a data management pipeline with feature engineering steps for the Titanic dataset.

    This function applies a series of transformations to both the train and test datasets,
    including data type conversions, missing value imputation, feature extraction, and
    dropping unnecessary columns. The goal is to prepare the data for downstream modeling.

    Args:
        test (pd.DataFrame): The test dataset, expected to contain columns such as 'Pclass', 'SibSp',
            'Parch', 'Fare', 'Cabin', and other Titanic-related features.
        train (pd.DataFrame): The training dataset, expected to contain the same columns as `test` plus
            the target variable 'Survived'.

    Returns:
        dict:
            A dictionary with the following keys:
            - "test" (pd.DataFrame): The processed test dataset.
            - "train" (pd.DataFrame): The processed train dataset.
            - "categorical_features" (list of str): The final list of categorical feature names.
            - "numerical_features" (list of str): The final list of numerical feature names.
            - "target" (str): The target column name ('Survived').

    Processing Steps Overview:
        1. Type conversion of specified columns (e.g., 'Pclass') in both datasets.
        2. Identification of numerical and categorical features.
        3. Imputation of missing 'Fare' values in the test set using information from the train set.
        4. Handling of missing 'Embarked' values in the train set.
        5. Extraction of 'Title' and 'LastName' from passenger names.
        6. Creation of simplified or grouped titles via 'manage_titles'.
        7. Feature engineering for 'Deck' based on cabin information.
        8. Removal of unused columns (e.g., 'Name', 'LastName', 'Title', 'Cabin', 'Ticket').
        9. Extraction of family-related features (e.g., 'Is_Alone', 'Family_Size').
        10. Final data type conversions and cleanup.
        11. Replacement of missing 'Age' values with median ages per group.

    Note:
        - This pipeline modifies the input DataFrames in place and returns the final versions.
        - It is tailored specifically to the Titanic dataset but can serve as a template for
          other datasets with similar preprocessing needs.
    """
    train = type_conversion(df=train, columns_to_convert=["Pclass"])
    test = type_conversion(df=test, columns_to_convert=["Pclass"])

    numerical_features_train, categorical_features_train = get_data_types(
        train, df_name="train", target_variable="Survived"
    )
    numerical_features_test, categorical_features_test = get_data_types(
        test, df_name="test", target_variable="Survived"
    )

    test = fare_missing_values_imputation(df_test=test, df_train=train)

    combined = create_combined_dataset(train_df=train, test_df=test)
    train = embarked_missing_value_imputation(train)

    train = extract_title_lastname(train)
    test = extract_title_lastname(test)

    train = manage_titles(train)
    test = manage_titles(test)

    categorical_features_train = add_features(
        var_list=categorical_features_train, features_to_add=["Simplified_Title"]
    )
    categorical_features_test = add_features(
        var_list=categorical_features_test, features_to_add=["Simplified_Title"]
    )

    test = extract_deck(test)
    train = extract_deck(train)

    categorical_features_train = add_features(
        var_list=categorical_features_train, features_to_add=["Deck"]
    )
    categorical_features_test = add_features(
        var_list=categorical_features_test, features_to_add=["Deck"]
    )

    to_delete = ["Name", "LastName", "Title", "Cabin", "Ticket"]
    categorical_features_train = delete_features(
        var_list=categorical_features_train, var_to_delete=to_delete
    )
    categorical_features_test = delete_features(
        var_list=categorical_features_test, var_to_delete=to_delete
    )

    test = extract_family_alone(test)
    train = extract_family_alone(train)

    train = type_conversion(df=train, columns_to_convert=["Is_Alone"])
    test = type_conversion(df=test, columns_to_convert=["Is_Alone"])

    numerical_features_train = add_features(
        var_list=numerical_features_train, features_to_add=["Family_Size"]
    )
    numerical_features_test = add_features(
        var_list=numerical_features_test, features_to_add=["Family_Size"]
    )

    categorical_features_train = add_features(
        var_list=categorical_features_train, features_to_add=["Is_Alone"]
    )
    categorical_features_test = add_features(
        var_list=categorical_features_test, features_to_add=["Is_Alone"]
    )

    to_delete = ["SibSp", "Parch"]
    numerical_features_train = delete_features(
        var_list=numerical_features_train, var_to_delete=to_delete
    )
    numerical_features_test = delete_features(
        var_list=numerical_features_test, var_to_delete=to_delete
    )

    combined = create_combined_dataset(train_df=train, test_df=test)
    train, test = replace_missing_age_with_median(combined_df=combined)

    categorical_features = categorical_features_train
    numerical_features = numerical_features_train
    target = "Survived"

    print("\nCategorical_features:", categorical_features)
    print("Numerical features:", numerical_features)
    print("target:", target)

    return {
        "test": test,
        "train": train,
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "target": target,
    }
