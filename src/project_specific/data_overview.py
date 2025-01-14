import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
from IPython.display import display
from io import StringIO
from typing import Dict
import logging



class DataOverview:
    """
    A class to perform data overview and exploration on pandas DataFrames.
    """

    def __init__(self):
        """
        Initializes the DataOverview class.
        """
        pass  # No initialization parameters needed for now

    def data_overview(self, df: pd.DataFrame, df_name: str) -> None:
        """
        Display basic information about the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to overview.
            df_name (str): The name of the DataFrame.
        """
        print(f"\n=== Data Overview: {df_name} ===")
        print(f"- Shape of the dataset: {df.shape}")
        print("- First five rows:")
        display(df.head())
        print("\n- Summary Statistics:")
        display(df.describe(include='all'))

    def missing_values_analysis(self, df: pd.DataFrame, df_name: str) -> None:
        """
        Analyze missing values in the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to analyze.
            df_name (str): The name of the DataFrame.
        """
        print(f"\n=== Missing Values Analysis: {df_name} ===")
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        print("- Missing values in each column:")
        print(missing)
        
        # Visualize missing values
        if missing.any():
            print("\n- Visualizing missing values:")
            msno.matrix(df)
            plt.show()
        else:
            print("No missing values found.")

    def duplicate_detection(self, df: pd.DataFrame, df_name: str) -> None:
        """
        Detect duplicate rows in the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to check.
            df_name (str): The name of the DataFrame.
        """
        print(f"\n=== Duplicate Detection: {df_name} ===")
        duplicates = df[df.duplicated()]
        print(f"Number of duplicate rows: {duplicates.shape[0]}")
        if duplicates.shape[0] > 0:
            print("Duplicate rows:")
            display(duplicates)

    def print_data_types(self, df: pd.DataFrame, df_name: str) -> None:
        """
        Print data types and categorize features into numerical and categorical.

        Args:
            df (pd.DataFrame): The DataFrame to analyze.
            df_name (str): The name of the DataFrame.
        """
        print(f"\n=== Data Types: {df_name} ===")
        print(df.dtypes)

        print("\n => Numerical and categorical features based on types")
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        print("\nNumerical Features:", numerical_features)
        print("Categorical Features:", categorical_features)

    def data_exploration(self, df_dict: Dict[str, pd.DataFrame]) -> None:
        """
        Perform data exploration on a dictionary of DataFrames.

        Args:
            df_dict (Dict[str, pd.DataFrame]): Dictionary with DataFrame names as keys and DataFrames as values.
        """
        print("Starting data exploration on multiple DataFrames.")
        for df_name, df in df_dict.items():
            self.data_overview(df, df_name)
            self.missing_values_analysis(df, df_name)
            self.duplicate_detection(df, df_name)
            self.print_data_types(df, df_name)
