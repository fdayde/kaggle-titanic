from .data_dictionnary import create_data_dictionnary_df
from .data_overview import DataOverview
from .data_preprocessing import (
    type_conversion,
    get_data_types,
    fare_missing_values_imputation,
    embarked_missing_value_imputation,
    delete_features,
    add_features,
    extract_title_lastname,
    manage_titles,
    extract_deck,
    extract_family_alone,
    create_combined_dataset,
    replace_missing_age_with_median,
    create_log_fare,
    add_age_group,
    run_dm_pipeline,
)
from .data_analysis import (
    plot_categorical_features_vs_target,
    plot_numerical_features_vs_target,
    chi2_categorical_features,
    chi2_all_categorical,
    analyze_correlation,
    cramers_v,
    chi_square_test_train,
    preprocessing_log_reg
)

__all__ = [
    "create_data_dictionnary_df",
    "DataOverview",
    "type_conversion",
    "get_data_types",
    "fare_missing_values_imputation",
    "embarked_missing_value_imputation",
    "delete_features",
    "add_features",
    "extract_title_lastname",
    "manage_titles",
    "extract_deck",
    "extract_family_alone",
    "create_combined_dataset",
    "replace_missing_age_with_median",
    "create_log_fare",
    "add_age_group",
    "run_dm_pipeline",
    "plot_categorical_features_vs_target",
    "plot_numerical_features_vs_target",
    "chi2_categorical_features",
    "chi2_all_categorical",
    "analyze_correlation",
    "cramers_v", 
    "chi_square_test_train",
    "preprocessing_log_reg"
]
