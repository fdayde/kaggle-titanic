import pandas as pd


def create_data_dictionnary_df():
    """
    Creates a DataFrame describing the Titanic dataset features, their definitions, keys, and additional notes.
    
    Returns:
    - pandas DataFrame with columns 'Variable', 'Definition', 'Key', and 'Note'
    """

    data = {
        "Variable": ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age",
                     "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"],
        "Definition": [
            "Passenger's unique identifier",
            "Survival status",
            "Ticket class",
            "Passenger's name",
            "Sex",
            "Age in years",
            "Number of siblings/spouses aboard",
            "Number of parents/children aboard",
            "Ticket number",
            "Passenger fare",
            "Cabin number",
            "Port of Embarkation"
        ],
        "Key": [
            "Unique ID",
            "0 = No, 1 = Yes",
            "1 = 1st, 2 = 2nd, 3 = 3rd",
            "",
            "'male' or 'female'",
            "",
            "0, 1, 2, ...",
            "0, 1, 2, ...",
            "",
            "",
            "",
            "C = Cherbourg, Q = Queenstown, S = Southampton"
        ],
        "Note": [
            "",
            "",
            "Proxy for socio-economic status",
            "",
            "",
            "Fractional if less than 1. If estimated, it is in the form xx.5",
            "Sibling = brother, sister, stepbrother, stepsister; Spouse = husband, wife (mistresses and fiancés were ignored)",
            "Parent = mother, father; Child = daughter, son, stepdaughter, stepson",
            "",
            "",
            "May have multiple cabin numbers",
            ""
        ]
    }
    
    df = pd.DataFrame(data)
    
    return df