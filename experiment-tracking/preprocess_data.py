# Import required libraries
print("Importing required libraries...")
import os
import pickle
import click
from zipfile import ZipFile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Data pre-processing functions
def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


# Read data
def load_data(filename: str):
    print("Loading data from the zip file...")

    # Load data from the zip file
    with ZipFile(filename) as zip_file:
        df = pd.read_csv(zip_file.open("survey_results_public.csv"))

    print("Data loaded successfully.")
    return df


def preprocess_data(df: pd.DataFrame):
    print("Preprocessing data...")

    # Drop 'ResponseId' column and remove duplicates
    df = df.drop(["ResponseId"], axis=1).drop_duplicates()

    # Set the target variable
    target = "ConvertedCompYearly"
    # Convert compensations into kUSD/year
    df[target] = df[target] * 1e-3

    print("Filtering outliers based on the target label...")
    # Filter outliers by selecting records with target label more than 1k USD/year
    df = df[df[target] > 1.0]

    # Further exclude 2% of smallest and 2% of highest salaries
    P = np.percentile(df[target], [2, 98])
    df = df[(df[target] > P[0]) & (df[target] < P[1])]

    print("Converting YearsCode, YearsCodePro, and WorkExp to integers...")

    # Convert YearsCode, YearsCodePro, and WorkExp to integers
    def clean_years(x):
        if x == "Less than 1 year":
            return 0
        elif x == "More than 50 years":
            return 51
        else:
            return x

    df["YearsCode"] = df["YearsCode"].apply(clean_years).fillna(-1).astype(int)
    df["YearsCodePro"] = df["YearsCodePro"].apply(clean_years).fillna(-1).astype(int)
    df["WorkExp"] = df["WorkExp"].fillna(-1).astype(int)

    # Fill NaN values with 'none'
    df = df.fillna("none")

    print("Dropping unused columns...")
    # Drop unused columns
    cols2drop = [
        "Q120",
        "MainBranch",
        "CodingActivities",
        "Knowledge_1",
        "Knowledge_2",
        "Knowledge_3",
        "Knowledge_4",
        "Knowledge_5",
        "Knowledge_6",
        "Knowledge_7",
        "Knowledge_8",
        "Frequency_1",
        "Frequency_2",
        "Frequency_3",
        "PurchaseInfluence",
        "TechList",
        "BuyNewTool",
        "Currency",
        "CompTotal",
        "LanguageWantToWorkWith",
        "DatabaseWantToWorkWith",
        "PlatformWantToWorkWith",
        "WebframeWantToWorkWith",
        "MiscTechWantToWorkWith",
        "ToolsTechWantToWorkWith",
        "NEWCollabToolsWantToWorkWith",
        "OpSysPersonal use",
        "OfficeStackAsyncWantToWorkWith",
        "OfficeStackSyncWantToWorkWith",
        "AISearchWantToWorkWith",
        "AIDevWantToWorkWith",
        "NEWSOSites",
        "SOVisitFreq",
        "SOAccount",
        "SOPartFreq",
        "SOComm",
        "SOAI",
        "AISelect",
        "AISent",
        "AIAcc",
        "AIBen",
        "AIToolInterested in Using",
        "AIToolCurrently Using",
        "AIToolNot interested in Using",
        "AINextVery different",
        "AINextNeither different nor similar",
        "AINextSomewhat similar",
        "AINextVery similar",
        "AINextSomewhat different",
        "SurveyLength",
        "SurveyEase",
        "TimeSearching",
        "TimeAnswering",
    ]
    df = df.drop(cols2drop, axis=1)

    print("Data preprocessing completed successfully.")
    return df


# def ref_data(df):
#     print("Creating monitoring reference files...")
#     # For monitoring reference only
#     train_data = df[:35000]
#     val_data = df[35000:]

#     print("monitoring reference files created. Saving now...")
#     return train_data, val_data


def prepare_data(df: pd.DataFrame):
    # Prepare data for modeling
    print("Preparing data for modeling...")
    target = "ConvertedCompYearly"
    y = df[target].values.reshape(
        -1,
    )
    X = df.drop([target], axis=1)
    categorical = df.select_dtypes(include=["object"]).columns
    categorical_idx = [list(X.columns).index(c) for c in categorical]
    numerical = ["YearsCode", "YearsCodePro", "WorkExp"]

    print("Data prepared for modeling...")
    return X, y, categorical, categorical_idx, numerical


def train_test_split_data(X, y):
    # Train-test split
    print("Creating Train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    print("Train-test split completed.")
    return X_train, X_test, y_train, y_test


@click.command()
@click.option("--raw_data_path", help="Location where the raw data was saved")
@click.option("--dest_path", help="Location where the resulting files will be saved")
def run_data_prep(
    raw_data_path: str,
    dest_path: str,
    dataset: str = "stack-overflow",
    year: str = "2023",
):
    # Load raw data
    # zip_file_path = '../data/raw/stack-overflow-developer-survey-2023.zip'
    df = load_data(
        os.path.join(raw_data_path, f"{dataset}-developer-survey-{year}.zip")
    )
    df = preprocess_data(df)

    # Line separator
    print("=" * 50)

    # train_data, val_data = ref_data(df)
    # print("train_data shape:", train_data.shape)
    # print("val_data shape:", val_data.shape)

    # train_data.to_csv('../data/processed/sf_train_reference.csv')
    # val_data.to_csv('../data/processed/sf_val_reference.csv')

    # # Line separator
    # print("=" * 50)

    X, y, categorical, categorical_idx, numerical = prepare_data(df)
    # X, y = prepare_data(df)

    # Line separator
    print("=" * 50)

    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    # Line separator
    print("=" * 50)

    # Create dest_path folder unless it already exists
    os.makedirs(dest_path, exist_ok=True)

    # Save processed datasets
    print("Saving processed datasets...")
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))
    dump_pickle(
        (categorical, categorical_idx, numerical),
        os.path.join(dest_path, "features.pkl"),
    )
    print("Saved! and All tasks completed now...")


if __name__ == "__main__":
    run_data_prep()
