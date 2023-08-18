import os
import pathlib
import pickle
import pandas as pd
import numpy as np
import scipy
import mlflow
from zipfile import ZipFile

from catboost import Pool, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from prefect import flow, task
from prefect.artifacts import create_markdown_artifact
from datetime import date


@task(retries=3, retry_delay_seconds=2)
# Data pre-processing functions
def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


@task(retries=3, retry_delay_seconds=2)
# Data pre-processing functions
def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@task(retries=3, retry_delay_seconds=2)
def load_data(filename: str) -> pd.DataFrame:
    print("Loading data from the zip file...")

    # Load data from the zip file
    with ZipFile(filename) as zip_file:
        df = pd.read_csv(zip_file.open("survey_results_public.csv"))

    print("Data loaded successfully.")
    return df


@task(retries=3, retry_delay_seconds=2)
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


@task(retries=3, retry_delay_seconds=2)
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


@task(retries=3, retry_delay_seconds=2)
def train_test_split_data(X, y):
    # Train-test split
    print("Creating Train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )
    print("Train-test split completed.")
    return X_train, X_test, y_train, y_test


@flow
def run_data_prep(
    raw_data_path: str = "data/raw",
    dest_path: str = "data/processed",
    dataset: str = "stack-overflow",
    year: str = "2023",
):
    # Load raw data
    df = load_data(
        os.path.join(raw_data_path, f"{dataset}-developer-survey-{year}.zip")
    )
    df = preprocess_data(df)

    # Line separator
    print("=" * 50)

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
    print("Saved! and All data processing tasks completed now...")


@flow
def train_best_model(data_path: str = "data/processed") -> None:
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))
    categorical, categorical_idx, numerical = load_pickle(
        os.path.join(data_path, "features.pkl")
    )

    """train a model with best hyperparams and write everything out"""

    print("Starting mlfow run...")
    with mlflow.start_run():
        best_params = {
            "iterations": 957,
            "depth": 4,
            "learning_rate": 0.18094505491408697,
            "l2_leaf_reg": 17,
            "verbose": 0,
            "loss_function": "RMSE",
            "random_seed": 14,
        }

        mlflow.log_params(best_params)
        print("Training model...This may take a while")
        train_pool = Pool(X_train, y_train, cat_features=categorical_idx)
        test_pool = Pool(X_test, y_test, cat_features=categorical_idx)
        model = CatBoostRegressor(**best_params)
        # Train the model
        model.fit(train_pool, plot=False)

        print("Evaluating model...")
        y_test_pred = model.predict(test_pool)
        # Compute RMSE scores for the model predictions
        rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
        print(f"RMSE score for test: {round(rmse_test, 2)} kUSD/year")

        print("Logging mlfow metrics...")
        # Log metrics
        mlflow.log_metric("rmse_test", rmse_test)

        pathlib.Path("model").mkdir(exist_ok=True)

        # Save the model in catboost cbm format
        model.save_model(
            fname="model/sf_catboost_model",
            format="cbm",
            export_parameters=None,
            pool=None,
        )
        # Pickle model in binary format
        with open("model/sf_catboost.bin", "wb") as f_out:
            pickle.dump(model, f_out)

        # Log model artifact
        print("Logging mlfow artifacts...")
        mlflow.log_artifact(local_path="model/", artifact_path=None)

        mlflow.catboost.log_model(model, artifact_path="models_mlflow")

        print("Finished mlfow run...")

        markdown__rmse_report = f"""# RMSE Report

        ## Summary

        Stackoverflow Compensation Prediction 

        ## RMSE CatBoost Model

        | Region    | RMSE |
        |:----------|-------:|
        | {date.today()} | {rmse_test:.2f} |
        """

        create_markdown_artifact(
            key="compensation-model-report", markdown=markdown__rmse_report
        )

    return None


@flow
def main_flow() -> None:
    """The main training pipeline"""

    # MLflow settings
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("catboost-stack-overflow-train")

    run_data_prep()

    train_best_model()


if __name__ == "__main__":
    main_flow()
