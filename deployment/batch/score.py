#!/usr/bin/env python
# coding: utf-8

# import libraries
import os
import sys
import mlflow
import pandas as pd
import numpy as np
from zipfile import ZipFile

from catboost import Pool, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from prefect import task, flow, get_run_logger
from prefect.context import get_run_context
from prefect.artifacts import create_markdown_artifact
from datetime import date


def get_paths(dataset, year, run_id):
    input_file = f"data/raw/{dataset}-developer-survey-{year:04d}.zip"
    output_file = f"data/score/score-{dataset}-{year:04d}-{run_id}.csv"

    return input_file, output_file


def load_data(filename: str) -> pd.DataFrame:
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


def load_model(run_id):
    logged_model = f"gs://mlflow-cb-stack-overflow/1/{run_id}/artifacts/models_mlflow"
    model = mlflow.pyfunc.load_model(logged_model)
    return model


@task
def apply_model(input_file, run_id, output_file):
    logger = get_run_logger()

    logger.info(f"reading data from {input_file}...")
    df = load_data(input_file)
    df = preprocess_data(df)

    logger.info(f"loading model with RUN_ID={run_id}...")
    model = load_model(run_id)

    logger.info(f"applying model...")
    y_pred = model.predict(df)

    df.to_csv("data/score/df_result_temp.csv", index=False)
    df_result = pd.read_csv("data/score/df_result_temp.csv")

    df_result["predicted_ConvertedCompYearly"] = y_pred
    df_result["model_version"] = run_id
    df_result["diff"] = (
        df_result["ConvertedCompYearly"] - df_result["predicted_ConvertedCompYearly"]
    )

    logger.info(f"writing score results to {output_file}...")
    df_result.to_csv(output_file, index=False)
    print("Model application completed successfully.")


@flow
def comp_prediction(dataset, year, run_id):
    input_file, output_file = get_paths(dataset, year, run_id)
    apply_model(input_file=input_file, run_id=run_id, output_file=output_file)


def run():
    dataset = sys.argv[1]  # "stack-overflow"
    year = int(sys.argv[2])  # 2023
    run_id = sys.argv[3]  # "8ac5e4553c464697a9d70d833458e3d2"

    comp_prediction(dataset=dataset, year=year, run_id=run_id)


if __name__ == "__main__":
    run()
