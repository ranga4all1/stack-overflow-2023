import os
import pickle
import click
import mlflow
from joblib import load, dump

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from catboost import Pool, CatBoostRegressor
from sklearn.metrics import mean_squared_error


EXPERIMENT_NAME = "catboost-stack-overflow-train"
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)


@click.command()
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote",
)
def run_register_model(top_n: int):
    client = MlflowClient()

    print("Retrieving top_n model runs and selecting best model...")
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=10,
        order_by=["metrics.test_rmse ASC"],
    )[0]

    run_id = best_run.info.run_id
    model_uri = f"gs://rh-mlflow-cb-stack-overflow/1/{run_id}/artifacts/models_mlflow"

    print(f"Registering model with run_id: {run_id} and model_uri: {model_uri}")
    mlflow.register_model(model_uri, name="catboost-best-model")
    print("Registering best model completed...")


if __name__ == "__main__":
    run_register_model()
