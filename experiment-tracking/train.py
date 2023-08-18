import os
import pickle
import click
import mlflow
import optuna

from optuna.samplers import TPESampler

import numpy as np
from catboost import Pool, CatBoostRegressor
from sklearn.metrics import mean_squared_error


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("catboost-stack-overflow-train")


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="data/processed",
    help="Location where the processed data was saved",
)
@click.option(
    "--num_trials",
    default=10,
    help="The number of parameter evaluations for the optimizer to explore",
)
def run_train_optimization(data_path: str, num_trials: int):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))
    categorical, categorical_idx, numerical = load_pickle(
        os.path.join(data_path, "features.pkl")
    )

    # Initialize Pool
    train_pool = Pool(X_train, y_train, cat_features=categorical_idx)
    test_pool = Pool(X_test, y_test, cat_features=categorical_idx)

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 500, 2000),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.5, log=True),
            "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 10, 25),
            "verbose": 0,
            "loss_function": "RMSE",
            "random_seed": 14,
        }

        print("Starting mlfow run...")

        with mlflow.start_run():
            print("Logging mlfow params...")
            # Log hyperparameters
            mlflow.log_params(params)
            print("Training model...This may take a while")
            # Model for training
            model = CatBoostRegressor(**params)
            # Train the model
            model.fit(train_pool, plot=False)
            print("Evaluating model...")
            y_test_pred = model.predict(test_pool)
            # Compute RMSE scores for the model predictions
            rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
            print(f"RMSE score for test: {round(rmse_test, 2)} kUSD/year")
            # Baseline scores (assumption - same prediction for all data samples)
            rmse_bs_test = mean_squared_error(
                y_test, [np.mean(y_train)] * len(y_test), squared=False
            )
            print(f"RMSE baseline score test: {round(rmse_bs_test, 2)} kUSD/year")
            print("Logging mlfow metrics...")
            # Log metrics
            mlflow.log_metric("rmse_test", rmse_test)
            mlflow.log_metric("rmse_bs_test", rmse_bs_test)

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

            print("Finished mlfow run...")

        return rmse_test

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=num_trials)


if __name__ == "__main__":
    run_train_optimization()
