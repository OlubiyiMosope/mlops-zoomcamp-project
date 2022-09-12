import os
import joblib
import argparse
from datetime import datetime

import pandas as pd
import numpy as np

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV  # RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

from prefect import flow, task, get_run_logger
# from prefect.task_runners import SequentialTaskRunner


# # @task
def full_prep_pipeline(num_transformer, cat_transformer):
    num_attribs = ["Length", "Diameter", "Height", "Whole_weight", "Shucked_weight", "Viscera_weight", "Shell_weight"]
    cat_attribs = ["Sex"]

    pipeline = ColumnTransformer([
        ("num", num_transformer(), num_attribs),
        ("cat", cat_transformer(), cat_attribs)
    ])
    return pipeline


# # @task
def load_data(path, pipeline):
    columns = ["Sex", "Length", "Diameter", "Height", "Whole_weight",
               "Shucked_weight", "Viscera_weight", "Shell_weight", "Rings"]
    target = "Rings"

    data = pd.read_csv(path)  # , names=columns
    X = data.drop(columns=target)
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=0.8, random_state=42)

    X_train = pipeline.fit_transform(X_train)
    X_test = pipeline.transform(X_test)

    return X_train, X_test, y_train, y_test, pipeline


@task
def train_model(X_train, X_test, y_train, y_test, pipeline):
    logger = get_run_logger()

    # save the data preprocessor pipeline object to external file.
    date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    pipeline_loc = f"mlflow_artifacts/preprocessor_{date}.pkl"
    joblib.dump(pipeline, pipeline_loc)

    parameters = [{"estimator": LinearRegression(), "param_grid": {"n_jobs": [-1],
                                                                   "fit_intercept": [True, False],
                                                                   "positive": [True, False]}},
                  {"estimator": Lasso(),
                   "param_grid": {"alpha": [0.1, 0.25, 0.5, 0.75, 1], "fit_intercept": [True, False]}},
                  {"estimator": Ridge(),
                   "param_grid": {"alpha": [0.1, 0.25, 0.5, 0.75, 1], "fit_intercept": [True, False]}},
                  {"estimator": ElasticNet(),
                   "param_grid": {"alpha": [0.1, 0.25, 0.5, 0.755, 1], "fit_intercept": [True, False],
                                  "l1_ratio": [0, 0.25, 0.5, 0.75, 1]}},
                  {"estimator": RandomForestRegressor(),
                   "param_grid": {"n_estimators": [3, 10, 30, 50], "max_features": [2, 4, 6, 8, 10]}},
                  ]

    for param in parameters:
        logger.info(f"Starting new run: {param['estimator']}...")
        # print(f"Starting new run: {param['estimator']}...")
        with mlflow.start_run():
            mlflow.set_tag("model", str(param['estimator']))

            # log the data preprocessor pipeline object as artifact.
            mlflow.log_artifact(pipeline_loc, artifact_path="preprocessor")

            search_cv = RandomizedSearchCV(param["estimator"], param["param_grid"],
                                           scoring='neg_mean_squared_error',
                                           return_train_score=True
                                           )
            search_cv.fit(X_train, y_train)

            # save the search cv object to external file.
            date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
            search_cv_loc = f"mlflow_artifacts/search_cv_{date}.pkl"
            joblib.dump(search_cv, search_cv_loc)
            mlflow.log_artifact(search_cv_loc, artifact_path="search_cv")

            # log best model and its hyperparameters to mlflow
            best_model = search_cv.best_estimator_

            logger.info("Logging params, metrics and best model...")
            mlflow.sklearn.log_model(best_model, artifact_path="models")
            mlflow.log_param("best_params", search_cv.best_params_)
            mlflow.log_param("best_index", search_cv.best_index_)
            mlflow.log_metric("validation_rmse", np.sqrt(-search_cv.best_score_))
            # evaluate the model on the test set
            test_rmse = mean_squared_error(y_test, best_model.predict(X_test), squared=False)
            mlflow.log_metric("test_rmse", test_rmse)

    logger.info("TRAINING DONE. EXITING...")
    return best_model, pipeline


@task
def register_models(experiment_name, registered_model, tracking_uri):
    logger = get_run_logger()

    logger.info(F"TRACKING SERVER: {tracking_uri}")
    client = MlflowClient(tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)

    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        filter_string="attributes.status='FINISHED'",
        run_view_type=ViewType.ACTIVE_ONLY,  # search for active runs only
        order_by=["metrics.test_rmse ASC"]
    )

    # The runs are ordered in ascending order of the test_rmse metric;
    #   the first run in `runs` is therefore the best model.
    # The model in this run will be registered to `Production` while the orders will be put in `Staging`.

    for run in runs:
        logger.info("getting model from run: {run.info.run_id}...")
        model_uri = f"runs:/{run.info.run_id}/model"
        model_version = mlflow.register_model(model_uri=model_uri, name=registered_model)

        # set tags for current run model version
        logger.info("setting tags to model...")
        client.set_model_version_tag(registered_model, model_version.version, "model", run.data.tags["model"])
        client.set_model_version_tag(registered_model, model_version.version, "experiment_id",
                                     run.info.experiment_id)

        # transition model version stage to "Staging"
        logger.info("Transitioning model version stage to 'Staging'...")
        client.transition_model_version_stage(
            name=registered_model,
            version=model_version.version,
            stage="Staging",
            archive_existing_versions=False
        )

        # Add description to current model version
        logger.info("Adding description to current model version...")
        client.update_model_version(
            name=registered_model,
            version=model_version.version,
            description=f"The model: {run.data.tags['model']}, \
            test_rmse: ~{np.round(run.data.metrics['test_rmse'], 3)}"
        )

    # Transition best model stage to `Production`
    # The best run is at index 0
    logger.info("Transitioning best model from `Staging` to `Production`...")
    best_run = runs[0]
    best_model_version = client.search_model_versions(f"run_id = '{best_run.info.run_id}'")[0]
    client.transition_model_version_stage(
        name=registered_model,
        version=best_model_version.version,
        stage="Production",
        archive_existing_versions=False
    )

    # Add description to `Production` stage model
    client.update_model_version(
        name=registered_model,
        version=best_model_version.version,
        description=f"The production model: {best_run.data.tags['model']}, \
                test_rmse: ~{np.round(best_run.data.metrics['test_rmse'], 3)}"
    )


@task
def make_best_model_pipeline(best_model, pipeline, experiment_name, tracking_uri):
    logger = get_run_logger()

    model = make_pipeline(
        pipeline,
        best_model
    )

    # creating new experiment
    new_exp_name = f"{experiment_name}_production"

    logger.info(f"Setting Experiment: {new_exp_name}")
    mlflow.set_experiment(f"{new_exp_name}")

    logger.info(f"Setting Tracking URI: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    logger.info("Logging Best Model")
    mlflow.sklearn.log_model(model, artifact_path="prod_models")


description = "Training and registering of abalone age prediction models."


@flow(description=description)  # , task_runner=SequentialTaskRunner()
def abalone_train_register(path, experiment_name, registered_model, tracking_server_host):
    logger = get_run_logger()

    tracking_uri = f"http://{tracking_server_host}:5000"
    # set the AWS profile credentials to use.
    os.environ["AWS_PROFILE"] = "default"
    logger.info(f"setting Tracking URI: {tracking_uri}...")
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"Setting experiment name: {experiment_name}...")
    mlflow.set_experiment(experiment_name)

    pipeline = full_prep_pipeline(num_transformer=MinMaxScaler, cat_transformer=OneHotEncoder)
    X_train, X_test, y_train, y_test, pipeline = load_data(path, pipeline)

    logger.info("training model...")
    best_model, pipeline = train_model(X_train, X_test, y_train, y_test, pipeline)
    logger.info("registering model...")
    register_models(experiment_name, registered_model, tracking_uri)
    make_best_model_pipeline(best_model, pipeline, experiment_name, tracking_uri)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("path", type=str,
                        help="Path to the location of the data.")
    parser.add_argument("experiment_name", type=str,
                        help="Name of Mlflow experiment.")
    parser.add_argument("registered_model", type=str,
                        help="Name of Mlflow Model Registry's registered model.")
    parser.add_argument("tracking_server_host", type=str,
                        help="Tracking server host's public IP address")

    args = parser.parse_args()

    abalone_train_register(args.path,
                           args.experiment_name,
                           args.registered_model,
                           args.tracking_server_host,
                           )


if __name__ == '__main__':
    main()
