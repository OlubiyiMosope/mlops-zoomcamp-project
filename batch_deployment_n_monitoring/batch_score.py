import json

import uuid
import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta

import mlflow
import pandas as pd

from prefect import task, flow, get_run_logger
from prefect.context import get_run_context
# from prefect.task_runners import SequentialTaskRunner

# from pymongo import MongoClient

from evidently import ColumnMapping

from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab, RegressionPerformanceTab

from evidently.model_profile import Profile
from evidently.model_profile.sections import (DataDriftProfileSection,
                                              RegressionPerformanceProfileSection)


def generate_uuids(n):
    # pylint: disable=unused-variable
    ride_ids = []
    for i in range(n):
        ride_ids.append(str(uuid.uuid4()))
    return ride_ids


def read_dataframe(filename: str):
    df = pd.read_csv(filename)
    df.loc[:, "Age"] = df.loc[:, "Rings"] + 1.5
    df["id"] = generate_uuids(len(df))
    return df


def prepare_features(df):
    features = ["Sex", "Length", "Diameter", "Height", "Whole_weight",
                "Shucked_weight", "Viscera_weight", "Shell_weight"]
    x = df[features]
    return x


def prepare_results(df, y_pred, run_id):
    df_result = df.copy()

    df_result["predicted_rings"] = y_pred
    df_result["diff_preds"] = df_result["Rings"] - df_result["predicted_rings"]
    df_result["model_version"] = run_id
    return df_result


def get_paths(run_date, run_id):
    # pylint: disable=line-too-long
    prev_month = run_date - relativedelta(months=1)
    year = prev_month.year
    month = prev_month.month

    input_file = f"s3://mlops-zoomcamp-datasets/live/reserved_{month}.csv"
    output_file = f"s3://mlops-zoomcamp-datasets/output-files/year={year:04d}/month={month:02d}/{run_id}.csv"

    return input_file, output_file


@task
def load_model(run_id, experiment_id):
    # pylint: disable=line-too-long

    logger = get_run_logger()

    logger.info(f"loading the model with RUN_ID={run_id} ...")
    logged_model = f"s3://mlops-zoomcamp-artifacts-remotes/artifacts/{experiment_id}/{run_id}/artifacts/prod_models"
    model = mlflow.pyfunc.load_model(logged_model)
    return model


def apply_model(input_file, model, output_file, run_id):
    logger = get_run_logger()

    logger.info(f"reading the data from {input_file} ...")
    df = read_dataframe(input_file)
    x = prepare_features(df)

    logger.info("applying the model...")
    y_pred = model.predict(x)

    df_result = prepare_results(df, y_pred, run_id)

    logger.info(f"saving the result to {output_file} ...")
    df_result.to_csv(output_file, index=False)

    return df_result


@task
def load_reference_data(filename, model):
    df_ref = read_dataframe(filename)
    x_ref = prepare_features(df_ref)
    df_ref["predicted_rings"] = model.predict(x_ref)
    return df_ref


@task
def run_evidently(df_ref, data):
    num_attribs = ["Length", "Diameter", "Height", "Whole_weight",
                   "Shucked_weight", "Viscera_weight", "Shell_weight"]
    cat_attribs = ["Sex"]
    profile = Profile(sections=[DataDriftProfileSection(), RegressionPerformanceProfileSection()])
    mapping = ColumnMapping(prediction="predicted_rings", numerical_features=num_attribs,
                            categorical_features=cat_attribs, target="Rings",
                            datetime_features=[]
                            )
    profile.calculate(df_ref, data, mapping)

    dashboard = Dashboard(tabs=[DataDriftTab(), RegressionPerformanceTab(verbose_level=0)])
    dashboard.calculate(df_ref, data, mapping)
    return json.loads(profile.json()), dashboard


@task
def abalone_age_prediction(run_id, model, run_date):
    input_file, output_file = get_paths(run_date, run_id)
    df_result = apply_model(input_file, model, output_file, run_id)
    return df_result


@task
def save_html_report(result, run_date):
    result[1].save(f"evidently_reports/evidently_report-{run_date.strftime('%Y_%m')}.html")


@flow  # (task_runner=SequentialTaskRunnerunner())
def predict_n_monitor(
        run_id: str,
        experiment_id: str,
        run_date: datetime = None):
    if run_date is None:
        ctx = get_run_context()
        run_date = ctx.flow_run.expected_start_time

    model = load_model(run_id, experiment_id)

    df_result = abalone_age_prediction(run_id, model, run_date)

    reference_data = load_reference_data(
        "s3://mlops-zoomcamp-datasets/reference-data/abalone_data.csv",
        model)
    monitoring_results = run_evidently(reference_data, df_result)

    save_html_report(monitoring_results, run_date)


def run():
    parser = argparse.ArgumentParser()

    parser.add_argument("run_id", type=str,  # "cdd99c2996c042ab9e7bf93845357d48"
                        help="Run-id of run that produced the model.")
    parser.add_argument("experiment_id", type=str,
                        help="Experiment-id of experiment that produced the model.")
    parser.add_argument("year", type=int,
                        help="Year of execution date.")
    parser.add_argument("month", type=int,
                        help="Month of execution date.")

    args = parser.parse_args()

    predict_n_monitor(run_id=args.run_id,
                      experiment_id=args.experiment_id,
                      run_date=datetime(year=args.year, month=args.month, day=1)
                      )


if __name__ == "__main__":
    run()
