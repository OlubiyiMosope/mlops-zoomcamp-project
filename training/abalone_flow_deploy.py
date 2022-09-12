import os

from abalone_flow import abalone_train_register

from prefect.deployments import Deployment
from prefect.filesystems import S3
from prefect.orion.schemas.schedules import CronSchedule


def storage_block(block_name):
    """
    Load Prefect storage block if it exists, else, create and then load the storage block.
    :param block_name: name of the storage block
    :return: loaded storage block
    """

    try:
        storage = S3.load(block_name)
        return storage
    except ValueError:
        print(f"Block does not exist. Creating storage block {block_name}.")

        aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

        block = S3(bucket_path="mlops-zoomcamp-prefect-storage", aws_access_key_id=aws_access_key_id,
                   aws_secret_access_key=aws_secret_access_key)
        block.save(block_name)
        storage = S3.load(block_name)
        return storage


parameters = {
    "path": "s3://mlops-zoomcamp-datasets/reference-data/abalone_data.csv",
    "experiment_name": "abalone-age-prediction",
    "registered_model": "abalone-age-regressor",
    "tracking_server_host": "3.129.87.23",
}


name = "train-register-block"
s_block = storage_block(name)


deployment = Deployment.build_from_flow(
    flow=abalone_train_register,
    name="abalone-model-training-practice",
    schedule=CronSchedule(cron="0 3 2 * *"),
    parameters=parameters,
    storage=s_block,
    output="abalone-deployment.yaml",
    work_queue_name="training-registry-workqueue-practice",
    tags=["ml", "abalone", "monitoring", "training", "registry"],
    path="abalone-workflows",
    description="Deployment to schedule the batch deployment and batch monitoring of abalone_age_prediction model."
)


if __name__ == "__main__":
    deployment.apply()
