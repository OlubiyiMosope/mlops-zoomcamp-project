import os

from batch_score import predict_n_monitor

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
    "run_id": "08b42e845ce74b0cbc5e6659a9952b97",
    "experiment_id": "5",
}


name = "prefect-s3-block"
s_block = storage_block(name)


deployment = Deployment.build_from_flow(
    flow=predict_n_monitor,
    name="abalone-model-training-monitoring",
    schedule=CronSchedule(cron="0 3 2 * *"),
    parameters=parameters,
    storage=s_block,
    output="abalone-deployment.yaml",
    work_queue_name="abalone-model-training-monitoring-workqueue",
    tags=["ml", "abalone", "monitoring"],
    path="abalone-workflows",
    description="Deployment to schedule the batch deployment and batch monitoring of abalone_age_prediction model."
)


if __name__ == "__main__":
    deployment.apply()
