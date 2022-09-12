from datetime import datetime
from dateutil.relativedelta import relativedelta

from prefect import flow

import batch_score


@flow
def abalone_prediction_backfill():
    start_date = datetime(year=2022, month=2, day=1)
    end_date = datetime(year=2022, month=9, day=1)

    d = start_date

    while d <= end_date:
        batch_score.predict_n_monitor(
            run_id="08b42e845ce74b0cbc5e6659a9952b97",
            experiment_id="5",
            run_date=d
        )

        d = d + relativedelta(months=1)


if __name__ == '__main__':
    abalone_prediction_backfill()
