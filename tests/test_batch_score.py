from datetime import datetime

import numpy as np
import pandas as pd

from batch_deployment_n_monitoring import batch_score  # , batch_score_deploy


def test_generate_uuids():
    n = 5
    ids = batch_score.generate_uuids(n)
    assert len(np.unique(ids)) == n


def test_prepare_results():
    shell = {
        "Sex": 2,
        "Length": 0.45,
        "Diameter": 0.325,
        "Height": 0.135,
        "Whole_weight": 0.438,
        "Shucked_weight": 0.18,
        "Viscera_weight": 0.113,
        "Shell_weight": 0.11,
        "Rings": 14,
    }
    df = pd.DataFrame(shell, index=[0])
    df_result = batch_score.prepare_results(df, 17, "4eff5834")
    assert len(df_result.columns) == len(df.columns) + 3


def test_get_path():
    # pylint: disable=unidiomatic-typecheck
    run_date = datetime(year=2022, month=9, day=1)
    run_id = "23e4356sdas84gba9red4356flk24rf64adf"

    in_file, out_file = batch_score.get_paths(run_date, run_id)
    assert type(in_file) and type(out_file) == str
