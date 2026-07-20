import pandas as pd

from training.train_cnn import _filter_rows_for_model


def test_role_model_filters_to_default_noninjured_rows() -> None:
    df = pd.DataFrame(
        [
            {"Class": "Default", "Injured": False, "Role": "civilian"},
            {"Class": "Default", "Injured": True, "Role": "medic"},
            {"Class": "Zombie", "Injured": False, "Role": "engineer"},
            {"Class": "Default", "Injured": "False", "Role": "medic"},
            {"Class": "Default", "Injured": 0, "Role": "engineer"},
        ]
    )

    filtered = _filter_rows_for_model(df, model_name="role")

    assert len(filtered) == 3
    assert filtered["Role"].tolist() == ["civilian", "medic", "engineer"]
