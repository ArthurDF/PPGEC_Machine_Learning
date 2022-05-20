import pytest
import wandb
import pandas as pd

# This is global so all tests are collected under the same run
run = wandb.init(project="Spotify", job_type="data_checks")

@pytest.fixture(scope="session")
def data():

    local_path = run.use_artifact("Spotify/preprocessed_data.csv:latest").file()
    df = pd.read_csv(local_path)

    return df

def test_data_length(data):
    """
    We test that we have enough data to continue
    """
    assert len(data) > 1000


def test_number_of_columns(data):
    """
    We test that we have enough data to continue
    """
    assert data.shape[1] == 14

def test_column_presence_and_type(data):
    required_columns = {
        "danceability": pd.api.types.is_float_dtype,
        "energy": pd.api.types.is_float_dtype,
        "key": pd.api.types.is_integer_dtype,
        "loudness": pd.api.types.is_float_dtype,
        "mode": pd.api.types.is_integer_dtype,
        "speechiness": pd.api.types.is_float_dtype,
        "acousticness": pd.api.types.is_float_dtype,
        "instrumentalness": pd.api.types.is_float_dtype,
        "liveness": pd.api.types.is_float_dtype,
        "valence": pd.api.types.is_float_dtype,
        "tempo": pd.api.types.is_float_dtype,
        "duration_ms": pd.api.types.is_integer_dtype,
        "time_signature": pd.api.types.is_integer_dtype,
        "genre": pd.api.types.is_object_dtype,
    }


    # Check column presence
    assert set(data.columns.values).issuperset(set(required_columns.keys()))

    for col_name, format_verification_funct in required_columns.items():

        assert format_verification_funct(data[col_name]), f"Column {col_name} failed test {format_verification_funct}"


def test_class_names(data):

    # Check that only the known classes are present
    known_classes = ['Dark Trap', 'Underground Rap', 'Trap Metal', 'Emo', 'Rap', 'RnB',
       'Pop', 'Hiphop', 'techhouse', 'techno', 'trance', 'psytrance',
       'trap', 'dnb', 'hardstyle']

    assert data["genre"].isin(known_classes).all()


def test_column_ranges(data):

    ranges = {
        "danceability": (0.0, 1.0),
        "energy": (0.0, 1.0),
        "key": (0, 11),
        "loudness": (-35., 5.),
        "mode": (0,1),
        "speechiness": (0.0, 1.0),
        "acousticness": (0.0, 1.0),
        "instrumentalness": (0.0, 1.0),
        "liveness": (0.0, 1.0),
        "valence": (0.0, 1.0),
        "tempo": (0.0, 300.0),
        "duration_ms": (0, 3.6e+6),
        "time_signature": (1, 6),
    }

    for col_name, (minimum, maximum) in ranges.items():

        assert data[col_name].dropna().between(minimum, maximum).all(), (
            f"Column {col_name} failed the test. Should be between {minimum} and {maximum}, "
            f"instead min={data[col_name].min()} and max={data[col_name].max()}"
        )