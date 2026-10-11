"""Sanity checks on the SQL feature pipeline and model inputs."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from load_data import build_database, load_features  # noqa: E402
from model import FEATURES, prepare  # noqa: E402


@pytest.fixture(scope="module")
def features(tmp_path_factory):
    db = build_database(tmp_path_factory.mktemp("db") / "lahman.db")
    return load_features(db, ROOT / "sql" / "build_features.sql")


def test_one_row_per_player_season(features):
    assert not features.duplicated(["playerID", "salary_year"]).any()


def test_rate_stats_are_valid(features):
    for col in ("AVG", "OBP", "SLG"):
        assert features[col].between(0, 4).all(), col


def test_pitchers_and_small_samples_excluded(features):
    assert (features["AB"] >= 100).all()


def test_salary_percentile_in_range(features):
    assert features["salary_pct_rank"].between(0, 1).all()


def test_prepared_frame_has_no_missing_features(features):
    df = prepare(features)
    assert df[FEATURES].notna().all().all()
