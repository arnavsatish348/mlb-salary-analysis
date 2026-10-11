"""End-to-end pipeline: load data -> SQL features -> models -> charts + results report."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from charts import feature_importance, predicted_vs_actual, salary_by_experience  # noqa: E402
from load_data import build_database, load_features  # noqa: E402
from model import TEST_START_YEAR, prepare, run_classification, run_regression  # noqa: E402

ROOT = Path(__file__).parent
FIG = ROOT / "reports" / "figures"


def main():
    db = build_database(ROOT / "data" / "lahman.db")
    df = prepare(load_features(db, ROOT / "sql" / "build_features.sql"))
    reg = run_regression(df)
    clf = run_classification(df)

    FIG.mkdir(parents=True, exist_ok=True)
    feature_importance(reg["importance"], FIG / "feature_importance.png")
    best_r2 = reg["metrics"]["Gradient boosting"]["r2"]
    predicted_vs_actual(reg["split"].y_test, reg["best_pred"], best_r2, FIG / "predicted_vs_actual.png")
    salary_by_experience(df, FIG / "salary_by_experience.png")

    results = {
        "rows": len(df),
        "players": int(df["playerID"].nunique()),
        "seasons": f"{df['salary_year'].min()}-{df['salary_year'].max()}",
        "train_rows": len(reg["split"].X_train),
        "test_rows": len(reg["split"].X_test),
        "test_years": f"{TEST_START_YEAR}-{df['salary_year'].max()}",
        "regression": reg["metrics"],
        "top_quartile_classifier": clf,
        "top_drivers": reg["importance"].head(5).round(3).to_dict(),
    }
    (ROOT / "reports" / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
