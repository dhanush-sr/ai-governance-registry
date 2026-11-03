"""Train the three example models and write their lineage records.

    python pipeline/train_models.py

For each model this writes registry/models/<id>/run.json (code commit,
dataset hashes, artifact hash, metrics) and eval_predictions.parquet, the
evidence the fairness gate recomputes from. Model binaries go to data/models/.
"""
import hashlib
import json
import pickle
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fairlearn.reductions import DemographicParity, ExponentiatedGradient
from folktables import ACSIncome
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split

DATA = Path("data")
MODELS = Path("registry/models")
SENSITIVE = ["SEX", "RAC1P"]
BLIND = [f for f in ACSIncome.features if f not in SENSITIVE]  # protected attributes are never inputs
TAXI_FEATURES = ["trip_distance", "PULocationID", "DOLocationID", "hour", "weekday"]


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def git_state():
    outputs = [":(exclude)registry/models/*/run.json", ":(exclude)registry/models/*/eval_predictions.parquet"]
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no", "--", ".", *outputs], text=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    return {"commit": commit, "dirty": bool(dirty.strip()), "script": "pipeline/train_models.py"}


def record(model_id, model, training_data, preds, metrics, code):
    out = MODELS / model_id
    artifact = DATA / "models" / f"{model_id}.pkl"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(pickle.dumps(model))
    preds.to_parquet(out / "eval_predictions.parquet", index=False)
    run = {
        "model_id": model_id,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code": code,
        "training_data": training_data,
        "artifact": {"path": str(artifact), "sha256": sha256(artifact)},
        "evaluation": {"predictions": "eval_predictions.parquet",
                       "sha256": sha256(out / "eval_predictions.parquet"),
                       "rows": len(preds), "metrics": metrics},
    }
    (out / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(model_id, metrics)


def income_models(manifest, code):
    df = pd.read_parquet(DATA / "acs_income_ca_2018.parquet")
    tr, te = train_test_split(df, test_size=0.2, random_state=0, stratify=df.label)
    data = [{"id": "acs-income-ca-2018", "sha256": manifest["acs-income-ca-2018"]["sha256"],
             "rows": len(tr), "split": "80% stratified train split, random_state=0"}]

    def evidence(y_pred):
        preds = te[SENSITIVE].assign(y_true=te.label.values, y_pred=y_pred.astype("int8"))
        return preds, {"accuracy": round(accuracy_score(te.label, y_pred), 4),
                       "positive_rate": round(float(y_pred.mean()), 4)}

    # B: fairness through unawareness only.
    base = HistGradientBoostingClassifier(random_state=0).fit(tr[BLIND], tr.label)
    record("income-baseline", base, data, *evidence(base.predict(te[BLIND])), code)

    # A: same inputs, trained under a demographic-parity constraint on SEX x RAC1P.
    fair = ExponentiatedGradient(HistGradientBoostingClassifier(random_state=0),
                                 DemographicParity(difference_bound=0.01))
    fair.fit(tr[BLIND], tr.label, sensitive_features=tr[SENSITIVE])
    record("income-fair", fair, data, *evidence(fair.predict(te[BLIND], random_state=0)), code)


def taxi_model(manifest, code):
    t = pd.read_parquet(DATA / "nyc_tlc_yellow_2023_01.parquet",
                        columns=["tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance",
                                 "PULocationID", "DOLocationID"])
    t["minutes"] = (t.tpep_dropoff_datetime - t.tpep_pickup_datetime).dt.total_seconds() / 60
    t = t[t.minutes.between(1, 180) & t.trip_distance.between(0.01, 100)]
    t = t.assign(hour=t.tpep_pickup_datetime.dt.hour, weekday=t.tpep_pickup_datetime.dt.weekday)
    tr, te = train_test_split(t.sample(600_000, random_state=0), test_size=1 / 6, random_state=0)
    m = HistGradientBoostingRegressor(random_state=0).fit(tr[TAXI_FEATURES], tr.minutes)
    y_pred = m.predict(te[TAXI_FEATURES])
    preds = pd.DataFrame({"y_true": te.minutes.values, "y_pred": y_pred}).astype("float32")
    data = [{"id": "nyc-tlc-yellow-2023-01", "sha256": manifest["nyc-tlc-yellow-2023-01"]["sha256"],
             "rows": len(tr), "split": "600k-trip random sample, 5/6 train, random_state=0"}]
    record("taxi-eta", m, data, preds, {"mae_minutes": round(mean_absolute_error(te.minutes, y_pred), 3)}, code)


if __name__ == "__main__":
    manifest = json.loads((DATA / "manifest.json").read_text())
    code = git_state()
    if code["dirty"]:
        print("WARNING: uncommitted changes; the lineage gate will block these models")
    income_models(manifest, code)
    taxi_model(manifest, code)
