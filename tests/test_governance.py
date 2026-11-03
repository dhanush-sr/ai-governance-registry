import numpy as np
import pandas as pd
import pytest
from fairlearn.metrics import demographic_parity_ratio, true_positive_rate, MetricFrame

from governance.fairness import metrics
from governance.risk import classify


def test_risk_tiers():
    assert classify({"use_case": "social_scoring"})[0] == "unacceptable"
    assert classify({"use_case": "creditworthiness"})[0] == "high"
    assert classify({"use_case": "other", "interacts_with_people": True})[0] == "limited"
    assert classify({"use_case": "other"})[0] == "minimal"
    with pytest.raises(ValueError):
        classify({"use_case": "made_up"})


def test_metrics_match_fairlearn():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"g": rng.integers(0, 3, 5000), "y_true": rng.integers(0, 2, 5000)})
    df["y_pred"] = (rng.random(5000) < 0.3 + 0.1 * df.g).astype(int)
    counts = df.assign(n=1, selected=df.y_pred, positives=df.y_true, true_pos=df.y_true * df.y_pred) \
               .groupby("g")[["n", "selected", "positives", "true_pos"]].sum().reset_index()
    m = metrics(counts, "g")
    assert m["disparate_impact"] == pytest.approx(
        demographic_parity_ratio(df.y_true, df.y_pred, sensitive_features=df.g))
    tpr = MetricFrame(metrics=true_positive_rate, y_true=df.y_true, y_pred=df.y_pred, sensitive_features=df.g)
    assert m["equal_opportunity_diff"] == pytest.approx(tpr.difference())


def test_gate_decisions_and_tamper_detection(tmp_path, monkeypatch):
    import shutil
    from governance import gate

    assert gate.main(["income-fair"]) == 0
    assert gate.main(["income-baseline"]) == 1
    assert gate.main(["taxi-eta"]) == 1

    # Editing the evidence after training must break lineage.
    reg = tmp_path / "registry"
    shutil.copytree("registry", reg)
    preds = reg / "models/income-fair/eval_predictions.parquet"
    df = pd.read_parquet(preds)
    df["y_pred"] = 1
    df.to_parquet(preds)
    monkeypatch.setattr(gate, "REGISTRY", reg)
    assert gate.main(["income-fair"]) == 1
