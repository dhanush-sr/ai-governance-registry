"""Scale benchmark for the fairness audit the gate runs.

    python -m pipeline.benchmark_spark

Replicates California ACSIncome with new person ids (x8, x32, x128, up to
25M rows), then scores every row with the income-baseline model and computes
per-group fairness counts, the same governance.fairness code the CI gate
uses. Timed for Spark at 1-8 cores and for single-process pandas.
Writes results/benchmark.csv and results/gate_timing.json.
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"  # one core per worker, so "cores" means cores (inherited by Spark workers)

import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from pyspark.sql import functions as F

from governance.fairness import group_counts, metrics, spark

DATA = Path("data")
SCALED = DATA / "scaled"
RESULTS = Path("results")
SCALES = [8, 32, 128]
CORES = [1, 2, 4, 8]
REPEATS = 2  # best of 2, so JVM warm-up doesn't count
BLIND = ["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP"]
MODEL = pickle.loads((DATA / "models" / "income-baseline.pkl").read_bytes())


def make_scaled(k):
    out = SCALED / f"x{k}.parquet"
    if not out.exists():
        base = pd.read_parquet(DATA / "acs_income_ca_2018.parquet")
        n = len(base)
        df = pd.concat([base.assign(person_id=r * n + base.index) for r in range(k)], ignore_index=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False, row_group_size=1_000_000)
    return out


def score(batches):
    for b in batches:
        yield pd.DataFrame({"SEX": b.SEX, "RAC1P": b.RAC1P, "y_true": b.label.astype("int32"),
                            "y_pred": MODEL.predict(b[BLIND]).astype("int32")})


def run_spark(session, path):
    df = session.read.parquet(str(path)).mapInPandas(score, "SEX int, RAC1P int, y_true int, y_pred int")
    return group_counts(df, ["SEX", "RAC1P"])


def run_pandas(path):
    b = pd.read_parquet(path)
    preds = next(score([b]))
    keys = ["SEX", "RAC1P"]
    return (preds.assign(n=1, selected=preds.y_pred, positives=preds.y_true, true_pos=preds.y_true * preds.y_pred)
                 .groupby(keys)[["n", "selected", "positives", "true_pos"]].sum().reset_index())


def best_time(fn):
    times, out = [], None
    for _ in range(REPEATS):
        t = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t)
    return min(times), out


def main():
    RESULTS.mkdir(exist_ok=True)
    paths = {k: make_scaled(k) for k in SCALES}
    rows = {k: pd.read_parquet(p, columns=["SEX"]).shape[0] for k, p in paths.items()}
    records = []

    def log(engine, cores, k, secs, counts):
        di = metrics(counts, "RAC1P", 100)["disparate_impact"]
        records.append({"engine": engine, "cores": cores, "scale": k, "rows": rows[k], "seconds": round(secs, 2),
                        "rows_per_sec": round(rows[k] / secs), "rac1p_disparate_impact": round(di, 4),
                        "parquet_mb": round(paths[k].stat().st_size / 2**20, 1)})
        print(records[-1], flush=True)

    for k in SCALES:
        secs, counts = best_time(lambda: run_pandas(paths[k]))
        log("pandas", 1, k, secs, counts)

    for cores in CORES:
        session = spark(cores=cores, app=f"bench-{cores}")
        session.range(1).agg(F.count("*")).collect()  # JVM start-up is not part of the job
        for k in (SCALES if cores == max(CORES) else [max(SCALES)]):
            secs, counts = best_time(lambda: run_spark(session, paths[k]))
            log("spark", cores, k, secs, counts)
        session.stop()

    pd.DataFrame(records).to_csv(RESULTS / "benchmark.csv", index=False)

    # Wall-clock of the real CI gate, Spark start-up included.
    gate = {}
    for m in ["income-fair", "income-baseline", "taxi-eta"]:
        t = time.perf_counter()
        subprocess.run([sys.executable, "-m", "governance.gate", m], capture_output=True)
        gate[m] = round(time.perf_counter() - t, 2)
    (RESULTS / "gate_timing.json").write_text(json.dumps(gate, indent=2) + "\n")
    print("gate seconds:", gate)


if __name__ == "__main__":
    main()
