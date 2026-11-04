"""Story-mode live demo. From the repo root:  uv run python -m demo.demo

Uses the real data, models and gate; nothing is mocked. The "cheating" steps
run on a throwaway copy of the registry, so the repo is never changed.
Needs `make data train` to have been run (data/ is local, not in git).
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"

import shutil
import tempfile
import time
import webbrowser
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from governance import gate
from governance.fairness import spark

B, DIM, G, R, Y, C, N = "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[0m"
RACE = {1: "White", 2: "Black", 3: "American Indian", 5: "Native tribe (other)", 6: "Asian",
        7: "Pacific Islander", 8: "Some other race", 9: "Two or more races"}
SCHOOL = {16: "high-school diploma", 19: "some college", 20: "associate degree", 21: "bachelor's degree",
          22: "master's degree", 23: "professional degree", 24: "doctorate"}
WORKER = {1: "private company", 2: "non-profit", 3: "local government", 4: "state government",
          5: "federal government", 6: "self-employed", 7: "own incorporated business"}
PLAIN = {
    "risk": "its risk level is wrong or banned",
    "documentation": "its paperwork is incomplete",
    "lineage": "we can't prove where it came from, or its evidence was changed",
    "fairness": "it treats groups of people too differently",
    "approval": "no independent person has signed it off",
}


def pause(next_title):
    input(f"\n{C}  ── Enter: {next_title} ──{N}")
    print("\033c", end="")
    print(f"{B}{next_title}{N}\n")


def bar(rate, color):
    return f"{color}{'█' * round(rate * 60)}{N} {rate * 100:4.0f}%"


def verdict(model_dir, policy, datasets):
    r = gate.check_model(model_dir, policy, datasets, spark)
    head = f"{G}✅ APPROVED FOR RELEASE{N}" if r["deployable"] else f"{R}⛔ BLOCKED{N}"
    print(f"  {B}{r['id']}{N}  ({r['tier']} risk)  →  {head}")
    for g in gate.GATES:
        if r["status"][g] == "fail":
            print(f"      {R}✗{N} {PLAIN[g]}")
            for msg in r["fails"][g]:
                print(f"        {DIM}{msg.replace('`', '')}{N}")
    return r


def main():
    policy = gate.load("registry/policy.yaml")
    datasets = {p.stem: gate.load(p) for p in Path("registry/datasets").glob("*.yaml")}

    print("\033c", end="")
    print(f"{B}AI Governance Registry with Deployment Gates{N}\n")
    print("  A bank wants to use AI to sort loan applications: it predicts whether an")
    print("  applicant earns over $50,000 a year. The EU AI Act calls credit decisions")
    print("  HIGH RISK, so before the AI goes live it must be documented, fair, traceable")
    print("  and approved. Our system checks all of that automatically, and blocks it if not.")
    print(f"\n  The bank's team has built two versions of the AI: {Y}baseline{N} and {Y}fair{N}.")

    # ── 1. one real person ───────────────────────────────────────────────
    pause("1 · Meet one real applicant")
    df = pd.read_parquet("data/acs_income_ca_2018.parquet")
    _, te = train_test_split(df, test_size=0.2, random_state=0, stratify=df.label)
    base = pd.read_parquet("registry/models/income-baseline/eval_predictions.parquet")
    fair = pd.read_parquet("registry/models/income-fair/eval_predictions.parquet")
    te = te.reset_index(drop=True).assign(base=base.y_pred, fair=fair.y_pred)
    assert (te.RAC1P.values == base.RAC1P.values).all()  # evidence rows line up with the census rows
    p = te[(te.RAC1P == 8) & (te.label == 1) & (te.base == 0) & (te.fair == 1) & (te.SCHL >= 21)].iloc[0]
    print(f"  {DIM}A real (anonymised) person from the US Census, California 2018:{N}\n")
    print(f"    Age {p.AGEP}, {SCHOOL.get(p.SCHL, 'degree')}, works {p.WKHP} hours a week at a {WORKER.get(p.COW, 'job')}.")
    print(f"    {'Woman' if p.SEX == 2 else 'Man'}, census race group: \"{RACE[p.RAC1P]}\".")
    print(f"    {B}Really earns over $50,000? YES{N}\n")
    print(f"    Baseline AI says:  {R}NO, not likely to earn $50k{N}")
    print(f"    Fair AI says:      {G}YES{N}")
    print(f"\n  {DIM}Neither AI is shown race or sex. So is this bad luck, or a pattern?{N}")

    # ── 2. the pattern ───────────────────────────────────────────────────
    pause("2 · Zoom out: 39,133 applicants the AI never saw during training")
    rates = te.groupby("RAC1P")[["base", "fair"]].mean()[te.RAC1P.value_counts().sort_index() >= 100]
    print(f"  How often each group is approved ({Y}baseline{N} vs {C}fair{N}):\n")
    for k, row in rates.iterrows():
        print(f"    {RACE[k]:<22}{Y}baseline{N} {bar(row.base, Y)}")
        print(f"    {'':<22}{C}fair    {N} {bar(row.fair, C)}")
    lo, hi = rates.base.min(), rates.base.max()
    print(f"\n  {B}The rule (\"four-fifths rule\"):{N} the least-approved group must get at least 80%")
    print(f"  of the approval rate of the most-approved group.")
    print(f"    Baseline: {lo*100:.0f}% ÷ {hi*100:.0f}% = {R}{lo/hi:.2f}{N}  (needs 0.80)")
    print(f"    Fair:     {rates.fair.min()*100:.0f}% ÷ {rates.fair.max()*100:.0f}% = {G}{rates.fair.min()/rates.fair.max():.2f}{N}")
    print(f"\n  {DIM}Removing race didn't remove the bias: job, birthplace and hours act as stand-ins.{N}")

    # ── 3. the gate ──────────────────────────────────────────────────────
    pause("3 · The teams submit their models. The system decides")
    print(f"  Every model must pass 5 checks: {B}risk level · paperwork · traceability · fairness · sign-off{N}")
    print(f"  {DIM}(The bank's manager already signed off the baseline. The system checks anyway.){N}\n")
    for m in ["income-fair", "income-baseline", "taxi-eta"]:
        verdict(Path("registry/models") / m, policy, datasets)
        print()
    print(f"  {DIM}taxi-eta is a low-risk taxi-time model from another team: no fairness or sign-off")
    print(f"  needed, but its paperwork and its dataset's privacy statement are missing.{N}")

    # ── 4. cheating ──────────────────────────────────────────────────────
    pause("4 · Can the team cheat? They edit their test results so the baseline looks fair")
    tmp = Path(tempfile.mkdtemp())
    shutil.copytree("registry", tmp / "registry")
    evidence = tmp / "registry/models/income-baseline/eval_predictions.parquet"
    e = pd.read_parquet(evidence)
    e["y_pred"] = e["y_true"]  # pretend the model was perfect for everyone
    e.to_parquet(evidence)
    print("  They overwrite the saved predictions with perfect ones, then resubmit.\n")
    gate.REGISTRY = tmp / "registry"
    verdict(gate.REGISTRY / "models/income-baseline", policy, datasets)
    print(f"\n  {DIM}When the model was trained, the system saved a fingerprint (sha256 hash) of the")
    print(f"  results. Change one byte and the fingerprint no longer matches.{N}")

    pause("5 · Or: the manager withdraws sign-off on the GOOD model")
    card = gate.REGISTRY / "models/income-fair/card.yaml"
    card.write_text(card.read_text().replace("status: approved", "status: requested"))
    verdict(gate.REGISTRY / "models/income-fair", policy, datasets)
    gate.REGISTRY = Path("registry")
    shutil.rmtree(tmp)

    # ── 5. scale ─────────────────────────────────────────────────────────
    pause("6 · A real bank scores millions of applications. Audit 1.6 million, live, with Spark")
    from pipeline.benchmark_spark import run_spark
    from governance.fairness import metrics
    t = time.perf_counter()
    counts = run_spark(spark(), "data/scaled/x8.parquet")
    secs = time.perf_counter() - t
    print(f"  Scored and audited {B}{int(counts.n.sum()):,}{N} applications in {B}{secs:.1f} s{N}.")
    print(f"  Fairness score at full scale: {R}{metrics(counts, 'RAC1P', 100)['disparate_impact']:.2f}{N}: the same verdict, BLOCKED.\n")
    print("  Measured earlier, 25 million applications:")
    print(f"    normal Python (pandas, 1 core)   {Y}{'█' * 43}{N} 85 s")
    print(f"    Spark, 8 cores                   {C}{'█' * 10}{N} 20 s   {B}4.3× faster{N}, identical result")

    # ── 6. where it runs ─────────────────────────────────────────────────
    pause("7 · This runs automatically on GitHub for every change")
    print("  One check per model: green = released, red = blocked, with the reason attached.")
    webbrowser.open("https://github.com/dhanush-sr/ai-governance-registry/actions")
    print(f"\n  {DIM}Under the hood (if asked): registry/policy.yaml = the rules · registry/models/*/card.yaml =")
    print(f"  what teams write · run.json = fingerprints · governance/gate.py = the 5 checks ·")
    print(f"  governance/fairness.py = the Spark fairness check · .github/workflows = GitHub setup{N}\n")


if __name__ == "__main__":
    main()
