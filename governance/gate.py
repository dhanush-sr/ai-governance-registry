"""Deployment gate.

    python -m governance.gate [MODEL_ID ...]

Checks each model in registry/models (or only those named) against
registry/policy.yaml, prints a Markdown report to stdout and exits 1 if
any checked model is blocked. CI pipes stdout into the job summary.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator, FormatChecker

from governance.risk import classify

REGISTRY = Path("registry")
GATES = ("risk", "documentation", "lineage", "fairness", "approval")
ICON = {"pass": "✅ pass", "fail": "❌ fail", "n/a": "➖ n/a"}


def load(path):
    return yaml.safe_load(Path(path).read_text())  # YAML is a JSON superset


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def get(doc, dotted):
    for key in dotted.split("."):
        doc = doc.get(key) if isinstance(doc, dict) else None
    return doc


def missing(doc, fields):
    return [f"missing `{f}`" for f in fields if get(doc, f) in (None, "", [], {})]


def schema_errors(doc, schema):
    v = Draft202012Validator(load(REGISTRY / "schemas" / schema), format_checker=FormatChecker())
    return [f"schema: `{'.'.join(map(str, e.path)) or '(root)'}` {e.message}" for e in v.iter_errors(doc)]


def commit_exists(sha):
    return subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"], capture_output=True).returncode == 0


def check_model(model_dir, policy, datasets, spark):
    card = load(model_dir / "card.yaml")
    fails = {g: [] for g in GATES}
    na = set()
    result = {"id": card.get("id", model_dir.name), "name": card.get("name", ""), "fairness": {}}

    # Risk: tier is computed from the use case, the declared tier must agree.
    try:
        tier, basis = classify(card.get("risk") or {})
    except ValueError as e:
        tier, basis = None, None
        fails["risk"].append(str(e))
    result.update(tier=tier, basis=basis)
    rules = policy["tiers"].get(tier, {})
    declared = get(card, "risk.declared_tier")
    if tier and declared and declared != tier:
        fails["risk"].append(f"declared `{declared}` but use case makes it `{tier}` ({basis})")
    if tier and not rules.get("deployable"):
        fails["risk"].append(f"{basis}: never deployable")

    # Documentation: card structure plus the fields this tier requires.
    fails["documentation"] += schema_errors(card, "model_card.schema.json")
    fails["documentation"] += missing(card, policy["required_fields"] + rules.get("required_fields", []))

    # Lineage: registered datasets, matching hashes, committed code, untampered evidence.
    preds = None
    run_file = model_dir / "run.json"
    if not run_file.exists():
        fails["lineage"].append("no `run.json`: model was not produced by the tracked pipeline")
    else:
        run = json.loads(run_file.read_text())
        commit = run["code"]["commit"]
        if not commit_exists(commit):
            fails["lineage"].append(f"training commit `{commit[:10]}` not found in git history")
        if run["code"]["dirty"]:
            fails["lineage"].append("trained from uncommitted code changes")
        for d in run["training_data"]:
            ds = datasets.get(d["id"])
            if ds is None:
                fails["lineage"].append(f"training dataset `{d['id']}` has no dataset card in registry/datasets")
                continue
            if get(ds, "source.sha256") != d["sha256"]:
                fails["lineage"].append(f"`{d['id']}`: trained on sha256 {d['sha256'][:12]}…, "
                                        f"card documents {str(get(ds, 'source.sha256'))[:12]}…")
            problems = schema_errors(ds, "dataset_card.schema.json") + missing(ds, policy["dataset_required_fields"])
            fails["lineage"] += [f"dataset `{d['id']}` {p}" for p in problems]
        p = model_dir / run["evaluation"]["predictions"]
        if not p.exists() or sha256(p) != run["evaluation"]["sha256"]:
            fails["lineage"].append("evaluation predictions missing or altered after training")
        else:
            preds = p
        result["metrics"] = run["evaluation"].get("metrics", {})

    # Fairness: recomputed from the prediction evidence, never self-reported.
    fair = rules.get("fairness")
    if not fair:
        na.add("fairness")
    elif preds is None:
        fails["fairness"].append("no trustworthy evaluation predictions to measure")
    elif not card.get("sensitive_attributes"):
        fails["fairness"].append("no `sensitive_attributes` declared")
    else:
        from governance.fairness import group_counts, metrics
        attrs = card["sensitive_attributes"]
        df = spark().read.parquet(str(preds))
        absent = [a for a in attrs + ["y_true", "y_pred"] if a not in df.columns]
        if absent:
            fails["fairness"].append(f"predictions lack columns {absent}")
        else:
            counts = group_counts(df, attrs)
            for a in attrs:
                m = metrics(counts, a, fair["min_group_size"])
                result["fairness"][a] = m
                if m["disparate_impact"] < fair["disparate_impact_min"]:
                    fails["fairness"].append(f"`{a}` disparate impact {m['disparate_impact']:.3f} "
                                             f"< {fair['disparate_impact_min']}")
                if m["equal_opportunity_diff"] > fair["equal_opportunity_diff_max"]:
                    fails["fairness"].append(f"`{a}` equal-opportunity gap {m['equal_opportunity_diff']:.3f} "
                                             f"> {fair['equal_opportunity_diff_max']}")

    # Approval: a human other than the owner signed off.
    if not rules.get("approval_required"):
        na.add("approval")
    else:
        a = card.get("approval") or {}
        if a.get("status") != "approved":
            fails["approval"].append(f"approval status is `{a.get('status', 'absent')}`")
        elif not a.get("approver") or a.get("approver") == card.get("owner"):
            fails["approval"].append("approver must be named and differ from the owner")

    result["status"] = {g: "n/a" if g in na and not fails[g] else "fail" if fails[g] else "pass" for g in GATES}
    result["fails"] = fails
    result["deployable"] = "fail" not in result["status"].values()
    return result


def report(results, datasets):
    out = ["# Model registry: deployment gate report", "",
           "| Model | Risk tier | " + " | ".join(g.capitalize() for g in GATES) + " | Decision |",
           "|---|---|" + "---|" * len(GATES) + "---|"]
    for r in results:
        decision = "**DEPLOY**" if r["deployable"] else "**BLOCKED**"
        out.append(f"| `{r['id']}` | {r['tier'] or '?'} | " +
                   " | ".join(ICON[r["status"][g]] for g in GATES) + f" | {decision} |")
    for r in results:
        out += ["", f"## `{r['id']}`: {r['name']}", "",
                f"- **Risk tier:** {r['tier']} ({r['basis']})",
                f"- **Evaluation metrics:** {json.dumps(r.get('metrics', {}))}"]
        problems = [f"- **{g}:** {msg}" for g in GATES for msg in r["fails"][g]]
        out += ["- **Blocking issues:**" if problems else "- **Blocking issues:** none"]
        out += ["  " + p for p in problems]
        for attr, m in r["fairness"].items():
            out += ["", f"Fairness by `{attr}`: disparate impact **{m['disparate_impact']:.3f}**, "
                        f"equal-opportunity gap **{m['equal_opportunity_diff']:.3f}**"
                        + (f" (excluded small groups: {m['excluded_groups']})" if m["excluded_groups"] else ""),
                    "", f"| {attr} | n | selection rate | TPR |", "|---|---|---|---|"]
            out += [f"| {g[attr]} | {g['n']} | {g['selection_rate']:.3f} | {g['tpr']:.3f} |" for g in m["groups"]]
    out += ["", "## Registered datasets", "", "| Dataset | Source | License | sha256 |", "|---|---|---|---|"]
    out += [f"| `{k}` | {get(d, 'source.url')} | {get(d, 'source.license')} | `{str(get(d, 'source.sha256'))[:12]}…` |"
            for k, d in sorted(datasets.items())]
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", help="model ids (default: all registered models)")
    args = ap.parse_args(argv)

    policy = load(REGISTRY / "policy.yaml")
    datasets = {p.stem: load(p) for p in sorted((REGISTRY / "datasets").glob("*.yaml"))}
    dirs = [REGISTRY / "models" / m for m in args.models] or sorted(p.parent for p in (REGISTRY / "models").glob("*/card.yaml"))

    session = None

    def spark():  # started only if some model needs a fairness check
        nonlocal session
        if session is None:
            from governance.fairness import spark as make
            session = make()
        return session

    results = [check_model(d, policy, datasets, spark) for d in dirs]
    print(report(results, datasets))
    return 0 if all(r["deployable"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
