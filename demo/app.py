"""Live demo app: walks through the brief and proves each requirement live.

    uv run python -m demo.app        -> opens the demo in your browser

Everything runs for real: the registry files, the EU AI Act classifier, the
gate, the trained models, Spark, and the live GitHub check results. Breaking a
model ("edit its results", ...) happens on a throwaway copy of the registry.
Needs `make data train benchmark` to have been run (data/ is local, not in git).
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"

import json
import pickle
import shutil
import tempfile
import threading
import time
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from governance import gate
from governance.fairness import metrics, spark
from governance.risk import HIGH_RISK, PROHIBITED, classify

REPO = "dhanush-sr/ai-governance-registry"
DEMO_BRANCHES = {"delete_limitations": "demo/break-fair-model", "edit_results": "demo/edit-fair-results",
                 "withdraw_approval": "demo/withdraw-approval", "claim_minimal": "demo/claim-low-risk"}  # one real PR per change
REQUIRED_CHECKS = ["tests", "gate (income-fair)"]  # as set in branch protection
BLIND = ["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP"]
RACE = {1: "White", 2: "Black", 3: "American Indian", 4: "Alaska Native", 5: "Native tribe (other)", 6: "Asian",
        7: "Pacific Islander", 8: "Some other race", 9: "Two or more races"}
CHOICES = {
    "SCHL": {14: "School, no diploma", 16: "High-school diploma", 19: "Some college", 20: "Associate degree",
             21: "Bachelor's degree", 22: "Master's degree", 23: "Professional degree", 24: "Doctorate"},
    "COW": {1: "Private company", 2: "Non-profit", 3: "Local government", 4: "State government",
            5: "Federal government", 6: "Self-employed", 7: "Own business"},
    "OCCP": {1021: "Software developer", 10: "Chief executive", 440: "Manager", 3255: "Registered nurse",
             2310: "School teacher", 9130: "Truck / delivery driver", 5240: "Customer service rep",
             4760: "Retail salesperson", 4220: "Janitor", 9620: "Warehouse labourer", 4020: "Cook",
             5400: "Receptionist", 4720: "Cashier"},
    "MAR": {1: "Married", 5: "Never married", 3: "Divorced", 2: "Widowed", 4: "Separated"},
    "POBP": {6: "California", 36: "New York", 48: "Texas", 303: "Mexico", 233: "Philippines", 207: "China",
             210: "India", 247: "Vietnam"},
    "RELP": {0: "Head of household", 1: "Spouse", 13: "Unmarried partner", 2: "Child"},
}
PLAIN = {"risk": "Its risk level is wrong, or it is banned.",
         "documentation": "Its description is incomplete.",
         "lineage": "We can't trust where it came from: something is missing or was changed.",
         "fairness": "It treats groups of people too differently.",
         "approval": "Nobody else has approved it."}
FIELD = {"name": "Name", "owner": "Owner", "version": "Version", "description": "What it does",
         "intended_use.primary_use": "Main use", "risk.declared_tier": "Declared risk level",
         "limitations": "Limitations", "intended_use.users": "Who will use it",
         "intended_use.out_of_scope": "What it must not be used for", "evaluation.description": "How it was tested",
         "sensitive_attributes": "Groups checked for bias", "ethical_considerations": "Ethical concerns",
         "human_oversight": "How humans oversee it", "transparency_notice": "\"You are talking to an AI\" notice",
         "source.url": "Download link", "source.license": "Licence", "source.sha256": "Fingerprint",
         "collection": "How it was collected", "personal_data": "Privacy statement"}
USE_LABEL = {
    "creditworthiness": "Credit scoring / loans", "employment": "Hiring or managing workers", "education": "Exams, admissions",
    "public_benefits": "Public benefits", "insurance_pricing": "Life/health insurance pricing", "law_enforcement": "Policing",
    "migration_border": "Border / asylum", "justice_democracy": "Courts, elections", "biometric_identification": "Face/fingerprint ID",
    "critical_infrastructure": "Power, water, traffic", "emergency_dispatch": "Emergency calls", "product_safety_component": "Safety part of a product",
    "social_scoring": "Scoring citizens' behaviour", "subliminal_manipulation": "Manipulating people", "exploiting_vulnerabilities": "Exploiting vulnerable people",
    "predictive_policing_profiling": "Predicting crime from profiles", "untargeted_facial_scraping": "Scraping faces from the internet",
    "emotion_recognition_workplace_education": "Emotion reading at work/school", "biometric_categorisation_sensitive": "Guessing race/religion from biometrics",
    "realtime_remote_biometric_id_public": "Live face ID in public", "other": "Anything else (e.g. taxi times, spam filter)"}
ABOUT = {"income-fair": "Fair loan AI", "income-baseline": "First loan AI", "taxi-eta": "Taxi trip-time AI"}

MODELS = {m: pickle.loads(Path(f"data/models/{m}.pkl").read_bytes()) for m in ["income-baseline", "income-fair"]}
_df = pd.read_parquet("data/acs_income_ca_2018.parquet")
_, TEST = train_test_split(_df, test_size=0.2, random_state=0, stratify=_df.label)
TEST = TEST.reset_index(drop=True)
POLICY = gate.load("registry/policy.yaml")
LOCK = threading.Lock()  # gate.REGISTRY is module state; one gate run at a time
CARD_FILES = sorted(str(p) for p in Path("registry").glob("**/*.yaml")) + [".github/workflows/governance-gate.yml"]
_overview = None


def plain_basis(tier, basis):
    ref = basis.split(" (")[0]
    return {"unacceptable": f"Banned by the EU AI Act ({ref})", "high": f"On the EU AI Act's high-risk list ({ref})",
            "limited": f"Must tell people it's an AI ({ref})", "minimal": "Not on any EU AI Act list"}.get(tier, basis)


def summarise(r, secs=0.0):
    return {"id": r["id"], "name": ABOUT.get(r["id"], r["id"]), "tier": r["tier"], "basis": plain_basis(r["tier"], r["basis"]),
            "ok": r["deployable"], "status": r["status"], "seconds": secs,
            "why": [("Its fairness can't be measured from changed evidence." if g == "fairness" and any("no trustworthy" in m for m in r["fails"][g])
                     else PLAIN[g]) for g in gate.GATES if r["status"][g] == "fail"],
            "detail": [m.replace("`", "") for g in gate.GATES for m in r["fails"][g]],
            "fairness": {a: {"di": v["disparate_impact"], "gap": v["equal_opportunity_diff"]} for a, v in r["fairness"].items()}}


def overview():
    """Registry contents plus a real gate run on every model (cached: the registry doesn't change mid-demo)."""
    global _overview
    with LOCK:
        if _overview is None:
            datasets = {p.stem: gate.load(p) for p in Path("registry/datasets").glob("*.yaml")}
            models = []
            for d in sorted(Path("registry/models").iterdir()):
                card = gate.load(d / "card.yaml")
                r = summarise(gate.check_model(d, POLICY, datasets, spark))
                models.append({**r, "owner": card.get("owner"), "version": card.get("version"),
                               "use_case": USE_LABEL.get(card["risk"]["use_case"], card["risk"]["use_case"]), "declared": card["risk"].get("declared_tier"),
                               "file": str(d / "card.yaml")})
            ds = [{"id": k, "name": v.get("name"), "url": gate.get(v, "source.url"), "license": gate.get(v, "source.license"),
                   "sha": str(gate.get(v, "source.sha256"))[:12], "privacy": bool(v.get("personal_data")),
                   "rows": v.get("rows"), "file": f"registry/datasets/{k}.yaml"} for k, v in sorted(datasets.items())]
            rules = {"all": [FIELD.get(f, f) for f in POLICY["required_fields"]],
                     "tiers": {t: {"deployable": r.get("deployable"), "fields": [FIELD.get(f, f) for f in r.get("required_fields", [])],
                                   "fairness": r.get("fairness"), "approval": r.get("approval_required")}
                               for t, r in POLICY["tiers"].items()},
                     "datasets": [FIELD.get(f, f) for f in POLICY["dataset_required_fields"]]}
            _overview = {"models": models, "datasets": ds, "rules": rules}
    return _overview


def classify_use(body):
    tier, basis = classify({"use_case": body["use_case"], "interacts_with_people": body.get("people", False),
                            "generates_synthetic_content": body.get("content", False)})
    r = POLICY["tiers"][tier]
    return {"tier": tier, "basis": plain_basis(tier, basis), "deployable": r.get("deployable"), "fields": [FIELD.get(f, f) for f in r.get("required_fields", [])],
            "fairness": bool(r.get("fairness")), "approval": bool(r.get("approval_required"))}


def github():
    def get(path):
        req = urllib.request.Request(f"https://api.github.com/repos/{REPO}/{path}", headers={"Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req, timeout=8) as f:
            return json.load(f)

    def checks(sha):
        runs = get(f"commits/{sha}/check-runs?per_page=50")["check_runs"]
        return sorted(({"name": c["name"], "status": c["status"], "conclusion": c["conclusion"], "url": c["html_url"]} for c in runs),
                      key=lambda c: c["name"])
    try:
        out = {"repo": REPO, "required": REQUIRED_CHECKS, "main": checks("main"), "prs": {}}
        for pr in get("pulls?state=open&per_page=50"):
            if pr["head"]["ref"] in DEMO_BRANCHES.values():
                out["prs"][pr["head"]["ref"]] = {"number": pr["number"], "title": pr["title"], "url": pr["html_url"],
                                                  "checks": checks(pr["head"]["sha"])}
        return out
    except Exception as e:  # offline: the UI falls back to links
        return {"error": str(e), "repo": REPO}


def chance_yes(model_id, X):
    m = MODELS[model_id]
    return m._pmf_predict(X)[:, 1] if model_id == "income-fair" else m.predict_proba(X)[:, 1]


def predict(person):
    X = pd.DataFrame([{k: int(person[k]) for k in BLIND}])
    return {m: {"chance": float(chance_yes(m, X)[0]), "yes": bool(chance_yes(m, X)[0] >= 0.5)} for m in MODELS}


def sample():
    pool = TEST[TEST.OCCP.isin(CHOICES["OCCP"]) & TEST.POBP.isin(CHOICES["POBP"]) & TEST.SCHL.isin(CHOICES["SCHL"])
                & TEST.COW.isin(CHOICES["COW"]) & TEST.RELP.isin(CHOICES["RELP"])]
    r = pool.sample(1).iloc[0]
    return {**{k: int(r[k]) for k in BLIND}, "truth": bool(r.label), "race": RACE[int(r.RAC1P)],
            "sex": "Female" if r.SEX == 2 else "Male"}


def audit():
    t = time.perf_counter()
    X = TEST[BLIND]
    preds = {"income-baseline": MODELS["income-baseline"].predict(X),
             "income-fair": MODELS["income-fair"].predict(X, random_state=0)}
    out = {"people": len(TEST), "seconds": time.perf_counter() - t}
    for m, y in preds.items():
        d = TEST[["RAC1P", "SEX", "label"]].assign(y_pred=y)
        counts = (d.assign(n=1, selected=d.y_pred, positives=d.label, true_pos=d.label * d.y_pred)
                   .groupby(["RAC1P", "SEX"])[["n", "selected", "positives", "true_pos"]].sum().reset_index())
        race = metrics(counts, "RAC1P", 100)
        out[m] = {"accuracy": float((y == TEST.label).mean()), "di": race["disparate_impact"],
                  "gap": race["equal_opportunity_diff"], "sex_di": metrics(counts, "SEX", 100)["disparate_impact"],
                  "groups": [{"group": RACE[int(g["RAC1P"])], "rate": float(g["selection_rate"]), "n": int(g["n"])}
                             for g in race["groups"] if g["n"] >= 100]}
    return out


TIER_PLAIN = {"unacceptable": "banned", "high": "high risk", "limited": "limited risk", "minimal": "low risk"}


def first_line(text, n=70):
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "…"


def run_gate(model_id, breaks):
    """Apply the chosen changes to a throwaway copy, run the real checker, and explain each rule."""
    tmp = Path(tempfile.mkdtemp())
    try:
        shutil.copytree("registry", tmp / "registry")
        d = tmp / "registry/models" / model_id
        card = d / "card.yaml"
        before = gate.load(card)
        changes = []
        if "edit_results" in breaks:
            e = pd.read_parquet(d / "eval_predictions.parquet")
            e["y_pred"] = e["y_true"]
            e.to_parquet(d / "eval_predictions.parquet")
            changes.append(["Its saved test results", "the AI's real answers", "replaced with perfect answers"])
        if "withdraw_approval" in breaks:
            card.write_text(card.read_text().replace("status: approved", "status: requested"))
            changes.append(["Approval", "approved", "requested (not approved yet)"])
        if "delete_limitations" in breaks:
            lines, skip = [], False
            for line in card.read_text().splitlines():
                if line.startswith("limitations:"):
                    skip = True
                    continue
                if skip and (line.startswith(" ") or not line):
                    continue
                skip = False
                lines.append(line)
            card.write_text("\n".join(lines) + "\n")
            changes.append(["Limitations section", first_line(before.get("limitations"), 60), "deleted"])
        if "claim_minimal" in breaks:
            card.write_text(card.read_text().replace("declared_tier: high", "declared_tier: minimal"))
            changes.append(["Risk level written on its card", "high", "minimal (low risk)"])
        if "fix_paperwork" in breaks:
            card.write_text(card.read_text().replace(
                "known_issues: Slow at airports, probably fine elsewhere.",
                "limitations: >\n  Less accurate for airport trips and rush hour; trained on January 2023 only."))
            ds = tmp / "registry/datasets/nyc-tlc-yellow-2023-01.yaml"
            ds.write_text(ds.read_text().replace("# personal_data:", "personal_data: >\n  Trip records only; no names or rider identities. Published by NYC.\n# was:"))
            changes.append(["Limitations section", "missing (an unofficial 'known issues' note)", "added"])
            changes.append(["Privacy note on its data card", "missing", "added"])
        with LOCK:
            gate.REGISTRY = tmp / "registry"
            datasets = {p.stem: gate.load(p) for p in (gate.REGISTRY / "datasets").glob("*.yaml")}
            t = time.perf_counter()
            r = gate.check_model(d, POLICY, datasets, spark)
            secs = time.perf_counter() - t
            gate.REGISTRY = Path("registry")
        out = summarise(r, secs)
        out["changes"] = changes
        out["rules"] = explain(r, gate.load(card), d, datasets)
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def explain(r, card, d, datasets):
    """For each of the five questions: what the rule book asks, and what this AI has."""
    tier = r["tier"]
    rules = POLICY["tiers"].get(tier, {})
    need = POLICY["required_fields"] + rules.get("required_fields", [])
    missing = [FIELD.get(f, f) for f in need if gate.get(card, f) in (None, "", [], {})]
    unofficial = [k for k in card if k not in json.loads(Path("registry/schemas/model_card.schema.json").read_text())["properties"]]
    run = json.loads((d / "run.json").read_text())
    saved = run["evaluation"]["sha256"]
    now = gate.sha256(d / run["evaluation"]["predictions"])
    ds_problems = []
    for t in run["training_data"]:
        ds = datasets.get(t["id"])
        if ds is None:
            ds_problems.append("its data has no card")
        else:
            ds_problems += [f"its data card has no {FIELD.get(f, f).lower()}" for f in POLICY["dataset_required_fields"] if gate.get(ds, f) in (None, "", [], {})]
    fair = r["fairness"].get("RAC1P")
    appr = card.get("approval") or {}
    use = USE_LABEL.get(gate.get(card, "risk.use_case"), gate.get(card, "risk.use_case"))
    return {
        "risk": {"asks": f"Used for {use.lower()}, so the EU AI Act says: {TIER_PLAIN.get(tier, tier)}. The card must say the same.",
                 "has": f"Its card says: {TIER_PLAIN.get(gate.get(card, 'risk.declared_tier'), gate.get(card, 'risk.declared_tier'))}"},
        "documentation": {"asks": f"{len(need)} sections filled in, including limitations" + (" and human oversight" if tier == "high" else ""),
                          "has": ("Missing: " + ", ".join(missing) if missing else f"All {len(need)} filled in")
                                 + (f". Unofficial section: {', '.join(unofficial)}" if unofficial else "")},
        "lineage": {"asks": "Test results unchanged since it was built (same fingerprint), and a complete card for its data",
                    "has": (f"Fingerprint {saved[:8]}… then, {now[:8]}… now" + (": changed" if now != saved else ": same"))
                           + ("; " + "; ".join(ds_problems) if ds_problems else "; data card complete")},
        "fairness": {"asks": "Least-approved group gets at least 80% of the best-approved group's rate (0.80)",
                     "has": f"Measured: {fair['disparate_impact']:.2f}" if fair else
                            ("Can't be measured: the results were changed" if r["status"]["fairness"] == "fail" else "Not needed at low risk")},
        "approval": {"asks": "Approved by someone who didn't build it",
                     "has": (f"{appr.get('status', 'no approval')} by {str(appr.get('approver', 'nobody')).split('@')[0]} (built by {str(card.get('owner')).split('@')[0]})"
                             if rules.get("approval_required") else "Not needed at low risk")},
    }


def scale():
    from pipeline.benchmark_spark import run_spark
    t = time.perf_counter()
    counts = run_spark(spark(), "data/scaled/x8.parquet")
    return {"rows": int(counts.n.sum()), "seconds": time.perf_counter() - t,
            "di": metrics(counts, "RAC1P", 100)["disparate_impact"]}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else json.dumps(body, default=str).encode()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path, _, query = self.path.partition("?")
        routes = {"/api/overview": overview, "/api/sample": sample, "/api/audit": audit, "/api/scale": scale, "/api/github": github}
        if path == "/":
            self.send(PAGE.replace("__CHOICES__", json.dumps(CHOICES)).replace("__DEMO__", json.dumps(DEMO_BRANCHES)).replace("__USES__", json.dumps(
                {"high": {k: USE_LABEL[k] for k in HIGH_RISK}, "banned": {k: USE_LABEL[k] for k in PROHIBITED},
                 "other": {"other": USE_LABEL["other"]}})), "text/html; charset=utf-8")
        elif path == "/api/file" and query.removeprefix("p=") in CARD_FILES:
            self.send(Path(query.removeprefix("p=")).read_text(), "text/plain; charset=utf-8")
        elif path in routes:
            self.send(routes[path]())
        else:
            self.send_error(404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        routes = {"/api/predict": lambda: predict(body), "/api/classify": lambda: classify_use(body),
                  "/api/gate": lambda: run_gate(body["model"], body.get("breaks", []))}
        if self.path in routes:
            self.send(routes[self.path]())
        else:
            self.send_error(404)


PAGE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Governance Registry</title><style>
:root{--ink:#1d1d1f;--ink2:#6e6e73;--ink3:#86868b;--line:#e5e5ea;--fill:#f5f5f7;--blue:#0071e3;--blue-h:#0077ed;--ok:#1f7a35;--bad:#c4131f;--okbg:#eaf6ec;--badbg:#fdeceb;--first:#eb6834;--fair:#2a78d6;
--ease:cubic-bezier(.16,1,.3,1);font-synthesis:none}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{font:17px/1.47 -apple-system,BlinkMacSystemFont,"SF Pro Text","Helvetica Neue",Arial,sans-serif;color:var(--ink);background:#fff;-webkit-font-smoothing:antialiased;font-variant-numeric:tabular-nums}
::selection{background:#cfe4fb}
:focus-visible{outline:3px solid rgba(0,113,227,.45);outline-offset:3px;border-radius:8px}
button{font:inherit;color:inherit;background:none;border:0;cursor:pointer}
a{color:var(--blue);text-decoration:none}a:hover{text-decoration:underline;text-underline-offset:3px}
svg.i{width:1em;height:1em;flex:none;vertical-align:-.13em}

/* frame */
.top{position:fixed;inset:0 0 auto;height:56px;display:flex;align-items:center;justify-content:space-between;padding:0 28px;background:rgba(255,255,255,.86);backdrop-filter:saturate(180%) blur(16px);z-index:5}
.brand{font-weight:600;font-size:15px}
.dots{display:flex;gap:6px;align-items:center}
.dot{width:8px;height:8px;border-radius:50%;background:#d2d2d7;transition:background .3s}
.dot.done{background:#8e8e93}.dot.on{width:22px;border-radius:4px;background:var(--ink)}
.dots button{padding:6px 2px;display:grid;place-items:center}
.count{font-size:13px;color:var(--ink2);min-width:74px;text-align:right}
main{min-height:100%;display:flex;justify-content:center;padding:120px 28px 140px}
.step{width:100%;max-width:1000px;display:none}
.rulelist{display:flex;flex-wrap:wrap;justify-content:center;gap:10px}
.rulelist span{background:var(--fill);border-radius:980px;padding:9px 16px 9px 10px;font-size:16px;display:inline-flex;align-items:center;gap:10px}
.rulelist b{width:24px;height:24px;border-radius:50%;background:#fff;display:grid;place-items:center;font-size:13px}
.pick{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:44px}
.choice{display:flex;flex-direction:column;gap:6px;text-align:left;padding:24px;border-radius:20px;background:var(--fill);transition:background .2s,transform .2s var(--ease)}
.choice:hover{background:#ebebf0;transform:translateY(-2px)}
.choice .cn{font-size:13px;color:var(--ink2)}.choice b{font-size:22px;letter-spacing:-.01em}.choice span{color:var(--ink2);font-size:15px}
.choice .go{margin-top:14px;color:var(--blue);font-size:15px;display:inline-flex;align-items:center;gap:4px}
h1.caseh{max-width:none}
.casebody{margin:36px auto 0}
.rt{border-top:1px solid var(--line)}
.rh,.rr{display:grid;grid-template-columns:1.05fr 1.35fr 1.35fr 128px;gap:18px;padding:14px 8px;border-bottom:1px solid var(--line);align-items:center}
.rh{font-size:13px;color:var(--ink2);font-weight:600;padding-top:10px;padding-bottom:10px}
.rr{font-size:15px;transition:background .3s}.rr b{font-size:16px}.rr span{color:var(--ink2)}.rr .has{color:var(--ink)}
.rr.fail .has{color:var(--bad);font-weight:600}.rr.na{opacity:.55}
.rr.flip{animation:flip 2.2s var(--ease)}
@keyframes flip{0%,40%{background:#fff4c2}100%{background:transparent}}
.res{display:inline-flex;align-items:center;gap:7px;font-weight:600;font-size:15px;justify-self:start}
.res.ok{color:var(--ok)}.res.bad{color:var(--bad)}.res.na{color:var(--ink3);font-weight:400}
.panel-h{font:600 22px/1.2 -apple-system,BlinkMacSystemFont,"SF Pro Display",sans-serif;letter-spacing:-.01em;text-align:center;margin-top:56px}
.hint{text-align:center;color:var(--ink2);font-size:15px;margin-top:6px}
.chips{display:flex;flex-wrap:wrap;justify-content:center;gap:10px;margin-top:18px}
.chip{padding:10px 18px;border-radius:980px;background:var(--fill);font-size:15px;transition:background .2s,color .2s}
.chip:hover{background:#e8e8ed}.chip.on{background:var(--ink);color:#fff}.chip.ghost{background:none;color:var(--blue)}
.diff{max-width:720px;margin:26px auto 0;padding:16px 20px;border-radius:16px;background:#fff8dc}
.dh{font-size:13px;font-weight:600;color:#7a5d00;margin-bottom:6px}
.dr{display:flex;align-items:center;gap:10px;flex-wrap:wrap;font-size:15px;padding:3px 0}
.dr span{font-weight:600;margin-right:6px}.dr s{color:var(--ink2)}.dr b{font-weight:600}
.pushbox{text-align:center;margin-top:30px}
@media (max-width:760px){.pick{grid-template-columns:1fr}.rh{display:none}.rr{grid-template-columns:1fr;gap:4px}}
.step.on{display:block;animation:rise .55s var(--ease) both}
@keyframes rise{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
@media (prefers-reduced-motion:reduce){.step.on{animation:none}.dot{transition:none}}
.foot{position:fixed;inset:auto 0 0;height:84px;display:flex;align-items:center;justify-content:center;gap:18px;background:linear-gradient(to top,#fff 70%,rgba(255,255,255,0));z-index:5}
.next{background:var(--blue);color:#fff;font-weight:500;font-size:17px;padding:12px 30px;border-radius:980px;min-width:148px;transition:background .2s}
.next:hover{background:var(--blue-h)}
.back{color:var(--blue);font-size:17px;padding:12px 18px}
.back[disabled]{visibility:hidden}

/* type */
h1{font:600 clamp(34px,4.4vw,52px)/1.08 -apple-system,BlinkMacSystemFont,"SF Pro Display","Helvetica Neue",sans-serif;letter-spacing:-.022em;text-align:center;text-wrap:balance;margin:0 auto 18px;max-width:18ch}
.lede{font-size:21px;line-height:1.38;color:var(--ink2);text-align:center;max-width:36em;margin:0 auto;text-wrap:pretty}
.lede b{color:var(--ink);font-weight:600}
.quote{font-size:15px;color:var(--ink3);text-align:center;margin:16px auto 0;max-width:42em}
.stage{margin:48px auto 0;max-width:720px}
.center{text-align:center}

/* actions */
.act{display:inline-flex;align-items:center;gap:8px;background:var(--fill);color:var(--ink);font-weight:500;padding:11px 20px;border-radius:980px;transition:background .2s}
.act:hover{background:#e8e8ed}.act[disabled]{opacity:.5;cursor:progress}
.act.primary{background:var(--ink);color:#fff}.act.primary:hover{background:#333336}
.link{color:var(--blue);font-size:15px;padding:4px 0}
.row-actions{display:flex;gap:12px;justify-content:center;flex-wrap:wrap}

/* lists */
.list{border-top:1px solid var(--line)}
.li{display:grid;grid-template-columns:1fr auto;gap:16px;align-items:center;padding:16px 4px;border-bottom:1px solid var(--line)}
.li .t{font-weight:600}.li .s{font-size:14px;color:var(--ink2);margin-top:2px}
.li.btnrow{cursor:pointer;width:100%;text-align:left}.li.btnrow:hover{background:#fafafa}
.group-h{font-size:13px;font-weight:600;color:var(--ink2);margin:30px 4px 8px}
.badge{display:inline-flex;align-items:center;gap:6px;font-size:14px;font-weight:600;padding:5px 12px;border-radius:980px;white-space:nowrap}
.badge.ok{background:var(--okbg);color:var(--ok)}.badge.bad{background:var(--badbg);color:var(--bad)}.badge.tier{background:var(--fill);color:var(--ink)}
.chev{color:var(--ink3);margin-left:10px}

/* flow on welcome */
.flow{display:flex;justify-content:center;align-items:center;gap:14px;flex-wrap:wrap;margin-top:44px;font-size:17px}
.flow span{padding:10px 18px;border-radius:980px;background:var(--fill)}
.flow .arr{background:none;padding:0;color:var(--ink3)}
.flow .end{background:none;padding:0;font-weight:600;white-space:nowrap}
.flow .end span{background:none;padding:0}

/* verdict */
.verdict{display:flex;align-items:center;justify-content:center;gap:12px;font:600 34px/1.1 -apple-system,BlinkMacSystemFont,"SF Pro Display",sans-serif;letter-spacing:-.015em;margin:28px 0 6px}
.verdict.ok{color:var(--ok)}.verdict.bad{color:var(--bad)}
.because{text-align:center;color:var(--ink2);font-size:17px;max-width:34em;margin:0 auto}
.checks{margin:28px auto 0;max-width:520px}
.ck{display:flex;align-items:center;gap:14px;padding:12px 4px;border-bottom:1px solid var(--line);font-size:17px}
.ck:first-child{border-top:1px solid var(--line)}
.ck .ic{width:26px;height:26px;border-radius:50%;display:grid;place-items:center;flex:none}
.ck.pass .ic{background:var(--okbg);color:var(--ok)}.ck.fail .ic{background:var(--badbg);color:var(--bad)}.ck.na .ic{background:var(--fill);color:var(--ink3)}
.ck.fail{font-weight:600}.ck.na{color:var(--ink3)}
.ck small{margin-left:auto;font-size:13px;color:var(--ink3);font-weight:400}

/* segmented */
.seg{display:inline-flex;background:var(--fill);border-radius:12px;padding:3px;gap:2px}
.seg button{padding:8px 16px;border-radius:9px;font-size:15px;color:var(--ink2)}
.seg button.on{background:#fff;color:var(--ink);font-weight:600;box-shadow:0 1px 3px rgba(0,0,0,.12)}

/* bars */
.bars{margin:30px auto 0;max-width:640px}
.br{display:grid;grid-template-columns:150px 1fr;gap:16px;align-items:center;margin:9px 0}
.br .n{font-size:14px;text-align:right;color:var(--ink2)}
.br .pair{display:grid;gap:3px}
.bar{height:12px;border-radius:0 6px 6px 0;position:relative;transform-origin:left;animation:grow .8s var(--ease) both}
.bar i{position:absolute;left:calc(100% + 8px);top:-3px;font:normal 12px/1 -apple-system,sans-serif;color:var(--ink2);white-space:nowrap}
@keyframes grow{from{transform:scaleX(0)}to{transform:none}}
@media (prefers-reduced-motion:reduce){.bar{animation:none}}
.legend{display:flex;gap:22px;justify-content:center;font-size:14px;color:var(--ink2);margin-top:18px}
.sw{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:7px}
.facts{display:flex;justify-content:center;gap:56px;margin-top:34px;flex-wrap:wrap}
.fact{text-align:center}.fact b{display:block;font:600 40px/1 -apple-system,BlinkMacSystemFont,"SF Pro Display",sans-serif;letter-spacing:-.02em}
.fact span{font-size:14px;color:var(--ink2)}

/* details */
details{margin:26px auto 0;max-width:640px}
summary{cursor:pointer;color:var(--blue);font-size:15px;list-style:none;text-align:center}
summary::-webkit-details-marker{display:none}
details[open] summary{margin-bottom:14px}
.detail-body{font-size:15px;color:var(--ink2)}
pre{background:var(--fill);color:var(--ink);border-radius:14px;padding:18px 20px;font:13px/1.55 ui-monospace,"SF Mono",Menlo,monospace;white-space:pre-wrap;max-height:360px;overflow:auto;margin-top:8px;text-align:left}
.tierrow{display:grid;grid-template-columns:120px 1fr;gap:16px;padding:12px 0;border-bottom:1px solid var(--line);font-size:15px;color:var(--ink)}
.tierrow b{font-weight:600}
select{font:inherit;font-size:17px;padding:10px 38px 10px 16px;border-radius:12px;border:1px solid #d2d2d7;background:#fff url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8'%3E%3Cpath d='M1 1l5 5 5-5' fill='none' stroke='%2386868b' stroke-width='1.6'/%3E%3C/svg%3E") no-repeat right 14px center;-webkit-appearance:none;appearance:none;color:var(--ink)}
.try{display:flex;gap:12px;justify-content:center;align-items:center;flex-wrap:wrap;margin-top:18px}
.result{text-align:center;margin-top:18px;min-height:52px;font-size:17px}
.spinner{display:inline-block;width:18px;height:18px;border:2px solid #d2d2d7;border-top-color:var(--ink);border-radius:50%;animation:sp .8s linear infinite;vertical-align:-3px;margin-right:10px}
@keyframes sp{to{transform:rotate(360deg)}}
.wait{text-align:center;color:var(--ink2);margin-top:30px}
.recap .li{grid-template-columns:28px 1fr auto}
.num{width:30px;height:30px;border-radius:50%;background:var(--fill);display:grid;place-items:center;font-weight:600;font-size:15px}
.recap .ok{color:var(--ok)}
.form{display:grid;grid-template-columns:1fr 1fr;gap:14px 18px;max-width:640px;margin:36px auto 0}
.form label{font-size:13px;color:var(--ink2);display:grid;gap:6px}
.form select,.form input{width:100%}
.form input{font:inherit;font-size:17px;padding:10px 16px;border-radius:12px;border:1px solid #d2d2d7}
.answers{display:flex;justify-content:center;gap:48px;margin-top:30px;flex-wrap:wrap}
.answer{text-align:center}.answer span{font-size:14px;color:var(--ink2)}.answer b{display:block;font:600 28px/1.2 -apple-system,BlinkMacSystemFont,"SF Pro Display",sans-serif;margin-top:4px}
.truth{text-align:center;margin-top:22px;font-size:15px;color:var(--ink2)}
@media (max-width:640px){.flow .arr{display:none}.flow{gap:8px}main{padding:96px 20px 130px}.br{grid-template-columns:96px 1fr}.form{grid-template-columns:1fr}.top{padding:0 16px}.dots{display:none}}
</style></head><body>
<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs>
<symbol id="i-ok" viewBox="0 0 16 16"><path d="M3.5 8.5l3 3 6-7" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></symbol>
<symbol id="i-no" viewBox="0 0 16 16"><path d="M4.5 4.5l7 7m0-7l-7 7" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></symbol>
<symbol id="i-na" viewBox="0 0 16 16"><path d="M4.5 8h7" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></symbol>
<symbol id="i-arrow" viewBox="0 0 16 16"><path d="M3 8h10m-4-4l4 4-4 4" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></symbol>
<symbol id="i-chev" viewBox="0 0 16 16"><path d="M6 3.5L10.5 8 6 12.5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></symbol>
</defs></svg>
<header class="top"><div class="brand">AI Governance Registry</div><nav class="dots" id="dots" aria-label="Steps"></nav><div class="count" id="count"></div></header>
<main id="main"></main>
<footer class="foot"><button class="back" id="back">Back</button><button class="next" id="next">Continue</button></footer>
<script>
const C = __CHOICES__, USES = __USES__, DEMO = __DEMO__;
const $ = s => document.querySelector(s);
const J = async (u, b) => (await fetch(u, b ? {method: "POST", body: JSON.stringify(b)} : {})).json();
const icon = (n, cls = "i") => `<svg class="${cls}"><use href="#i-${n}"/></svg>`;
const pct = x => (x * 100).toFixed(0) + "%";
const esc = s => String(s).replace(/[<&>]/g, c => ({"<": "&lt;", "&": "&amp;", ">": "&gt;"}[c]));
const wait = msg => `<p class="wait"><span class="spinner"></span>${msg}</p>`;
const TIER = {unacceptable: "Banned", high: "High risk", limited: "Limited risk", minimal: "Low risk"};
let O = null;  // overview: registry + live gate results

const CASES = {
  "income-fair": {n: 1, title: "The fair loan AI", short: "Built carefully, with a fairness rule.",
    lede: "It guesses who earns over $50,000, to help a bank check loan applications. Loans are <b>high risk</b>, so all five rules apply.",
    changes: [["delete_limitations", "Delete its limitations section"], ["edit_results", "Secretly edit its test results"], ["withdraw_approval", "The manager takes back approval"], ["claim_minimal", "Call it low risk to skip rules"]]},
  "income-baseline": {n: 2, title: "The first loan AI", short: "Same job, built the quick way.",
    lede: "Same job as Case 1, but the team only hid race and sex from it, and a manager approved it. Loans are <b>high risk</b>, so all five rules apply.",
    changes: [["edit_results", "Edit its test results so it looks fair"], ["claim_minimal", "Call it low risk to skip the fairness rule"]]},
  "taxi-eta": {n: 3, title: "The taxi AI", short: "Low risk, but sloppy paperwork.",
    lede: "It guesses how long a taxi ride takes. That's <b>low risk</b>, so only the first three rules apply.",
    changes: [["fix_paperwork", "Fix its paperwork"]]},
};
const RULES = [["risk", "How risky is it?"], ["documentation", "Is its description complete?"], ["lineage", "Can we trust where it came from?"], ["fairness", "Is it fair to everyone?"], ["approval", "Did someone else approve it?"]];
let model = "income-fair", applied = null, last = null;

const STEPS = [
 { // case study
  html: () => `<h1>No AI goes live without passing a check.</h1>
   <p class="lede">A company builds AI programs. Its rule: <b>no AI is released</b> unless it is properly described, its risk is known, and someone else has approved it. We built the checker that enforces this, automatically.</p>
   <div class="flow"><span>An AI and its description</span><svg class="i arr"><use href="#i-arrow"/></svg><span>Checked against the rule book</span><svg class="i arr"><use href="#i-arrow"/></svg><span class="end"><b style="color:var(--ok)">Released</b> <span style="color:var(--ink3);font-weight:400">or</span> <b style="color:var(--bad)">Blocked</b></span></div>
   <div class="stage"><div class="group-h center">The rule book asks every AI five things</div><div class="rulelist">${RULES.map(([, q], i) => `<span><b>${i + 1}</b>${q}</span>`).join("")}</div></div>`, next: "Pick a case"},
 { // pick
  html: () => `<h1>Pick an AI to check.</h1>
   <p class="lede">We made three example AIs. Each one goes through the same rule book.</p>
   <div class="pick">${Object.entries(CASES).map(([id, c]) => `<button class="choice" onclick="openCase('${id}')"><span class="cn">Case ${c.n}</span><b>${c.title}</b><span>${c.short}</span><span class="go">Check it ${icon("chev")}</span></button>`).join("")}</div>`,
  next: "Open Case 1", onNext: () => openCase("income-fair")},
 { // the case
  html: () => `<div id="case"></div>`, run: () => renderCase(), next: "Next: bigger scale"},
 { // scale
  html: () => `<h1>It works for millions of applications.</h1>
   <p class="lede">A real bank checks millions of applications. Our fairness check runs on <b>Apache Spark</b>, which splits the work across the computer's processor cores.</p>
   <div class="stage center"><button class="act primary" id="runscale">Check 1.6 million applications now</button><div id="scaleout"></div>
   <div class="group-h center" style="margin-top:46px">Measured on 25 million applications</div>
   <div class="bars" style="margin-top:12px"><div class="br"><div class="n">Normal Python</div><div class="bar" style="width:72%;background:var(--first)"><i>85 s</i></div></div><div class="br"><div class="n">Spark, 8 cores</div><div class="bar" style="width:17%;background:var(--fair)"><i>20 s · 4.3× faster, same answer</i></div></div></div>
   <p class="quote" style="margin-top:28px">The millions are the census people copied with new IDs, so this tests speed, not new findings.</p></div>`,
  run: () => { $("#runscale").onclick = async () => { $("#runscale").disabled = true; $("#scaleout").innerHTML = wait("Spark is checking 1.6 million applications…");
   const r = await J("/api/scale"); $("#runscale").disabled = false;
   $("#scaleout").innerHTML = `<div class="facts"><div class="fact"><b>${r.rows.toLocaleString()}</b><span>applications checked</span></div><div class="fact"><b>${r.seconds.toFixed(1)} s</b><span>time taken</span></div><div class="fact"><b style="color:var(--bad)">${r.di.toFixed(2)}</b><span>first loan AI's fairness: still blocked</span></div></div>`; }; }},
 { // summary
  html: () => `<h1>Every AI is checked before it's released.</h1>
   <p class="lede">One rule book. Every change goes through GitHub, which runs the checker and refuses anything that breaks a rule.</p>
   <div class="stage recap"><div class="list">
    ${[["Case 1: Fair loan AI", "Released. Any change that breaks a rule is refused on GitHub.", true, "income-fair"], ["Case 2: First loan AI", "Blocked: unfair (0.30, needs 0.80)", false, "income-baseline"], ["Case 3: Taxi AI", "Blocked: paperwork missing. Fixing it releases it.", false, "taxi-eta"]].map(([a, b, ok, id]) =>
      `<button class="li btnrow" onclick="openCase('${id}')"><span style="color:${ok ? "var(--ok)" : "var(--bad)"}">${icon(ok ? "ok" : "no")}</span><div><div class="t">${a}</div><div class="s">${b}</div></div>${icon("chev", "i chev")}</button>`).join("")}
   </div>
   <details><summary>See every AI and dataset description (the registry)</summary><div id="reg"></div><div id="cardview"></div></details>
   <details><summary>See the real rule book</summary><div class="detail-body" id="rules"></div></details></div>`, next: "Try the AI yourself",
  run: async () => { const o = await overview(); $("#rules").innerHTML = rulesHTML(o.rules);
   const row = (t, s, file) => `<button class="li btnrow" onclick="card('${file}')"><div><div class="t">${t}</div><div class="s">${s}</div></div>${icon("chev", "i chev")}</button>`;
   $("#reg").innerHTML = `<div class="list">` + o.models.map(m => row(m.name, `${m.use_case} · ${TIER[m.tier]}`, m.file)).join("") + o.datasets.map(d => row(d.name, `${(d.rows || 0).toLocaleString()} rows · public data`, d.file)).join("") + `</div>`; }},
 { // try
  html: () => `<h1>Try the loan AI yourself.</h1>
   <p class="lede">Describe a person and see what both loan AIs guess. They are never told race or sex.</p>
   <div class="form" id="form"></div><div class="row-actions" style="margin-top:22px"><button class="act primary" id="ask">Ask the AIs</button><button class="act" id="real">Load a real person</button></div><div id="answers"></div>`, next: "Back to the start",
  run: () => { const F = [["AGEP", "Age"], ["SCHL", "Education"], ["OCCP", "Job"], ["COW", "Employer"], ["WKHP", "Hours a week"], ["MAR", "Marital status"], ["POBP", "Born in"], ["RELP", "Role at home"]];
   $("#form").innerHTML = F.map(([k, l]) => `<label>${l}${C[k] ? `<select id="${k}">${Object.entries(C[k]).map(([v, t]) => `<option value="${v}">${t}</option>`).join("")}</select>` : `<input id="${k}" type="number" value="${k == "AGEP" ? 35 : 40}">`}</label>`).join("");
   const ask = async truth => { const r = await J("/api/predict", Object.fromEntries(F.map(([k]) => [k, $("#" + k).value])));
     const a = (m, n) => `<div class="answer"><span>${n} says</span><b style="color:${r[m].yes ? "var(--ok)" : "var(--bad)"}">${r[m].yes ? "Yes, over $50k" : "No"}</b><span>${pct(r[m].chance)} sure it's a yes</span></div>`;
     $("#answers").innerHTML = `<div class="answers">${a("income-baseline", "First loan AI")}${a("income-fair", "Fair loan AI")}</div>` +
       (truth ? `<p class="truth">A real person from the US census (${truth.sex}, race group "${truth.race}", never shown to the AI). Really earns over $50k: <b style="color:${truth.truth ? "var(--ok)" : "var(--bad)"}">${truth.truth ? "Yes" : "No"}</b></p>` : `<p class="truth">Change the job or the hours and ask again.</p>`); };
   $("#ask").onclick = () => ask(null);
   $("#real").onclick = async () => { const p = await J("/api/sample"); F.forEach(([k]) => $("#" + k).value = p[k]); ask(p); }; }},
];

window.openCase = id => { model = id; applied = null; last = null; loaded.delete(2); go(2); };

function renderCase() {
  const c = CASES[model];
  $("#case").innerHTML = `<h1 class="caseh">Case ${c.n}: ${c.title}</h1><p class="lede">${c.lede}</p>
    <div class="casebody"><div id="table">${wait("Checking it against the rule book…")}</div>
    <div class="panel-h">Now change something</div>
    <p class="hint">Pick a change. We apply it to a copy and check again, live.</p>
    <div class="chips">${c.changes.map(([k, l]) => `<button class="chip" data-k="${k}" onclick="change('${k}')">${l}</button>`).join("")}<button class="chip ghost" data-k="" onclick="change(null)">Undo</button></div>
    <div id="diff"></div><div id="push"></div>
    <div class="row-actions" style="margin-top:34px"><button class="link" onclick="go(1)">Pick another case</button></div></div>`;
  check();
}
window.change = k => { applied = k; document.querySelectorAll(".chip").forEach(b => b.classList.toggle("on", (b.dataset.k || null) == k && k != null)); check(); };

async function check() {
  const t = $("#table"); t.style.opacity = .45;
  const r = await J("/api/gate", {model, breaks: applied ? [applied] : []}); t.style.opacity = 1;
  const flipped = g => last && last.status[g] != r.status[g];
  const mark = s => s == "pass" ? `<span class="res ok">${icon("ok")}Yes</span>` : s == "fail" ? `<span class="res bad">${icon("no")}No</span>` : `<span class="res na">${icon("na")}Not needed</span>`;
  t.innerHTML = `<div class="rt"><div class="rh"><span>Rule</span><span>The rule book asks</span><span>This AI has</span><span></span></div>
    ${RULES.map(([g, q]) => `<div class="rr ${r.status[g] == "n/a" ? "na" : r.status[g]} ${flipped(g) ? "flip" : ""}"><b>${q}</b><span>${r.rules[g].asks}</span><span class="has">${r.rules[g].has}</span>${mark(r.status[g])}</div>`).join("")}</div>
    <div class="verdict ${r.ok ? "ok" : "bad"}">${icon(r.ok ? "ok" : "no")} ${r.ok ? "Released" : "Blocked"}</div>
    <p class="because">${r.ok ? "Every rule that applies is met, so it can go live." : r.why.join(" ")}</p>
    ${model == "income-baseline" && !applied ? `<details id="whyfair"><summary>See why it's unfair</summary><div id="fairout"></div></details>` : ""}`;
  $("#whyfair") && $("#whyfair").addEventListener("toggle", showFair, {once: true});
  $("#diff").innerHTML = r.changes.length ? `<div class="diff"><div class="dh">What changed</div>${r.changes.map(([what, a, b]) => `<div class="dr"><span>${what}</span><s>${esc(a)}</s>${icon("arrow")}<b>${esc(b)}</b></div>`).join("")}</div>` : "";
  $("#push").innerHTML = `<div class="pushbox"><button class="act primary" onclick="push()">${applied ? "Push this change to GitHub" : "See this AI on GitHub"}</button><p class="hint">${applied ? "Changes reach the real system through GitHub, as a pull request." : "GitHub runs the same checker and only releases AIs that pass."}</p><div id="pr"></div></div>`;
  last = r;
}

window.push = async () => {
  const out = $("#pr"); out.innerHTML = wait("Asking GitHub…");
  const g = await J("/api/github"), branch = model == "income-fair" && DEMO[applied];
  if (g.error) { out.innerHTML = `<p class="because">GitHub can't be reached right now.</p>`; return; }
  const NAME = {"tests": "Our automatic tests", "gate (income-fair)": "Fair loan AI checker", "gate (income-baseline)": "First loan AI checker", "gate (taxi-eta)": "Taxi AI checker"};
  const row = c => { const ok = c.conclusion == "success", req = g.required.includes(c.name), done = c.status == "completed";
    return `<div class="ck ${!done ? "na" : ok ? "pass" : "fail"}" ${req ? "" : 'style="opacity:.45;font-weight:400"'}><span class="ic">${done ? icon(ok ? "ok" : "no") : '<span class="spinner" style="margin:0;width:12px;height:12px"></span>'}</span>${NAME[c.name] || c.name}<small>${req ? "must pass" : "not required"}</small></div>`; };
  if (branch && g.prs[branch]) { const pr = g.prs[branch], cs = pr.checks.filter(c => c.name != "discover");
    const running = cs.some(c => c.status != "completed"), blocked = cs.some(c => g.required.includes(c.name) && c.conclusion == "failure");
    out.innerHTML = `<p class="because" style="margin-top:20px">We pushed exactly this change earlier as <b>pull request #${pr.number}</b>. GitHub ran the checker on it:</p>
      <div class="checks">${cs.map(row).join("")}</div>
      <div class="verdict ${blocked ? "bad" : running ? "" : "ok"}">${running ? "Still checking…" : blocked ? icon("no") + " Merging is blocked" : icon("ok") + " Allowed"}</div>
      <p class="because">${blocked ? "The change can't get into the real system. Nobody can override this, not even the owner." : ""}</p>
      <div class="row-actions" style="margin-top:18px"><a class="act" href="${pr.url}" target="_blank">See it on GitHub</a></div>`; return; }
  const mine = g.main.find(c => c.name == "gate (" + model + ")"), ok = mine && mine.conclusion == "success";
  const note = applied == "fix_paperwork" ? " With this fix pushed, its check would turn green and it could be released."
    : applied ? " This change doesn't help: the checker still says no." : "";
  out.innerHTML = `<p class="because" style="margin-top:20px">On GitHub, this AI's own check on the current version:</p>
    <div class="checks">${mine ? row(mine) : ""}</div>
    <div class="verdict ${ok ? "ok" : "bad"}">${icon(ok ? "ok" : "no")} ${ok ? "Released" : "Not released"}</div>
    <p class="because">${ok ? "Its check passed, so GitHub's release step ran." : "Its check failed, so GitHub skips the release step. It never goes live."}${note}</p>
    ${mine ? `<div class="row-actions" style="margin-top:18px"><a class="act" href="${mine.url}" target="_blank">See it on GitHub</a></div>` : ""}`;
};

async function showFair(e) {
  if (!e.target.open) return; $("#fairout").innerHTML = wait("Testing both loan AIs on 39,133 real people…");
  const r = await J("/api/audit"), b = r["income-baseline"], f = r["income-fair"];
  $("#fairout").innerHTML = `<p class="because">How often each group gets a "yes":</p><div class="legend"><span><i class="sw" style="background:var(--first)"></i>First loan AI</span><span><i class="sw" style="background:var(--fair)"></i>Fair loan AI</span></div>
   <div class="bars">${b.groups.map((g, i) => `<div class="br"><div class="n">${g.group}</div><div class="pair"><div class="bar" style="width:${g.rate * 140}%;background:var(--first)"><i>${pct(g.rate)}</i></div><div class="bar" style="width:${f.groups[i].rate * 140}%;background:var(--fair)"><i>${pct(f.groups[i].rate)}</i></div></div></div>`).join("")}</div>
   <p class="because" style="margin-top:20px">Least-approved ÷ best-approved: first loan AI <b style="color:var(--bad)">${b.di.toFixed(2)}</b>, fair loan AI <b style="color:var(--ok)">${f.di.toFixed(2)}</b>. Being fair costs some accuracy: ${pct(b.accuracy)} → ${pct(f.accuracy)}.</p>`;
}

async function overview() { return O || (O = await J("/api/overview")); }
function rulesHTML(R) {
  const T = R.tiers, row = (n, t) => `<div class="tierrow"><b>${n}</b><span>${t}</span></div>`;
  return `<p style="margin-bottom:10px">Every AI card: ${R.all.join(", ").toLowerCase()}. Every data card: ${R.datasets.join(", ").toLowerCase()}.</p>` +
    row("Banned", "Can never be released.") +
    row("High risk", `Also: ${T.high.fields.join(", ").toLowerCase()}; pass the fairness checks; approved by someone who didn't build it.`) +
    row("Limited risk", `Also: ${T.limited.fields.join(", ").toLowerCase()}.`) + row("Minimal risk", "Only the basics.") +
    `<div class="row-actions" style="margin-top:16px"><button class="link" onclick="card('registry/policy.yaml')">Open the rule book</button><button class="link" onclick="card('.github/workflows/governance-gate.yml')">Open the GitHub setup</button></div>`;
}
window.card = async (p, el) => { const t = await (await fetch("/api/file?p=" + encodeURIComponent(p))).text(), v = $("#cardview");
  v.innerHTML = `<div class="group-h">${esc(p)}</div><pre>${esc(t)}</pre>`; v.scrollIntoView({behavior: "smooth", block: "center"}); };

let cur = 0;
const main = $("#main");
main.innerHTML = STEPS.map((s, i) => `<section class="step" id="s${i}" aria-label="Step ${i + 1}"></section>`).join("");
$("#dots").innerHTML = STEPS.map((_, i) => `<button aria-label="Go to step ${i + 1}" onclick="go(${i})"><span class="dot" id="d${i}"></span></button>`).join("");
const loaded = new Set();
window.go = i => {
  cur = Math.max(0, Math.min(STEPS.length - 1, i));
  document.querySelectorAll(".step").forEach(s => s.classList.remove("on"));
  const sec = $("#s" + cur);
  if (!loaded.has(cur)) { sec.innerHTML = STEPS[cur].html(); loaded.add(cur); STEPS[cur].run && STEPS[cur].run(); }
  sec.classList.add("on");
  STEPS.forEach((_, j) => { $("#d" + j).className = "dot" + (j == cur ? " on" : j < cur ? " done" : ""); });
  $("#count").textContent = `${cur + 1} of ${STEPS.length}`;
  $("#back").disabled = cur == 0;
  $("#next").textContent = STEPS[cur].next || "Continue";
  window.scrollTo({top: 0}); history.replaceState(null, "", "#" + cur);
};
$("#next").onclick = () => STEPS[cur].onNext ? STEPS[cur].onNext() : go(cur == STEPS.length - 1 ? 0 : cur + 1);
$("#back").onclick = () => go(cur - 1);
document.addEventListener("keydown", e => { if (e.target.closest("select,input")) return;
  if (e.key == "ArrowRight") go(cur + 1); if (e.key == "ArrowLeft") go(cur - 1); });
overview();
go(+location.hash.slice(1) || 0);
</script></body></html>
"""


if __name__ == "__main__":
    threading.Thread(target=overview, daemon=True).start()  # warm up Spark + cache the gate results
    for port in range(8600, 8700):  # first free port
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    print(f"Live demo running at http://localhost:{port}  (Ctrl+C to stop)", flush=True)
    webbrowser.open(f"http://localhost:{port}")
    server.serve_forever()
