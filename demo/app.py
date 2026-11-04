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
DEMO_BRANCH = "demo/break-fair-model"
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
         "documentation": "Its paperwork is incomplete.",
         "lineage": "We can't prove where it came from, or its evidence was changed.",
         "fairness": "It treats groups of people too differently.",
         "approval": "No independent person has signed it off."}
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
            "why": [PLAIN[g] for g in gate.GATES if r["status"][g] == "fail"],
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
        prs = get(f"pulls?state=all&head={REPO.split('/')[0]}:{DEMO_BRANCH}")
        pr = prs[0] if prs else None
        out = {"repo": REPO, "required": REQUIRED_CHECKS, "main": checks("main")}
        if pr:
            out["pr"] = {"number": pr["number"], "title": pr["title"], "url": pr["html_url"], "state": pr["state"],
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


def run_gate(model_id, breaks):
    tmp = Path(tempfile.mkdtemp())
    try:
        shutil.copytree("registry", tmp / "registry")
        d = tmp / "registry/models" / model_id
        card = d / "card.yaml"
        if "edit_results" in breaks:
            e = pd.read_parquet(d / "eval_predictions.parquet")
            e["y_pred"] = e["y_true"]
            e.to_parquet(d / "eval_predictions.parquet")
        if "withdraw_approval" in breaks:
            card.write_text(card.read_text().replace("status: approved", "status: requested"))
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
        if "claim_minimal" in breaks:
            card.write_text(card.read_text().replace("declared_tier: high", "declared_tier: minimal"))
        with LOCK:
            gate.REGISTRY = tmp / "registry"
            datasets = {p.stem: gate.load(p) for p in (gate.REGISTRY / "datasets").glob("*.yaml")}
            t = time.perf_counter()
            r = gate.check_model(d, POLICY, datasets, spark)
            secs = time.perf_counter() - t
            gate.REGISTRY = Path("registry")
        return summarise(r, secs)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


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
            self.send(PAGE.replace("__CHOICES__", json.dumps(CHOICES)).replace("__USES__", json.dumps(
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


PAGE = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>AI Governance Registry: live demo</title><style>
:root{--ink:#0b0b0b;--muted:#52514e;--line:#e2e1dc;--bg:#f6f5f2;--base:#eb6834;--fair:#2a78d6;--ok:#0a7d0a;--bad:#c62f2f}
*{box-sizing:border-box;margin:0}body{font:17px/1.5 -apple-system,"Helvetica Neue",Arial,sans-serif;color:var(--ink);background:var(--bg)}
header{background:white;border-bottom:1px solid var(--line);padding:14px 36px;display:flex;align-items:center;gap:28px;position:sticky;top:0;z-index:2}
header h1{font-size:19px;white-space:nowrap}nav{display:flex;gap:4px;flex-wrap:wrap}nav button{border:0;background:none;font:inherit;font-size:15px;padding:9px 13px;border-radius:10px;cursor:pointer;color:var(--muted)}
nav button.on{background:var(--ink);color:white}main{padding:28px 36px;max-width:1440px}.tab{display:none}.tab.on{display:block}
h2{font-size:30px;margin-bottom:6px}h3{font-size:19px;margin:22px 0 10px}.sub{color:var(--muted);margin-bottom:22px;max-width:80ch}
.panel{background:white;border:1px solid var(--line);border-radius:16px;padding:22px}.grid{display:grid;grid-template-columns:420px 1fr;gap:24px;align-items:start}
label{display:block;font-size:14px;color:var(--muted);margin:12px 0 4px}select,input{width:100%;font:inherit;padding:9px 12px;border:1px solid #cfcec8;border-radius:10px;background:white}
.btn{border:0;font:inherit;font-weight:600;padding:11px 18px;border-radius:12px;cursor:pointer;background:var(--ink);color:white;margin:14px 8px 0 0;text-decoration:none;display:inline-block}
.btn.light{background:#ecebe6;color:var(--ink)}.btn:disabled{opacity:.5;cursor:wait}.yes{color:var(--ok)}.no{color:var(--bad)}.note{color:var(--muted);font-size:14px;margin-top:12px}
table{border-collapse:collapse;width:100%;background:white;border:1px solid var(--line);border-radius:14px;overflow:hidden}
th,td{text-align:left;padding:12px 14px;border-bottom:1px solid var(--line);vertical-align:top;font-size:15px}th{background:#faf9f7;color:var(--muted);font-weight:600}
tr.click{cursor:pointer}tr.click:hover{background:#f7f6f3}.pill{display:inline-block;padding:3px 10px;border-radius:999px;font-weight:700;font-size:13px;white-space:nowrap}
.pill.ok{background:#e3f4e3;color:var(--ok)}.pill.bad{background:#fbe4e4;color:var(--bad)}.pill.tier{background:#e8f0fb;color:#1c5cab}.pill.ban{background:#2b2a28;color:white}
pre{background:#161615;color:#f0efec;border-radius:12px;padding:18px;font-size:13px;white-space:pre-wrap;max-height:520px;overflow:auto}
.brief{font-size:24px;line-height:1.4;background:white;border-left:6px solid var(--fair);padding:20px 26px;border-radius:0 14px 14px 0;margin-bottom:24px;max-width:1100px}
.check-row{display:grid;grid-template-columns:44px 1fr 1.3fr 150px;gap:16px;align-items:center;background:white;border:1px solid var(--line);border-radius:14px;padding:14px 18px;margin:8px 0}
.tick{width:36px;height:36px;border-radius:50%;background:#e3f4e3;color:var(--ok);display:grid;place-items:center;font-weight:800;font-size:19px}
.check-row b{font-size:17px}.check-row span{color:var(--muted);font-size:15px}.check-row .btn{margin:0;padding:8px 14px;font-size:14px;text-align:center}
.flow{display:flex;gap:10px;align-items:center;margin:4px 0 22px;flex-wrap:wrap}.fbox{background:white;border:2px solid var(--line);border-radius:12px;padding:10px 14px;font-size:15px}.arrow{color:var(--muted);font-size:22px}
.chain{display:flex;align-items:center;gap:12px;flex-wrap:wrap;font-size:17px}.chain .step{background:white;border:1px solid var(--line);border-radius:12px;padding:10px 14px}.chain small{display:block;color:var(--muted);font-size:12px}
.tiers{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.tierbox{background:white;border:1px solid var(--line);border-radius:14px;padding:16px;font-size:14px}.tierbox b{font-size:17px;display:block;margin-bottom:6px}.tierbox ul{padding-left:18px}
.lights div{display:flex;gap:12px;align-items:center;padding:8px 0;border-bottom:1px solid var(--line)}.dot{width:28px;height:28px;border-radius:50%;display:grid;place-items:center;font-weight:800;flex:none}
.pass .dot{background:#e3f4e3;color:var(--ok)}.fail .dot{background:#fbe4e4;color:var(--bad)}.fail{font-weight:600}.na .dot{background:#efeeea;color:#8a8984}.na{color:#8a8984}
.decision{font-size:38px;font-weight:800;margin:4px 0 10px}.why{color:var(--bad);padding-left:20px;margin-top:10px}code{font-size:12px;color:var(--muted);display:block;margin-top:10px;white-space:pre-wrap}
.models{display:flex;gap:10px;flex-wrap:wrap}.chip{border:2px solid var(--line);background:white;border-radius:12px;padding:10px 14px;cursor:pointer;font:inherit;text-align:left}.chip.on{border-color:var(--ink)}.chip small{display:block;color:var(--muted);font-size:13px}
.breaks label{display:flex;gap:10px;align-items:center;color:var(--ink);font-size:16px;margin:8px 0}.breaks input{width:auto}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:16px 0}.stat{background:white;border:1px solid var(--line);border-radius:14px;padding:16px}.stat b{display:block;font-size:30px}.stat small{color:var(--muted)}
.grp{display:grid;grid-template-columns:190px 1fr;gap:14px;align-items:center;margin:5px 0}.gname{text-align:right;font-size:15px}
.b{height:18px;border-radius:0 4px 4px 0;position:relative}.b span{position:absolute;left:calc(100% + 6px);top:-2px;font-size:13px;white-space:nowrap}
.answers{display:grid;grid-template-columns:1fr 1fr;gap:16px}.ans{border-radius:16px;padding:20px;background:white;border:2px solid var(--line)}.ans h3{font-size:15px;color:var(--muted);margin:0}
.verdict{font-size:30px;font-weight:800;margin:6px 0}.meter{height:12px;border-radius:6px;background:#ecebe6;overflow:hidden;margin-top:8px}.meter i{display:block;height:100%}
.truth{margin-top:16px;padding:14px 18px;border-radius:14px;background:#fffbe9;border:1px solid #f2e3a3}
.gh{display:grid;grid-template-columns:1fr 1fr;gap:20px}.ghcheck{display:flex;justify-content:space-between;align-items:center;padding:10px 0;border-bottom:1px solid var(--line)}
.banner{border-radius:14px;padding:16px 20px;font-size:19px;font-weight:700;margin:12px 0}.banner.bad{background:#fbe4e4;color:var(--bad)}.banner.ok{background:#e3f4e3;color:var(--ok)}
.spin{display:inline-block;width:18px;height:18px;border:3px solid #ccc;border-top-color:var(--ink);border-radius:50%;animation:s .8s linear infinite;vertical-align:-3px;margin-right:8px}@keyframes s{to{transform:rotate(360deg)}}
.legend span{margin-right:20px;font-size:14px}.sw{display:inline-block;width:14px;height:14px;border-radius:3px;vertical-align:-2px;margin-right:6px}
</style></head><body>
<header><h1>AI Governance Registry</h1><nav>
<button class="on" data-t="brief">0 · The task</button><button data-t="registry">1 · Registry</button><button data-t="risk">2 · Risk level</button>
<button data-t="gate">3 · Release check</button><button data-t="github">4 · On GitHub</button><button data-t="evidence">5 · Proof in numbers</button><button data-t="try">6 · Try the AI</button></nav></header>
<main>

<section class="tab on" id="brief"><h2>Our task, and where we prove each part</h2>
<div class="brief"><small style="display:block;color:var(--muted);font-size:14px;font-weight:600">CASE STUDY</small>"A company needs every model to be <b>documented</b>, <b>risk-rated</b> and <b>approved</b> before release."</div>
<div class="flow"><div class="fbox">A team builds an AI and writes its card</div><div class="arrow">→</div><div class="fbox">Uploads it to GitHub</div><div class="arrow">→</div>
<div class="fbox">Our checker runs by itself</div><div class="arrow">→</div><div class="fbox">5 checks, using the rule book</div><div class="arrow">→</div><div class="fbox"><b class="yes">✓ Released</b> or <b class="no">✗ Blocked</b></div></div>
<div id="checklist"><p class="note"><span class="spin"></span>Loading live status…</p></div></section>

<section class="tab" id="registry"><h2>1 · The registry</h2><p class="sub">Every AI and every dataset has a <b>card</b>: a short document about it, kept as a file on GitHub. The status is checked live. Click a row to read its card.</p>
<h3>AI cards</h3><div id="modelTable"></div><h3>Data cards</h3><div id="dataTable"></div>
<h3>The rule book: what each risk level needs</h3><div id="rules"></div>
<p style="margin-top:6px"><b>The checker's settings:</b> <button class="btn light" onclick="show('registry/policy.yaml')">Open the rule book (policy.yaml)</button><button class="btn light" onclick="show('.github/workflows/governance-gate.yml')">Open the GitHub setup (workflow)</button></p><div id="cardView" style="margin-top:20px"></div></section>

<section class="tab" id="risk"><h2>2 · Risk level (EU AI Act)</h2><p class="sub">The team only says what the AI is <b>used for</b>. Our checker uses the EU AI Act to decide how risky it is. So no one can call a loan AI "harmless" to skip the rules.</p>
<div id="riskChains"></div><h3>Try it with any use</h3><div class="grid"><div class="panel"><label>What is the AI used for?</label><select id="useCase"></select>
<label style="display:flex;gap:8px;align-items:center;color:var(--ink)"><input type="checkbox" id="people" style="width:auto">It chats with people</label>
<label style="display:flex;gap:8px;align-items:center;color:var(--ink)"><input type="checkbox" id="content" style="width:auto">It creates images, text or audio</label>
<button class="btn" id="classifyBtn">Check the risk level</button></div><div class="panel" id="classifyOut"><p class="note">Pick a use and press the button.</p></div></div></section>

<section class="tab" id="gate"><h2>3 · Release check: three example AIs</h2><p class="sub">This runs our real checker. Tick a box to break something first, and see if it gets caught. (Your files are not changed.)</p>
<div class="grid"><div class="panel"><b>Choose an AI</b><div class="models" id="models" style="margin-top:12px"></div>
<div class="breaks" style="margin-top:20px"><b>Break something? (you can skip this)</b>
<label><input type="checkbox" value="edit_results">Secretly edit its test results to look perfect</label>
<label><input type="checkbox" value="withdraw_approval">The manager withdraws approval</label>
<label><input type="checkbox" value="delete_limitations">Delete the "limitations" section</label>
<label><input type="checkbox" value="claim_minimal">Claim it's "minimal risk" to dodge the rules</label></div>
<button class="btn" id="gateBtn">Submit for release</button></div><div class="panel" id="gateOut"><p class="note">Choose an AI and press "Submit for release".</p></div></div></section>

<section class="tab" id="github"><h2>4 · The same checks, by themselves, on GitHub</h2>
<p class="sub">GitHub runs our checker every time someone changes something, with one check per AI. This is the <b>live</b> result from GitHub: the current version, and a proposed change where someone removed a section from the fair AI's card.</p><div id="ghOut"><p class="note"><span class="spin"></span>Asking GitHub…</p></div></section>

<section class="tab" id="evidence"><h2>5 · Proof in numbers</h2><p class="sub">The course asks for real measurements, not just claims. Both buttons run live.</p>
<h3>Responsible AI: is the loan AI fair? Test both versions on 39,133 real people</h3><button class="btn" id="auditBtn">Run the fairness test</button><div id="auditOut"></div>
<h3 style="margin-top:34px">Big data: run the same fairness check on 1.6 million applications</h3>
<p class="note" style="margin-top:0">We use Apache Spark, which shares the work across the computer's cores. The 1.6 million are the census people copied with new IDs.</p>
<button class="btn" id="scaleBtn">Check 1.6 million applications now</button><div id="scaleOut"></div>
<div class="panel" style="margin-top:20px;max-width:860px"><b>Measured earlier on 25 million applications</b>
<div class="grp"><div class="gname">Normal Python, 1 core</div><div class="b" style="width:85%;background:var(--base)"><span>85 s</span></div></div>
<div class="grp"><div class="gname">Spark, 8 cores</div><div class="b" style="width:20%;background:var(--fair)"><span>20 s: 4.3× faster, same answer</span></div></div></div></section>

<section class="tab" id="try"><h2>6 · Try the loan AI yourself</h2><p class="sub">This is the example AI our checker tests. It guesses if a person earns over $50,000. It is never told their race or sex.</p>
<div class="grid"><div class="panel" id="form"></div><div><div class="answers" id="answers"></div><div id="truth"></div></div></div></section>
</main>
<script>
const C = __CHOICES__, USES = __USES__;
const $ = s => document.querySelector(s), J = async (u, b) => (await fetch(u, b ? {method: "POST", body: JSON.stringify(b)} : {})).json();
const go = t => { document.querySelectorAll("nav button,.tab").forEach(e => e.classList.remove("on"));
  document.querySelector(`nav button[data-t=${t}]`).classList.add("on"); $("#" + t).classList.add("on"); window.scrollTo(0, 0); if (t == "github") loadGH(); };
document.querySelectorAll("nav button").forEach(b => b.onclick = () => go(b.dataset.t));
const busy = (el, msg) => el.innerHTML = `<p class="note" style="margin-top:16px"><span class="spin"></span>${msg}</p>`;
const pct = x => (x * 100).toFixed(0) + "%";
const tierPill = t => `<span class="pill ${t == "unacceptable" ? "ban" : "tier"}">${t == "unacceptable" ? "BANNED" : t.toUpperCase() + " RISK"}</span>`;
const verdictPill = ok => `<span class="pill ${ok ? "ok" : "bad"}">${ok ? "✓ Released" : "✗ Blocked"}</span>`;

// ---- 0 brief + 1 registry + 2 risk chains (one live gate run on every model)
J("/api/overview").then(o => {
  const rel = o.models.filter(m => m.ok).length, fair = o.models.find(m => m.id == "income-fair"), base = o.models.find(m => m.id == "income-baseline");
  const taxi = o.models.find(m => m.id == "taxi-eta");
  const rows = [
    ["What to do", "\u201cBuild a registry of model and dataset cards\u201d", `${o.models.length} AI cards and ${o.datasets.length} data cards, kept on GitHub`, "go('registry')"],
    ["", "\u201c…with a risk classification based on the NIST AI RMF or the EU AI Act\u201d", `Risk level worked out from what each AI is used for: ${o.models.map(m => m.tier).join(", ")}`, "go('risk')"],
    ["", "\u201cAdd CI checks that block deployment…\u201d", "GitHub runs our checker on every change. If it fails, the change can't go in", "go('github')"],
    ["", "\u201c…when fairness…\u201d requirements are not met", `The first loan AI is blocked: fairness score ${base.fairness.RAC1P.di.toFixed(2)} (needs 0.80)`, "runGate('income-baseline', [])"],
    ["", "\u201c…documentation…\u201d requirements are not met", "The taxi AI is blocked: its card has no \u201climitations\u201d section", "runGate('taxi-eta', [])"],
    ["", "\u201c…or lineage requirements are not met\u201d", "If test results are changed after training, the AI is blocked", "runGate('income-fair', ['edit_results'])"],
    ["Submit", "\u201cRegistry\u201d", "The registry folder: all the cards and the rule book", "go('registry')"],
    ["", "\u201cCI gate configuration\u201d", "The rule book (policy.yaml) and the GitHub setup file", "show('.github/workflows/governance-gate.yml')"],
    ["", "\u201cThree example models that pass or fail\u201d", `${rel} released and ${o.models.length - rel} blocked, each for a different reason`, "go('gate')"],
    ["Also required by the course", "Responsible AI, proven with numbers", `Fairness measured on 39,133 people: ${base.fairness.RAC1P.di.toFixed(2)} vs ${fair.fairness.RAC1P.di.toFixed(2)}`, "go('evidence')"],
    ["", "Big data, proven with numbers", "Spark checks 1.6 million applications live, and was 4.3× faster on 25 million", "go('evidence')"],
    ["", "Only public data", "US Census survey · New York taxi trips", "go('registry')"]];
  $("#checklist").innerHTML = rows.map(([sec, a, b, act]) => (sec ? `<h3>${sec}</h3>` : "") +
    `<div class="check-row"><div class="tick">✓</div><b>${a}</b><span>${b}</span><button class="btn" onclick="${act}">Show me →</button></div>`).join("");
  $("#modelTable").innerHTML = `<table><tr><th>AI</th><th>Owner</th><th>Version</th><th>Used for</th><th>Risk level</th><th>Status</th></tr>` +
    o.models.map(m => `<tr class="click" onclick="show('${m.file}')"><td><b>${m.name}</b><br><small>${m.id}</small></td><td>${m.owner}</td><td>${m.version}</td><td>${m.use_case}</td><td>${tierPill(m.tier)}</td><td>${verdictPill(m.ok)}${m.why.length ? "<br><small class='no'>" + m.why.join("<br>") + "</small>" : ""}</td></tr>`).join("") + "</table>";
  $("#dataTable").innerHTML = `<table><tr><th>Dataset</th><th>Where it came from</th><th>Licence</th><th>Rows</th><th>Fingerprint</th><th>Privacy note</th></tr>` +
    o.datasets.map(d => `<tr class="click" onclick="show('${d.file}')"><td><b>${d.name}</b></td><td><small>${d.url}</small></td><td>${d.license}</td><td>${(d.rows || 0).toLocaleString()}</td><td><code style="margin:0">${d.sha}…</code></td><td>${d.privacy ? "<span class='yes'>✓ present</span>" : "<span class='no'>✗ missing</span>"}</td></tr>`).join("") + "</table>";
  const R = o.rules, T = R.tiers;
  $("#rules").innerHTML = `<p class="note" style="margin:0 0 10px"><b>Every AI card needs:</b> ${R.all.join(" · ")}<br><b>Every data card needs:</b> ${R.datasets.join(" · ")}</p><div class="tiers">
    <div class="tierbox"><b>${tierPill("unacceptable")}</b>Can never be released.</div>
    <div class="tierbox"><b>${tierPill("high")}</b>Also needs:<ul>${T.high.fields.map(f => `<li>${f}</li>`).join("")}<li>Fair: the least-approved group gets at least ${T.high.fairness.disparate_impact_min * 100}% of the most-approved group's rate</li><li>Fair: among people who qualify, groups differ by at most ${T.high.fairness.equal_opportunity_diff_max * 100} points</li><li>Approved by someone who didn't build it</li></ul></div>
    <div class="tierbox"><b>${tierPill("limited")}</b>Also needs:<ul>${T.limited.fields.map(f => `<li>${f}</li>`).join("")}</ul></div>
    <div class="tierbox"><b>${tierPill("minimal")}</b>Only the basics.</div></div>
    <p class="note">From <code style="display:inline">registry/policy.yaml</code>. Risk levels come from the EU AI Act. The 80% rule is an existing US fairness standard.</p>`;
  $("#riskChains").innerHTML = o.models.map(m => `<div class="panel" style="margin-bottom:12px"><div class="chain"><b style="min-width:170px">${m.name}</b>
    <div class="step"><small>The team says it's for</small>${m.use_case}</div><div class="arrow">→</div><div class="step"><small>EU AI Act says</small>${m.basis}</div><div class="arrow">→</div>
    <div class="step"><small>Our checker decides</small>${tierPill(m.tier)}</div><div class="step"><small>The team wrote</small>${m.declared} ${m.declared == m.tier ? "<span class='yes'>✓ matches</span>" : "<span class='no'>✗ mismatch</span>"}</div></div></div>`).join("");
});
window.show = async p => { go("registry"); const t = await (await fetch("/api/file?p=" + encodeURIComponent(p))).text();
  $("#cardView").innerHTML = `<h3>${p}</h3><pre>${t.replace(/[<&]/g, c => ({"<": "&lt;", "&": "&amp;"}[c]))}</pre>`; $("#cardView").scrollIntoView({behavior: "smooth"}); };

// ---- 2 classify
$("#useCase").innerHTML = [["High risk (Annex III)", USES.high], ["Banned (Article 5)", USES.banned], ["Other", USES.other]].map(([g, o]) =>
  `<optgroup label="${g}">${Object.entries(o).map(([k, v]) => `<option value="${k}">${v}</option>`).join("")}</optgroup>`).join("");
$("#classifyBtn").onclick = async () => { const r = await J("/api/classify", {use_case: $("#useCase").value, people: $("#people").checked, content: $("#content").checked});
  $("#classifyOut").innerHTML = `<div class="decision">${tierPill(r.tier)}</div><p>${r.basis}</p>${!r.deployable ? "<div class='banner bad'>Banned: it can never be released.</div>" :
    `<p class="note">On top of the basics, it needs:</p><ul style="padding-left:20px;margin-top:6px">${r.fields.map(f => `<li>${f}</li>`).join("")}${r.fairness ? "<li>Pass the fairness checks</li>" : ""}${r.approval ? "<li>Independent sign-off</li>" : ""}${!r.fields.length && !r.fairness ? "<li>Nothing extra</li>" : ""}</ul>`}`; };

// ---- 3 gate
const Q = {risk: "How risky is it? (EU AI Act)", documentation: "Is the paperwork complete?", lineage: "Can we trace where it came from?", fairness: "Is it fair to every group?", approval: "Did someone independent approve it?"};
let chosen = "income-fair";
$("#models").innerHTML = [["income-fair", "Fair loan AI", "high risk"], ["income-baseline", "First loan AI", "high risk"], ["taxi-eta", "Taxi trip-time AI", "minimal risk"]].map(([id, n, s]) =>
  `<button class="chip ${id == chosen ? "on" : ""}" data-id="${id}">${n}<small>${s}</small></button>`).join("");
document.querySelectorAll(".chip").forEach(c => c.onclick = () => { document.querySelectorAll(".chip").forEach(x => x.classList.remove("on")); c.classList.add("on"); chosen = c.dataset.id; });
$("#gateBtn").onclick = () => runGate(chosen, [...document.querySelectorAll(".breaks input:checked")].map(i => i.value));
window.runGate = async (model, breaks) => { go("gate"); chosen = model;
  document.querySelectorAll(".chip").forEach(c => c.classList.toggle("on", c.dataset.id == model));
  document.querySelectorAll(".breaks input").forEach(i => i.checked = breaks.includes(i.value));
  $("#gateBtn").disabled = true; busy($("#gateOut"), "Running the 5 checks…"); const r = await J("/api/gate", {model, breaks}); $("#gateBtn").disabled = false;
  $("#gateOut").innerHTML = `<small style="color:var(--muted)">${r.id} · ${r.basis}</small><div class="decision ${r.ok ? "yes" : "no"}">${r.ok ? "✓ Released" : "✗ Blocked"}</div>
   <div class="lights">${Object.entries(Q).map(([g, q]) => `<div class="${r.status[g] == "n/a" ? "na" : r.status[g]}"><span class="dot">${{pass: "✓", fail: "✗", "n/a": "–"}[r.status[g]]}</span>${q}${r.status[g] == "n/a" ? " (not needed for this risk level)" : ""}</div>`).join("")}</div>
   ${r.why.length ? `<ul class="why">${r.why.map(w => `<li>${w}</li>`).join("")}</ul><code>${r.detail.join("\n")}</code>` : ""}
   ${r.fairness.RAC1P ? `<p class="note">Fairness score by race ${r.fairness.RAC1P.di.toFixed(2)} (needs 0.80) · by sex ${r.fairness.SEX.di.toFixed(2)}</p>` : ""}`; };

// ---- 4 github
let ghLoaded = false;
window.loadGH = async () => { if (ghLoaded) return; ghLoaded = true;
  const g = await J("/api/github"), repo = `https://github.com/${g.repo}`;
  if (g.error) { $("#ghOut").innerHTML = `<div class="panel">Couldn't reach GitHub (${g.error}).<br><a class="btn" href="${repo}/actions" target="_blank">Open GitHub Actions →</a><a class="btn light" href="${repo}/pulls?q=is%3Apr" target="_blank">Open pull requests →</a></div>`; return; }
  const icon = c => c.status != "completed" ? "<span class='spin'></span>running" : c.conclusion == "success" ? "<b class='yes'>✓ passed</b>" : c.conclusion == "skipped" ? "skipped" : "<b class='no'>✗ failed</b>";
  const list = cs => cs.filter(c => c.name != "discover").map(c => `<div class="ghcheck"><a href="${c.url}" target="_blank">${c.name}</a><span>${g.required.includes(c.name) ? "<small style='color:var(--muted)'>required · </small>" : ""}${icon(c)}</span></div>`).join("");
  const blocked = cs => g.required.some(n => (cs.find(c => c.name == n) || {}).conclusion == "failure");
  const pr = g.pr;
  $("#ghOut").innerHTML = `<div class="gh"><div class="panel"><b>Current version</b>${list(g.main)}<p class="note">The fair AI passes and is released. The other two are blocked on purpose.</p><a class="btn light" href="${repo}/actions" target="_blank">Open on GitHub →</a></div>
    <div class="panel">${pr ? `<b>Proposed change #${pr.number}: ${pr.title}</b>${list(pr.checks)}
    ${blocked(pr.checks) ? "<div class='banner bad'>✗ This change can't go in: a required check failed</div>" : "<div class='banner ok'>Checks passing</div>"}
    <a class="btn" href="${pr.url}" target="_blank">Open it on GitHub →</a>` : "<b>No demo pull request found.</b>"}</div></div>`; };

// ---- 5 evidence
$("#auditBtn").onclick = async () => { $("#auditBtn").disabled = true; busy($("#auditOut"), "Testing both AIs on 39,133 people…"); const r = await J("/api/audit"); $("#auditBtn").disabled = false;
  const b = r["income-baseline"], f = r["income-fair"];
  $("#auditOut").innerHTML = `<p class="note">Both AIs answered for ${r.people.toLocaleString()} people in ${r.seconds.toFixed(1)} s.</p>
   <div class="stats"><div class="stat"><small>First AI: fairness score</small><b class="no">${b.di.toFixed(2)}</b><small>needs 0.80 or more, so blocked</small></div>
   <div class="stat"><small>Fair AI: fairness score</small><b class="yes">${f.di.toFixed(2)}</b><small>passes</small></div>
   <div class="stat"><small>First AI: accuracy</small><b>${pct(b.accuracy)}</b></div><div class="stat"><small>Fair AI: accuracy</small><b>${pct(f.accuracy)}</b><small>the cost of being fair</small></div></div>
   <div class="panel"><div class="legend" style="margin-bottom:10px"><span><i class="sw" style="background:var(--base)"></i>First AI</span><span><i class="sw" style="background:var(--fair)"></i>Fair AI</span><span style="color:var(--muted)">How often each race group gets a "yes"</span></div>
   ${b.groups.map((g, i) => `<div class="grp"><div class="gname">${g.group}<br><small style="color:var(--muted)">${g.n.toLocaleString()} people</small></div><div>
     <div class="b" style="width:${g.rate * 150}%;background:var(--base)"><span>${pct(g.rate)}</span></div><div class="b" style="width:${f.groups[i].rate * 150}%;background:var(--fair);margin-top:3px"><span>${pct(f.groups[i].rate)}</span></div></div></div>`).join("")}
   <p class="note">Neither AI sees race. The first AI still picked it up from other clues, like job, birthplace and hours worked.</p></div>`; };
$("#scaleBtn").onclick = async () => { $("#scaleBtn").disabled = true; busy($("#scaleOut"), "Spark is checking 1.6 million applications…"); const r = await J("/api/scale"); $("#scaleBtn").disabled = false;
  $("#scaleOut").innerHTML = `<div class="stats"><div class="stat"><small>Applications checked</small><b>${r.rows.toLocaleString()}</b></div><div class="stat"><small>Time taken</small><b>${r.seconds.toFixed(1)} s</b></div>
   <div class="stat"><small>Fairness score (first AI)</small><b class="no">${r.di.toFixed(2)}</b><small>same answer at full size: blocked</small></div></div>`; };

// ---- 6 try
const FIELDS = [["AGEP","Age"],["SCHL","Education"],["OCCP","Job"],["COW","Employer"],["WKHP","Hours worked per week"],["MAR","Marital status"],["POBP","Born in"],["RELP","Role in household"]];
$("#form").innerHTML = FIELDS.map(([k, l]) => `<label>${l}</label>` + (C[k] ? `<select id="${k}">${Object.entries(C[k]).map(([v, t]) => `<option value="${v}">${t}</option>`).join("")}</select>`
  : `<input id="${k}" type="number" value="${k == "AGEP" ? 35 : 40}">`)).join("") + `<button class="btn" id="askBtn">Ask the AI</button><button class="btn light" id="realBtn">Load a real applicant</button>`;
async function ask(truth) { const r = await J("/api/predict", Object.fromEntries(FIELDS.map(([k]) => [k, $("#" + k).value])));
  $("#answers").innerHTML = [["income-baseline", "First AI", "var(--base)"], ["income-fair", "Fair AI", "var(--fair)"]].map(([m, n, c]) =>
    `<div class="ans"><h3>${n} says</h3><div class="verdict ${r[m].yes ? "yes" : "no"}">${r[m].yes ? "✓ Yes, over $50k" : "✗ No"}</div><small>Confidence it's a yes: ${pct(r[m].chance)}</small><div class="meter"><i style="width:${r[m].chance * 100}%;background:${c}"></i></div></div>`).join("");
  $("#truth").innerHTML = truth ? `<div class="truth">A <b>real person</b> from the US Census (${truth.sex}, race group "${truth.race}", never shown to the AI). Really earns over $50k? <b class="${truth.truth ? "yes" : "no"}">${truth.truth ? "YES" : "NO"}</b></div>`
    : `<p class="note">Change any answer and press "Ask the AI" again.</p>`; }
$("#askBtn").onclick = () => ask(null);
$("#realBtn").onclick = async () => { const p = await J("/api/sample"); FIELDS.forEach(([k]) => $("#" + k).value = p[k]); ask(p); };
ask(null);
</script></body></html>"""


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
