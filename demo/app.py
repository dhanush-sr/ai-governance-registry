"""Live demo app: use the models, audit them, and push them through the release gate.

    uv run python -m demo.app        -> opens the demo in your browser

Everything runs for real: the trained models, the real gate, Spark. Breaking a
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
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from governance import gate
from governance.fairness import group_counts, metrics, spark

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

MODELS = {m: pickle.loads(Path(f"data/models/{m}.pkl").read_bytes()) for m in ["income-baseline", "income-fair"]}
_df = pd.read_parquet("data/acs_income_ca_2018.parquet")
_, TEST = train_test_split(_df, test_size=0.2, random_state=0, stratify=_df.label)
TEST = TEST.reset_index(drop=True)
POLICY = gate.load("registry/policy.yaml")
LOCK = threading.Lock()  # gate.REGISTRY is module state; one gate run at a time


def chance_yes(model_id, X):
    m = MODELS[model_id]
    return m._pmf_predict(X)[:, 1] if model_id == "income-fair" else m.predict_proba(X)[:, 1]


def predict(person):
    X = pd.DataFrame([{k: int(person[k]) for k in BLIND}])
    out = {}
    for m in MODELS:
        p = float(chance_yes(m, X)[0])
        out[m] = {"chance": p, "yes": bool(p >= 0.5)}
    return out


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
    secs = time.perf_counter() - t
    out = {"people": len(TEST), "seconds": secs}
    for m, y in preds.items():
        d = TEST[["RAC1P", "SEX", "label"]].assign(y_pred=y)
        counts = d.rename(columns={"label": "y_true"}).assign(n=1, selected=d.y_pred, positives=d.label,
                                                              true_pos=d.label * d.y_pred)
        counts = counts.groupby(["RAC1P", "SEX"])[["n", "selected", "positives", "true_pos"]].sum().reset_index()
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
        if "edit_results" in breaks and (d / "eval_predictions.parquet").exists():
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
        return {"id": r["id"], "tier": r["tier"], "basis": r["basis"], "ok": r["deployable"], "status": r["status"],
                "seconds": secs, "why": [PLAIN[g] for g in gate.GATES if r["status"][g] == "fail"],
                "detail": [m.replace("`", "") for g in gate.GATES for m in r["fails"][g]],
                "fairness": {a: {"di": v["disparate_impact"], "gap": v["equal_opportunity_diff"]} for a, v in r["fairness"].items()}}
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
        data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/":
            self.send(PAGE.replace("__CHOICES__", json.dumps(CHOICES)), "text/html; charset=utf-8")
        elif self.path == "/api/sample":
            self.send(sample())
        elif self.path == "/api/audit":
            self.send(audit())
        elif self.path == "/api/scale":
            self.send(scale())
        else:
            self.send_error(404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/predict":
            self.send(predict(body))
        elif self.path == "/api/gate":
            self.send(run_gate(body["model"], body.get("breaks", [])))
        else:
            self.send_error(404)


PAGE = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>AI Release Checkpoint: live demo</title><style>
:root{--ink:#0b0b0b;--muted:#52514e;--line:#e2e1dc;--bg:#f6f5f2;--base:#eb6834;--fair:#2a78d6;--ok:#0a7d0a;--bad:#c62f2f}
*{box-sizing:border-box;margin:0}body{font:17px/1.5 -apple-system,"Helvetica Neue",Arial,sans-serif;color:var(--ink);background:var(--bg)}
header{background:white;border-bottom:1px solid var(--line);padding:18px 40px;display:flex;align-items:center;gap:40px}
header h1{font-size:22px}nav{display:flex;gap:6px}nav button{border:0;background:none;font:inherit;padding:10px 16px;border-radius:10px;cursor:pointer;color:var(--muted)}
nav button.on{background:var(--ink);color:white}main{padding:30px 40px;max-width:1400px}.tab{display:none}.tab.on{display:block}
h2{font-size:30px;margin-bottom:6px}.sub{color:var(--muted);margin-bottom:24px;max-width:75ch}
.grid{display:grid;grid-template-columns:420px 1fr;gap:26px;align-items:start}.panel{background:white;border:1px solid var(--line);border-radius:16px;padding:24px}
label{display:block;font-size:14px;color:var(--muted);margin:12px 0 4px}select,input{width:100%;font:inherit;padding:9px 12px;border:1px solid #cfcec8;border-radius:10px;background:white}
.btn{border:0;font:inherit;font-weight:600;padding:12px 20px;border-radius:12px;cursor:pointer;background:var(--ink);color:white;margin:16px 8px 0 0}
.btn.light{background:#ecebe6;color:var(--ink)}.btn:disabled{opacity:.5;cursor:wait}
.answers{display:grid;grid-template-columns:1fr 1fr;gap:18px}.ans{border-radius:16px;padding:22px;background:white;border:2px solid var(--line)}
.ans h3{font-size:16px;color:var(--muted);font-weight:600}.verdict{font-size:34px;font-weight:800;margin:6px 0}.yes{color:var(--ok)}.no{color:var(--bad)}
.meter{height:12px;border-radius:6px;background:#ecebe6;overflow:hidden;margin-top:8px}.meter i{display:block;height:100%}
.truth{margin-top:18px;padding:16px 20px;border-radius:14px;background:#fffbe9;border:1px solid #f2e3a3}.note{color:var(--muted);font-size:14px;margin-top:12px}
.grp{display:grid;grid-template-columns:190px 1fr;gap:14px;align-items:center;margin:5px 0}.gname{text-align:right;font-size:15px}
.b{height:18px;border-radius:0 4px 4px 0;position:relative}.b span{position:absolute;left:calc(100% + 6px);top:-2px;font-size:13px;white-space:nowrap}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:18px 0}.stat{background:white;border:1px solid var(--line);border-radius:14px;padding:16px}
.stat b{display:block;font-size:30px}.stat small{color:var(--muted)}
.models{display:flex;gap:10px;flex-wrap:wrap}.chip{border:2px solid var(--line);background:white;border-radius:12px;padding:12px 16px;cursor:pointer;font:inherit;text-align:left}
.chip.on{border-color:var(--ink)}.chip small{display:block;color:var(--muted);font-size:13px}
.breaks label{display:flex;gap:10px;align-items:center;color:var(--ink);font-size:16px;margin:8px 0}.breaks input{width:auto}
.lights div{display:flex;gap:12px;align-items:center;padding:9px 0;border-bottom:1px solid var(--line);font-size:17px}
.dot{width:30px;height:30px;border-radius:50%;display:grid;place-items:center;font-weight:800;flex:none}
.pass .dot{background:#e3f4e3;color:var(--ok)}.fail .dot{background:#fbe4e4;color:var(--bad)}.fail{font-weight:600}.na .dot{background:#efeeea;color:#8a8984}.na{color:#8a8984}
.decision{font-size:40px;font-weight:800;margin:6px 0 12px}.why{color:var(--bad);padding-left:20px;margin-top:10px}code{font-size:12px;color:var(--muted);display:block;margin-top:10px;white-space:pre-wrap}
.spin{display:inline-block;width:18px;height:18px;border:3px solid #ccc;border-top-color:var(--ink);border-radius:50%;animation:s .8s linear infinite;vertical-align:-3px;margin-right:8px}@keyframes s{to{transform:rotate(360deg)}}
.legend span{margin-right:20px;font-size:14px}.sw{display:inline-block;width:14px;height:14px;border-radius:3px;vertical-align:-2px;margin-right:6px}
</style></head><body>
<header><h1>AI Release Checkpoint <span style="color:var(--muted);font-weight:400">· live demo</span></h1>
<nav><button class="on" data-t="t1">1 · Try the AI</button><button data-t="t2">2 · Fairness audit</button><button data-t="t3">3 · Release checkpoint</button><button data-t="t4">4 · Bank scale</button></nav></header>
<main>
<section class="tab on" id="t1"><h2>Try the bank's loan AI</h2>
<p class="sub">The bank uses AI to guess whether an applicant earns over $50,000. It has two versions: the <b style="color:var(--base)">first AI</b>, built the "obvious" way, and a <b style="color:var(--fair)">fair version</b>. Neither is ever told an applicant's race or sex.</p>
<div class="grid"><div class="panel" id="form"></div><div><div class="answers" id="answers"></div><div id="truth"></div></div></div></section>

<section class="tab" id="t2"><h2>Is it a pattern? Audit both AIs on every applicant</h2>
<p class="sub">This runs both AIs live on 39,133 real applicants they never saw while learning, then counts how often each group gets a "yes". <b>The rule:</b> the least-approved group must get at least 80% of the most-approved group's rate.</p>
<button class="btn" id="auditBtn">Run the audit</button><div id="auditOut"></div></section>

<section class="tab" id="t3"><h2>Submit an AI for release</h2>
<p class="sub">This runs the real checkpoint: risk level, paperwork, traceability, fairness and sign-off. Tick a box to break something first and see if it gets caught (it runs on a throwaway copy).</p>
<div class="grid"><div class="panel"><b>1 · Choose an AI</b><div class="models" id="models" style="margin-top:12px"></div>
<div class="breaks" style="margin-top:22px"><b>2 · Break something? (optional)</b>
<label><input type="checkbox" value="edit_results">Secretly edit its test results to look perfect</label>
<label><input type="checkbox" value="withdraw_approval">The manager withdraws approval</label>
<label><input type="checkbox" value="delete_limitations">Delete the "limitations" section from its info sheet</label>
<label><input type="checkbox" value="claim_minimal">Claim it's "minimal risk" to dodge the rules</label></div>
<button class="btn" id="gateBtn">Submit for release</button></div><div class="panel" id="gateOut"><p class="note">Choose an AI and press "Submit for release".</p></div></div></section>

<section class="tab" id="t4"><h2>Bank-sized volumes</h2>
<p class="sub">A real bank scores millions of applications. The fairness check runs on <b>Apache Spark</b>, which splits the work across the computer's processor cores. Press the button to score and audit <b>1.6 million</b> applications live (the census people copied with new IDs).</p>
<button class="btn" id="scaleBtn">Audit 1.6 million applications now</button><div id="scaleOut"></div>
<div class="panel" style="margin-top:26px;max-width:820px"><b>Measured earlier on 25 million applications</b>
<div class="grp"><div class="gname">Normal Python, 1 core</div><div class="b" style="width:85%;background:var(--base)"><span>85 s</span></div></div>
<div class="grp"><div class="gname">Spark, 8 cores</div><div class="b" style="width:20%;background:var(--fair)"><span>20 s: 4.3× faster, same answer</span></div></div></div></section>
</main>
<script>
const C = __CHOICES__;
const $ = s => document.querySelector(s);
document.querySelectorAll("nav button").forEach(b => b.onclick = () => {
  document.querySelectorAll("nav button,.tab").forEach(e => e.classList.remove("on"));
  b.classList.add("on"); $("#" + b.dataset.t).classList.add("on"); });
const busy = (el, msg) => el.innerHTML = `<p class="note" style="margin-top:20px"><span class="spin"></span>${msg}</p>`;
const pct = x => (x * 100).toFixed(0) + "%";

// ---- tab 1
const FIELDS = [["AGEP","Age"],["SCHL","Education"],["OCCP","Job"],["COW","Employer"],["WKHP","Hours worked per week"],["MAR","Marital status"],["POBP","Born in"],["RELP","Role in household"]];
$("#form").innerHTML = FIELDS.map(([k, l]) => `<label>${l}</label>` + (C[k]
  ? `<select id="${k}">${Object.entries(C[k]).map(([v, t]) => `<option value="${v}">${t}</option>`).join("")}</select>`
  : `<input id="${k}" type="number" value="${k == "AGEP" ? 35 : 40}">`)).join("") +
  `<button class="btn" id="askBtn">Ask the AI</button><button class="btn light" id="realBtn">Load a real applicant</button>`;
async function ask(truth) {
  const person = Object.fromEntries(FIELDS.map(([k]) => [k, $("#" + k).value]));
  const r = await (await fetch("/api/predict", {method: "POST", body: JSON.stringify(person)})).json();
  $("#answers").innerHTML = [["income-baseline", "First AI", "var(--base)"], ["income-fair", "Fair version", "var(--fair)"]].map(([m, n, c]) =>
    `<div class="ans"><h3>${n} says</h3><div class="verdict ${r[m].yes ? "yes" : "no"}">${r[m].yes ? "✓ Yes, likely over $50k" : "✗ No"}</div>
     <small>Confidence it's a yes: ${pct(r[m].chance)}</small><div class="meter"><i style="width:${r[m].chance * 100}%;background:${c}"></i></div></div>`).join("");
  $("#truth").innerHTML = truth ? `<div class="truth">This is a <b>real person</b> from the US Census (${truth.sex}, census race group "${truth.race}", never shown to the AI).
     Do they really earn over $50k? <b class="${truth.truth ? "yes" : "no"}">${truth.truth ? "YES" : "NO"}</b></div>` :
     `<p class="note">Change any answer and press "Ask the AI" again. Try changing the job or the hours.</p>`;
}
$("#askBtn").onclick = () => ask(null);
$("#realBtn").onclick = async () => { const p = await (await fetch("/api/sample")).json();
  FIELDS.forEach(([k]) => $("#" + k).value = p[k]); ask(p); };
ask(null);

// ---- tab 2
$("#auditBtn").onclick = async () => { $("#auditBtn").disabled = true; busy($("#auditOut"), "Running both AIs on 39,133 applicants…");
  const r = await (await fetch("/api/audit")).json(); $("#auditBtn").disabled = false;
  const b = r["income-baseline"], f = r["income-fair"];
  const bars = b.groups.map((g, i) => `<div class="grp"><div class="gname">${g.group}<br><small style="color:var(--muted)">${g.n.toLocaleString()} people</small></div><div>
     <div class="b" style="width:${g.rate * 150}%;background:var(--base)"><span>${pct(g.rate)}</span></div>
     <div class="b" style="width:${f.groups[i].rate * 150}%;background:var(--fair);margin-top:3px"><span>${pct(f.groups[i].rate)}</span></div></div></div>`).join("");
  $("#auditOut").innerHTML = `<p class="note">Scored ${r.people.toLocaleString()} applicants with both AIs in ${r.seconds.toFixed(1)} s.</p>
   <div class="stats"><div class="stat"><small>First AI: fairness score</small><b class="no">${b.di.toFixed(2)}</b><small>needs at least 0.80</small></div>
   <div class="stat"><small>Fair version: fairness score</small><b class="yes">${f.di.toFixed(2)}</b><small>needs at least 0.80</small></div>
   <div class="stat"><small>First AI: accuracy</small><b>${pct(b.accuracy)}</b></div><div class="stat"><small>Fair version: accuracy</small><b>${pct(f.accuracy)}</b><small>the price of fairness</small></div></div>
   <div class="panel"><div class="legend" style="margin-bottom:10px"><span><i class="sw" style="background:var(--base)"></i>First AI</span><span><i class="sw" style="background:var(--fair)"></i>Fair version</span>
   <span style="color:var(--muted)">How often each race group gets a "yes"</span></div>${bars}
   <p class="note">Neither AI sees race. The first AI still learned it from stand-ins like job, birthplace and hours worked.</p></div>`; };

// ---- tab 3
const MODELS = [["income-fair", "Fair loan AI", "high risk"], ["income-baseline", "First loan AI", "high risk"], ["taxi-eta", "Taxi trip-time AI", "low risk, other team"]];
let chosen = "income-fair";
$("#models").innerHTML = MODELS.map(([id, n, s]) => `<button class="chip ${id == chosen ? "on" : ""}" data-id="${id}">${n}<small>${s}</small></button>`).join("");
document.querySelectorAll(".chip").forEach(c => c.onclick = () => { document.querySelectorAll(".chip").forEach(x => x.classList.remove("on")); c.classList.add("on"); chosen = c.dataset.id; });
const Q = {risk: "How risky is it? (EU AI Act)", documentation: "Is the paperwork complete?", lineage: "Can we trace where it came from?", fairness: "Is it fair to every group?", approval: "Did someone independent approve it?"};
$("#gateBtn").onclick = async () => { const breaks = [...document.querySelectorAll(".breaks input:checked")].map(i => i.value);
  $("#gateBtn").disabled = true; busy($("#gateOut"), "Running the 5 checks…");
  const r = await (await fetch("/api/gate", {method: "POST", body: JSON.stringify({model: chosen, breaks})})).json(); $("#gateBtn").disabled = false;
  $("#gateOut").innerHTML = `<small style="color:var(--muted)">${r.id} · ${r.tier} risk: ${r.basis}</small>
   <div class="decision ${r.ok ? "yes" : "no"}">${r.ok ? "✓ Released" : "✗ Blocked"}</div>
   <div class="lights">${Object.entries(Q).map(([g, q]) => `<div class="${r.status[g] == "n/a" ? "na" : r.status[g]}"><span class="dot">${{pass: "✓", fail: "✗", "n/a": "–"}[r.status[g]]}</span>${q}${r.status[g] == "n/a" ? " (not needed at this risk level)" : ""}</div>`).join("")}</div>
   ${r.why.length ? `<ul class="why">${r.why.map(w => `<li>${w}</li>`).join("")}</ul><code>${r.detail.join("\n")}</code>` : ""}
   ${r.fairness.RAC1P ? `<p class="note">Fairness score by race: ${r.fairness.RAC1P.di.toFixed(2)} (needs 0.80) · by sex: ${r.fairness.SEX.di.toFixed(2)}</p>` : ""}
   <p class="note">Checked in ${r.seconds.toFixed(1)} s.</p>`; };

// ---- tab 4
$("#scaleBtn").onclick = async () => { $("#scaleBtn").disabled = true; busy($("#scaleOut"), "Spark is scoring and auditing 1.6 million applications…");
  const r = await (await fetch("/api/scale")).json(); $("#scaleBtn").disabled = false;
  $("#scaleOut").innerHTML = `<div class="stats"><div class="stat"><small>Applications scored and audited</small><b>${r.rows.toLocaleString()}</b></div>
   <div class="stat"><small>Time taken</small><b>${r.seconds.toFixed(1)} s</b></div><div class="stat"><small>Fairness score (first AI)</small><b class="no">${r.di.toFixed(2)}</b><small>same verdict as the small test: blocked</small></div></div>`; };
</script></body></html>"""


if __name__ == "__main__":
    threading.Thread(target=lambda: spark().range(1).count(), daemon=True).start()  # warm up Spark in the background
    for port in range(8600, 8700):  # first free port
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    print(f"Live demo running at http://localhost:{port}  (Ctrl+C to stop)", flush=True)
    webbrowser.open(f"http://localhost:{port}")
    server.serve_forever()
