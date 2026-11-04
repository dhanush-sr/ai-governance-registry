"""Build demo/index.html, a one-page visual walkthrough for presenting.

    uv run python -m demo.build_page

Every number comes from running the real gate (and real data) at build time,
including the two "cheating" scenarios, which run on a throwaway copy.
"""
import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from governance import gate
from governance.fairness import spark

OUT = Path("demo/index.html")
RACE = {1: "White", 2: "Black", 3: "American Indian", 5: "Native tribe (other)", 6: "Asian",
        7: "Pacific Islander", 8: "Some other race", 9: "Two or more races"}
SCHOOL = {16: "high-school diploma", 19: "some college", 20: "associate degree", 21: "bachelor's degree",
          22: "master's degree", 23: "professional degree", 24: "doctorate"}
WORKER = {1: "a private company", 2: "a non-profit", 3: "local government", 4: "state government",
          5: "federal government", 6: "self-employed", 7: "own business"}
CHECKS = [("risk", "How risky is it?", "Worked out from what it's used for, using EU law"),
          ("documentation", "Is the paperwork complete?", "Every section the rule book requires"),
          ("lineage", "Can we trace where it came from?", "Data and results match their saved fingerprints"),
          ("fairness", "Is it fair to every group?", "Re-measured by the checker, never self-reported"),
          ("approval", "Did someone independent approve it?", "Signed off by a person who didn't build it")]
PLAIN = {"risk": "Its risk level is wrong, or it's banned.",
         "documentation": "Its paperwork is incomplete.",
         "lineage": "We can't prove where it came from, or its evidence was changed.",
         "fairness": "It treats groups of people too differently.",
         "approval": "No independent person has signed it off."}
ABOUT = {"income-fair": "Loan pre-screen, built with a fairness rule",
         "income-baseline": "Loan pre-screen, built the \"obvious\" way",
         "taxi-eta": "Taxi trip-time estimate (another team)"}


def check(model_dir, policy, datasets):
    r = gate.check_model(model_dir, policy, datasets, spark)
    return {"id": r["id"], "tier": r["tier"], "ok": r["deployable"], "status": r["status"],
            "why": [PLAIN[g] for g in gate.GATES if r["status"][g] == "fail"],
            "detail": [m.replace("`", "") for g in gate.GATES for m in r["fails"][g]]}


def collect():
    policy = gate.load("registry/policy.yaml")
    datasets = {p.stem: gate.load(p) for p in Path("registry/datasets").glob("*.yaml")}
    models = [check(Path("registry/models") / m, policy, datasets) for m in ["income-fair", "income-baseline", "taxi-eta"]]

    tmp = Path(tempfile.mkdtemp())
    shutil.copytree("registry", tmp / "registry")
    gate.REGISTRY = tmp / "registry"
    ev = gate.REGISTRY / "models/income-baseline/eval_predictions.parquet"
    e = pd.read_parquet(ev)
    e["y_pred"] = e["y_true"]
    e.to_parquet(ev)
    cheat_edit = check(gate.REGISTRY / "models/income-baseline", policy, datasets)
    card = gate.REGISTRY / "models/income-fair/card.yaml"
    card.write_text(card.read_text().replace("status: approved", "status: requested"))
    cheat_sign = check(gate.REGISTRY / "models/income-fair", policy, datasets)
    gate.REGISTRY = Path("registry")
    shutil.rmtree(tmp)

    df = pd.read_parquet("data/acs_income_ca_2018.parquet")
    _, te = train_test_split(df, test_size=0.2, random_state=0, stratify=df.label)
    base = pd.read_parquet("registry/models/income-baseline/eval_predictions.parquet")
    fair = pd.read_parquet("registry/models/income-fair/eval_predictions.parquet")
    te = te.reset_index(drop=True).assign(base=base.y_pred, fair=fair.y_pred)
    p = te[(te.RAC1P == 8) & (te.label == 1) & (te.base == 0) & (te.fair == 1) & (te.SCHL >= 21)].iloc[0]
    sizes = te.RAC1P.value_counts()
    rates = te.groupby("RAC1P")[["base", "fair"]].mean().loc[[k for k in RACE if sizes.get(k, 0) >= 100]]
    acc = {m: json.loads(Path(f"registry/models/{m}/run.json").read_text())["evaluation"]["metrics"]["accuracy"]
           for m in ["income-baseline", "income-fair"]}
    bench = pd.read_csv("results/benchmark.csv")
    big = bench[bench.rows == bench.rows.max()]
    return {
        "models": models, "cheat_edit": cheat_edit, "cheat_sign": cheat_sign, "acc": acc, "n_test": len(te),
        "person": {"age": int(p.AGEP), "school": SCHOOL.get(int(p.SCHL), "a degree"), "hours": int(p.WKHP),
                   "work": WORKER.get(int(p.COW), "a job"), "sex": "woman" if p.SEX == 2 else "man", "race": RACE[int(p.RAC1P)]},
        "rates": [{"group": RACE[k], "base": float(r.base), "fair": float(r.fair), "n": int(sizes[k])} for k, r in rates.iterrows()],
        "pandas": float(big[big.engine == "pandas"].seconds.iloc[0]),
        "spark": {int(c): float(s) for c, s in big[big.engine == "spark"][["cores", "seconds"]].values},
        "rows_big": int(bench.rows.max()), "rows_base": int(len(df)),
    }


def lights(status):
    icon = {"pass": ("ok", "✓"), "fail": ("bad", "✗"), "n/a": ("na", "–")}
    return "".join(f'<div class="light {icon[status[g]][0]}"><span>{icon[status[g]][1]}</span>{q}</div>'
                   for g, q, _ in CHECKS)


def model_card(m, title=None):
    verdict = '<div class="verdict ok">✓ Released</div>' if m["ok"] else '<div class="verdict bad">✗ Blocked</div>'
    why = "".join(f"<li>{w}</li>" for w in m["why"])
    detail = "<br>".join(m["detail"])
    return f"""<div class="card">
      <div class="cardhead"><div><b>{title or ABOUT[m['id']]}</b><small>{m['id']} · {m['tier']} risk</small></div>{verdict}</div>
      <div class="lights">{lights(m['status'])}</div>
      {f'<ul class="why">{why}</ul><details><summary>Exact reason from the checker</summary><code>{detail}</code></details>' if m['why'] else '<p class="why ok">Passed every check that applies to it.</p>'}
    </div>"""


def page(d):
    p = d["person"]
    lo, hi = min(r["base"] for r in d["rates"]), max(r["base"] for r in d["rates"])
    flo, fhi = min(r["fair"] for r in d["rates"]), max(r["fair"] for r in d["rates"])
    scale = 1 / 0.55
    bars = "".join(f"""<div class="grp"><div class="gname">{r['group']}<small>{r['n']:,} people</small></div><div class="gbars">
        <div class="b base" style="width:{r['base']*scale*100:.1f}%"><span>{r['base']*100:.0f}%</span></div>
        <div class="b fair" style="width:{r['fair']*scale*100:.1f}%"><span>{r['fair']*100:.0f}%</span></div></div></div>""" for r in d["rates"])
    sp8 = d["spark"][max(d["spark"])]
    speed = "".join(f"""<div class="srow"><div class="slabel">{lab}</div><div class="sbar {cls}" style="width:{s/d['pandas']*100:.1f}%"><span>{s:.0f} s</span></div></div>"""
                    for lab, cls, s in [("Normal way (1 core)", "base", d["pandas"])] +
                    [(f"Spark, {c} core{'s' if c > 1 else ''}", "fair", s) for c, s in sorted(d["spark"].items())])
    checks = "".join(f'<div class="check"><div class="num">{i}</div><b>{q}</b><p>{hint}</p></div>'
                     for i, (_, q, hint) in enumerate(CHECKS, 1))
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Governance Registry with Deployment Gates</title><style>
:root {{ --ink:#0b0b0b; --muted:#52514e; --line:#e2e1dc; --bg:#fcfcfb; --base:#eb6834; --fair:#2a78d6; --ok:#0a7d0a; --bad:#c62f2f; }}
* {{ box-sizing:border-box; margin:0; }}
html {{ scroll-snap-type:y mandatory; scroll-behavior:smooth; }}
body {{ font:20px/1.5 -apple-system,"Helvetica Neue",Arial,sans-serif; color:var(--ink); background:var(--bg); }}
section {{ min-height:100vh; scroll-snap-align:start; padding:7vh 8vw 6vh; display:flex; flex-direction:column; justify-content:center; border-bottom:1px solid var(--line); }}
.step {{ color:var(--fair); font-weight:700; letter-spacing:.06em; text-transform:uppercase; font-size:16px; }}
h1 {{ font-size:clamp(36px,4.6vw,64px); line-height:1.1; margin:.3em 0 .4em; }}
h2 {{ font-size:clamp(30px,3.4vw,46px); line-height:1.15; margin:.25em 0 .6em; }}
.lead {{ font-size:clamp(20px,1.7vw,26px); color:var(--muted); max-width:62ch; }}
small {{ display:block; color:var(--muted); font-size:14px; font-weight:400; }}
.flow {{ display:flex; align-items:stretch; gap:14px; margin:10px 0 34px; flex-wrap:wrap; }}
.fbox {{ flex:1; min-width:170px; background:white; border:2px solid var(--line); border-radius:16px; padding:18px 20px; font-size:18px; }}
.fbox b {{ display:block; font-size:20px; margin-bottom:4px; }}
.arrow {{ align-self:center; font-size:30px; color:var(--muted); }}
.fbox.end {{ border-color:var(--ink); }}
.checks {{ display:grid; grid-template-columns:repeat(5,1fr); gap:14px; }}
.check {{ background:white; border:2px solid var(--line); border-radius:16px; padding:18px; }}
.check .num {{ width:34px; height:34px; border-radius:50%; background:var(--fair); color:white; display:grid; place-items:center; font-weight:700; margin-bottom:10px; }}
.check b {{ font-size:19px; }} .check p {{ font-size:15px; color:var(--muted); margin-top:6px; }}
.person {{ display:grid; grid-template-columns:1.2fr 1fr; gap:30px; align-items:center; }}
.pcard {{ background:white; border:2px solid var(--line); border-radius:20px; padding:30px 34px; font-size:24px; }}
.pcard .truth {{ margin-top:18px; padding-top:18px; border-top:1px solid var(--line); font-weight:700; }}
.ai {{ display:flex; flex-direction:column; gap:16px; }}
.aianswer {{ border-radius:16px; padding:22px 26px; font-size:24px; background:white; border:2px solid var(--line); }}
.aianswer b {{ font-size:30px; }} .no {{ color:var(--bad); }} .yes {{ color:var(--ok); }}
.legend {{ display:flex; gap:26px; margin-bottom:14px; font-size:17px; }}
.sw {{ display:inline-block; width:16px; height:16px; border-radius:4px; vertical-align:-2px; margin-right:7px; }}
.grp {{ display:grid; grid-template-columns:230px 1fr; gap:16px; align-items:center; margin:6px 0; }}
.gname {{ font-size:17px; text-align:right; }} .gname small {{ font-size:13px; }}
.gbars {{ display:flex; flex-direction:column; gap:3px; }}
.b, .sbar {{ height:20px; border-radius:0 4px 4px 0; position:relative; min-width:2px; }}
.b span, .sbar span {{ position:absolute; left:calc(100% + 8px); top:-2px; font-size:15px; color:var(--ink); white-space:nowrap; }}
.base {{ background:var(--base); }} .fair {{ background:var(--fair); }}
.rule {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; margin-top:26px; }}
.rulebox {{ background:white; border:2px solid var(--line); border-radius:16px; padding:18px 22px; font-size:19px; }}
.big {{ font-size:40px; font-weight:800; }}
.cards {{ display:grid; grid-template-columns:repeat(3,1fr); gap:20px; }}
.card {{ background:white; border:2px solid var(--line); border-radius:18px; padding:22px; display:flex; flex-direction:column; gap:14px; }}
.cardhead {{ display:flex; justify-content:space-between; align-items:flex-start; gap:10px; }}
.cardhead b {{ font-size:19px; }}
.verdict {{ font-weight:800; padding:6px 12px; border-radius:10px; white-space:nowrap; font-size:17px; }}
.verdict.ok {{ background:#e3f4e3; color:var(--ok); }} .verdict.bad {{ background:#fbe4e4; color:var(--bad); }}
.lights {{ display:flex; flex-direction:column; gap:6px; }}
.light {{ font-size:15px; display:flex; align-items:center; gap:10px; color:var(--muted); }}
.light span {{ width:24px; height:24px; border-radius:50%; display:grid; place-items:center; font-weight:800; font-size:14px; flex:none; }}
.light.ok span {{ background:#e3f4e3; color:var(--ok); }} .light.bad span {{ background:#fbe4e4; color:var(--bad); }}
.light.bad {{ color:var(--ink); font-weight:600; }} .light.na span {{ background:#efeeea; color:#8a8984; }}
.why {{ font-size:16px; padding-left:20px; color:var(--bad); }} p.why.ok {{ color:var(--ok); padding:0; }}
details {{ font-size:13px; color:var(--muted); }} summary {{ cursor:pointer; }} code {{ display:block; margin-top:6px; font-size:12px; }}
.two {{ display:grid; grid-template-columns:1fr 1fr; gap:22px; }}
.scenario {{ font-size:18px; color:var(--muted); margin-bottom:10px; }}
.srow {{ display:grid; grid-template-columns:230px 1fr; gap:16px; align-items:center; margin:12px 0; }}
.slabel {{ text-align:right; font-size:18px; }} .sbar {{ height:30px; }} .sbar span {{ top:3px; font-size:18px; }}
.links a {{ display:inline-block; margin:8px 12px 8px 0; padding:14px 22px; border-radius:12px; background:var(--ink); color:white; text-decoration:none; font-weight:600; }}
.files {{ margin-top:30px; display:grid; grid-template-columns:auto 1fr; gap:6px 24px; font-size:17px; }}
.files code {{ display:inline; font-size:16px; margin:0; color:var(--ink); }}
.note {{ font-size:15px; color:var(--muted); margin-top:18px; }}
@media (max-width:1000px) {{ .checks, .cards, .two, .person, .rule {{ grid-template-columns:1fr; }} }}
</style></head><body>

<section>
  <div class="step">AI Governance Registry with Deployment Gates</div>
  <h1>Can this AI be trusted<br>with people's loans?</h1>
  <p class="lead">A bank wants an AI to sort loan applications. EU law calls that <b>high risk</b>: before it's used, it must be documented, fair, traceable and approved.<br><br>
  Our system checks every AI <b>automatically</b>, and blocks any that fails.</p>
</section>

<section>
  <div class="step">How it works</div>
  <h2>Every AI goes through the same checkpoint</h2>
  <div class="flow">
    <div class="fbox"><b>1 · Team builds an AI</b>and fills in its info sheet: what it's for, its limits, who approved it</div><div class="arrow">→</div>
    <div class="fbox"><b>2 · Uploads it to GitHub</b>the training script also saves a "seal" (fingerprint) of its test results</div><div class="arrow">→</div>
    <div class="fbox"><b>3 · Our checker runs</b>automatically, on every change, using a written rule book</div><div class="arrow">→</div>
    <div class="fbox end"><b>4 · Decision</b><span class="yes">✓ Released</span> or <span class="no">✗ Blocked</span>, with the reason</div>
  </div>
  <div class="checks">{checks}</div>
  <p class="note">The rules come from the EU AI Act and an existing fairness standard (the "four-fifths rule"), not from us. The checker never takes the team's word: it works things out itself.</p>
</section>

<section>
  <div class="step">A real example</div>
  <h2>Meet one applicant</h2>
  <div class="person">
    <div class="pcard">A real, anonymised person from the US Census (California, 2018):<br><br>
      <b>{p['age']}-year-old {p['sex']}</b>, {p['school']}, works {p['hours']} hours a week at {p['work']}.
      <div class="truth">Really earns over $50,000? <span class="yes">YES</span></div></div>
    <div class="ai">
      <div class="aianswer">The bank's first AI says<br><b class="no">✗ No</b></div>
      <div class="aianswer">The fair version says<br><b class="yes">✓ Yes</b></div>
      <p class="note">Neither AI is ever shown her race (census group: "{p['race']}") or sex. So is this bad luck, or a pattern?</p>
    </div>
  </div>
</section>

<section>
  <div class="step">The pattern</div>
  <h2>How often each group gets a "yes"</h2>
  <div class="legend"><span><i class="sw base"></i>First AI (the "obvious" way)</span><span><i class="sw fair"></i>Fair version</span><span style="color:var(--muted)">{d['n_test']:,} applicants the AI never saw while learning</span></div>
  {bars}
  <div class="rule">
    <div class="rulebox"><b>The rule:</b> the least-approved group must get at least <b>80%</b> of the most-approved group's rate.</div>
    <div class="rulebox">First AI: {lo*100:.0f}% ÷ {hi*100:.0f}% = <span class="big no">{lo/hi:.2f}</span> &nbsp; Fair: {flo*100:.0f}% ÷ {fhi*100:.0f}% = <span class="big yes">{flo/fhi:.2f}</span></div>
  </div>
  <p class="note">Hiding race didn't remove the bias: job type, birthplace and hours worked act as stand-ins. The price of fairness: accuracy drops from {d['acc']['income-baseline']*100:.1f}% to {d['acc']['income-fair']*100:.1f}%.</p>
</section>

<section>
  <div class="step">The checker's decision</div>
  <h2>Three AI programs submitted. One released.</h2>
  <div class="cards">{''.join(model_card(m) for m in d['models'])}</div>
  <p class="note">A manager had already approved the first AI. It's blocked anyway. Low-risk programs (the taxi one) skip the fairness and sign-off checks; the rules scale with risk.</p>
</section>

<section>
  <div class="step">Can someone cheat?</div>
  <h2>We tried. It caught both.</h2>
  <div class="two">
    <div><p class="scenario">The team overwrites its test results with perfect ones, so the unfair AI looks fair.</p>{model_card(d['cheat_edit'], 'First AI, with edited results')}</div>
    <div><p class="scenario">The manager withdraws sign-off on the good AI.</p>{model_card(d['cheat_sign'], 'Fair AI, approval withdrawn')}</div>
  </div>
  <p class="note">Why the first one fails: when the AI was trained, a fingerprint of its results was saved. Change even one character and the fingerprint no longer matches.</p>
</section>

<section>
  <div class="step">Bank-sized volumes</div>
  <h2>Checking {d['rows_big']/1e6:.0f} million applications</h2>
  <p class="lead" style="margin-bottom:20px">A real bank scores millions of applications, so the fairness check runs on <b>Apache Spark</b>, which splits the work across many processor cores at once.</p>
  {speed}
  <p class="note">{d['pandas']/sp8:.1f}× faster on 8 cores, and the fairness score came out identical at every size. The {d['rows_big']/1e6:.0f} million rows are the {d['rows_base']:,} census people copied with new IDs, which tests speed, not new findings.</p>
</section>

<section>
  <div class="step">See it for real</div>
  <h2>It runs on GitHub for every change</h2>
  <p class="lead">One check per AI program: green means released, red means blocked, with the reason attached.</p>
  <div class="links"><a href="https://github.com/dhanush-sr/ai-governance-registry/actions">GitHub checks →</a><a href="https://github.com/dhanush-sr/ai-governance-registry/blob/main/REGISTRY.md">Full registry report →</a><a href="https://github.com/dhanush-sr/ai-governance-registry">Code →</a></div>
  <div class="files">
    <b>The rule book</b><span><code>registry/policy.yaml</code></span>
    <b>Info sheets</b><span><code>registry/models/*/card.yaml</code> · <code>registry/datasets/*.yaml</code></span>
    <b>Fingerprints</b><span><code>registry/models/*/run.json</code></span>
    <b>The checker</b><span><code>governance/gate.py</code> (5 checks) · <code>governance/risk.py</code> (EU law) · <code>governance/fairness.py</code> (Spark)</span>
    <b>GitHub setup</b><span><code>.github/workflows/governance-gate.yml</code></span>
  </div>
</section>
</body></html>"""


if __name__ == "__main__":
    OUT.write_text(page(collect()))
    print("wrote", OUT)
