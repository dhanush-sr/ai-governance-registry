# Project context and handoff

Topic 32, **AI Governance Registry with Deployment Gates**. Submission **Option A** (public GitHub repo).
Last updated 2026-11-02. About 35% done: the design is fixed and the core code is written, but **none of the gate code has been run yet**.

---

## 1. The assignment

**Case study.** A company needs every model to be documented, risk-rated and approved before release.

**What to do.** Build a registry of model and dataset cards with a risk classification based on the NIST AI RMF or the EU AI Act. Add CI checks that block deployment when fairness, documentation or lineage requirements are not met.

**Topic deliverables ("Submit"):**
1. Registry
2. CI gate configuration
3. Three example models that pass or fail

**Course-wide requirements:**
- Both components must be backed by *measured evidence, not description*:
  - **Big data systems** (scale, distributed processing, streaming, or storage design)
  - **Responsible AI** (fairness, privacy, transparency, safety, or governance)
- Only public, synthetic or openly licensed data. No real personal data.
- **Option A deliverables:**
  - README with an architecture diagram and setup steps
  - Reproducible pipeline (scripts or Docker Compose)
  - Results notebook
  - 3 to 5 page technical report
  - Model or data card
  - 3-minute demo video (the owner is doing this one later; not our job for now)

The course datasets doc (`Project_Datasets.docx`, gitignored) says this for topic 32: *"Three models trained on datasets from other tracks; example model cards on Hugging Face. Reuse existing work and document it properly."*

---

## 2. Decisions already made (don't re-open them)

| Decision | Choice | Why |
|---|---|---|
| Framework | **EU AI Act** tiers (unacceptable / high / limited / minimal) | The brief asks for NIST *or* EU. EU tiers are rule-based and easy to automate. A NIST mapping table in the report is optional. |
| Registry form | YAML cards in git, JSON Schema for structure, `policy.yaml` for per-tier rules | The owner asked for "best, not overcomplicated". There's no web UI or database. The generated `REGISTRY.md` is the human view. |
| CI | GitHub Actions, public repo on the owner's GitHub account | |
| Big data angle | **Spark fairness evaluation at scale.** The CI gate and the at-scale audit run the *same* `governance/fairness.py` code | The governance topic has no natural big data, so the gate's own fairness job is what we scale and benchmark. |
| Data | ACS 2018 PUMS **California only** (Folktables ACSIncome, 195,665 rows) and NYC TLC yellow taxi trips for **Jan 2023** (3.07 M rows) | Both are in the course doc. The national ACS file (562 MB) was too slow to download (see §7). Scale comes from replicating CA with new IDs, which the course doc explicitly allows. |
| Pipeline | Plain Python scripts plus a Makefile, no Docker | The Docker daemon isn't running, and the disk is tight. |
| Python | 3.12 via `uv` (system Python is 3.14, which PySpark isn't tested on). PySpark 4.2.1 needs **Java 17** | |

### The three example models

All three are designed so that each gate visibly blocks something.

| Model id | Story | Expected gate result |
|---|---|---|
| `income-fair` | Credit pre-screen (Annex III 5(b), **high-risk**). SEX and RAC1P are *not* inputs, and the model is trained with fairlearn `ExponentiatedGradient` under a `DemographicParity(0.01)` constraint on SEX × RAC1P. Approved. | ✅ **DEPLOY** |
| `income-baseline` | Same task and inputs, plain HistGradientBoosting with "fairness through unawareness" only. A human approved it anyway. Narrative: *human sign-off alone didn't catch it; the automated gate did.* | ❌ **fairness**: RAC1P disparate impact 0.30 and equal-opportunity gap 0.33 |
| `taxi-eta` | Trip-duration regressor (**minimal risk**). Its training dataset was never registered, and it has a free-form `known_issues` field instead of `limitations`. | ❌ **documentation** (schema error plus missing `limitations`) and ❌ **lineage** (no dataset card for `nyc-tlc-yellow-2023-01`) |

Owner and approver emails are fictional (`@example.org`). No real people.

---

## 3. Architecture (what the gate does)

```
registry/
  policy.yaml                      per-tier rules = the "CI gate configuration" (together with the workflow YAML)
  schemas/*.schema.json            card structure (additionalProperties:false catches typo'd fields)
  datasets/<id>.yaml               dataset cards (datasheets)
  models/<id>/card.yaml            human-written model card
  models/<id>/run.json             WRITTEN BY PIPELINE: code commit, dataset sha256s, artifact sha256, metrics
  models/<id>/eval_predictions.parquet   WRITTEN BY PIPELINE: y_true, y_pred, SEX, RAC1P (the fairness evidence)
governance/
  risk.py       classify(card["risk"]) -> (tier, legal basis). Use case -> Art. 5 / Annex III / Art. 50 lookup
  fairness.py   spark(); group_counts(df, keys) = one Spark scan; metrics(counts, attr, min_group_size)
  gate.py       CLI: python -m governance.gate [MODEL_ID ...] -> Markdown report on stdout, exit 1 if any blocked
pipeline/
  download_data.py   downloads both sources, writes data/manifest.json with source sha256s
  train_models.py    trains the 3 models, writes run.json + eval_predictions.parquet per model
```

**The five gates in `gate.py`**, with rules from `policy.yaml`:
1. **risk.** The tier is *computed* from `risk.use_case`, and `risk.declared_tier` must match it, so nobody can label a credit model "minimal". `unacceptable` is never deployable.
2. **documentation.** The card must pass the JSON Schema, plus the common `required_fields` and the tier's `required_fields` (dotted paths; empty values count as missing).
3. **lineage.**
   - `run.json` exists.
   - The training commit exists in git (`git cat-file -e`) and the tree was not dirty.
   - Every training dataset has a dataset card whose `source.sha256` equals the hash recorded at training time, and that card passes its schema and required fields.
   - The `eval_predictions.parquet` sha256 is unchanged since training.
4. **fairness** (high tier only). The gate **recomputes** fairness from `eval_predictions.parquet` with Spark, for every attribute in `sensitive_attributes`, and checks:
   - disparate impact ≥ 0.8 (four-fifths rule);
   - equal-opportunity (TPR) gap ≤ 0.1.
   
   Groups with fewer than 100 rows are excluded from disparate impact, and groups with fewer than 100 positives are excluded from the TPR gap. They are still reported. Spark is started only if some model needs this check.
5. **approval** (high tier only). `approval.status == approved`, and the approver is named and differs from the owner.

A gate is `n/a` when the tier doesn't require it. A model deploys only if no gate shows `fail`.

**Lineage hashes are of the *source* files** (the census zip and the TLC parquet), which don't change upstream. That keeps reruns on another machine consistent with the committed dataset cards.

---

## 4. Current state of the files

| File | State |
|---|---|
| `pyproject.toml`, `uv.lock`, `.gitignore` | Done. `uv sync` works. |
| `pipeline/download_data.py` | **Done and run.** `data/` holds both parquets and `manifest.json`. |
| `registry/policy.yaml`, `registry/schemas/*.json` | Written. |
| `registry/models/*/card.yaml` (all 3) | Written. |
| `governance/risk.py`, `fairness.py`, `gate.py` | Written, **never executed**. Expect small bugs. |
| `pipeline/train_models.py` | Written, **never executed**. The same model code was validated in a scratch experiment (§5). |
| `registry/datasets/acs-income-ca-2018.yaml` | **Missing.** Draft below. |
| tests, benchmark, CI workflow, notebook, README, report, Makefile | **Not started.** |
| git | `git init` done on `main`, **no commits yet**. |

Source hashes from `data/manifest.json`:
- `acs-income-ca-2018`: `6194ade52892ddb0e41903f7d3b49399842f08eeebd42e6b19ee8e417affbf79` (195,665 rows)
- `nyc-tlc-yellow-2023-01`: `32df6f67578fa86c484a6b5ef23a5281992ff085521082340b0f9e5889e9a572` (3,066,766 rows)

### Draft: `registry/datasets/acs-income-ca-2018.yaml`

Do **not** create a card for the TLC dataset. Its absence is what makes `taxi-eta` fail lineage.

```yaml
id: acs-income-ca-2018
name: Folktables ACSIncome, California 2018
description: >
  Person records from the US Census Bureau 2018 ACS 1-year PUMS for California,
  filtered and labelled with the Folktables ACSIncome task: predict whether
  personal income exceeds USD 50,000.
source:
  url: https://www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year/csv_pca.zip
  license: Public domain (US Government work, 17 U.S.C. 105); Census Bureau terms of use
  sha256: 6194ade52892ddb0e41903f7d3b49399842f08eeebd42e6b19ee8e417affbf79
  retrieved: 2026-10-29
collection: >
  ACS is a continuous, mandatory household survey. PUMS is the public-use
  microdata sample (about 1% of the population per year).
preprocessing: >
  Folktables adult_filter (AGEP > 16, PINCP > 100, WKHP > 0, PWGTP >= 1);
  10 features AGEP COW SCHL MAR OCCP POBP RELP WKHP SEX RAC1P; label PINCP > 50000;
  missing values coded -1. Script: pipeline/download_data.py
personal_data: >
  De-identified public-use microdata: no names or addresses, geography coarsened
  to PUMAs of 100k+ people, Census disclosure avoidance applied. Race is
  special-category data (GDPR Art. 9); used only to measure and correct bias.
  Never attempt re-identification.
sensitive_attributes: [SEX, RAC1P]
known_biases: >
  Survey non-response differs by group. SEX is binary as recorded in 2018.
  RAC1P codes: 1 White, 2 Black, 3 American Indian, 4 Alaska Native,
  5 AIAN tribe specified, 6 Asian, 7 Native Hawaiian/Pacific Islander,
  8 Some other race, 9 Two or more races.
rows: 195665
```

---

## 5. Results so far (scratch experiment, same code as `train_models.py`)

California hold-out set, 39,133 rows:

| | Accuracy | SEX disparate impact | SEX TPR gap | RAC1P disparate impact | RAC1P TPR gap | Gate |
|---|---|---|---|---|---|---|
| `income-baseline` | **0.825** | 0.802 | 0.015 | **0.299** | **0.331** | fails on race |
| `income-fair` | **0.747** | 0.977 | 0.075 | 0.885 | 0.054 | passes |

- **Fairness costs about 7.8 accuracy points.** That's a result to discuss in the report, not hide.
- **The baseline's SEX disparate impact is 0.802, right on the threshold.** It fails on race, not sex.
- **The baseline's low race disparate impact comes from RAC1P = 8 ("Some other race").** That group's selection rate is 0.146 against 0.487 for Asian respondents. In California, many Hispanic/Latino respondents choose "Some other race"; treat this as a hypothesis in the report, not a fact.
- **Some test groups are tiny:** RAC1P 4 has 3 rows (excluded from disparate impact). RAC1P 5 (114 rows, 26 positives) and 7 (137 rows, 39 positives) are included in disparate impact but excluded from the TPR gap.
- **Training is slow:** the `ExponentiatedGradient` fit takes **about 5 min** (101 base predictors), the baseline about 1.5 s. Expect a large `income-fair.pkl`; check its size given the disk situation.
- **Predictions are randomized:** `ExponentiatedGradient.predict` is randomized, so always pass `random_state=0`.
- **Warnings are noise:** fairlearn prints pandas FutureWarnings, which are harmless.

---

## 6. Next steps, in order

1. **Write the ACS dataset card** using the draft in §4.
2. **Add `tests/test_governance.py`.** One small file with plain asserts:
   - The risk classifier returns each tier: `social_scoring` → unacceptable, `creditworthiness` → high, a chatbot (`interacts_with_people`) → limited, `other` → minimal. An unknown use case raises.
   - `fairness.metrics` matches fairlearn's `demographic_parity_ratio` and the TPR gap on a small random DataFrame.
   - Optionally, the gate produces the expected decisions for the 3 models.
3. **Commit all code first**, then train. `train_models.py` records `git rev-parse HEAD` and blocks lineage if the tree is dirty; its own outputs (`run.json`, `eval_predictions.parquet`) are excluded from the dirty check.
4. **Train:** `uv run python pipeline/train_models.py` (about 6 min). Then commit the `run.json` and `eval_predictions.parquet` files.
5. **Run the gate:** `uv run python -m governance.gate`. Fix bugs until it matches the expected results in §2.
   - Save the registry view with `uv run python -m governance.gate > REGISTRY.md`.
   - Check that Spark doesn't print to stdout and pollute the Markdown; Spark logs should go to stderr.
6. **Write the benchmark, `pipeline/benchmark_spark.py`.** This is the big data evidence.
   - Replicate CA ACSIncome with new person IDs at ×8, ×32 and ×128, which gives about 1.6 M, 6.3 M and 25 M rows. Write each as Parquet and check `df -h` first.
   - Score each with the saved model via a Spark `pandas_udf`, then run `group_counts` and `metrics`.
   - Time it against row count and against core count (`local[1]`, `[2]`, `[4]`, `[8]`).
   - Use a single-process pandas version as the baseline, capped at about 25 M rows to avoid swapping.
   - Also time the CI gate itself.
   - Write the results to `results/benchmark.csv`.
   - Honest limitation for the report: this is single-node Spark local mode, and the code would run unchanged with `--master spark://...`.
7. **Write `.github/workflows/governance-gate.yml`:**
   - A `discover` job lists the model ids, then a **matrix job runs one model per entry**, so `income-fair` shows green and the other two red in the PR UI.
   - Steps:
     1. `actions/checkout` with `fetch-depth: 0` (needed for the commit lineage check)
     2. `actions/setup-java` with Temurin 17
     3. `astral-sh/setup-uv`
     4. Python 3.12
     5. `uv sync`
     6. `pytest`
     7. `python -m governance.gate ${{ matrix.model }} | tee -a $GITHUB_STEP_SUMMARY`
   - Set `defaults.run.shell: bash` so pipefail is on.
   - A final step, "Deploy (simulated)", runs only on `main`; it is skipped automatically when the gate step fails.
   - The README must explain that the workflow is red **by design**, because two of the examples are meant to be blocked.
   - Required status checks (branch protection) are a repo setting, not YAML; document it.
8. **Build `notebooks/results.ipynb`.** It reads the gate results, run metrics and `results/benchmark.csv`, and plots:
   - selection rate by group for the baseline vs the fair model;
   - the accuracy-vs-fairness trade-off;
   - runtime vs rows, speedup vs cores, and Spark vs pandas.
   
   Execute it with `jupyter nbconvert --execute --inplace` so the outputs are saved.
9. **Write the README:** a Mermaid architecture diagram, setup steps, how to run, a results summary, and a link to `REGISTRY.md`. Add a `Makefile` with the targets `data`, `train`, `gate`, `benchmark`, `notebook` and `test`.
10. **Write the report** (3–5 pages, `report/report.md` → PDF). pandoc and LaTeX are **not installed**. Convert Markdown to HTML, then print to PDF with headless Chrome (installed at `/Applications/Google Chrome.app`). Outline:
    - problem
    - architecture
    - EU AI Act classification table (article references are in `governance/risk.py`)
    - policy and gates
    - the 3-model experiment
    - scaling results
    - limitations
    - ethics
    
    An optional table can map gates to NIST AI RMF: GOVERN 1.6 (inventory), MAP 1.1 (intended purpose), MEASURE 2.11 (fairness), MANAGE 1.1 (go/no-go).
11. **Push to GitHub. Ask the owner first**, because the repo is public. Their `gh` CLI is already logged in. Then open a PR that shows the blocked checks, and take screenshots for the report and video.

**The model/data card deliverable** is already covered: `registry/models/income-fair/card.yaml`, the ACS dataset card, and the rendered `REGISTRY.md`.

---

## 7. Setup and gotchas

```bash
cd bdsrae_mini_proj
uv sync --python 3.12                      # creates .venv (~0.9 GB, PySpark is most of it)
java -version                              # must be 17+
uv run python pipeline/download_data.py    # skip if data/ already exists (gitignored)
uv run python pipeline/train_models.py
uv run python -m governance.gate           # run from the repo root (paths are relative)
```

- **Disk.** The owner's Mac was at 98–100% full (it dropped to 1.2 GB free during work; it's about 6 GB after cleaning caches). Swap usage is heavy (8 GB+), so keep `SPARK_DRIVER_MEMORY` at 2–4g and check `df -h` before writing the scaled Parquet files.
- **Network is slow (~100 KB/s) and census.gov stalls.** Downloads use `curl` with resume and retries. The Census PUMS API now needs an API key, so it isn't used. On the same machine, reuse `data/` instead of re-downloading.
- **The uv cache is locked** by long-running `uvx serena` MCP processes. Don't run `uv cache prune --force`; it can break them.
- **`Project_Datasets.docx` is gitignored** (`*.docx`). It's course material, not a deliverable.
- **Owner's preferences:** keep it simple, with the fewest files and no extra infrastructure. Do everything except the demo video. Ask before pushing anything public.
