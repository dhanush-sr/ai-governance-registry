#!/usr/bin/env bash
# Guided live demo. Run from the repo root:  ./demo/run_demo.sh
# Press Enter to advance. Every step runs the real system.
# The "break it" steps work on a throwaway copy, so the repo is never changed.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="$PWD/.venv/bin/python"
B=$'\e[1m'; G=$'\e[32m'; R=$'\e[31m'; C=$'\e[36m'; N=$'\e[0m'

step() { echo; read -rp "${C}── press Enter: $1 ──${N}"; clear; echo "${B}$1${N}"; echo; }
# Real gate output, trimmed to the decision table and the blocking reasons (exit 1 = blocked is expected).
gate() { { "$PY" -m governance.gate "$@" 2>/dev/null || true; } | awk '
           /^## Registered/ {stop=1} stop {next}
           /^\| (Model|`)/ {print} /^## `/ {split($0,a,"`"); m=a[2]} /^  - / {sub(/^  - /,""); printf "  %-16s %s\n", m, $0}' \
         | sed -e "s/✅ pass/${G}✅ pass${N}/g" -e "s/❌ fail/${R}❌ fail${N}/g" \
               -e "s/\*\*DEPLOY\*\*/${G}DEPLOY${N}/" -e "s/\*\*BLOCKED\*\*/${R}BLOCKED${N}/" -e 's/\*\*//g'; }

clear
echo "${B}AI Governance Registry with Deployment Gates${N}"
echo "No model is released unless it is documented, risk-rated, fair, traceable and approved."

step "1. The registry: plain files in git"
ls registry registry/models registry/datasets | sed 's/^/  /'

step "2. The rule book: what a HIGH-risk model must prove (registry/policy.yaml)"
sed -n '/^  high:/,/approval_required/p' registry/policy.yaml

step "3. A model card: what the team writes (registry/models/income-fair/card.yaml)"
grep -E '^(id|name|owner):|use_case|declared_tier|sensitive_attributes|status|approver' registry/models/income-fair/card.yaml

step "4. What the training pipeline records: hashes the gate checks (run.json)"
"$PY" -c "import json; r=json.load(open('registry/models/income-fair/run.json'))
print('training commit :', r['code']['commit'][:12], '(dirty:', str(r['code']['dirty'])+')')
print('dataset sha256  :', r['training_data'][0]['sha256'][:16], '…')
print('evidence sha256 :', r['evaluation']['sha256'][:16], '…')
print('accuracy        :', r['evaluation']['metrics']['accuracy'])"

step "5. Run the gate on all three models"
gate

step "6. Why was income-baseline blocked? Selection rate by race group"
open results/fig_selection_rates.png 2>/dev/null || true
echo "Chart opened. Baseline: one group selected at 15%, another at 49% → disparate impact 0.30 (needs ≥ 0.8)."
echo "Fair model: every group between 35% and 40% → 0.89, at a cost of 7.8 accuracy points."

T=$(mktemp -d)
rsync -a --exclude .venv --exclude data ./ "$T/"
step "7. Break it live: the approver withdraws sign-off on the GOOD model"
(cd "$T" && sed -i.bak 's/status: approved/status: requested/' registry/models/income-fair/card.yaml && gate income-fair)

step "8. Break it live: someone edits the evidence after training"
(cd "$T" && mv registry/models/income-fair/card.yaml.bak registry/models/income-fair/card.yaml && "$PY" -c "
import pandas as pd; p='registry/models/income-fair/eval_predictions.parquet'
d=pd.read_parquet(p); d['y_pred']=1; d.to_parquet(p)" && gate income-fair)
rm -rf "$T"

step "9. Big data: the same fairness check, in Spark, at scale"
column -s, -t < results/benchmark.csv | cut -c1-90
open results/fig_scaling.png 2>/dev/null || true
echo; echo "25M rows: pandas 85 s → Spark 8 cores 20 s (4.3× faster), identical result every time."

step "10. The same gates on GitHub, one check per model"
open "https://github.com/dhanush-sr/ai-governance-registry/actions" 2>/dev/null || true
echo "Opened GitHub Actions: gate (income-fair) green and deployed; the other two red."
echo; echo "${B}Done.${N}"
