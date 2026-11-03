# Demo video narration script

Generated voice-over: macOS `say` (Samantha). Re-record over `demo/demo.mp4` with your own voice if you prefer.

## Slide 1

This is our AI governance registry with deployment gates. Every model must be documented, risk-rated under the EU AI Act, and approved before release, and continuous integration enforces it.

## Slide 2

Governance usually lives in meetings and spreadsheets, so it gets skipped. We turned the policy into code, with one principle: the gate verifies, it does not trust. The risk tier is computed, fairness is recomputed from evidence, and lineage is checked against hashes recorded at training time.

## Slide 3

The registry is plain files in git. Owners write model cards, the training pipeline writes lineage and evaluation evidence, and the governance team owns a policy with rules per risk tier. The gate runs five checks: risk, documentation, lineage, fairness, and approval.

## Slide 4

Prohibited practices under Article 5 are never deployable. Annex three uses, like credit scoring, are high risk, and need extra documentation, a fairness check, and independent approval. Chatbots are limited risk; everything else is minimal. If the declared tier disagrees with the computed one, the gate blocks.

## Slide 5

We built three example models. Two are income pre-screens for a fictional lender, trained on public census data. The baseline drops sex and race as inputs, and a human approved it. The second adds a demographic parity constraint. The third is a minimal-risk taxi model with sloppy documentation.

## Slide 6

Here is the real gate output. The fair model passes all five gates, and deploys. The baseline is blocked on fairness. The taxi model is blocked on documentation and lineage: no limitations section, and its dataset card lacks a privacy statement.

## Slide 7

Removing protected attributes is not enough. Through proxies like occupation, the baseline selects one group at fifteen percent and another at forty nine: a disparate impact of point three, against a threshold of point eight. The constrained model keeps every group between thirty five and forty percent, at a cost of eight accuracy points.

## Slide 8

Lineage is tamper-evident. A test edits the fair model's predictions after training, and the gate blocks it. Other tests check the risk tiers and match our metrics against fairlearn.

## Slide 9

For big data, the fairness gate is a Spark job, and the same code scales. On twenty five million rows, single-process pandas takes eighty five seconds. Spark on eight cores takes under twenty: a four point three times speed-up, with identical results at every scale. In C I, the gate costs about five seconds per model.

## Slide 10

In GitHub Actions, each model gets its own check. The release step runs only when every gate passes, and required checks block the merge.

## Slide 11

Limitations: Spark ran on one machine, the large data is replicated, and the risk questionnaire is self-reported. The gates are a minimum, auditable standard, not a replacement for human review. Thank you.
