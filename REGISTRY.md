# Model registry: deployment gate report

| Model | Risk tier | Risk | Documentation | Lineage | Fairness | Approval | Decision |
|---|---|---|---|---|---|---|---|
| `income-baseline` | high | ✅ pass | ✅ pass | ✅ pass | ❌ fail | ✅ pass | **BLOCKED** |
| `income-fair` | high | ✅ pass | ✅ pass | ✅ pass | ✅ pass | ✅ pass | **DEPLOY** |
| `taxi-eta` | minimal | ✅ pass | ❌ fail | ❌ fail | ➖ n/a | ➖ n/a | **BLOCKED** |

## `income-baseline`: Income eligibility pre-screen (unconstrained)

- **Risk tier:** high (Annex III 5(b) (creditworthiness))
- **Evaluation metrics:** {"accuracy": 0.8254, "positive_rate": 0.4109}
- **Blocking issues:**
  - **fairness:** `RAC1P` disparate impact 0.299 < 0.8
  - **fairness:** `RAC1P` equal-opportunity gap 0.331 > 0.1

Fairness by `SEX`: disparate impact **0.802**, equal-opportunity gap **0.015**

| SEX | n | selection rate | TPR |
|---|---|---|---|
| 1 | 20674 | 0.453 | 0.794 |
| 2 | 18459 | 0.364 | 0.779 |

Fairness by `RAC1P`: disparate impact **0.299**, equal-opportunity gap **0.331** (excluded small groups: [4])

| RAC1P | n | selection rate | TPR |
|---|---|---|---|
| 1 | 24200 | 0.449 | 0.807 |
| 2 | 1770 | 0.358 | 0.724 |
| 3 | 245 | 0.322 | 0.698 |
| 4 | 3 | 0.333 | 1.000 |
| 5 | 114 | 0.246 | 0.692 |
| 6 | 6557 | 0.487 | 0.825 |
| 7 | 137 | 0.321 | 0.667 |
| 8 | 4520 | 0.146 | 0.494 |
| 9 | 1587 | 0.365 | 0.759 |

## `income-fair`: Income eligibility pre-screen (fairness-constrained)

- **Risk tier:** high (Annex III 5(b) (creditworthiness))
- **Evaluation metrics:** {"accuracy": 0.7469, "positive_rate": 0.3881}
- **Blocking issues:** none

Fairness by `SEX`: disparate impact **0.977**, equal-opportunity gap **0.075**

| SEX | n | selection rate | TPR |
|---|---|---|---|
| 1 | 20674 | 0.392 | 0.634 |
| 2 | 18459 | 0.383 | 0.709 |

Fairness by `RAC1P`: disparate impact **0.885**, equal-opportunity gap **0.054** (excluded small groups: [4])

| RAC1P | n | selection rate | TPR |
|---|---|---|---|
| 1 | 24200 | 0.397 | 0.657 |
| 2 | 1770 | 0.382 | 0.682 |
| 3 | 245 | 0.351 | 0.698 |
| 4 | 3 | 0.333 | 1.000 |
| 5 | 114 | 0.360 | 0.731 |
| 6 | 6557 | 0.390 | 0.680 |
| 7 | 137 | 0.372 | 0.564 |
| 8 | 4520 | 0.352 | 0.654 |
| 9 | 1587 | 0.369 | 0.708 |

## `taxi-eta`: Taxi trip duration estimate

- **Risk tier:** minimal (no Art. 5, Art. 6 or Art. 50 trigger)
- **Evaluation metrics:** {"mae_minutes": 2.818}
- **Blocking issues:**
  - **documentation:** schema: `(root)` Additional properties are not allowed ('known_issues' was unexpected)
  - **documentation:** missing `limitations`
  - **lineage:** dataset `nyc-tlc-yellow-2023-01` missing `personal_data`

## Registered datasets

| Dataset | Source | License | sha256 |
|---|---|---|---|
| `acs-income-ca-2018` | https://www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year/csv_pca.zip | Public domain (US Government work, 17 U.S.C. 105); Census Bureau terms of use | `6194ade52892…` |
| `nyc-tlc-yellow-2023-01` | https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet | NYC Open Data terms of use (public, free to reuse) | `32df6f67578f…` |

