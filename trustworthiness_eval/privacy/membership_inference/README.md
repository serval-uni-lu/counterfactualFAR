# Membership inference

These scripts run membership inference attacks (LOSS, population, RMIA) against the saved RFR
profitability models. They never retrain or replace the target.

| Script | What it does |
|---|---|
| `loss.py` | LOSS attack (Yeom et al.). A row is called a member when its loss is unusually low. |
| `population_attack.py` | Population attack (ml_privacy_meter attack-P). Each row's loss is compared with a held-out half of the test-period rows. |
| `rmia.py` | Offline RMIA (Zarifzadeh et al.). Uses reference models fitted on halves of the target's training rows. |
| `evaluate.py` | Re-evaluates the saved scores at block level, (asset, time block), because neighbouring days are near-duplicates. |
| `regression_signals.py` | Shared code for the attacks. |
| `plots.py` | Figures, saved under `stats/plots/membership_inference/`. |

Results are in `results/<model>/`.

## Does the shared model need a membership inference evaluation?

**Short answer:** not as a legal requirement. The model is not trained on personal data, so the trees
contain none. The attacks here are a **memorisation and robustness audit**, not a privacy
requirement. Before sharing the model, though, remove the cached data table in the saved file
(see below).

### What the model is trained on

Each training row is **one asset on one date**:

- **Features:** the 30 KPIs computed from that asset's closing prices (`close_prices.csv`:
  `ISIN`, `timestamp`, `closePrice`).
- **Target:** the asset's return over the next 6 months.

See `algorithms/profitability_prediction.py` (`train`). The set of assets comes only from the price file
(`data/filter/asset/asset_with_test_price.py`). No customer ID, transaction, holding or customer
attribute (segment, risk profile, capacity, channel) goes into training.

Customers only come in **after** the model. Each customer's list is the global ranking minus the
assets that customer already traded. That step lives in `recommend()`, not in the saved model.

### Why a membership inference test is not required

- **Membership is not about a person.** Here, a successful attack would reveal "asset X on date D was
  in the training set". That's a fact about a public market price, so there's no personal data for GDPR
  to protect.
- **The regulatory expectation doesn't apply.** The EDPB (Opinion 28/2024 on AI models) expects
  resistance to membership inference and data extraction to be shown when a model **was trained on
  personal data** and is claimed to be anonymous. This model was not.
- **The model memorises little by design.** `min_samples_leaf=96`, so every leaf prediction averages at
  least 96 training rows.

The attack results are still worth reporting. A model whose training rows can be told apart from
unseen ones fits them much more closely than new data, which is the generalisation gap seen from an
attacker's side.

### What the saved model file contains

Inspected file: `artifacts_for_counterfactuals/rfr_n-193_leaf-96_depth-30_kpi-full_short_internal_kpis/profitability_recommendation_pipeline_2019-09-26_..._.pkl`
(`RFRKPIModel`).

| Part | Content | Personal data? |
|---|---|---|
| `model` / `pipeline`: `RandomForestRegressor`, 193 trees, 30 features | Split thresholds on KPI values, and leaf averages of 6-month returns | No |
| Hyperparameters, `kpi_features`, `kpi_type`, `k` | Configuration | No |
| `transformer.last_kpis_df_` | **54,772 rows** of `col_item` (ISIN), `col_timestamp`, `col_rating` (price) and 40 KPIs: a copy of the data from the last time the model computed KPIs | No, but it's raw data |

`last_kpis_df_` holds no personal data, but it ships bank's price series inside the model. That may
conflict with the dataset licence or data-sharing agreement. Set it to `None` **in a copy** before
sharing. The model does not need it to make predictions.

### Before sharing

- **Remove the cache.** Share a copy with `transformer.last_kpis_df_ = None`, and keep the original
  target unchanged.
- **Share the model only, not the per-customer outputs.** `results/rfr/*_recs.txt` and
  `artifacts_for_counterfactuals/*/user_recs_*.csv` contain `col_user` (a pseudonymised customer ID,
  which is still personal data under GDPR Recital 26). Each list is the global ranking minus the
  customer's traded assets, so comparing the two gives back that customer's trading history.
- **Re-assess if the inputs change.** The answer above only holds while the inputs are prices. If
  customer features (risk profile, segment, capacity) or transaction-based training (e.g. collaborative
  filtering) are added, the training data becomes personal and a membership inference evaluation becomes
  necessary.
- **Pickle format.** A pickle can run code when it's loaded, and this one was saved with scikit-learn
  1.9.0. Give recipients the exact versions, or export with `skops`.

## Other FAR-Trans models: is membership inference needed?

Whether a model needs a membership inference evaluation depends on what its `train()` reads. The
table below is based on the code in `far-trans/algorithms/`. There are three cases.

### 1. Trained on prices only: not needed

| Model | Trained on |
|---|---|
| `profitability_prediction` (this project's RFR and the other regressors) | Price-based KPIs → future return |
| `profitability_classification` | Price-based KPIs → profitable or not |
| `profitability_learning_to_rank` (LETOR) | Price-based KPIs → rank by return |
| `rankingbased` | Ranks assets by a single KPI; nothing is learned |
| `random` | Nothing |

These models are in the same position as the RFR above. A successful attack would reveal only an asset
on a date.

### 2. Trained on customer data: needed

| Model | Trained on | What the shared model holds |
|---|---|---|
| `ubknn` (CAS) | Each customer's transactions (`data.train`) | **The full customer × asset matrix and customer similarities.** Sharing the model shares the transactions. It is not anonymous, and no attack result can make it so. |
| `cps` | Customer profiles file + transactions | **Stored per-customer profiles.** Same issue as `ubknn`. |
| `beta_recsys` (MF, LightGCN) | Customer × asset purchases | One learned vector per customer ID and one per asset. The customer vectors are personal data. Sharing only the asset vectors still needs an evaluation, because published attacks on recommenders target exactly that. |
| `arm` (association rules) | Customers' baskets | Rules with support and confidence. A low-support rule can come from one or two customers' baskets. |
| `popularity`, `kpi_popularity` | Number of customers who bought each asset | Counts per asset. Low counts and repeated releases (one per date) can single out customers. |
| `grouppop` (`CustomerProfileGroupBuilder`) | Customer attributes + purchases, per group | Popularity by demographic group. Small groups leak membership and attributes. |

### 3. Hybrids: depends on their parts

| Model | Needed? |
|---|---|
| `hybrid_lambdamart` | **Yes.** Its labels are what customers actually bought (`prev_data.test`), and its features are each customer's scores from the base models. |
| `hybrid_rankagg` | Only if any of the models it combines is in case 2. |

### How the evaluation must change for these models

- **The unit is the customer, not the row.** The question becomes "was **this customer** in training?",
  so all of a customer's rows are grouped and scored together. The row and (asset, time block) setup in
  this folder doesn't answer that question.
- **Don't share `ubknn` or `cps`.** These models store the data itself, so a membership inference
  evaluation is beside the point. Only their outputs could be shared, after anonymisation.
- **Reduce the risk in the count-based models.** For `arm`, `popularity`, `kpi_popularity` and
  `grouppop`, the options are a minimum support or count threshold, a minimum group size, or noise added
  to the counts (differential privacy).
