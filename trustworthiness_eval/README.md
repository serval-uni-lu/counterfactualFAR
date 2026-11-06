# Trustworthiness evaluation

Produces the metrics for the tuned
RFR recommender (`rfr_n-193_leaf-96_depth-30_kpi-full_short_internal_kpis`) and its counterfactual (CF)
explanations, over the 61 time points.

The result is `outputs/<model>/reported metrics/reported_metrics_table.md` (on the counterfactual price
series) and `reported_metrics_table_real.md` (on actual prices), with one figure per property.

## How to run

From the repo root. Each script writes to `outputs/<model>/`; the last one builds the table and figures.

```bash
# actual prices, every time point -> outputs/<model>/all_dates/
python trustworthiness_eval/robustness.py            # market shocks: J@10, SVR@10 under shock
python trustworthiness_eval/attribution.py           # SHAP stability and faithfulness
python trustworthiness_eval/monitoring.py            # PSI of the score and the input features
python trustworthiness_eval/cf_quality_all_dates.py  # CF validity, proximity, sparsity, plausibility
python trustworthiness_eval/cf_tests_all_dates.py    # stability, fairness, privacy, suitability of the CFs
python trustworthiness_eval/privacy/holdings_inference/holdings_inference.py --model <model>

# counterfactual price series -> outputs/<model>/synthetic/
python trustworthiness_eval/synthetic_eval.py
python trustworthiness_eval/attribution.py --synthetic

# table and figures -> outputs/<model>/reported metrics/
python trustworthiness_eval/reported_metrics.py
```

The per-time-point metrics on actual prices are read from
`results/rfr/rfr_tuned_full_short_internal_kpis_<date>_metrics.csv` (`recommendation.py`, then
`compute_risk_metrics.py`). Suitability (PRIIPs risk classes, profile limits, SVR@k) is shared code in
`metrics/suitability.py`. `common.py` holds the helpers the CF scripts share (loading the CFs, CF quality,
re-checking each CF through the saved model).

## Outputs

| Subfolder | Files | Made by |
|---|---|---|
| `all_dates/` (actual prices) | `robustness_market_shocks.csv` | `robustness.py` |
| | `attribution_stability.csv`, `attribution_faithfulness.csv` | `attribution.py` |
| | `monitoring_psi.csv` | `monitoring.py` |
| | `cf_quality.csv` | `cf_quality_all_dates.py` |
| | `cf_test_{stability,fairness,privacy,suitability}.csv` | `cf_tests_all_dates.py` |
| | `holdings_inference.csv` | `privacy/holdings_inference/holdings_inference.py` |
| `synthetic/` (counterfactual price series) | `synthetic_metrics.csv`, `synthetic_shocks.csv`, `synthetic_monitoring.csv` | `synthetic_eval.py` |
| | `attribution_stability.csv`, `attribution_faithfulness.csv` | `attribution.py --synthetic` |
| `reported metrics/` | the two tables (`.md` and `.csv`), figures (`real/` = actual prices) | `reported_metrics.py` |

`synthetic_eval.py`, `cf_quality_all_dates.py` and `cf_tests_all_dates.py` keep a per-date cache
(`*by_date/`) next to their output so an interrupted run resumes; delete it once the run is complete.
`attribution.py` run on some dates replaces those dates in its files and keeps the others.

## Privacy

`privacy/holdings_inference/` gives the Privacy rows of the table. `privacy/membership_inference/` holds
the membership inference attacks (LOSS, population, RMIA) on the fixed target model; see its README.

## Method notes

- **Validity re-check.** Each CF at the recommendation date is fed back through the saved model on the
  full price history, as the recommender scores assets, and counts as valid only if the re-scored value
  lands in the requested range.
- **Feature attribution.** The score has no customer input, so SHAP is computed per asset and date, and
  a customer–asset pair inherits its asset's explanation. Stability uses perturbations of the price
  history (noise of 0.1% and 0.25% on daily closes, 2% of days dropped), because the doc's customer-data
  perturbations leave every score unchanged. Faithfulness replaces features with their training median,
  which can create KPI combinations that never occur together; the random-feature baseline is replaced
  the same way.
- **Counterfactual flip rate** is 0 by construction: segment and capacity are not inputs of the
  recommender, and each customer's list only drops assets they already hold.
