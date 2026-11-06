Data: counterfactual price series (cf_timeseries) in place of the real prices over each window's test period; asset-days without a counterfactual keep the real price. Customer profiles and transactions are those of FAR-Trans. Rows on the 21-day counterfactual windows say so in Basis. Real-data values: reported_metrics_table_real.md.

| Property | Metric | Value | Reference or threshold | Check | Basis |
|---|---|---|---|---|---|
| Accuracy | R² of the 6-month return prediction (test rows) | -1.153 | 0 = predicting the average |  | mean of 61 time points, counterfactual series |
| Accuracy | RMSE (test rows) | 43.4% | percentage points of return |  | mean of 61 time points, counterfactual series |
| Accuracy | MAE (test rows) | 20.9% | percentage points of return |  | mean of 61 time points, counterfactual series |
| Accuracy | nDCG@10 | 0.039 | random 0.011 | pass | mean of 61 time points, counterfactual series |
| Accuracy | Monthly ROI@10 | 0.77% | risk-matched market 0.14%; market 0.81%; customer average -1.1% | pass | mean of 61 time points, counterfactual series |
| Accuracy | SR@10 (Sharpe ratio) | 0.418 | risk-matched market 0.973; market 2.681 | fail | mean of 61 time points, counterfactual series |
| Accuracy | Vol@10 (annualised volatility) | 112% |  |  | mean of 61 time points, counterfactual series |
| Accuracy | %prof@10 | 55.3% |  |  | mean of 61 time points, counterfactual series |
| Suitability | SVR@10, filtered list | 0.0% | must be 0 at every time point (max 0.0%); conditional if it fails under the sensitivity classifications | conditional pass | mean of 61 time points, counterfactual series |
| Suitability | SVR@10, filtered, 1-year volatility window | 32.3% | sensitivity check, not binding: above 0 makes the pass conditional |  | mean of 61 time points, counterfactual series |
| Suitability | SVR@10, filtered, limits one class stricter | 78.8% | sensitivity check, not binding: above 0 makes the pass conditional |  | mean of 61 time points, counterfactual series |
| Suitability | SVR@10, unfiltered ranking | 81.5% |  |  | mean of 61 time points, counterfactual series |
| Suitability | Customers with an unsuitable asset (unfiltered) | 81.5% |  |  | mean of 61 time points, counterfactual series |
| Suitability | Assets that changed class since the previous time point | 2.7% |  |  | mean of 61 time points, counterfactual series |
| Suitability | Assets within 10% of a class boundary | 24.9% |  |  | mean of 61 time points, counterfactual series |
| Suitability | Counterfactuals that move the asset to a higher risk class | 1.7% |  |  | mean of 61 time points, 21-day counterfactual windows |
| Suitability | Explained assets whose counterfactual class is above the customer's limit | 5.9% |  |  | mean of 61 time points, 21-day counterfactual windows |
| Fairness | nDCG@10 gap by segment | 0.014 | 0 = parity; significant at 23% of time points |  | mean of 61 time points, counterfactual series |
| Fairness | nDCG@10 gap by capacity band | 0.023 | 0 = parity; significant at 15% of time points |  | mean of 61 time points, counterfactual series |
| Fairness | nDCG@10 gap by risk profile | 0.016 | 0 = parity; significant at 3% of time points |  | mean of 61 time points, counterfactual series |
| Fairness | SVR@10 (unfiltered) gap by segment | 22.4% | 0 = parity; significant at 100% of time points |  | mean of 61 time points, counterfactual series |
| Fairness | SVR@10 (unfiltered) gap by capacity band | 46.4% | 0 = parity; significant at 100% of time points |  | mean of 61 time points, counterfactual series |
| Fairness | SVR@10 (unfiltered) gap by risk profile | 100% | 0 = parity; significant at 100% of time points |  | mean of 61 time points, counterfactual series |
| Fairness | Monthly ROI@10 gap within risk profile, by segment | 0.09% | 0 = parity |  | mean of 61 time points, counterfactual series |
| Fairness | Monthly ROI@10 gap within risk profile, by capacity band | 0.13% | 0 = parity |  | mean of 61 time points, counterfactual series |
| Fairness | Gini of exposure (top-10) | 0.981 | 0 = even, 1 = concentrated |  | mean of 61 time points, counterfactual series |
| Fairness | Catalogue coverage (top-10) | 3.3% |  |  | mean of 61 time points, counterfactual series |
| Fairness | In-house share / eligible share | 0.067 | 1 = proportional |  | mean of 61 time points, counterfactual series |
| Fairness | %prof@10, in-house products | 66.7% |  |  | mean of 61 time points, counterfactual series |
| Fairness | %prof@10, third-party products | 55.3% |  |  | mean of 61 time points, counterfactual series |
| Fairness | Share of the top-10 explained: gap by segment | 10.0% | 0 = parity; significant at 95% of time points |  | mean of 61 time points, 21-day counterfactual windows |
| Fairness | Share of the top-10 explained: gap by capacity band | 18.8% | 0 = parity; significant at 95% of time points |  | mean of 61 time points, 21-day counterfactual windows |
| Fairness | Share of the top-10 explained: gap by risk profile | 54.0% | 0 = parity; significant at 100% of time points |  | mean of 61 time points, 21-day counterfactual windows |
| Fairness | Proximity of the explanations: gap by segment | 1.3% | 0 = parity; significant at 100% of time points |  | mean of 61 time points, 21-day counterfactual windows |
| Fairness | Proximity of the explanations: gap by capacity band | 2.4% | 0 = parity; significant at 95% of time points |  | mean of 61 time points, 21-day counterfactual windows |
| Fairness | Proximity of the explanations: gap by risk profile | 6.2% | 0 = parity; significant at 100% of time points |  | mean of 61 time points, 21-day counterfactual windows |
| Fairness | Sparsity of the explanations: gap by segment | 4.1% | 0 = parity; significant at 80% of time points |  | mean of 61 time points, 21-day counterfactual windows |
| Fairness | Sparsity of the explanations: gap by capacity band | 8.7% | 0 = parity; significant at 92% of time points |  | mean of 61 time points, 21-day counterfactual windows |
| Fairness | Sparsity of the explanations: gap by risk profile | 22.1% | 0 = parity; significant at 100% of time points |  | mean of 61 time points, 21-day counterfactual windows |
| Robustness | nDCG@10, calm vs stress | 0.048 vs 0.020 (drop 59%) | worst time point 0.003 |  | mean of 61 time points, counterfactual series |
| Robustness | Monthly ROI@10, calm vs stress | 1.6% vs -1.1% (drop 169%) | worst time point -6.0% |  | mean of 61 time points, counterfactual series |
| Robustness | SR@10, calm vs stress | 0.693 vs -0.191 (drop 128%) | worst time point -2.072 |  | mean of 61 time points, counterfactual series |
| Robustness | J@10 before vs after, equity shock | 0.357 | 1 = unchanged |  | mean of 61 time points, counterfactual series |
| Robustness | SVR@10 under equity shock, filtered | 0.0% | must be 0 (max 0.0%) | pass | mean of 61 time points, counterfactual series |
| Robustness | Assets moved to a higher risk class, equity shock | 6.9% |  |  | mean of 61 time points, counterfactual series |
| Robustness | J@10 before vs after, rates shock | 0.586 | 1 = unchanged |  | mean of 61 time points, counterfactual series |
| Robustness | SVR@10 under rates shock, filtered | 0.0% | must be 0 (max 0.0%) | pass | mean of 61 time points, counterfactual series |
| Robustness | Assets moved to a higher risk class, rates shock | 1.7% |  |  | mean of 61 time points, counterfactual series |
| Robustness | J@10 before vs after, combined shock | 0.192 | 1 = unchanged |  | mean of 61 time points, counterfactual series |
| Robustness | SVR@10 under combined shock, filtered | 0.0% | must be 0 (max 0.0%) | pass | mean of 61 time points, counterfactual series |
| Robustness | Assets moved to a higher risk class, combined shock | 8.5% |  |  | mean of 61 time points, counterfactual series |
| Robustness | J@10, counterfactual scores vs model scores (filtered list) | 0.750 | 1 = unchanged; lists unchanged for 17.7% of customers |  | mean of 61 time points, 21-day counterfactual windows |
| Robustness | Kendall tau, counterfactual scores vs model scores | 0.900 | 1 = same order |  | mean of 61 time points, 21-day counterfactual windows |
| Transparency | Counterfactual validity, re-checked end to end | 93.1% |  |  | mean of 61 time points |
| Transparency | Counterfactual proximity (median relative price change) | 1.2% |  |  | mean of 61 time points |
| Transparency | Counterfactual sparsity (share of the 21 prices changed) | 92.6% | below 50% = actionable |  | mean of 61 time points |
| Transparency | Counterfactual plausibility | 94.1% |  |  | mean of 61 time points |
| Transparency | SHAP stability, Spearman (noise 0.1%) | 0.983 | 1 = identical |  | mean of 61 time points, counterfactual series |
| Transparency | SHAP stability, top-5 overlap (noise 0.1%) | 0.956 | 1 = identical |  | mean of 61 time points, counterfactual series |
| Transparency | SHAP stability, Spearman (noise 0.25%) | 0.951 | 1 = identical |  | mean of 61 time points, counterfactual series |
| Transparency | SHAP stability, top-5 overlap (noise 0.25%) | 0.905 | 1 = identical |  | mean of 61 time points, counterfactual series |
| Transparency | SHAP stability, Spearman (2% days dropped) | 0.983 | 1 = identical |  | mean of 61 time points, counterfactual series |
| Transparency | SHAP stability, top-5 overlap (2% days dropped) | 0.957 | 1 = identical |  | mean of 61 time points, counterfactual series |
| Transparency | SHAP faithfulness, top 1 vs random | 14.7 | above 1 = faithful |  | mean of 61 time points, counterfactual series |
| Transparency | SHAP faithfulness, top 5 vs random | 3.996 | above 1 = faithful |  | mean of 61 time points, counterfactual series |
| Transparency | SHAP faithfulness, top 10 vs random | 2.181 | above 1 = faithful |  | mean of 61 time points, counterfactual series |
| Privacy | Counterfactual windows within 0.1% of a past price window of the asset | 81.8% | real windows: 12.0% |  | mean of 61 time points, 21-day counterfactual windows |
| Privacy | Counterfactual windows that copy a past price window exactly | 23.8% |  |  | mean of 61 time points, 21-day counterfactual windows |
| Privacy | Holdings recovered, top-10 | 7.1% |  |  | mean of 61 time points, counterfactual series |
| Privacy | Precision of the inferred holdings, top-10 | 100% |  |  | mean of 61 time points, counterfactual series |
| Privacy | Customers with at least one holding exposed, top-10 | 29.4% |  |  | mean of 61 time points, counterfactual series |
| Privacy | Holdings recovered, full list | 100% |  |  | mean of 61 time points, counterfactual series |
| Privacy | Precision of the inferred holdings, full list | 100% |  |  | mean of 61 time points, counterfactual series |
| Privacy | Customers with at least one holding exposed, full list | 98.2% |  |  | mean of 61 time points, counterfactual series |
| Monitoring | PSI of the model score | 0.147 | review above 0.25 |  | mean of 61 time points, counterfactual series |
| Monitoring | Highest PSI among the 5 most important features | 0.592 | review above 0.25; triggered at 45 of 61 time points |  | mean of 61 time points, counterfactual series |
