# Credit Model Validation

Before a bank uses a credit model, a separate team checks whether its results hold up. I applied that process to my own two models.

## Why this project

At work, I compared forecast credit losses with what actually happened and assessed portfolios against their underwriting assumptions, so I've seen how far assumptions can drift from reality. Banks require every model to be checked by someone other than its builder before it's used. With two models of my own, I wanted to apply the same discipline to them, including to my own claims.

## What I set out to do

Test whether my scorecard and XGBoost results are reproducible, statistically real and robust, and correct anything that isn't.

## What I did

- Reloaded both saved models and reproduced their reported results
- Tested whether XGBoost's improvement is real, using 1,000 paired bootstrap resamples
- Put error bars on calibration, and checked 11 borrower segments and 101,503 new applicants
- Re-tested my lending strategy's cut-offs on unseen borrowers and stress-tested its business assumptions

## Results

- **21 findings: 14 Pass, 7 Monitor, 0 Fail.** Both models are fit for use, with conditions
- **XGBoost's improvement is small but real:** +0.0069 AUC, ahead in all 1,000 resamples
- **The scorecard under-weights payment history**, and understates risk for its riskiest borrowers
- **I corrected my own earlier claims:** about seven-eighths of the profit uplift depends on a balance assumption (+37.2% falls to +4.7% with a flat balance), and at the profit-maximizing cut-off XGBoost is −0.4%, not +0.1%

<p align="center"><img src="outputs/calibration_error_bars.png" width="55%" alt="Predicted vs actual default rate by decile with 95% error bars for both models"></p>

## How it connects

It reviews my [credit risk scorecard](https://github.com/pooja003-cloud/credit-risk-scorecard) and my [lending strategy](https://github.com/pooja003-cloud/credit-lending-strategy) project.

## Run it

Download `cs-training.csv` and `cs-test.csv` from [Kaggle's Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit/data), save them as `data/credit_data.csv` and `data/cs-test.csv`, then run `pip install -r requirements.txt` and `python validate.py`.

## Full details

The findings table, recommendations, evidence for every check and limitations are in [REPORT.md](REPORT.md).
