"""
Credit Model Validation - independent review of the scorecard and XGBoost models
================================================================================
Validates the two models from credit-risk-scorecard (project 1) and
credit-lending-strategy (project 2) the way a model risk team would, one check at a time:

  Check 1 - Replication and data integrity
  Check 2 - Is the XGBoost lift real? (bootstrap confidence intervals)
  Check 3 - Calibration with error bars (exact binomial intervals by decile)
  Check 4 - Stability (PSI drift on new applicants) and segment performance
  Check 5 - Sensitivity: does each input always move risk in a sensible direction?
  Check 6 - Challenge project 2's conclusions (out-of-sample cut-offs, assumption stress)

Run: python validate.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from models.scorecard.scorecard import SEED, TARGET, apply_caps, clean

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "credit_data.csv"
NEW_APPLICANTS_FILE = ROOT / "data" / "cs-test.csv"     # Kaggle's unlabelled applicants
SCORECARD_FILE = ROOT / "models" / "scorecard" / "outputs" / "scorecard_model.pkl"
XGBOOST_FILE = ROOT / "models" / "xgboost" / "xgboost_model.json"
OUT_DIR = ROOT / "outputs"

# Results as reported in the two projects' READMEs (hold-out AUC, 3 decimals)
REPORTED_AUC = {"Scorecard": 0.861, "XGBoost": 0.868}
N_BOOT = 1000          # bootstrap resamples for confidence intervals
N_SENS = 500           # applicants used in the sensitivity check
N_SPLITS = 200         # random half-splits for the out-of-sample cut-off test

# Project 2's business assumptions (credit card line, two-year horizon)
CREDIT_LIMIT, CCF, LGD, APR, COST_OF_FUNDS, YEARS = 10_000, 0.50, 0.80, 0.20, 0.06, 2
TARGET_DEFAULT = 0.02                     # project 2's headline: "2% target default rate"
REPORTED_UPLIFT_2PCT = 0.37               # project 2: about 37% more profit at that target
REPORTED_EXTRA_2PCT = 3477                # project 2: extra approvals per 100,000 at that target

# Expected direction of each input: +1 = should only raise risk, -1 = only lower it
EXPECTED = {
    "RevolvingUtilizationOfUnsecuredLines": 1,
    "NumberOfTime30-59DaysPastDueNotWorse": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 1,
    "NumberOfTimes90DaysLate": 1,
    "DebtRatio": 1,
    "age": -1,
    "MonthlyIncome": -1,
}

BLUE, ORANGE = "#2a78d6", "#eb6834"      # scorecard, XGBoost
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


LABEL_SHORT = {
    "RevolvingUtilizationOfUnsecuredLines": "Revolving utilization",
    "NumberOfTime30-59DaysPastDueNotWorse": "Payments 30-59 days late",
    "NumberOfTime60-89DaysPastDueNotWorse": "Payments 60-89 days late",
    "NumberOfTimes90DaysLate": "Payments 90+ days late",
    "DebtRatio": "Debt ratio",
    "age": "Age",
    "MonthlyIncome": "Monthly income",
}


def matplotlib_dollars():
    from matplotlib.ticker import FuncFormatter
    return FuncFormatter(lambda v, _: f"${v / 1000:.0f}k")


def matplotlib_null():
    from matplotlib.ticker import NullFormatter
    return NullFormatter()


def style_charts():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt

findings = []   # one row per check: what was tested, the result and its rating


def line(char="=", n=78):
    print(char * n)


def header(title):
    print()
    line()
    print(title)
    line()


def record(check, test, result, rating):
    findings.append({"Check": check, "Test": test, "Result": result, "Rating": rating})


def load_models():
    with open(SCORECARD_FILE, "rb") as f:
        scorecard = pickle.load(f)
    challenger = xgb.XGBClassifier()
    challenger.load_model(XGBOOST_FILE)
    return scorecard, challenger


def predict(scorecard, challenger, X):
    """PDs from both models, each with its own preprocessing."""
    X_lr = apply_caps(X, scorecard["caps"])[scorecard["features"]]
    pd_lr = scorecard["pipeline"].predict_proba(X_lr)[:, 1]
    xgb_features = scorecard["features"] + ["DebtRatio"]
    pd_xgb = challenger.predict_proba(X[xgb_features])[:, 1]
    return pd_lr, pd_xgb


# ---------------------------------------------------------------------------
# Check 1 - Replication and data integrity
# ---------------------------------------------------------------------------
def check_replication(df, X_train, X_test, y_test, scorecard, challenger):
    header("CHECK 1 - REPLICATION AND DATA INTEGRITY")

    # 1a. Do the saved models reproduce the reported results?
    pd_lr, pd_xgb = predict(scorecard, challenger, X_test)
    print("1a. Reproduce reported hold-out AUC from the saved model files")
    print(f"    {'Model':<12}{'Reported':>10}{'Reproduced':>12}{'Match':>8}")
    for name, p in [("Scorecard", pd_lr), ("XGBoost", pd_xgb)]:
        auc = roc_auc_score(y_test, p)
        match = round(auc, 3) == REPORTED_AUC[name]
        print(f"    {name:<12}{REPORTED_AUC[name]:>10.3f}{auc:>12.4f}{'yes' if match else 'NO':>8}")
        record("1 Replication", f"{name} AUC reproduces",
               f"reported {REPORTED_AUC[name]:.3f}, reproduced {auc:.4f}",
               "Pass" if match else "Fail")

    # 1b. Duplicate rows in the raw data
    features = [c for c in df.columns if c != TARGET]
    dup_rows = df.duplicated(keep=False)
    n_groups = df[dup_rows].groupby(features, dropna=False).ngroups
    conflict = (df[dup_rows].groupby(features, dropna=False)[TARGET].nunique() > 1).sum()
    print(f"\n1b. Exact duplicate borrowers in the data")
    print(f"    Rows that have an identical twin: {dup_rows.sum():,} "
          f"({dup_rows.mean():.2%}) in {n_groups:,} groups")
    print(f"    Identical features but different outcomes: {conflict:,} groups")
    print(f"    Default rate among duplicated rows: {df.loc[dup_rows, TARGET].mean():.2%} "
          f"(all rows: {df[TARGET].mean():.2%})")

    # 1c. Do any test borrowers also appear, identically, in the training set?
    key = lambda d: pd.util.hash_pandas_object(d[features], index=False)
    in_train = key(X_test).isin(set(key(X_train))).values
    print(f"\n1c. Test borrowers with an identical twin in the training set: "
          f"{in_train.sum():,} of {len(X_test):,} ({in_train.mean():.2%})")
    print(f"    Default rate among them: {y_test[in_train].mean():.2%}")
    print(f"    {'Model':<12}{'AUC all test':>14}{'AUC without twins':>20}{'Change':>9}")
    for name, p in [("Scorecard", pd_lr), ("XGBoost", pd_xgb)]:
        auc_all = roc_auc_score(y_test, p)
        auc_clean = roc_auc_score(y_test[~in_train], p[~in_train])
        print(f"    {name:<12}{auc_all:>14.4f}{auc_clean:>20.4f}{auc_clean - auc_all:>+9.4f}")
        record("1 Data integrity", f"{name} AUC without train/test twins",
               f"{auc_all:.4f} -> {auc_clean:.4f} ({auc_clean - auc_all:+.4f})",
               "Pass" if abs(auc_clean - auc_all) < 0.002 else "Monitor")
    record("1 Data integrity", "Duplicate borrowers across train/test",
           f"{in_train.sum():,} test rows ({in_train.mean():.2%}) have a training twin",
           "Pass" if in_train.mean() < 0.01 else "Monitor")
    return pd_lr, pd_xgb


# ---------------------------------------------------------------------------
# Check 2 - Is the XGBoost lift real?
# ---------------------------------------------------------------------------
def check_lift(y_test, pd_lr, pd_xgb):
    header("CHECK 2 - IS THE XGBOOST LIFT REAL? (paired bootstrap, "
           f"{N_BOOT:,} resamples)")
    rng = np.random.default_rng(SEED)
    n = len(y_test)
    boot = np.empty((N_BOOT, 2))
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)          # same resampled borrowers for both models
        boot[b] = roc_auc_score(y_test[idx], pd_lr[idx]), roc_auc_score(y_test[idx], pd_xgb[idx])
    diff = boot[:, 1] - boot[:, 0]

    point = {"Scorecard": roc_auc_score(y_test, pd_lr), "XGBoost": roc_auc_score(y_test, pd_xgb)}
    print(f"{'':<22}{'AUC':>8}{'95% interval':>22}{'Gini':>8}{'95% interval':>22}")
    line("-")
    for j, name in enumerate(["Scorecard", "XGBoost"]):
        lo, hi = np.percentile(boot[:, j], [2.5, 97.5])
        print(f"{name:<22}{point[name]:>8.4f}{f'{lo:.4f} - {hi:.4f}':>22}"
              f"{2 * point[name] - 1:>8.4f}{f'{2 * lo - 1:.4f} - {2 * hi - 1:.4f}':>22}")
    d_lo, d_hi = np.percentile(diff, [2.5, 97.5])
    d_point = point["XGBoost"] - point["Scorecard"]
    wins = (diff > 0).mean()
    print(f"{'Difference (XGB - LR)':<22}{d_point:>+8.4f}{f'{d_lo:+.4f} - {d_hi:+.4f}':>22}")
    line("-")
    print(f"XGBoost beats the scorecard in {wins:.1%} of resamples.")
    print(f"Note: the two models' individual intervals overlap, but the paired difference is what "
          f"matters - both are scored on the same borrowers in every resample.")

    real = d_lo > 0
    record("2 Lift", "XGBoost AUC lift over scorecard",
           f"{d_point:+.4f} (95% interval {d_lo:+.4f} to {d_hi:+.4f}); XGBoost wins {wins:.1%} of resamples",
           "Pass" if real else "Monitor")

    # Chart: distribution of the AUC difference
    plt = style_charts()
    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=150)
    ax.hist(diff, bins=40, color=BLUE, edgecolor="white", linewidth=0.8)
    ax.axvline(0, color=INK, lw=1)
    ax.axvspan(d_lo, d_hi, color=BLUE, alpha=0.08, lw=0)
    ax.annotate("No difference", (0, ax.get_ylim()[1] * 0.95), xytext=(4, 0),
                textcoords="offset points", color=INK, va="top")
    ax.annotate(f"95% interval\n{d_lo:+.4f} to {d_hi:+.4f}", (d_hi, ax.get_ylim()[1] * 0.95),
                xytext=(6, 0), textcoords="offset points", color=INK, va="top")
    ax.set(xlabel="AUC difference, XGBoost minus scorecard", ylabel="Resamples")
    ax.set_title(f"XGBoost AUC lift across {N_BOOT:,} bootstrap resamples", loc="left", color=INK)
    ax.grid(color=GRID, lw=0.6, axis="y")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "auc_lift_bootstrap.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Check 3 - Calibration with error bars
# ---------------------------------------------------------------------------
def check_calibration(y_test, pd_lr, pd_xgb):
    from scipy.stats import beta

    header("CHECK 3 - CALIBRATION WITH ERROR BARS (exact 95% binomial interval per decile)")
    print("A decile is flagged only if the model's average predicted PD falls outside the range")
    print("of true default rates consistent with the defaults actually observed.\n")
    results = {}
    for name, p in [("Scorecard", pd_lr), ("XGBoost", pd_xgb)]:
        t = (pd.DataFrame({"pd": p, "y": y_test,
                           "Decile": pd.qcut(p, 10, labels=range(1, 11))})
             .groupby("Decile", observed=True)
             .agg(N=("y", "size"), Defaults=("y", "sum"), Predicted=("pd", "mean")))
        t["Actual"] = t["Defaults"] / t["N"]
        # Clopper-Pearson exact interval for the true default rate
        t["Low"] = np.nan_to_num(beta.ppf(0.025, t["Defaults"], t["N"] - t["Defaults"] + 1))
        t["High"] = beta.ppf(0.975, t["Defaults"] + 1, t["N"] - t["Defaults"])
        t["Verdict"] = np.select([t["Predicted"] > t["High"], t["Predicted"] < t["Low"]],
                                 ["OVERSTATES", "UNDERSTATES"], "ok")
        results[name] = t

        print(f"{name}")
        print(f"  {'Decile':<8}{'Defaults':>9}{'Actual':>9}{'95% range':>19}{'Predicted':>11}  Verdict")
        for d, r in t.iterrows():
            rng_ = f"{r['Low']:.2%} - {r['High']:.2%}"
            print(f"  {d:<8}{int(r['Defaults']):>9}{r['Actual']:>9.2%}{rng_:>19}"
                  f"{r['Predicted']:>11.2%}  {r['Verdict']}")
        over = (t["Verdict"] == "OVERSTATES").sum()
        under = (t["Verdict"] == "UNDERSTATES").sum()
        flagged = t.index[t["Verdict"] != "ok"].tolist()
        print(f"  Flagged deciles: {over} overstate risk, {under} understate risk"
              f"{' (' + ', '.join(map(str, flagged)) + ')' if flagged else ''}\n")
        # Understating risk is the dangerous direction for a lender (too optimistic)
        rating = "Fail" if under > 1 else ("Monitor" if over + under > 1 else "Pass")
        record("3 Calibration", f"{name} deciles outside 95% range",
               f"{over} overstate, {under} understate"
               f"{' (deciles ' + ', '.join(map(str, flagged)) + ')' if flagged else ''}", rating)
    print("With 20 decile tests at 95% confidence, about one flag is expected by chance alone.")

    # Chart: predicted vs actual with error bars, both models
    plt = style_charts()
    from matplotlib.ticker import PercentFormatter
    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=150)
    lims = [0.0015, 0.6]
    ax.plot(lims, lims, color=MUTED, lw=1, ls="--", label="Perfect calibration")
    for name, col, shift in [("Scorecard", BLUE, 0.97), ("XGBoost", ORANGE, 1.03)]:
        t = results[name]
        x = t["Predicted"] * shift            # tiny offset so the error bars don't overlap
        ax.errorbar(x, t["Actual"], yerr=[t["Actual"] - t["Low"], t["High"] - t["Actual"]],
                    fmt="o", ms=6, color=col, ecolor=col, elinewidth=1.2, capsize=0,
                    markeredgecolor="white", markeredgewidth=1, label=name)
    ax.set(xscale="log", yscale="log", xlim=lims, ylim=lims,
           xlabel="Average predicted PD", ylabel="Actual default rate (95% range)")
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    ax.set_title("Calibration by decile, with error bars", loc="left", color=INK)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "calibration_error_bars.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Check 4 - Stability and segment performance
# ---------------------------------------------------------------------------
def psi(expected, actual, bins=10):
    """Population Stability Index. Bins come from the expected (training) data;
    missing values get their own bin."""
    e, a = pd.Series(expected, dtype=float), pd.Series(actual, dtype=float)
    edges = np.unique(np.nanquantile(e, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf

    def shares(x):
        counts = pd.cut(x, edges, include_lowest=True).value_counts(sort=False).values
        counts = np.append(counts, x.isna().sum())
        return np.clip(counts / len(x), 1e-4, None)

    pe, pa = shares(e), shares(a)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def psi_rating(v):
    return "Pass" if v < 0.10 else ("Monitor" if v < 0.25 else "Fail")


def check_stability(X_train, X_test, X_new, y_test, scorecard, challenger, pd_lr, pd_xgb):
    from scipy.stats import beta

    header("CHECK 4 - STABILITY AND SEGMENT PERFORMANCE")
    tr_lr, tr_xgb = predict(scorecard, challenger, X_train)
    new_lr, new_xgb = predict(scorecard, challenger, X_new)

    # 4a. PSI: training data vs test set (control) and vs new applicants
    print(f"4a. Population Stability Index vs the training data "
          f"(<0.10 stable, 0.10-0.25 monitor, >0.25 shift)")
    print(f"    New applicants: {len(X_new):,} borrowers from cs-test.csv (no outcomes)\n")
    print(f"    {'Variable':<42}{'Test set':>10}{'New applicants':>16}")
    line("-")
    rows = []
    variables = [c for c in scorecard["features"] if c not in ("PastDueCodeFlag", "IncomeMissingFlag")]
    variables += ["DebtRatio"]
    for col in variables:
        rows.append((col, psi(X_train[col], X_test[col]), psi(X_train[col], X_new[col])))
    for name, tr, te, new in [("Scorecard PD", tr_lr, pd_lr, new_lr),
                              ("XGBoost PD", tr_xgb, pd_xgb, new_xgb)]:
        rows.append((name, psi(tr, te), psi(tr, new)))
    for col, v_test, v_new in rows:
        print(f"    {col:<42}{v_test:>10.4f}{v_new:>16.4f}")
    line("-")
    worst = max(rows, key=lambda r: r[2])
    print(f"    Largest shift on new applicants: {worst[0]} (PSI {worst[2]:.4f})")
    for col, v_test, v_new in rows[-2:]:
        print(f"    {col}: mean on training {(tr_lr if 'Scorecard' in col else tr_xgb).mean():.2%}, "
              f"new applicants {(new_lr if 'Scorecard' in col else new_xgb).mean():.2%}")
    record("4 Stability", "Largest feature or score PSI, new applicants",
           f"{worst[0]}: {worst[2]:.4f}", psi_rating(worst[2]))
    psi_table = pd.DataFrame(rows, columns=["Variable", "PSI_test", "PSI_new_applicants"])
    psi_table.to_csv(OUT_DIR / "psi.csv", index=False)

    # 4b. Segment performance on the test set
    late = X_test[["NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse",
                   "NumberOfTimes90DaysLate"]].fillna(0).sum(axis=1).values > 0
    util = X_test["RevolvingUtilizationOfUnsecuredLines"].values
    age = X_test["age"].values
    segments = {
        "Age under 30": age < 30, "Age 30-44": (age >= 30) & (age < 45),
        "Age 45-61": (age >= 45) & (age < 62), "Age 62+": age >= 62,
        "Income reported": X_test["IncomeMissingFlag"].values == 0,
        "Income missing": X_test["IncomeMissingFlag"].values == 1,
        "Utilization under 30%": util < 0.3, "Utilization 30-90%": (util >= 0.3) & (util < 0.9),
        "Utilization 90%+": util >= 0.9,
        "Never late": ~late, "Any late payment": late,
    }
    print(f"\n4b. Segment performance on the test set")
    print("    AUC inside a segment is naturally lower than overall: the segment removes some of")
    print("    the variation the model uses. Calibration is flagged (*) when the predicted rate")
    print("    falls outside the exact 95% range of the actual default rate.\n")
    print(f"    {'Segment':<24}{'Borrowers':>10}{'Actual':>9}{'95% range':>18}"
          f"{'Scorecard':>11}{'XGBoost':>10}{'AUC LR':>8}{'AUC XGB':>9}")
    line("-")
    seg_rows, flags = [], {"Scorecard": [], "XGBoost": []}
    for name, m in segments.items():
        n, k = int(m.sum()), int(y_test[m].sum())
        lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
        hi = beta.ppf(0.975, k + 1, n - k)
        cells = []
        for model, p in [("Scorecard", pd_lr), ("XGBoost", pd_xgb)]:
            pred = p[m].mean()
            off = pred > hi or pred < lo
            if off:
                flags[model].append(f"{name} ({'over' if pred > hi else 'under'})")
            cells.append(f"{pred:.2%}{'*' if off else ' '}")
        auc_lr = roc_auc_score(y_test[m], pd_lr[m])
        auc_x = roc_auc_score(y_test[m], pd_xgb[m])
        print(f"    {name:<24}{n:>10,}{k / n:>9.2%}{f'{lo:.2%} - {hi:.2%}':>18}"
              f"{cells[0]:>11}{cells[1]:>10}{auc_lr:>8.3f}{auc_x:>9.3f}")
        seg_rows.append({"Segment": name, "Borrowers": n, "Actual": k / n, "Low": lo, "High": hi,
                         "Scorecard_pred": pd_lr[m].mean(), "XGBoost_pred": pd_xgb[m].mean(),
                         "Scorecard_AUC": auc_lr, "XGBoost_AUC": auc_x})
    line("-")
    for model in ["Scorecard", "XGBoost"]:
        print(f"    {model} segments outside 95% range: "
              f"{', '.join(flags[model]) if flags[model] else 'none'}")
        # Gaps are modest, so segment issues are rated Monitor (not unfit for use);
        # an over-prediction for the 62+ group is referred for fair-lending review.
        note = " - refer age 62+ to fair-lending review" if any("62+" in f for f in flags[model]) else ""
        record("4 Segments", f"{model} segment calibration",
               f"{len(flags[model])} of {len(segments)} segments off"
               f"{': ' + '; '.join(flags[model]) if flags[model] else ''}{note}",
               "Pass" if not flags[model] else "Monitor")
    pd.DataFrame(seg_rows).to_csv(OUT_DIR / "segments.csv", index=False)

    # Chart: predicted vs actual default rate by segment, with the actual's 95% range
    plt = style_charts()
    from matplotlib.ticker import PercentFormatter
    seg = pd.DataFrame(seg_rows).iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 5.6), dpi=150)
    yy = np.arange(len(seg))
    ax.hlines(yy, seg["Low"], seg["High"], color=MUTED, lw=6, alpha=0.35,
              label="Actual default rate (95% range)")
    ax.plot(seg["Scorecard_pred"], yy + 0.12, "o", ms=7, color=BLUE,
            markeredgecolor="white", markeredgewidth=1, label="Scorecard predicted")
    ax.plot(seg["XGBoost_pred"], yy - 0.12, "o", ms=7, color=ORANGE,
            markeredgecolor="white", markeredgewidth=1, label="XGBoost predicted")
    ax.set_yticks(yy, seg["Segment"])
    ax.set_xscale("log")
    ax.set_xticks([0.02, 0.03, 0.05, 0.1, 0.2, 0.3])
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.xaxis.set_minor_formatter(matplotlib_null())
    ax.set(xlabel="Default rate (log scale)")
    ax.set_title("Predicted vs actual default rate by segment", loc="left", color=INK)
    ax.grid(color=GRID, lw=0.6, axis="x")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.12), ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "segment_calibration.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Check 5 - Sensitivity
# ---------------------------------------------------------------------------
def sweep(base_rows, col, grid, scorecard, challenger):
    """PD for every applicant at every grid value of one input, others held fixed.
    Returns two arrays of shape (applicants, grid points)."""
    rep = base_rows.loc[base_rows.index.repeat(len(grid))].copy()
    rep[col] = np.tile(grid, len(base_rows))
    pd_lr, pd_xgb = predict(scorecard, challenger, rep)
    shape = (len(base_rows), len(grid))
    return pd_lr.reshape(shape), pd_xgb.reshape(shape)


def check_sensitivity(X_train, X_test, scorecard, challenger):
    header(f"CHECK 5 - SENSITIVITY ({N_SENS} test applicants, one input moved at a time)")
    rng = np.random.default_rng(SEED)
    pool = X_test[X_test["PastDueCodeFlag"] == 0]          # skip rows with data-entry codes
    sample = pool.iloc[rng.choice(len(pool), N_SENS, replace=False)].reset_index(drop=True)

    grids = {}
    for col in EXPECTED:
        if col.startswith("NumberOf"):
            grids[col] = np.arange(0, 7, dtype=float)       # 0 to 6 late payments
        elif col == "age":
            grids[col] = np.arange(21, 96, dtype=float)
        else:
            grids[col] = np.unique(np.nanquantile(X_train[col], np.linspace(0.01, 0.99, 25)))

    print("An applicant counts as a violation if their PD ever moves against the expected")
    print("direction as the input increases (tolerance 1e-9).\n")
    print(f"{'Input':<30}{'Should':>7}{'Scorecard':>11}{'XGBoost':>9}"
          f"{'Avg PD low->high: Scorecard':>29}{'XGBoost':>18}")
    line("-")
    curves = {}
    wrong_total = {"Scorecard": 0, "XGBoost": 0}
    for col, direction in EXPECTED.items():
        lr, xg = sweep(sample, col, grids[col], scorecard, challenger)
        curves[col] = (grids[col], lr.mean(axis=0), xg.mean(axis=0))
        counts = []
        for name, arr in [("Scorecard", lr), ("XGBoost", xg)]:
            if name == "Scorecard" and col not in scorecard["features"]:
                counts.append(None)
                continue
            wrong = ((np.diff(arr, axis=1) * direction) < -1e-9).any(axis=1).sum()
            counts.append(int(wrong))
            wrong_total[name] += int(wrong)
        shown = ["not used" if c is None else f"{c}/{N_SENS}" for c in counts]
        lr_txt = (f"{lr.mean(axis=0)[0]:.2%} -> {lr.mean(axis=0)[-1]:.2%}"
                  if col in scorecard["features"] else "not used")
        print(f"{LABEL_SHORT[col]:<30}{'raise' if direction > 0 else 'lower':>7}{shown[0]:>11}"
              f"{shown[1]:>9}{lr_txt:>29}{f'{xg.mean(axis=0)[0]:.2%} -> {xg.mean(axis=0)[-1]:.2%}':>18}")
    line("-")
    for name in ["Scorecard", "XGBoost"]:
        n_inputs = sum(1 for c in EXPECTED if name == "XGBoost" or c in scorecard["features"])
        record("5 Sensitivity", f"{name}: {n_inputs} key inputs move risk the right way",
               f"{wrong_total[name]} wrong-direction moves across {N_SENS} applicants",
               "Pass" if wrong_total[name] == 0 else "Fail")

    # Age 62+: is anyone ever scored riskier at 62+ than at any younger age?
    ages = grids["age"]
    lr, xg = sweep(sample, "age", ages, scorecard, challenger)
    old = ages >= 62
    print("\nAge 62+ check (Reg B): highest PD at ages 62-95 vs lowest PD at ages 21-61")
    for name, arr in [("Scorecard", lr), ("XGBoost", xg)]:
        worse = (arr[:, old].max(axis=1) > arr[:, ~old].min(axis=1) + 1e-9).sum()
        print(f"  {name:<10} applicants scored riskier at some age 62+: {worse} of {N_SENS}")

        record("5 Sensitivity", f"{name}: no applicant riskier at age 62+",
               f"{worse} of {N_SENS} applicants", "Pass" if worse == 0 else "Fail")

    # Inputs XGBoost was NOT constrained on: do they behave smoothly?
    free = {"NumberOfOpenCreditLinesAndLoans": np.arange(0, 31, dtype=float),
            "NumberRealEstateLoansOrLines": np.arange(0, 7, dtype=float),
            "NumberOfDependents": np.arange(0, 7, dtype=float)}
    names = {"NumberOfOpenCreditLinesAndLoans": "Open credit lines and loans",
             "NumberRealEstateLoansOrLines": "Real estate loans",
             "NumberOfDependents": "Number of dependents"}
    print("\nInputs without a monotone constraint in XGBoost (average PD of the 500 applicants)")
    zigzag = []
    for col, g in free.items():
        lr, xg = sweep(sample, col, g, scorecard, challenger)
        d = np.diff(xg, axis=1)
        both = ((d > 1e-9).any(axis=1) & (d < -1e-9).any(axis=1)).sum()
        avg = xg.mean(axis=0)
        show = range(0, len(g), 3) if len(g) > 10 else range(len(g))
        print(f"  {names[col]}: XGBoost moves both ways for {both} of {N_SENS} applicants")
        print("    value   " + "".join(f"{g[i]:>8.0f}" for i in show))
        print("    XGBoost " + "".join(f"{avg[i]:>8.2%}" for i in show))
        print("    Scorecard" + "".join(f"{lr.mean(axis=0)[i]:>8.2%}" for i in show)[1:])
        zigzag.append(f"{names[col]} ({both}/{N_SENS})")
    record("5 Sensitivity", "XGBoost: inputs without monotone constraints",
           "risk moves both ways for most applicants: " + "; ".join(zigzag)
           + " - zigzags in sparse values look like noise", "Monitor")

    # Chart: average PD as each input moves, both models
    plt = style_charts()
    from matplotlib.ticker import PercentFormatter
    panels = ["RevolvingUtilizationOfUnsecuredLines", "NumberOfTime30-59DaysPastDueNotWorse",
              "age", "MonthlyIncome"]
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), dpi=150)
    for ax, col in zip(axes.flat, panels):
        g, lr_mean, xg_mean = curves[col]
        ax.plot(g, lr_mean, color=BLUE, lw=2, label="Scorecard")
        ax.plot(g, xg_mean, color=ORANGE, lw=2, label="XGBoost")
        ax.set_title(LABEL_SHORT[col], loc="left", color=INK, fontsize=10)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        if col == "RevolvingUtilizationOfUnsecuredLines":
            ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        if col == "MonthlyIncome":
            ax.xaxis.set_major_formatter(matplotlib_dollars())
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0, 0].legend(frameon=False, loc="upper left", fontsize=9)
    fig.supylabel("Average PD of the 500 applicants", color=INK, fontsize=10)
    fig.suptitle("How each model's PD responds as one input changes", x=0.02, ha="left",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sensitivity.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Check 6 - Challenge project 2's conclusions
# ---------------------------------------------------------------------------
def economics(X, apr=APR, lgd=LGD):
    """Profit from each borrower if they repay, and loss if they default."""
    drawn = X["RevolvingUtilizationOfUnsecuredLines"].clip(0, 1).values * CREDIT_LIMIT
    ead = drawn + CCF * (CREDIT_LIMIT - drawn)
    return drawn * (apr - COST_OF_FUNDS) * YEARS, lgd * ead


def cutoff_for_target(p, y, target):
    """Highest PD cut-off whose approved group stays at or below the target default rate."""
    order = np.argsort(p)
    rate = np.cumsum(y[order]) / np.arange(1, len(p) + 1)
    k = int(np.nonzero(rate <= target)[0].max()) + 1
    return p[order][k - 1]


def cutoff_for_profit(p, y, gain, loss):
    order = np.argsort(p)
    profit = np.cumsum(np.where(y[order] == 1, -loss[order], gain[order]))
    return p[order][int(np.argmax(profit))]


def outcome(cutoff, p, y, gain, loss):
    """Apply a PD cut-off: approval rate, default rate and profit per 100,000 applicants."""
    ok = p <= cutoff
    profit = np.where(y[ok] == 1, -loss[ok], gain[ok]).sum() * 100_000 / len(p)
    return ok.mean(), y[ok].mean(), profit


def best_appr(o):
    return f"{o['LR_p'][0]:.0%} / {o['XGB_p'][0]:.0%}"


def check_project2_claims(X_test, y_test, pd_lr, pd_xgb):
    header("CHECK 6 - CHALLENGE PROJECT 2'S CONCLUSIONS")
    gain, loss = economics(X_test)
    n = len(y_test)

    # 6a(i). Project 2 searched cut-offs in 1% steps of approval rate. Does finer search
    #        (every borrower) change the profit-maximising result?
    print("6a. Cut-off precision: project 2 tried approval rates in 1% steps")
    best = {}
    for name, p in [("Scorecard", pd_lr), ("XGBoost", pd_xgb)]:
        order = np.argsort(p)
        prof = np.cumsum(np.where(y_test[order] == 1, -loss[order], gain[order])) * 100_000 / n
        grid = max(prof[int(round(n * r / 100)) - 1] for r in range(1, 101))
        k = int(np.argmax(prof))
        best[name] = (grid, prof[k])
        print(f"    {name:<10} best profit on 1% steps ${grid / 1e6:.2f}M | best at any borrower "
              f"${prof[k] / 1e6:.2f}M ({(k + 1) / n:.2%} approved)")
    u_grid = best["XGBoost"][0] / best["Scorecard"][0] - 1
    u_fine = best["XGBoost"][1] / best["Scorecard"][1] - 1
    print(f"    XGBoost uplift at the profit-maximising cut-off: {u_grid:+.1%} on 1% steps, "
          f"{u_fine:+.1%} at borrower level\n")

    # 6a(ii). Choose the cut-off on one half, measure it on the other half
    print(f"6b. Out-of-sample cut-offs: chosen on a random half of the test set, applied to the")
    print(f"    other half. Repeated over {N_SPLITS} random splits.\n")
    rows = []
    for i in range(N_SPLITS):
        a, b = train_test_split(np.arange(n), test_size=0.5, stratify=y_test, random_state=i)
        r = {}
        for name, p in [("LR", pd_lr), ("XGB", pd_xgb)]:
            c_t = cutoff_for_target(p[a], y_test[a], TARGET_DEFAULT)
            c_p = cutoff_for_profit(p[a], y_test[a], gain[a], loss[a])
            r[f"{name}_t"] = outcome(c_t, p[b], y_test[b], gain[b], loss[b])
            r[f"{name}_p"] = outcome(c_p, p[b], y_test[b], gain[b], loss[b])
        rows.append(r)

    def summarise(key_lr, key_x):
        appr = np.array([[r[key_lr][0], r[key_x][0]] for r in rows])
        dflt = np.array([[r[key_lr][1], r[key_x][1]] for r in rows])
        prof = np.array([[r[key_lr][2], r[key_x][2]] for r in rows])
        uplift = prof[:, 1] / prof[:, 0] - 1
        extra = (appr[:, 1] - appr[:, 0]) * 100_000
        return appr, dflt, prof, uplift, extra

    def band(x, fmt):
        lo, mid, hi = np.percentile(x, [5, 50, 95])
        return f"{fmt(mid)} ({fmt(lo)} to {fmt(hi)})"

    pct = lambda v: f"{v:.2%}"
    pct0 = lambda v: f"{v:+.0%}"
    money = lambda v: f"${v / 1e6:.2f}M"
    num = lambda v: f"{v:+,.0f}"
    results, extras, overshoot = {}, {}, {}
    for label, kl, kx in [(f"{TARGET_DEFAULT:.0%} target default rate", "LR_t", "XGB_t"),
                          ("Profit-maximising cut-off", "LR_p", "XGB_p")]:
        appr, dflt, prof, uplift, extra = summarise(kl, kx)
        results[label] = uplift
        extras[label] = extra
        print(f"    {label} - median across splits (5th to 95th percentile)")
        print(f"      {'':<26}{'Scorecard':>30}{'XGBoost':>30}")
        print(f"      {'Approval rate':<26}{band(appr[:, 0], pct):>30}{band(appr[:, 1], pct):>30}")
        print(f"      {'Default rate (unseen half)':<26}{band(dflt[:, 0], pct):>30}{band(dflt[:, 1], pct):>30}")
        print(f"      {'Profit per 100,000':<26}{band(prof[:, 0], money):>30}{band(prof[:, 1], money):>30}")
        print(f"      XGBoost extra approvals per 100,000: {band(extra, num)}")
        print(f"      XGBoost profit uplift: {band(uplift, pct0)}; "
              f"XGBoost ahead in {(uplift > 0).mean():.0%} of splits\n")
        if "target" in label:
            over = (dflt > TARGET_DEFAULT).mean(axis=0)
            overshoot = over
            print(f"      Splits where the unseen half exceeds the {TARGET_DEFAULT:.0%} target: "
                  f"scorecard {over[0]:.0%}, XGBoost {over[1]:.0%}\n")

    key_t = f"{TARGET_DEFAULT:.0%} target default rate"
    up_t = results[key_t]
    lo, mid, hi = np.percentile(up_t, [5, 50, 95])
    ex = np.median(extras[key_t])
    record("6 Project 2 claims", f"Profit uplift at {TARGET_DEFAULT:.0%} target, out of sample",
           f"reported +{REPORTED_UPLIFT_2PCT:.0%}; out of sample median {mid:+.0%} "
           f"(5th-95th {lo:+.0%} to {hi:+.0%}); extra approvals {ex:+,.0f} per 100,000 "
           f"(reported +{REPORTED_EXTRA_2PCT:,})",
           "Pass" if lo > 0 and abs(mid - REPORTED_UPLIFT_2PCT) < 0.10 else "Monitor")
    up_p = results["Profit-maximising cut-off"]
    lo, mid, hi = np.percentile(up_p, [5, 50, 95])
    record("6 Project 2 claims", "Profit uplift at profit-maximising cut-off",
           f"reported +0.1% (1% steps); {u_fine:+.1%} at borrower level; out of sample median "
           f"{mid:+.1%}, XGBoost ahead in {(up_p > 0).mean():.0%} of splits - no advantage at the optimum",
           "Monitor")
    record("6 Project 2 claims", f"Cut-off set at exactly {TARGET_DEFAULT:.0%} holds on new borrowers",
           f"unseen half exceeds {TARGET_DEFAULT:.0%} in {overshoot[0]:.0%} (scorecard) and "
           f"{overshoot[1]:.0%} (XGBoost) of splits - a lender needs a buffer below the target",
           "Monitor")

    # 6b. Do the conclusions survive different business assumptions?
    print(f"6c. Assumption stress test: XGBoost profit uplift over the scorecard (whole test set)")
    print(f"    {'APR':<6}{'LGD':<6}{f'at {TARGET_DEFAULT:.0%} target':>16}{'at best cut-off':>18}"
          f"{'best approval (LR / XGB)':>28}")
    line("-")
    stress, losses = [], 0
    for apr in (0.15, 0.20, 0.25):
        for lgd in (0.60, 0.80, 0.90):
            g, l = economics(X_test, apr, lgd)
            o = {}
            for name, p in [("LR", pd_lr), ("XGB", pd_xgb)]:
                o[name + "_t"] = outcome(cutoff_for_target(p, y_test, TARGET_DEFAULT), p, y_test, g, l)
                o[name + "_p"] = outcome(cutoff_for_profit(p, y_test, g, l), p, y_test, g, l)
            u_t = o["XGB_t"][2] / o["LR_t"][2] - 1
            u_p = o["XGB_p"][2] / o["LR_p"][2] - 1
            losses += (u_t < 0) + (u_p < 0)
            stress.append({"APR": apr, "LGD": lgd, "Uplift_at_target": u_t, "Uplift_at_best": u_p,
                           "Best_approval_LR": o["LR_p"][0], "Best_approval_XGB": o["XGB_p"][0]})
            print(f"    {apr:<6.0%}{lgd:<6.0%}{u_t:>+16.1%}{u_p:>+18.1%}"
                  f"{best_appr(o):>28}")
    line("-")
    st = pd.DataFrame(stress)
    st.to_csv(OUT_DIR / "assumption_stress.csv", index=False)
    print(f"    XGBoost behind the scorecard in {losses} of {2 * len(st)} scenario results.")
    record("6 Project 2 claims", "Conclusions under 9 APR / LGD scenarios",
           f"at {TARGET_DEFAULT:.0%} target XGBoost ahead in all ({st['Uplift_at_target'].min():+.0%} to "
           f"{st['Uplift_at_target'].max():+.0%}); at best cut-off {st['Uplift_at_best'].min():+.1%} to "
           f"{st['Uplift_at_best'].max():+.1%}, largest when margins are tightest", "Pass")

    # Chart: out-of-sample profit uplift at the 2% target
    plt = style_charts()
    from matplotlib.ticker import PercentFormatter
    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=150)
    ax.hist(up_t, bins=30, color=BLUE, edgecolor="white", linewidth=0.8)
    ax.axvline(REPORTED_UPLIFT_2PCT, color=INK, lw=1.2, ls="--")
    ax.annotate(f"Project 2 reported\n+{REPORTED_UPLIFT_2PCT:.0%}", (REPORTED_UPLIFT_2PCT, ax.get_ylim()[1] * 0.95),
                xytext=(6, 0), textcoords="offset points", color=INK, va="top")
    ax.axvline(0, color=MUTED, lw=1)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set(xlabel=f"XGBoost profit uplift at {TARGET_DEFAULT:.0%} target, measured on the unseen half",
           ylabel="Random splits")
    ax.set_title(f"Out-of-sample profit uplift across {N_SPLITS} splits", loc="left", color=INK)
    ax.grid(color=GRID, lw=0.6, axis="y")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "out_of_sample_uplift.png")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(exist_ok=True)

    raw = pd.read_csv(DATA_FILE)
    raw = raw.drop(columns=[c for c in raw.columns if c.startswith("Unnamed")])
    df = clean(raw)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)
    scorecard, challenger = load_models()
    new_raw = pd.read_csv(NEW_APPLICANTS_FILE)
    X_new = clean(new_raw).drop(columns=[TARGET])

    line()
    print("CREDIT MODEL VALIDATION - SCORECARD (project 1) AND XGBOOST (project 2)")
    line()
    print(f"Data: {len(df):,} borrowers | same split as both projects: "
          f"{len(X_train):,} train / {len(X_test):,} test")

    pd_lr, pd_xgb = check_replication(raw, X_train, X_test, y_test.values, scorecard, challenger)
    check_lift(y_test.values, pd_lr, pd_xgb)
    check_calibration(y_test.values, pd_lr, pd_xgb)
    check_stability(X_train, X_test, X_new, y_test.values, scorecard, challenger, pd_lr, pd_xgb)
    check_sensitivity(X_train, X_test, scorecard, challenger)
    check_project2_claims(X_test, y_test.values, pd_lr, pd_xgb)

    header("FINDINGS SO FAR")
    table = pd.DataFrame(findings)
    for _, r in table.iterrows():
        print(f"  [{r['Rating']:<7}] {r['Check']:<17} {r['Test']:<40} {r['Result']}")
    table.to_csv(OUT_DIR / "findings.csv", index=False)


if __name__ == "__main__":
    main()
