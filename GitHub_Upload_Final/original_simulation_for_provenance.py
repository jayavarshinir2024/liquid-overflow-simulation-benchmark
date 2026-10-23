"""
Simulation study: fixed vs. adaptive vs. learned fusion for overflow-risk warning.

ALL DATA ARE SIMULATED. Results describe algorithm behaviour under the
generator's assumptions, not real-world performance.

What this script does (compared with the original notebook)
-----------------------------------------------------------
1. Larger, balanced dataset (default 1000 cycles, 50 % overflow) with harder
   non-overflow cases (vigorous simmer, stirring) and sensor dropout.
2. Cycle-level TRAIN / VALIDATION / TEST split (60/20/20, stratified).
   Weights are fitted on train; thresholds are chosen on validation only;
   the test set is used once for reporting.
3. Causal warning time: a warning with 3-sample persistence is raised at the
   THIRD qualifying sample (the original code stamped the first one, which
   inflated lead time by 2 s for every method).
4. Baselines: ToF-only, ToF + rate-of-change, logistic regression (learned weights),
   gradient boosting with 10 s trend features, plus the equal-weight and
   heuristic-adaptive methods from the paper. Label horizons for the learned
   models were chosen on validation data only (see config).
5. Matched operating point: each method's threshold is the one that maximises
   validation lead time subject to validation false-alarm rate <= FAR_TARGET
   and detection >= MIN_DETECTION.
6. Multiple seeds (each seed regenerates data, split and fits), reported as
   mean and 95 % CI; paired Wilcoxon tests across seeds.
7. Drop-one-modality ablation (equal-weight and logistic).
8. Robustness: thresholds fixed from the nominal validation set, then applied to
   test sets with higher noise and more sensor dropout.
9. Lead-time / detection vs. false-alarm curves.

Run:  python overflow_simulation_study.py --seeds 10 --out results
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
N_CYCLES = 1000
P_OVERFLOW = 0.50
DURATION_S = 360                  # 1 Hz sampling, so index == time in seconds
PERSISTENCE = 3
THRESHOLDS = np.arange(1, 100)    # candidate thresholds on the 0-100 score
FAR_TARGET = 0.05                 # validation false-alarm budget
MIN_DETECTION = 0.90              # validation detection requirement
SPLIT = (0.6, 0.2, 0.2)           # train / validation / test
ROBUST_CYCLES = 300
# Frame labels for learned models: 'overflow within the next H seconds'.
# H was chosen on VALIDATION data only (3 development seeds, test sets untouched):
# a 30 s horizon works best for the linear model (60 s labels are too ambiguous
# for it), while gradient boosting handles the 60 s horizon better.
LOGIT_HORIZON_S = 30
GBM_HORIZON_S = 60

ACOUSTIC, HEIGHT, TEMP, HUMID = 0, 1, 2, 3
FEATURE_NAMES = ["acoustic", "height", "temperature", "humidity"]
TIME = np.arange(DURATION_S)

# Illustrative normalisation bounds (design choices, NOT calibrated physics).
# Acoustic: simulated peak frequency in Hz; height: ToF distance in cm;
# temperature in deg C; humidity in % RH.


def to_risk(a_hz, d_cm, t_c, h_pct):
    r_a = np.clip((a_hz - 65.0) / 335.0, 0, 1)
    r_h = np.clip((40.0 - d_cm) / 13.0, 0, 1)
    r_t = np.clip((t_c - 75.0) / 25.0, 0, 1)
    r_m = np.clip((h_pct - 40.0) / 48.0, 0, 1)
    return np.stack([r_a, r_h, r_t, r_m], axis=-1)


# --------------------------------------------------------------------------
# Synthetic generator
# --------------------------------------------------------------------------
def _hold(x, start, length):
    """Sample-and-hold gap: sensor repeats its last value (dropout)."""
    end = min(start + length, len(x))
    if start > 0 and end > start:
        x[start:end] = x[start - 1]


def simulate_cycle(rng, has_overflow, noise_mult=1.0, dropout_prob=0.2):
    heating_rate = rng.uniform(0.80, 1.20)
    foam_rate = rng.uniform(0.70, 1.30)
    noise_scale = rng.uniform(0.80, 1.40) * noise_mult
    acoustic_shift = rng.uniform(0.60, 1.30)
    vessel_offset = rng.uniform(-1.5, 1.5)

    temperature = 25 + 74 * (1 - np.exp(-TIME / (82 / heating_rate)))
    temperature += rng.normal(0, 0.45 * noise_scale, DURATION_S)
    temperature = np.clip(temperature, 15, 105)

    if has_overflow:
        overflow_time = int(rng.integers(270, 321))
        foam_start = overflow_time - int(rng.integers(45, 81))
        elapsed = np.maximum(TIME - foam_start, 0)
        foam = np.where(
            TIME > foam_start,
            np.minimum(13, 0.10 * foam_rate * elapsed
                       + 0.004 * foam_rate * elapsed ** 1.35),
            0.0)
        a_start = overflow_time - int(rng.integers(55, 91))
        a_second = overflow_time - int(rng.integers(10, 26))
        s1 = 1 / (1 + np.exp(-(TIME - a_start) / 7))
        s2 = 1 / (1 + np.exp(-(TIME - a_second) / 4))
        acoustic = 65 + 155 * acoustic_shift * s1 + 180 * s2
    else:
        overflow_time = -1
        # Hard negatives: vigorous simmer with sizeable but sub-rim foam.
        mild_start = int(rng.integers(170, 250))
        max_foam = rng.uniform(1.5, 6.5)
        elapsed = np.maximum(TIME - mild_start, 0)
        foam = np.where(TIME > mild_start,
                        np.minimum(max_foam, 0.03 * foam_rate * elapsed), 0.0)
        bg = 1 / (1 + np.exp(-(TIME - 190 / heating_rate) / 28))
        acoustic = 65 + rng.uniform(70, 150) * acoustic_shift * bg
        acoustic += 15 * np.sin(TIME / rng.uniform(18, 35))

    distance = 40 + vessel_offset - foam
    distance += rng.normal(0, 0.25 * noise_scale, DURATION_S)
    acoustic += rng.normal(0, 18 * noise_scale, DURATION_S)

    # Stirring episodes (both classes): short acoustic bursts and surface dips.
    if rng.random() < 0.5:
        for _ in range(int(rng.integers(1, 4))):
            start = int(rng.integers(60, DURATION_S - 20))
            length = int(rng.integers(5, 16))
            env = np.zeros(DURATION_S)
            env[start:start + length] = np.hanning(length + 2)[1:-1]
            acoustic += rng.uniform(60, 160) * env
            distance -= rng.uniform(0.5, 3.0) * env

    distance = np.clip(distance, 8, 50)
    acoustic = np.clip(acoustic, 10, 900)

    humidity = 40 + 48 / (1 + np.exp(-(TIME - 155 / heating_rate) / 24))
    humidity += rng.normal(0, 1.4 * noise_scale, DURATION_S)
    humidity = np.clip(humidity, 20, 100)

    # Sensor dropout: ToF and/or microphone freeze for a few seconds.
    if rng.random() < dropout_prob:
        for _ in range(int(rng.integers(1, 4))):
            start = int(rng.integers(30, DURATION_S - 15))
            length = int(rng.integers(3, 11))
            which = rng.integers(0, 3)           # 0 ToF, 1 acoustic, 2 both
            if which in (0, 2):
                _hold(distance, start, length)
            if which in (1, 2):
                _hold(acoustic, start, length)

    return acoustic, distance, temperature, humidity, overflow_time


def generate_dataset(seed, n_cycles=N_CYCLES, noise_mult=1.0, dropout_prob=0.2):
    rng = np.random.default_rng(seed)
    n_over = int(round(n_cycles * P_OVERFLOW))
    has = np.array([True] * n_over + [False] * (n_cycles - n_over))
    rng.shuffle(has)
    A = np.empty((n_cycles, DURATION_S)); D = np.empty_like(A)
    Tm = np.empty_like(A); H = np.empty_like(A)
    event_time = np.empty(n_cycles, dtype=int)
    for i in range(n_cycles):
        A[i], D[i], Tm[i], H[i], event_time[i] = simulate_cycle(
            rng, bool(has[i]), noise_mult, dropout_prob)
    R = to_risk(A, D, Tm, H)                                  # (N, T, 4)
    return R, has, event_time


def imminent_labels(has, event_time, horizon):
    gap = event_time[:, None] - TIME[None, :]
    return (has[:, None] & (gap > 0) & (gap <= horizon)).astype(int)


def stratified_split(rng, has, fractions=SPLIT):
    parts = {"train": [], "val": [], "test": []}
    for label in (True, False):
        idx = np.flatnonzero(has == label)
        rng.shuffle(idx)
        n_tr = int(round(fractions[0] * len(idx)))
        n_va = int(round(fractions[1] * len(idx)))
        parts["train"].append(idx[:n_tr])
        parts["val"].append(idx[n_tr:n_tr + n_va])
        parts["test"].append(idx[n_tr + n_va:])
    return {k: np.concatenate(v) for k, v in parts.items()}


# --------------------------------------------------------------------------
# Scoring methods (each returns an (N, T) array on a 0-100 scale)
# --------------------------------------------------------------------------
def score_fixed(R, cols=(0, 1, 2, 3)):
    return 100.0 * R[..., list(cols)].mean(axis=-1)


def score_adaptive(R):
    """Heuristic adaptive weights exactly as defined in the paper (Sec. 3.4.2)."""
    a, h, t, m = (R[..., k] for k in range(4))
    primary = np.maximum(a, h)
    q = np.stack([0.30 * (1 + 2 * a),
                  0.40 * (1 + 2 * h),
                  0.20 * (0.25 + 0.75 * primary),
                  0.10 * (0.25 + 0.75 * primary)], axis=-1)
    w = q / q.sum(axis=-1, keepdims=True)
    return 100.0 * (w * R).sum(axis=-1)


def score_tof_only(R):
    return 100.0 * R[..., HEIGHT]


def score_tof_roc(R, lag=10):
    """Hand-designed baseline: ToF level blended with its 10 s rise (causal)."""
    rh = R[..., HEIGHT]
    prev = np.concatenate([np.repeat(rh[:, :1], lag, axis=1), rh[:, :-lag]], axis=1)
    rise = np.clip((rh - prev) / 0.2, 0, 1)
    return 100.0 * np.clip(0.5 * rh + 0.5 * rise, 0, 1)


def _rise(x, lag=10):
    """Causal change over the last `lag` seconds (first samples padded)."""
    prev = np.concatenate([np.repeat(x[:, :1], lag, axis=1), x[:, :-lag]], axis=1)
    return x - prev


def _features(R, cols, temporal):
    feats = [R[..., list(cols)]]
    if temporal:
        feats.append(np.stack([_rise(R[..., ACOUSTIC]), _rise(R[..., HEIGHT])], axis=-1))
    return np.concatenate(feats, axis=-1)


def fit_logistic(R_train, y_train, cols=(0, 1, 2, 3)):
    """Frame-level logistic regression on the risk features (learned weights)."""
    cols = list(cols)
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    clf.fit(R_train[..., cols].reshape(-1, len(cols)), y_train.ravel())

    def scorer(R):
        p = clf.predict_proba(R[..., cols].reshape(-1, len(cols)))[:, 1]
        return 100.0 * p.reshape(R.shape[:-1])
    scorer.coef_ = clf.coef_[0]
    return scorer


def fit_gbm(R_train, y_train):
    """Gradient boosting on the four risk features plus 10 s acoustic/height trends."""
    F = _features(R_train, (0, 1, 2, 3), temporal=True)
    k = F.shape[-1]
    clf = HistGradientBoostingClassifier(max_depth=3, max_iter=100,
                                         learning_rate=0.1, random_state=0)
    clf.fit(F.reshape(-1, k), y_train.ravel())

    def scorer(R):
        Fr = _features(R, (0, 1, 2, 3), temporal=True)
        p = clf.predict_proba(Fr.reshape(-1, k))[:, 1]
        return 100.0 * p.reshape(R.shape[:-1])
    return scorer


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def sweep(scores, has, event_time, thresholds=THRESHOLDS, persistence=PERSISTENCE):
    """Cycle-level metrics for every threshold (vectorised, causal warning)."""
    T = scores.shape[1]
    windows = [scores[:, k:T - (persistence - 1) + k] for k in range(persistence)]
    pm = np.minimum.reduce(windows)            # min over each 3-sample window
    cm = np.maximum.accumulate(pm, axis=1)     # first time a window clears thr
    n_thr = len(thresholds)
    det_rate = np.empty(n_thr); far = np.empty(n_thr)
    mean_lead = np.full(n_thr, np.nan); median_lead = np.full(n_thr, np.nan)
    precision = np.full(n_thr, np.nan)
    n_over, n_non = has.sum(), (~has).sum()
    for j, thr in enumerate(thresholds):
        reached = cm[:, -1] >= thr
        first = (cm >= thr).argmax(axis=1)
        wt = np.where(reached, first + persistence - 1, -1)   # warning at 3rd sample
        detected = has & (wt >= 0) & (wt < event_time)
        false_alarm = (~has) & reached
        det_rate[j] = detected.sum() / n_over
        far[j] = false_alarm.sum() / n_non
        if detected.any():
            lead = (event_time - wt)[detected]
            mean_lead[j] = lead.mean(); median_lead[j] = np.median(lead)
        if detected.sum() + false_alarm.sum() > 0:
            precision[j] = detected.sum() / (detected.sum() + false_alarm.sum())
    return dict(det=det_rate, far=far, lead=mean_lead,
                median_lead=median_lead, precision=precision)


def select_threshold(val):
    """Highest validation lead time subject to FAR / detection constraints."""
    ok = (val["far"] <= FAR_TARGET) & (val["det"] >= MIN_DETECTION)
    if ok.any():
        lead = np.where(ok, np.nan_to_num(val["lead"], nan=-1), -np.inf)
        return int(np.argmax(lead))
    ok = val["far"] <= FAR_TARGET
    if ok.any():
        det = np.where(ok, val["det"], -np.inf)
        return int(np.argmax(det))
    return len(val["far"]) - 1


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------
def build_methods(R_train, has_tr, ev_tr):
    y_logit = imminent_labels(has_tr, ev_tr, LOGIT_HORIZON_S)
    y_gbm = imminent_labels(has_tr, ev_tr, GBM_HORIZON_S)
    logistic = fit_logistic(R_train, y_logit)
    methods = {
        "Fixed (equal weights)": score_fixed,
        "Adaptive (heuristic)": score_adaptive,
        "ToF only": score_tof_only,
        "ToF + rate of change": score_tof_roc,
        "Logistic (learned weights)": logistic,
        "Gradient boosting (+10 s trends)": fit_gbm(R_train, y_gbm),
    }
    return methods, logistic


def build_ablation(R_train, has_tr, ev_tr):
    y_logit = imminent_labels(has_tr, ev_tr, LOGIT_HORIZON_S)
    methods = {}
    for k, name in enumerate(FEATURE_NAMES):
        cols = [c for c in range(4) if c != k]
        methods[f"Fixed w/o {name}"] = (lambda R, cols=tuple(cols): score_fixed(R, cols))
        methods[f"Logistic w/o {name}"] = fit_logistic(R_train, y_logit, cols)
    return methods


def run_seed(seed, robust_conditions):
    R, has, ev = generate_dataset(seed)
    split = stratified_split(np.random.default_rng(seed + 10_000), has)
    tr, va, te = split["train"], split["val"], split["test"]
    methods, logistic = build_methods(R[tr], has[tr], ev[tr])
    ablation = build_ablation(R[tr], has[tr], ev[tr])

    rows, curves = [], {}
    chosen = {}

    def evaluate(group, methods_, condition, R_test, has_t, ev_t, keep_curves=False):
        for name, fn in methods_.items():
            if name not in chosen:
                val = sweep(fn(R[va]), has[va], ev[va])
                chosen[name] = select_threshold(val)
            j = chosen[name]
            res = sweep(fn(R_test), has_t, ev_t)
            rows.append(dict(
                seed=seed, group=group, condition=condition, method=name,
                threshold=int(THRESHOLDS[j]),
                detection_rate=res["det"][j], false_alarm_rate=res["far"][j],
                precision=res["precision"][j], mean_lead_s=res["lead"][j],
                median_lead_s=res["median_lead"][j],
                n_overflow=int(has_t.sum()), n_non_overflow=int((~has_t).sum())))
            if keep_curves:
                curves[name] = res

    evaluate("main", methods, "nominal", R[te], has[te], ev[te], keep_curves=True)
    evaluate("ablation", ablation, "nominal", R[te], has[te], ev[te])
    for label, (noise_mult, dropout_prob) in robust_conditions.items():
        Rr, hr, er = generate_dataset(seed + 777, ROBUST_CYCLES,
                                         noise_mult, dropout_prob)
        evaluate("robustness", methods, label, Rr, hr, er)
    return rows, curves, logistic.coef_


def ci95(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def summarise(df, by):
    metrics = ["detection_rate", "false_alarm_rate", "precision",
               "mean_lead_s", "median_lead_s", "threshold"]
    out = df.groupby(by)[metrics].agg(["mean", ci95])
    out.columns = [f"{m}_{s if isinstance(s, str) else s.__name__}"
                   for m, s in out.columns]
    return out.reset_index()


def fmt(df, by, metrics):
    s = summarise(df, by)
    out = s[by].copy()
    for m in metrics:
        scale = 100 if "rate" in m or m == "precision" else 1
        out[m] = [f"{scale * a:.1f} ± {scale * b:.1f}"
                  for a, b in zip(s[f"{m}_mean"], s[f"{m}_ci95"])]
    return out


def paired_tests(df, pairs):
    nominal = df[(df.group == "main")]
    out = []
    for a, b in pairs:
        xa = nominal[nominal.method == a].sort_values("seed")
        xb = nominal[nominal.method == b].sort_values("seed")
        for metric in ("mean_lead_s", "detection_rate", "false_alarm_rate"):
            d = xa[metric].to_numpy() - xb[metric].to_numpy()
            try:
                p = stats.wilcoxon(xa[metric], xb[metric]).pvalue
            except ValueError:               # all differences are zero
                p = np.nan
            out.append(dict(method_a=a, method_b=b, metric=metric,
                            mean_diff_a_minus_b=np.nanmean(d), wilcoxon_p=p,
                            n_seeds=len(d)))
    return pd.DataFrame(out)


def plot_curves(all_curves, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name in all_curves[0]:
        far = np.mean([c[name]["far"] for c in all_curves], axis=0)
        det = np.mean([c[name]["det"] for c in all_curves], axis=0)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lead = np.nanmean([c[name]["lead"] for c in all_curves], axis=0)
        axes[0].plot(100 * far, 100 * det, label=name)
        axes[1].plot(100 * far, lead, label=name)
    axes[0].set_xlabel("False-alarm rate on non-overflow cycles (%)")
    axes[0].set_ylabel("Detection rate (%)")
    axes[0].set_title("Detection vs. false alarms (test, mean over seeds)")
    axes[1].set_xlabel("False-alarm rate on non-overflow cycles (%)")
    axes[1].set_ylabel("Mean warning lead time (s, detected cycles)")
    axes[1].set_title("Lead time vs. false alarms (test, mean over seeds)")
    axes[1].axvline(100 * FAR_TARGET, ls=":", color="grey")
    axes[0].axvline(100 * FAR_TARGET, ls=":", color="grey")
    for ax in axes:
        ax.set_xlim(0, 30)
    axes[1].set_ylim(0, 60)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "fig_tradeoff_curves.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    robust_conditions = {
        "noise x2": (2.0, 0.2),
        "noise x3": (3.0, 0.2),
        "dropout in 80% of cycles": (1.0, 0.8),
    }
    rows, all_curves, coefs = [], [], []
    for k in range(args.seeds):
        seed = 1000 + k
        r, c, w = run_seed(seed, robust_conditions)
        rows += r; all_curves.append(c); coefs.append(w)
        print(f"seed {seed} done")

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "per_seed_results.csv", index=False)

    main_tbl = fmt(df[df.group == "main"], ["method"],
                   ["detection_rate", "false_alarm_rate", "precision",
                    "mean_lead_s", "median_lead_s", "threshold"])
    abl_tbl = fmt(df[df.group == "ablation"], ["method"],
                  ["detection_rate", "false_alarm_rate", "mean_lead_s"])
    rob_tbl = fmt(df[df.group == "robustness"], ["condition", "method"],
                  ["detection_rate", "false_alarm_rate", "mean_lead_s"])
    main_tbl.to_csv(outdir / "table_main.csv", index=False)
    abl_tbl.to_csv(outdir / "table_ablation.csv", index=False)
    rob_tbl.to_csv(outdir / "table_robustness.csv", index=False)

    tests = paired_tests(df, [("Fixed (equal weights)", "Adaptive (heuristic)"),
                              ("Logistic (learned weights)", "Fixed (equal weights)"),
                              ("Logistic (learned weights)", "ToF only"),
                              ("Gradient boosting (+10 s trends)", "Fixed (equal weights)"),
                              ("Gradient boosting (+10 s trends)", "Adaptive (heuristic)")])
    tests.to_csv(outdir / "table_paired_tests.csv", index=False)
    plot_curves(all_curves, outdir)

    coef = pd.DataFrame(coefs, columns=FEATURE_NAMES)
    coef.agg(["mean", "std"]).to_csv(outdir / "logistic_coefficients.csv")

    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    n_test = df[df.group == "main"].iloc[0]
    print(f"\nTest set per seed: {n_test.n_overflow} overflow / "
          f"{n_test.n_non_overflow} non-overflow cycles; {args.seeds} seeds")
    print("\n=== MAIN (mean ± 95% CI over seeds; %, s) ===")
    print(main_tbl.to_string(index=False))
    print("\n=== ABLATION ===");  print(abl_tbl.to_string(index=False))
    print("\n=== ROBUSTNESS (thresholds fixed from nominal validation) ===")
    print(rob_tbl.to_string(index=False))
    print("\n=== PAIRED WILCOXON ACROSS SEEDS ===")
    print(tests.to_string(index=False))
    print("\nLogistic coefficients (mean over seeds):")
    print(coef.mean().round(2).to_string())


if __name__ == "__main__":
    main()
