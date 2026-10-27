# Result fields

`per_seed_results_v2.csv`: one row per seed, condition and method (630 rows).
Rates are fractions in [0,1], not percentages. `n_event` and `n_nonevent`
record the actual test composition. `validation_feasible` states whether
both the validation detection and false-alarm requirements were met.

`detection_rate`: first warning in the 60 s before crossing / event cycles.
`false_alarm_rate`: any qualifying warning / non-event cycles.
`early_warning_rate`: first warning more than 60 s before crossing / event cycles.
`precision`: timely / (timely + premature + non-event warnings).
`mean_lead_s`, `median_lead_s`: conditional on timely detections; NaN if none.
`any_advance_detection`: secondary fraction with any pre-crossing first warning.
`threshold`: nominal-validation selected score, possibly above the maximum score.

`summary_v2.csv`: mean, t-based CI half-width and number of defined seeds for
each metric. A confidence interval is mean ± half-width. Do not interpret a
half-width as a standard deviation. Rates remain fractions in this file.

`cycle_warnings_v2.csv.gz`: one row per evaluated cycle/method (180180 rows).
`event_time=-1` means no crossing within the observation window.
`warning_time=-1` means no qualifying three-sample window in the trace.
The stored warning time is the first score threshold crossing in the full
generated trace; if it is at or after `event_time`, it is excluded from
advance/timely evaluation. It is not evidence of a physical post-event alarm.
`timely` records the primary warning outcome. Scores do not use future samples.

`cycle_seed` reconstructs the individual trace with `make_cycle(seed,scenario)`.
`freeboard_cm` is the known simulated initial distance to the rim.
The `seed` column identifies the full independent run; it is different from
the individual `cycle_seed`.

`threshold_curves_v2.csv.gz`: full nominal test sweeps. Candidate thresholds
are data-dependent and differ by seed. They are descriptive only. Plotting
averages outcomes sharing a false-alarm rate within each seed, then interpolates
onto a common false-alarm grid and averages seeds. Curves need not be monotonic
in timely detection because very early first warnings are not timely detections.

`paired_tests_holm_v2.csv`: differences are method_a minus method_b. Rate
differences are fractions; multiply by 100 for percentage points. Lead
differences are seconds. Tests use the asymptotic signed-rank approximation,
followed by Holm adjustment across the defined comparisons. Values are
exploratory with ten seeds and do not measure physical-device superiority.

## Added sensitivity outputs

The sensitivity_results/ CSV files use the original metric definitions. per_seed_sensitivity.csv adds test_seed to identify the fresh dataset paired with each trained seed. summary_sensitivity.csv reports means and t-based 95% CI half-widths across ten trained seed fits. cycle_warnings_sensitivity.csv.gz contains 270,000 cycle/method records. calibration_paired_differences.csv reports geometry-error minus fresh-nominal differences and CI half-widths in native metric units (multiply rates by 100 for percentage points). Conditions share cycle seeds; do not treat their observations as independent.
