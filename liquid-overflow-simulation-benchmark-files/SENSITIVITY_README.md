# Post hoc sensitivity extension

This extension tests how the original nine frozen warning pipelines respond to changes in the illustrative equations and geometry calibration. It uses no physical data and does not validate a kitchen device.

## Protocol and interpretation

The extension was planned after viewing the original results and before this extension was run. It is exploratory, not externally preregistered. Models are refitted on the original training cycles (seeds 24000–24009). Before any added test, the script asserts exact agreement of selected parameters and thresholds and numerical agreement of every original nominal metric, as well as the original dataset hashes. Neither models nor thresholds are tuned on the added conditions.

Test seeds 75000–75009 each produce 500 cycles. Six conditions share the same 5,000 cycle seeds: fresh nominal, heating exponents 0.8 and 1.2, growth exponents 1.5 and 2.5, and freeboard-estimation error. This gives 30,000 condition-cycles, not 30,000 independent trajectories.

Heating is T(t)=25+74[1-exp(-(t/tau)^q)], with nominal q=1. Growth uses g*b(t-1)^p, with nominal p=2; all other terms are unchanged. Labels follow the first rim crossing of each new trajectory. In the freeboard-error condition, r_est=r*U(0.8,1.2) affects only normalized height. Raw proxies, latent states, and event labels match fresh nominal exactly.

## Reproduce

Run from this project directory with the recorded requirements and original results_v2/ retained:

```bash
python -m unittest test_scientific_invariants test_sensitivity_invariants -v
python audit_results.py
python sensitivity_study.py
python audit_sensitivity.py
python plot_sensitivity.py
```

The extension overwrites sensitivity_results/ when rerun. To use a separate output directory, use `--out regenerated_sensitivity`; the audit and plotting scripts read the supplied sensitivity_results/ directory by default. The original source overflow_study_v2.py and original results_v2/ were not changed for this extension.

## Outputs

- sensitivity_manifest.json: seeds, equations, versions, design status and source hashes.
- frozen_settings_and_hashes.json: fixed settings, operating thresholds and generated-array hashes.
- per_seed_sensitivity.csv: 540 seed/condition/method rows.
- summary_sensitivity.csv: 54 condition/method rows; means and t-based 95% CI half-widths across ten fits.
- cycle_warnings_sensitivity.csv.gz: 270,000 cycle/method records.
- calibration_paired_differences.csv: geometry-error minus fresh-nominal paired differences; descriptive mean and 95% interval half-width.
- fig_workflow and fig_sensitivity, PNG and SVG: evaluation workflow and selected-method sensitivity results.
- sensitivity_scientific_checks.txt, sensitivity_audit.txt, original_results_reaudit.txt and sensitivity_execution_log.txt: recorded verification output.

All nine methods are retained in the CSV files; the figure shows four representative pipelines. Lead is conditional on timely detection. Detection rates are averaged across seed fits, not pooled as independent cycle observations. The sensitivity conditions are alternatives within the same uncalibrated generator family, not external physical validation.

## Key finding

Fresh nominal ToF-only detection was 99.5%, falling to 66.4% with freeboard-estimation error of ±20%. Its false-alarm rate increased from 6.0% to 8.4%. This exposes dependence on geometry calibration rather than establishes a hardware error specification.
