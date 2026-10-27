# Overflow warning simulation study version 2

This package contains a corrected, rerunnable **synthetic benchmark**. It is not
a tested kitchen device. No physical sensor data were supplied or collected.
The parameter values are illustrative assumptions, not fits to the cited papers.

## What changed

1. An event is the first simulated surface-height sample that reaches the
   vessel rim. Labels are computed from the latent trajectory, never assigned
   in advance and then used to construct a desired acoustic signature.
2. The benchmark includes acoustic-only, ToF-only, acoustic plus ToF, equal
   four-channel fusion, the original adaptive heuristic, optimized constant
   weights, a tuned ToF/trend rule, logistic regression, and gradient boosting.
3. Both learned models use the same six features, the same 60 s target, the
   same available training frames, and balanced training weights. Their tuning
   is confined to an inner split of the training cycles. The constant-weight
   grid and ToF/trend rule use that same inner tuning subset.
4. The first warning must fall in the 60 s before a simulated crossing to count
   as a timely detection. Earlier warnings are counted separately, not rewarded
   as successful long-lead predictions. Monitoring ends at crossing or 360 s.
5. Seven test conditions examine nominal behaviour and six distinct stresses.
   Nominal validation thresholds are frozen for stress tests. A full manifest,
   cycle splits, per-cycle warning records, generator hashes, dependency
   versions, and fixed seeds are included.

## Run

Use Python 3.12 with the versions in `requirements.txt`. A clean virtual
environment is recommended. The versions reproduce the execution environment;
their availability on another platform has not been tested here.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -p 'test_*.py' -v
python audit_results.py
python overflow_study_v2.py --seeds 10 --out regenerated_results
```

The full run was executed with one numerical-computation thread. Set
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` before running if
needed. The script also limits thread pools during evaluation.

Add `--save-data` to export all raw and normalized simulated arrays as NPZ.
This can require hundreds of megabytes. Arrays are reproducible from seeds,
so the default package supplies hashes and a small trace example instead.
Recreate one dataset with `generate(seed, n, scenario)` in the source. Per-cycle
seeds in `cycle_warnings_v2.csv.gz` identify individual traces; use
`make_cycle(cycle_seed, scenario)` to reconstruct one.

## Files

- `overflow_study_v2.py`: generator, tuning, evaluation, summaries and figures.
- `test_scientific_invariants.py`: ten checks for event consistency, causal
  features, disjoint splits, label masking, persistence and metric behaviour.
- `results_v2/run_manifest.json`: complete run configuration and versions.
- `results_v2/configuration_SEED.json`: selected hyperparameters, weights,
  validation feasibility, split indices and array hashes for each seed.
- `results_v2/per_seed_results_v2.csv`: selected-threshold test metrics.
- `results_v2/cycle_warnings_v2.csv.gz`: audit trail of event/warning times.
- `results_v2/table_main_v2.csv`: nominal summary, mean and t-based 95% CI.
- `results_v2/table_stress_v2.csv`: stressed-condition summaries.
- `results_v2/paired_tests_holm_v2.csv`: paired exploratory tests and Holm p-values.
- `results_v2/threshold_curves_v2.csv.gz`: descriptive test threshold sweeps.
- `results_v2/example_traces.csv`: one event and one non-event trace.
- `MODELLING_ASSUMPTIONS.md`: equations, evaluation and scope limitations.
- `PHYSICAL_VALIDATION_PLAN.md`: future recording protocol; not completed work.
- `original_*_for_provenance.*`: original supplied script/table, not the v2 results.

The initial integer-threshold diagnostic was superseded after an audit found
a score-resolution artifact for saturated probabilities. See the modelling
specification; only the final rerun supplies the manuscript results.

## Research interpretation

Do not choose the winner or change assumptions to make adaptive fusion win.
If a simpler baseline performs better, report it. Ten seeds quantify variation
within this generator, not uncertainty about the real world. The source was
written before this evaluation, but was not externally preregistered.

The old manuscript numbers and figures are superseded for this revised model.
Use only the v2 results when describing this benchmark. Detection definitions,
event prevalence and generator assumptions differ, so the old and new headline
percentages are not directly comparable.

## Source and results availability

Repository: https://github.com/jayavarshinir2024/liquid-overflow-simulation-benchmark

This repository contains the source and generated outputs for the synthetic
benchmark. It is a research draft, not a validated device or an accepted paper.
The code and results can also accompany a manuscript as supplementary material.
No archival DOI or software license is assigned here. Select a license only
after all authors approve it.

AI assisted development included code, experimental design, analysis and text
revision in this version. A disclosure limited to language polishing would not
describe that scope. Authors must review the work and follow the target
journal's actual policy before submission.

## Added sensitivity study

See [SENSITIVITY_README.md](SENSITIVITY_README.md) for the post hoc extension with 5,000 fresh matched cycle seeds, six conditions, frozen nominal pipelines, complete per-cycle records, and two manuscript figures. Run `python sensitivity_study.py`, `python audit_sensitivity.py`, and `python plot_sensitivity.py` from this directory. The 15 scientific checks cover both the original benchmark and this extension. Original results and source remain unchanged.
