# What is complete and what you should do next

The computation and manuscript have been rebuilt as a **simulation benchmark**.
Do not submit this as a working or experimentally validated overflow device.

| Requested fix | Status |
|---|---|
| 1. Consistent event definition | Implemented and checked: first generated rim crossing determines the label. All results rerun. Physical dynamics remain uncalibrated. |
| 2. Stronger comparisons | Implemented: nine pipelines, missing single/two-channel baselines, tuned constant weights, common learned inputs and horizon, separate inner tuning and final validation. |
| 3. Validate assumptions | Computational sensitivity tests completed for six stresses. Physical validation remains unperformed because no real recordings were supplied. A recording plan is included. |
| 4. Reproducibility | Code, pinned environment, fixed seeds, splits, selected settings, hashes, per-cycle audit records, tables, figures, checks, and rerun instructions included. Public repository: https://github.com/jayavarshinir2024/liquid-overflow-simulation-benchmark. No DOI has been assigned. |

The main results changed after correcting the model and threshold search.
In the final nominal benchmark, timely detection was 99.3% for ToF alone,
70.1% for the original adaptive rule, and 94.0% for optimized constant weights.
Their non-event false-alarm rates were 6.4%, 5.9%, and 5.7%, respectively;
these are different operating points, not an exactly matched false-alarm test.
Logistic regression and boosting achieved 95.8% and 96.4% detection with
longer conditional lead times (17.9 and 22.4 s). These results concern only
the specified synthetic model, which gives distance a direct relation to the
rim-based event. They do not establish real ToF accuracy or general superiority.

## Read these first

1. The separately supplied journal manuscript: revised manuscript with new methods,
   results, tables, figure, limitations, and expanded AI disclosure.
2. `MODELLING_ASSUMPTIONS.md`: exactly what was assumed and what the metrics mean.
3. `README.md`: run instructions and contents of the reproducibility package.
4. `PHYSICAL_VALIDATION_PLAN.md`: what measurements are needed next.

## Your next decisions

- Review the generator and outcomes with your faculty guide. Ask whether a
  simulation-only comparison is suitable for your intended submission.
- If physical evidence is required, share the hardware available before buying
  parts. The plan starts with feasibility recordings rather than a promised
  large dataset or guaranteed performance.
- If proceeding as simulation-only, target a journal that accepts this scope,
  follow its author instructions, and supply the complete supplementary package.
- Review and approve the AI disclosure. Assistance now included code, design,
  analysis, debugging and writing; a language-only declaration would be incomplete.
- Before submission, all authors should understand and verify the code and
  conclusions. A code archive and passing checks do not establish novelty,
  journal acceptance, or a particular similarity/detection score.

The earlier manuscript's 97.9–99.9% fusion result belongs to a different
generator and event definition. Do not mix those results with the new version.
