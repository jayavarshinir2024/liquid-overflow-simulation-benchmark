# Physical validation still to be performed

No real recordings have been supplied. Do not describe this plan as completed
experiments. Faculty approval and lab supervision should precede a hardware
campaign. The benchmark does not implement mains or gas control.

## Minimum useful first recording campaign

1. Confirm actual microphone, distance sensor, processor, temperature sensor,
   humidity sensor and acquisition software. Document models, ranges, sampling
   rates, filtering, mounting positions, and clock synchronization.
2. Record initial fill, vessel rim position, vessel dimensions/material, liquid
   composition, heat setting, initial temperature and background conditions.
   Distinguish liquid or contact temperature from ambient air temperature.
3. Start with a supervised feasibility pilot using milk and a non-overflowing
   reference condition. Check whether airborne sound and ToF traces remain
   usable near steam before promising a large collection campaign. Water-only
   recordings must not be the sole non-event class: otherwise liquid identity
   may substitute for learning overflow. Include milk runs that remain below
   the rim under appropriate controlled conditions.
4. Preserve raw audio at a documented audio sampling rate, native-rate distance,
   temperature and humidity values, valid/invalid flags and timestamps. An
   acoustic frequency feature sampled at 1 Hz is not a raw audio recording.
5. Use synchronized video of the vessel and a visible timing marker as an
   independent event reference. Define the event before analysis as first
   surface/foam contact with or crossing of a designated rim reference.
   Record annotation uncertainty and have a second person review ambiguous
   events. Document whether a trial was stopped before a crossing; do not
   label a stopped trial as a full-duration non-event without explanation.
6. Include matched event and non-event conditions across vessels, fill levels,
   heat settings, background sounds and stirring. Count independent heating
   trials, not overlapping audio windows, as the experimental units.
7. Use pilot data to estimate achievable variance and decide trial numbers
   with the faculty guide. A small pilot establishes feasibility only; it
   cannot certify low false-alarm rates or broad safety performance.
8. Reserve whole trials, sessions and preferably at least one vessel/condition
   for untouched testing. Fit feature normalization, model settings and
   thresholds using training/validation data only. Do not tune on the final
   test recordings. Keep a dated record of all exclusions and parameter changes.

## Suggested trial metadata columns

trial_id, session_id, date, operator, liquid, composition, liquid_batch,
initial_volume_ml, vessel_id, vessel_material, rim_reference_cm,
initial_freeboard_cm, heater_type, heater_setting, initial_temperature_c,
temperature_measurement_location, sensor_models, mounting_geometry,
audio_sample_rate_hz, distance_sample_rate_hz, ambient_temperature_c,
ambient_humidity_pct, background_condition, stirring_intervals,
event_time_s, event_annotation_uncertainty_s, stop_time_s, stop_reason,
split_assignment, data_quality_notes.

## Decisions after the pilot

If ToF fails on steam/foam, change or reposition the sensing method and report
the observed failure. If airborne sound has weak event information, retain
the acoustic-only baseline and report that result. Calibrate or reject the
synthetic assumptions using measurements; do not selectively retain only
recordings that resemble the desired simulator.

The immediate deliverable from the pilot is a documented dataset and an
honest feasibility assessment. It is not a ready-to-deploy safety device.
