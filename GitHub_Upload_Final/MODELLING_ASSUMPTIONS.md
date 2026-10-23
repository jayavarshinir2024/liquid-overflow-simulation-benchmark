# Model and evaluation specification

## A consistency repair, not a validated physical model

The original script assigned a labelled overflow time independently of its
foam trajectory. Version 2 instead derives the event from a surface reaching
the rim. This repairs internal consistency. It does not establish that the
surface model accurately represents boiling milk, a particular vessel, or a
particular sensor. There is no experimental calibration.

## Latent state

Each cycle lasts 360 s at 1 Hz. Independently draw a thermal time constant
tau ~ U(65,100) s, foamability f ~ U(0,1), collapse coefficient
c ~ U(0.006,0.018) per second, and initial freeboard r ~ U(6,12) cm.

T(t) = 25 + 74[1-exp(-t/tau)] degrees C.

b(t) = clip((T(t)-85)/14,0,1).

g = 0.025 + 0.18 f cm/s.

h(0)=0; h(t)=min(r,max(0,h(t-1)+g*b(t-1)^2-c*h(t-1))).

An event occurs at the first sample h(t)>=r. Otherwise the cycle is a non-event
within the observation window. This is a right-limited monitoring task, not a
claim that no overflow could occur after 360 s. Event prevalence is an outcome
of the generator; cycles are not forced into a 50/50 balance. The cap only
retains a bounded post-event trajectory, which is excluded from training and
advance-warning evaluation. There is no mass/energy balance or outflow model.

## Measurement channels

Nominal frequency proxy: A(t)=65+gain*[150*b(t)+9*h(t)]+noise, with
gain~U(0.8,1.2) and independent Gaussian noise SD 12 Hz.
Distance: d(t)=40-h(t)+Gaussian noise SD 0.15 cm.
Temperature proxy: T(t)+Gaussian noise SD 0.4 C.
Relative humidity proxy: 40+45*b(t)+Gaussian noise SD 1.2 percentage points.

These equations are illustrative correlations with a shared latent state,
not measured transduction laws. The temperature proxy tracks the assumed
bulk thermal state; an ambient BME280 must not be claimed to measure that
state. Frequency is a generated scalar, not raw microphone audio. Independent
Gaussian errors omit real bandwidth, response-time and cross-channel effects.

Half of cycles receive 1–3 measurement disturbance episodes, each 5–15 s:
Hann-shaped acoustic increases of 30–100 Hz and distance decreases of
0.2–1 cm. In 20% of cycles, both acoustic and distance channels freeze at the
preceding value for one 3–10 s episode. These measurement artifacts leave
latent event times unchanged.

Normalize acoustic by clip((A-65)/300), height by clip((40-d)/r), temperature
by clip((Tproxy-75)/25), humidity by clip((H-40)/45), all into [0,1]. Height
normalization assumes known initial sensor-to-surface distance (40 cm) and
freeboard. A real unit would need to measure these quantities; no perfect
geometry calibration should be assumed in physical claims.

## Comparisons and tuning

Nine methods are specified in source. Constant weights are searched over
nonnegative multiples of 0.25 summing to one (35 choices). The ToF/trend
baseline searches level weights 0,0.25,0.5,0.75,1 and trend scales
0.05,0.1,0.2. Both learned models use four channels plus causal 10 s acoustic
and height changes, and predict a crossing within 60 s. Logistic C candidates:
0.1,1,10. Boosting depth candidates: 2,3,5, with 100 iterations and fixed
learning rate 0.1, L2=1, and early stopping disabled. Other parameters use the
pinned scikit-learn defaults. Logistic features are standardized inside each
training fit. Both models use balanced training weights.

Per seed, stratified cycle splits are approximately 60/20/20 (rounding is
class-specific). An inner two-thirds/one-third split of training cycles chooses
rule/model settings. Learned models are then refitted on all training cycles.
Only every third frame is used for training; all test frames are evaluated.
Frames at or after the first crossing are excluded from training. Final
operating thresholds use the separate validation cycles only.

Thresholds are derived separately from each method’s validation scores: 257
quantiles of causal three-sample minima, per-cycle maximum boundaries, and
for event cycles the maxima before and within the 60 s warning window. Both
each boundary and its next representable higher float are included. The grid
also includes the global maximum and a strictly higher no-warning threshold.
This is an approximate search; it is not exhaustive over every frame value.
Candidate grids are recomputed on the inner tuning split during parameter
selection and on the final validation split during operating-point selection.
Test sweeps are descriptive and never choose thresholds.

An initial audit run used integer thresholds on the 0–100 scale and showed
that saturated learned probabilities could be suppressed simply because the
grid was too coarse. The score-aware search corrects that methodological
artifact for all methods. The final results rerun every seed; the initial
diagnostic log is retained separately and is not used as final evidence. The selection maximizes mean timely lead subject to validation
non-event false alarms <=5% and timely detection >=90%. If infeasible, choose
the highest timely detection satisfying the false-alarm constraint; ties prefer
fewer early warnings, then longer lead, then the lower threshold. Feasibility
is recorded per method and seed rather than assumed.

## Outcomes

A warning is the third consecutive qualifying sample. Only the first warning
is evaluated in a cycle; re-arming and repeated alarms are not modelled.
For event cycles, monitoring ends at crossing. A first warning 1–60 s before
crossing is timely; one more than 60 s before is premature and does not count
as a timely detection. Warnings at/after crossing are missed detections.
For non-event cycles, any warning in 360 s is a false alarm.

Detection = timely warned event cycles / event cycles.
False-alarm rate = warned non-event cycles / non-event cycles.
Early-warning rate = prematurely warned event cycles / event cycles.
Precision = timely warnings / (timely + premature + non-event warnings).
Lead time is conditional on timely detection, not averaged across misses.
Any-advance detection is saved as a secondary metric for transparency.
This precision definition is not the original paper's precision definition.

T-based 95% intervals summarize variation of per-seed metrics. They can extend
outside physical rate bounds and should not be interpreted as binomial
confidence intervals or evidence of real-device reliability. Undefined lead
times/precision are recorded as NaN and the number of defined seeds is saved.
Three specified method pairs across three metrics are tested with two-sided
Wilcoxon signed-rank tests using the asymptotic approximation, then Holm
adjustment across the nine tests. Small-seed p-values remain exploratory.

## Stress scenarios

All retain nominal trained parameters and thresholds. Each condition uses
separate seeds and 300 new cycles, not the same cycles with a paired perturbation.

| Condition | Change |
|---|---|
| Noise x3 | Multiply all measurement-noise standard deviations by three |
| Weak acoustic | Remove the height-dependent term from the acoustic proxy |
| ToF freeze | In 80% of cycles freeze ToF for 90 s starting at a random time 100–269 s |
| Distance bias | Reduce measured distance by 1 cm; retain nominal geometry calibration |
| Faster foam | Multiply growth by 1.5 and reduce acoustic height coupling from 9 to 2 |
| Small freeboard | Draw freeboard from 3–6 cm; supply that geometry to normalization |

This is sensitivity analysis over specified alternatives, not physical
validation or a proof of generalization to every liquid, vessel or failure.
