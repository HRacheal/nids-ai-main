# Results

All numbers come from the committed metric files
(`artifacts/metrics/track_{a,b}/*.json`, `artifacts/metrics/comparison.json`).
The full generated tables are in `reports/results_tables.md`, and the
figures in `reports/figures/`. Models were selected on validation data
only, and each final model was evaluated once on the test set. Unless
stated otherwise, tables report **seed 0**, with "±" giving the standard
deviation over seeds 0, 1 and 2 and a 95 % bootstrap interval.

**Training data.** Track A models (RF, Isolation Forest, MLP, 1D-CNN, its
random-order ablation, Autoencoder) were trained on the **full** training
split. Track B models and the leave-one-class-out experiment used
**reduced** training data (`docs/sampling.md`). Test sets are always
complete.

## 1. Track A: multiclass (official test set, 82,332 records, 10 classes)

| Model | Accuracy | Macro-F1 (seed 0 ± sd) | 95 % CI | Macro precision | Macro recall | Macro-F1, non-overlapping test | Train s / seed | Predict µs / record | Parameters |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Random Forest** | **0.754** | **0.529 ± 0.001** | [0.515, 0.541] | 0.539 | 0.574 | **0.561** | 77 | 23.5 | – |
| MLP | 0.700 | 0.468 ± 0.006 | [0.459, 0.477] | 0.461 | 0.581 | 0.457 | 188 | 3.8 | 33,226 |
| 1D-CNN | 0.702 | 0.469 ± 0.002 | [0.459, 0.479] | 0.455 | 0.566 | 0.476 | 479 | 36.9 | 58,250 |
| 1D-CNN, random feature order | 0.704 | 0.468 ± 0.004 | [0.461, 0.475] | 0.454 | 0.589 | 0.472 | 441 | 37.4 | 58,250 |

Seed means of macro-F1: RF 0.528, MLP 0.461, 1D-CNN 0.469, random-order
1D-CNN 0.461.

**Paired tests (seed 0, Holm-corrected McNemar).** The Random Forest beats
every network by about 0.06 macro-F1. The 95 % paired-bootstrap intervals
are [+0.048, +0.073] against the MLP and [+0.046, +0.073] against the
1D-CNN, and the McNemar p-values are numerically zero (below 10⁻³⁰⁰). The MLP and 1D-CNN do
not differ in macro-F1 (Δ = −0.001, CI [−0.011, +0.009]). McNemar still
finds a difference (p = 0.005), so they make different errors at about the
same overall rate.

**Feature-order ablation.** The 1D-CNN with its numeric features in
meaningful groups scores the same as one with a random order: seed-0
Δ = +0.001, CI [−0.007, +0.009]. The seed means differ by 0.007 (0.469 vs
0.461), which is within about two seed standard deviations. McNemar
(p = 0.02 after Holm) shows the two make somewhat different errors, but
not fewer. The grouped
order gives the convolutional model no measurable advantage. This agrees
with the MLP, which has no notion of order, scoring the same.

**Per-class F1.** Generic (0.98) and Normal (0.76–0.86) are easy.
Reconnaissance (0.79–0.87) and Exploits (0.67–0.71) are moderate. Fuzzers
(≈0.35), DoS (0.30–0.45), Shellcode (0.26–0.47), Worms (0.27–0.60),
Backdoor (≤0.11) and Analysis (≤0.04) are hard for every model. Analysis
and Backdoor test records are mostly predicted as other attacks: in
UNSW-NB15 their feature vectors overlap heavily with DoS and Exploits. The
Random Forest's most frequent errors are Normal→Fuzzers (7,213 records),
Exploits→Backdoor (1,877) and DoS→Backdoor (1,807). See
`reports/figures/models/per_class_f1_track_a.png` and the confusion
matrices.

**Overlap with training data.** On test records whose exact feature vector
never occurs in the training set, macro-F1 is similar to the full test set:
RF 0.561 (full 0.529), 1D-CNN 0.476 (0.469), MLP 0.457 (0.468). The
headline numbers are therefore not inflated by memorised duplicates.

## 2. Track A: attack vs normal

| Model | Training | Precision | Detection rate | F1 | FPR | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| Random Forest (from multiclass) | supervised | 0.840 | 0.971 | **0.901** | 0.226 | 0.980 | 0.984 |
| MLP (from multiclass) | supervised | 0.759 | 0.996 | 0.862 | 0.387 | 0.984 | 0.988 |
| 1D-CNN (from multiclass) | supervised | 0.768 | 0.996 | 0.867 | 0.369 | 0.984 | 0.988 |
| RF (binary) | supervised | 0.815 | 0.988 | 0.893 | 0.276 | 0.980 | 0.983 |
| **Autoencoder** (bottleneck 32) | normal traffic only | **0.907** | 0.850 | 0.878 | **0.107** | 0.943 | 0.947 |
| Isolation Forest | normal traffic only | 0.899 | 0.606 | 0.724 | 0.084 | 0.855 | 0.875 |

The supervised models detect almost every attack, but they raise false
alarms on 23–39 % of normal test records, mostly normal traffic predicted as
Fuzzers. The Autoencoder never saw an attack. It catches 85 % of attacks at
a 10.7 % false-positive rate, and its ROC-AUC over three seeds is
0.935 ± 0.006. Its threshold was set for a 5 % false-positive rate on
validation normal traffic. On the test set the rate is about twice that,
because normal traffic in the official test partition differs from the
training partition, a known property of UNSW-NB15. The Autoencoder flags
>88 % of Generic, Exploits, DoS, Analysis and Backdoor records, but only 33 %
of Reconnaissance and 17 % of Shellcode records. It is far stronger than the
Isolation Forest (ROC-AUC 0.943 vs 0.855).

## 3. Track B: chronological test block (366,522 flows), reduced training data

| Model | Accuracy | Macro-F1 (seed 0 ± sd) | 95 % CI (time blocks) | Macro precision | Macro recall | Attack-vs-normal F1 | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| LSTM (L = 4) | 0.973 | 0.492 ± 0.005 | [0.475, 0.509] | 0.457 | 0.569 | 0.949 | 0.0138 |
| **Per-flow MLP** | 0.974 | **0.522 ± 0.009** | [0.505, 0.542] | 0.537 | 0.604 | 0.951 | 0.0131 |
| Per-flow RF | 0.979 | 0.510 ± 0.001 | [0.501, 0.517] | 0.578 | 0.515 | 0.964 | 0.0060 |
| Context baseline | 0.978 | 0.455 ± 0.002 | [0.434, 0.479] | 0.614 | 0.453 | 0.962 | 0.0037 |
| Oracle context (upper bound) | 0.982 | 0.516 ± 0.004 | [0.499, 0.538] | 0.629 | 0.500 | 0.975 | 0.0024 |

Seed means of macro-F1: LSTM 0.499, per-flow MLP 0.520, per-flow RF 0.510.
The LSTM window length L = 4 was chosen on validation data (val macro-F1
0.483, against 0.478 for L = 16).

**Does flow history help?** Not on this data. The LSTM is significantly
*worse* than the same features without history (per-flow MLP: Δ = −0.031,
paired time-block bootstrap CI [−0.050, −0.013]). It is not distinguishable
from the per-flow RF (Δ = −0.018, CI [−0.035, +0.001]). Even the oracle,
which is given the true labels of the host's previous flows, does not beat
the per-flow MLP overall. It improves Fuzzers (0.65 vs 0.52) and Exploits,
but loses on DoS. The context baseline's logistic combiner, fitted on the
validation block, collapses on DoS (F1 0.04), whose share differs between
the validation and test blocks. Within one chronological block, a host's
recent flows carry little extra information about the category of its next
flow beyond what the flow's own features already contain.

All Track B models are excellent at attack vs normal: false-positive rates
of 0.2–1.4 %, compared with 8–39 % in Track A. The chronological blocks are
far more homogeneous than the official Track A partition. Track B numbers
come from reduced training data and a different split, so they must not be
compared directly with Track A.

## 4. Leave one attack class out (Track A, reduced training data)

For each attack category, the class was removed from training and
validation, and the preprocessor and the RF/MLP were refitted. The unseen
class's test records were then classified. Full table:
`reports/results_tables.md`; figure:
`reports/figures/models/leave_one_class_out.png`.

* **Most unseen attacks are still flagged as attacks.** For 8 of the 9
  classes, ≥ 85 % (RF) and ≥ 97 % (MLP) of the unseen class's records are
  predicted as *some* attack. They usually get a related label (DoS ↔
  Exploits/Backdoor, Reconnaissance → Shellcode).
* **Fuzzers is the exception.** 61 % (RF) and 41 % (MLP) of unseen Fuzzers
  records are classified as Normal. Fuzzers is also the class most confused
  with Normal in the closed-set models.
* **Unknown-class rejection is weak.** The max-softmax threshold (5 % of
  known validation records rejected) catches 0–45 % of unseen-class
  records. It works for Analysis (33–45 %), Backdoor (21–43 %) and DoS
  (16–35 %), but not for Generic, Worms or Shellcode (≤ 4 %). It also
  rejects 1.5–8.5 % of known-class test records.
* The **Autoencoder**, for which every attack is unseen, flags ≥ 84 % of 6
  classes but only 17–33 % of Shellcode and Reconnaissance. The closed-set
  models catch those two classes as "some attack", so the two detector types
  complement each other.

## 5. Explainability (Track A)

| Check | Result |
|---|---|
| RF TreeSHAP additivity (max \|Σ SHAP + E[f] − f(x)\|) | 8.4 × 10⁻⁴ |
| Integrated Gradients completeness, median relative error | MLP 0.0049, 1D-CNN 0.0060 |
| Samples | SHAP 200 test records (20/class), IG 500 (50/class), permutation 9,000 validation records |

One-hot features are summed back to their 42 original features. The
features ranked highly by every method are `service`, `proto`, `smean`,
`sttl` and `ct_dst_src_ltm`, with `sbytes` (RF) and `dttl` (networks) close
behind. `sttl`/`dttl` (time-to-live) are known artefacts of the UNSW-NB15
testbed rather than general attack signatures (`docs/dataset_profile.md`).

Agreement between methods (Spearman over the 42 features) ranges from 0.21
to 0.74. The highest are MLP-IG vs MLP-permutation (0.74), MLP-IG vs
1D-CNN-IG (0.70) and RF-SHAP vs Autoencoder error share (0.71). RF-SHAP and
RF-permutation share 8 of their top-10 features. Explanations agree on the
dominant features, but the ranking beyond the top five depends on the method
and should not be over-interpreted. Local explanations for one DoS, one
Exploits and one Reconnaissance record are in `reports/figures/xai/`.

## 6. Reduced-training-data study (validation only)

The same RF and MLP configurations, seed 0, trained on minority-preserving
stratified samples of the Track A training split. Scores are validation
macro-F1; the full-data reference uses the same configuration and seed.

| Training data | Records | RF val macro-F1 (drop) | RF fit s | MLP val macro-F1 (drop) | MLP fit s |
|---|---:|---:|---:|---:|---:|
| full | 150,291 | 0.6496 | 77 | 0.5852 | 188 |
| 70 % | 106,491 | 0.6427 (−0.007) | 33 | 0.5766 (−0.009) | 32 |
| **50 %** | 77,292 | 0.6390 (−0.011) | 20 | 0.5703 (−0.015) | 20 |
| 30 % | 48,095 | 0.6237 (−0.026) | 11 | 0.5657 (−0.020) | 13 |

Under the rule fixed in advance (≤ 0.02 drop for both models), 50 % was
used for leave-one-class-out. The 30 % sample fails the rule for the RF.
The full-data fit times in this table were measured in the original
training runs, which used a different mix of background load, so they are
only indicative.

## 7. Measured run times (this laptop, 4 CPU threads)

| Step | Measured |
|---|---|
| 1D-CNN final, per seed (full data) | 479 s (mean of 3) |
| Random-order ablation, per seed | 441 s |
| Autoencoder search + 3 seeds | 11 s per final seed |
| Track B, all five models, 3 seeds (reduced data) | 715 s in total; LSTM 34 s per final seed (L = 4) |
| Sampling study (6 trainings) | 139 s |
| Leave-one-class-out (18 trainings, per-fold preprocessing) | 288 s |
| Explainability | 229 s |
| Model comparison | 44 s |

Track B was never trained on the full training block, so no measured
speed-up from the data reduction can be reported. The reduction's effect on
run time is an estimate only. Peak memory was not recorded per run, because
the helper failed silently on Windows; it is fixed now and tested. During
Track B training, the OS reported about 1.7 GB private memory for the
process (1.8 GB working set).
