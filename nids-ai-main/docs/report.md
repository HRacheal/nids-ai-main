# DeepShield AI: Deep Learning for Network Intrusion Detection on UNSW-NB15

*Evaluation, explainability and a security-monitoring dashboard*

DSA4050_XA Deep Learning project · [Author name, student ID] · November 2026

---

## Abstract

DeepShield AI extends an open-source network intrusion detection project
(NIDS-AI) into a reproducible deep-learning study and a local monitoring
application built on the UNSW-NB15 dataset. Four neural families were
compared with tree-based and anomaly-detection baselines under a
leakage-controlled protocol: a multilayer perceptron (MLP), a 1D
convolutional network (1D-CNN), an autoencoder, and an LSTM over per-host
flow sequences. Two evaluation tracks were used: the official train/test
partition, and a chronological split of the 2.5 M full records. Selection
used validation data only; every final model was trained with three seeds
and tested once. On the official partition, a Random Forest outperformed
both supervised networks in macro-F1: 0.529 against 0.468 (MLP) and 0.469
(1D-CNN), paired 95 % CI of the difference [+0.05, +0.07]. A feature-order
ablation showed that the 1D-CNN gains nothing from its grouped feature
order. The autoencoder, trained on normal traffic only, detected 85 % of
attacks at an 11 % false-positive rate (ROC-AUC 0.94), far better than an
Isolation Forest (ROC-AUC 0.86). On the chronological track, the LSTM was
significantly *worse* than the same features without history (Δ macro-F1
−0.031, CI [−0.050, −0.013]). Even an oracle given the true labels of
earlier flows did not beat the per-flow model. In a leave-one-attack-class-out
experiment, unseen attacks were mostly still flagged as some attack, with
one exception: Fuzzers was largely missed as normal traffic. A max-softmax
"unknown" rule caught 0–45 % of unseen-class records. SHAP and Integrated
Gradients agreed on the dominant features (service, protocol, mean packet
size, TTL, connection counters) and passed numerical sanity checks. The
models are served by a local Flask dashboard. It analyses uploaded records
with per-record explanations, replays the chronological test block with
ground truth isolated from the models, maps detections to candidate MITRE
ATT&CK techniques, and exports reports. The main conclusion is a cautious
one: on UNSW-NB15, deep models did not beat a well-tuned Random Forest for
multiclass detection, but deep anomaly detection and explanation add real
value to a monitoring workflow.

## 1. Introduction

Machine-learning intrusion detection is often reported with near-perfect
accuracy. Such numbers frequently come from accuracy on imbalanced data,
from random splits that put duplicate flows on both sides, or from tuning on
the test set. The original NIDS-AI project, for example, advertised 99 %
accuracy. This project asks a narrower question, answered under a strict
protocol:

1. Do deep architectures (MLP, 1D-CNN, LSTM, autoencoder) detect and
   classify attacks in UNSW-NB15 better than classical baselines (Random
   Forest, Isolation Forest), measured by macro-F1 and paired statistical
   tests rather than accuracy?
2. Does structure help? The questions are whether the feature order matters
   to a 1D-CNN, and whether a host's recent flow history matters to an LSTM.
3. How do the models behave on attack classes they never saw in training?
4. Are their explanations consistent across methods, and numerically sound?
5. Can the models be delivered in a usable, honest monitoring application?

Contributions: a leakage-controlled two-track evaluation; three-seed results
with bootstrap intervals and Holm-corrected tests; a feature-order ablation;
an oracle analysis of sequence context; a leave-one-class-out study with
per-fold preprocessing; cross-method explanation agreement; and a local
dashboard with ground-truth isolation.

## 2. Background and related work

**UNSW-NB15** [1, 2] was generated in 2015 at UNSW Canberra. The IXIA
PerfectStorm tool mixed real normal traffic with synthetic attacks from nine
families, and Argus and Bro-IDS turned the packets into 49-attribute flow
records. It was designed to replace the outdated KDD'99. It still has known
problems: a single attacker subnet, testbed artefacts such as TTL values,
duplicate records, and a train/test partition whose class mixes differ
[2, 5]. Sommer and Paxson [13] argued that anomaly detection in networks is
hard because of the "closed world" assumption: classifiers are evaluated on
the same attack types they were trained on. The leave-one-class-out
experiment here tests exactly that assumption.

**Models.** Random Forests [6] are strong on tabular data. Isolation
Forests [7] isolate anomalies with random splits. Autoencoders detect
anomalies by reconstruction error. LSTMs [8] model sequences. A 1D-CNN
assumes local structure along the input axis, which tabular features do not
naturally have. That assumption is tested here with an ablation. Training
uses Adam [9].

**Evaluation.** Macro-F1 weights every class equally, which suits a
heavily imbalanced dataset. Paired comparisons use McNemar's test, as
recommended by Dietterich [10], with Holm's correction [11], and a paired
bootstrap. The max-softmax-probability rule [14] is the standard baseline
for detecting out-of-distribution inputs.

**Explainability.** SHAP [3] attributes predictions to features through
Shapley values; TreeSHAP is exact for tree ensembles. Integrated
Gradients [4] attributes a network's output to its inputs and satisfies
completeness. **MITRE ATT&CK** [12] catalogues adversary techniques. It is
used here only to suggest *candidate* techniques for a detected category.

## 3. Data

| | Track A: official partition | Track B: full records, chronological |
|---|---|---|
| Records | 175,341 train / 82,332 test | 2,540,047 (4 CSV files) |
| Features | 42 (186 after encoding) | 41 (194 after encoding) |
| Split | train 150,291 / validation 25,050 (grouped stratified fold) / official test | per session 70/15/15 % by flow end time, 300 s gaps: 1,778,062 / 366,323 / 366,522 |
| Used for | RF, Isolation Forest, MLP, 1D-CNN, autoencoder, leave-one-class-out, explanations | LSTM and comparisons; dashboard replay |

The official download was blocked (HTTP 403). The files therefore came from
Kaggle mirrors, which were verified against every official count
(`docs/dataset_profile.md` §2). The licence permits academic use but does
not mention redistribution, so the dataset is neither committed nor
packaged.

Key data issues found in profiling: 38.6 % of Track A training records are
exact duplicates; 10.4 % of test feature vectors also occur in training; the
`ct_*` connection counters cannot be reproduced from their documented
definition and partly count *later* connections; and all 321,283 attack
records come from four testbed hosts. Identity columns (IP addresses, ports,
timestamps) are therefore never model inputs. They are used only to order
and group Track B flows.

## 4. Method

### 4.1 Leakage control

* The preprocessor (median imputation with missing flags, `log1p` on 32
  heavy-tailed features, standardisation, clipping to ±10, one-hot encoding
  with an "other" level) is fitted on training data only. A unit test
  checks this.
* The Track A validation fold is grouped by feature vector, so no duplicate
  crosses the train/validation boundary; measured overlap is zero.
* Track B blocks are separated by 300 s gaps. LSTM windows contain only the
  target flow and earlier flows of the same source host in the same block.
* The test sets were used once per final model, after all choices were
  made on validation data.

### 4.2 Models and selection

Hyperparameters were searched with seed 0 on validation macro-F1. The
autoencoder was selected on validation ROC-AUC. The selected configuration
was retrained with seeds 0, 1 and 2. Early stopping restored the epoch with
the lowest validation loss.

| Model | Selected configuration |
|---|---|
| Random Forest | 150 trees, `max_features` 0.3, `min_samples_leaf` 1, balanced-subsample class weights |
| Isolation Forest | normal training records only; threshold = 95th percentile of validation normal scores |
| MLP | 186 → 128 → 64 → 10 (BatchNorm, ReLU, dropout 0.2), Adam lr 10⁻³, batch 512, 1/√frequency class weights |
| 1D-CNN | numeric features as a 1-channel sequence in meaningful groups; Conv(32)–Conv(64), kernel 5, plus a dense branch for one-hot features; 58,250 parameters |
| 1D-CNN ablation | identical, numeric features randomly permuted (seed 1234) |
| Autoencoder | 186 → 128 → 64 → 32 → 64 → 128 → 186, MSE on normal training traffic; threshold = 95th percentile of validation normal error |
| FlowLSTM (Track B) | Linear(194 → 64) → LSTM(64) → Linear(10) on the last step; window length L ∈ {4, 16}, L = 4 selected |
| Per-flow MLP / RF (Track B) | the Track A selections, no history |
| Context baseline / oracle (Track B) | logistic regression on [per-flow MLP probabilities, mean of the previous L−1 flows' predicted (baseline) or **true** (oracle) labels] |

### 4.3 Compute constraints and reduced settings

All experiments ran on one 4-core laptop CPU with 7.7 GB RAM, shared with
other applications. A first full-grid run was stopped by the operating
system for low memory. The approved response had three parts, all
documented in `docs/models.md` and `docs/sampling.md`:

1. A narrower hyperparameter search. The largest MLP and CNN variants were
   dropped, and the LSTM search used L ∈ {4, 16}.
2. Resumable runs, with every finished run cached and every epoch
   checkpointed.
3. Reduced **training** data for the last experiments, at the project
   owner's request:
   * Track B used the most recent 50 % of each session's training period.
     This keeps the time order, and no window crosses a block boundary.
   * Leave-one-class-out used a 50 % stratified sample in which classes
     under 2,000 records are kept whole. The fraction was confirmed on
     validation data with a rule fixed in advance.

Test sets were never reduced.

### 4.4 Leave one attack class out

For each attack class c, its records were removed from training and
validation. The preprocessor was refitted without c, and the RF and MLP
were retrained on the remaining nine classes. On c's test records we report
the share predicted Normal (missed), the share predicted as another attack,
and the share rejected as "unknown". A record is unknown if its maximum
softmax probability is below the threshold that rejects 5 % of known-class
validation records.

### 4.5 Statistics and explanations

Results report 95 % stratified bootstrap intervals of macro-F1, paired
bootstrap intervals of differences, and exact McNemar tests with Holm
correction. For Track B, bootstrap resampling uses 10-minute time blocks
because consecutive flows are correlated, and McNemar is not used.
Explanations:
* TreeSHAP for the RF, with additivity checked;
* Integrated Gradients (64 steps, mean-record baseline) for the MLP and
  1D-CNN, with completeness checked;
* per-feature reconstruction-error share for the autoencoder;
* grouped permutation importance on validation data.

One-hot attributions are summed back to the original 42 features.

## 5. Results

Full tables are in `docs/results.md` and `reports/results_tables.md`.

### 5.1 Multiclass detection (Track A)

| Model | Accuracy | Macro-F1 (± seed sd) | 95 % CI | Attack-vs-normal F1 |
|---|---:|---:|---:|---:|
| Random Forest | 0.754 | **0.529 ± 0.001** | [0.515, 0.541] | 0.901 |
| MLP | 0.700 | 0.468 ± 0.006 | [0.459, 0.477] | 0.862 |
| 1D-CNN | 0.702 | 0.469 ± 0.002 | [0.459, 0.479] | 0.867 |
| 1D-CNN, random order | 0.704 | 0.468 ± 0.004 | [0.461, 0.475] | 0.866 |

The Random Forest is significantly better than both networks: Δ macro-F1
+0.06, with paired CIs excluding zero and Holm-corrected McNemar
p ≈ 0. The MLP and 1D-CNN are indistinguishable in macro-F1 (Δ −0.001, CI
[−0.011, +0.009]). Shuffling the CNN's feature order changes macro-F1 by
+0.001 (CI [−0.007, +0.009]). The CNN's convolutional bias therefore finds
no exploitable local structure in the grouped feature order. Per class,
Generic and Normal are easy and Reconnaissance and Exploits moderate.
Fuzzers, DoS, Shellcode, Worms, Backdoor and Analysis remain hard for every
model; Analysis and Backdoor F1 is ≤ 0.11. The dominant error is
normal traffic predicted as Fuzzers. Results on the 73,791 test records
with no exact duplicate in training are similar (RF 0.561), so duplicates
do not inflate the headline numbers.

### 5.2 Anomaly detection (Track A)

The autoencoder never saw an attack. It reached ROC-AUC 0.943 (three seeds:
0.935 ± 0.006), precision 0.907, detection rate 0.850 and a false-positive
rate of 0.107. The Isolation Forest reached ROC-AUC 0.855 and a detection
rate of 0.606 at a false-positive rate of 0.084. The supervised models
detect 97–99.6 % of attacks, but at false-positive rates of 23–39 %. The
autoencoder's false-positive rate is about twice its 5 % validation target,
a direct trace of the shift between UNSW-NB15's official training and test
partitions.

### 5.3 Sequence context (Track B)

| Model | Macro-F1 (± seed sd) | 95 % CI (time blocks) | False-positive rate |
|---|---:|---:|---:|
| LSTM (L = 4) | 0.492 ± 0.005 | [0.475, 0.509] | 1.4 % |
| Per-flow MLP | **0.522 ± 0.009** | [0.505, 0.542] | 1.3 % |
| Per-flow RF | 0.510 ± 0.001 | [0.501, 0.517] | 0.6 % |
| Context baseline | 0.455 ± 0.002 | [0.434, 0.479] | 0.4 % |
| Oracle context (upper bound) | 0.516 ± 0.004 | [0.499, 0.538] | 0.2 % |

History did not help. The LSTM is significantly worse than the per-flow
MLP (Δ −0.031, CI [−0.050, −0.013]) and not distinguishable from the
per-flow RF. The oracle, which knows the true categories of the host's
previous flows, also fails to beat the per-flow MLP overall. It helps
Fuzzers and Exploits but hurts DoS. In UNSW-NB15, a host's recent flows add
little information beyond each flow's own features. A likely reason is that
the four attacker hosts interleave many attack types. All Track B models
have far lower false-positive rates than Track A. The chronological blocks
are more homogeneous than the official partition, and Track B models were
trained on reduced data, so the two tracks must not be compared directly.

### 5.4 Unseen attack classes

When a class is removed from training, its test records are usually still
flagged as *some* attack. This holds for 8 of 9 classes: ≥ 85 % (RF) and
≥ 97 % (MLP). The records then receive a related label, e.g. DoS ↔
Exploits/Backdoor. The exception is **Fuzzers**: 61 % (RF) and 41 % (MLP)
of unseen Fuzzers records are classified as Normal. The max-softmax unknown
rule is unreliable. It rejects 33–45 % of unseen Analysis records but ≤ 4 %
of unseen Generic, Shellcode or Worms records, while rejecting 1.5–8.5 %
of known-class records. The autoencoder flags ≥ 84 % of six classes but only
17–33 % of Shellcode and Reconnaissance, which the closed-set models do
catch. The two detector types are therefore complementary.

### 5.5 Explanations

TreeSHAP additivity error was 8.4 × 10⁻⁴, and the median Integrated
Gradients completeness error was 0.005 (MLP) and 0.006 (1D-CNN). Every
method ranks `service`, `proto`, `smean`, `sttl` and `ct_dst_src_ltm` among
the most important features. Spearman agreement between methods ranges from
0.21 to 0.74. `sttl`/`dttl` are testbed artefacts: in UNSW-NB15, attack
traffic has characteristic TTL values. Explanations therefore expose a
shortcut that would not transfer to a real network, which is a useful
finding in itself.

### 5.6 Application

The local dashboard (`python -m deepshield.app`, 127.0.0.1 only) offers:
* analysis of uploaded CSV/JSON records with any Track A model, with input
  validation and a per-record explanation;
* a time-ordered replay of the Track B test block, with the LSTM, per-flow
  MLP or per-flow RF. Models receive label-free `FlowRecord` objects, while
  ground truth travels separately to an evaluation monitor; a test shows
  that shuffling labels does not change predictions;
* an alert history with severity rules, stored in SQLite;
* candidate MITRE ATT&CK techniques, validated against the ATT&CK STIX
  bundle (v20.1);
* model-comparison and explanation pages, and PDF/CSV reports.

The application does not capture live traffic. That limitation is stated
in the interface and the documentation. The suite of 67 automated tests
passes, including an end-to-end dashboard test, and the original NIDS-AI
code is preserved byte-for-byte and verified.

## 6. Discussion

**Deep learning did not win the classification task.** With 42 engineered
flow features, many of them categorical or counters, the Random Forest's
axis-aligned splits match the data well. The networks' smooth decision
functions and class weighting trade precision for recall on rare classes:
the MLP's macro recall is slightly higher than the RF's (0.58 against 0.57),
but its macro precision is much lower (0.46 against 0.54), mostly because
normal traffic is labelled as attacks. This is consistent with the wider literature on tabular data,
where tree ensembles remain strong baselines. The narrower search forced by
memory limits may cost the networks a little, but the 0.06 gap is far
larger than the differences between the architectures that were searched
(at most 0.023 validation macro-F1 within the MLP and CNN grids).

**Where deep learning helped.** The autoencoder is the best detector that
needs no attack labels. That makes it the most relevant model for novel
attacks, and it complements the supervised models on the classes they miss.

**Structure assumptions should be tested, not assumed.** Neither the
CNN's feature order nor the LSTM's host history added measurable
information, and the oracle analysis shows this is a property of the data,
not a training failure. These negative results are among the study's most
transferable findings. Sequence models for intrusion detection should be
justified by an oracle or ablation of this kind before they are deployed.

**Operational reading.** In Track A, false alarms rather than misses are
the problem: 23–39 % of normal traffic flagged by supervised models. A
practical deployment would combine a supervised classifier for category
labels with the autoencoder, using its calibrated threshold, for alerting,
and would recalibrate thresholds on local normal traffic.

## 7. Limitations and threats to validity

* **Dataset.** Synthetic testbed traffic from 2015 with a single attacker
  subnet. The TTL and `ct_*` artefacts mean the results may not transfer to
  real networks. The files came from verified third-party mirrors.
* **Residual leakage.** The `ct_*` counters partly count later
  connections. Gaps between blocks reduce, but cannot provably remove,
  look-ahead in Track B.
* **Reduced search and data.** The searches were narrower than planned, and
  Track B and leave-one-class-out used reduced training data. Track B has
  no full-data counterpart, and leave-one-class-out used one sample (seed
  0). The data reduction cost the RF 0.011 and the MLP 0.015 validation
  macro-F1 in the sampling study.
* **Statistics.** Three seeds give a rough estimate of seed variance. The
  paired tests compare seed-0 models.
* **Scope.** The application analyses records and replays a dataset; it is
  not a live network sensor. MITRE mappings are candidates for an analyst,
  not attributions.

## 8. Ethics, licensing and reproducibility

The dataset is used under its academic licence and is not redistributed.
The five required citations are given below and in
`docs/dataset_profile.md`. The original NIDS-AI project (MIT licence) is
credited in `CREDITS.md`. It is preserved under the git tag
`original-nids-ai` and as a verified byte-for-byte copy in
`legacy/nids-ai-original/`. The dashboard binds to the local machine only
and loads no external resources.

Reproduction:
1. obtain the data (`docs/dataset_setup.md`);
2. run `scripts\prepare_data.py`;
3. run `scripts\run_experiments.py`, which executes seven resumable steps;
4. run `python -m pytest`.

Every configuration tried, with its validation score, is committed under
`artifacts/metrics/`.

## 9. Conclusion

Under a strict protocol, the deep models in this study did not outperform a
Random Forest for multiclass intrusion detection on UNSW-NB15, and neither
feature order (1D-CNN) nor host flow history (LSTM) added information. Deep
learning contributed most as an unsupervised anomaly detector, and through
explanations that were checked numerically and exposed dataset shortcuts.
Unseen attacks are mostly still flagged by closed-set models, except
Fuzzers, which resembles normal traffic. The resulting application packages
these models in an honest, locally run monitoring dashboard. Future work:
evaluation on newer and real-network datasets (e.g. the NetFlow versions of
[5]); threshold recalibration on local normal traffic; open-set methods
stronger than max-softmax rejection; and a wider hyperparameter search on
better hardware.

## References

1. N. Moustafa and J. Slay, "UNSW-NB15: a comprehensive data set for network
   intrusion detection systems (UNSW-NB15 network data set)," *Military
   Communications and Information Systems Conference (MilCIS)*, 2015.
2. N. Moustafa and J. Slay, "The evaluation of Network Anomaly Detection
   Systems: Statistical analysis of the UNSW-NB15 data set and the comparison
   with the KDD99 data set," *Information Security Journal: A Global
   Perspective*, 2016.
3. S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model
   predictions," *Advances in Neural Information Processing Systems (NeurIPS)*,
   2017.
4. M. Sundararajan, A. Taly and Q. Yan, "Axiomatic attribution for deep
   networks," *International Conference on Machine Learning (ICML)*, 2017.
5. M. Sarhan, S. Layeghy, N. Moustafa and M. Portmann, "NetFlow datasets for
   machine learning-based network intrusion detection systems," *BDTA/WiCON*,
   2020.
6. L. Breiman, "Random forests," *Machine Learning*, 45(1), 2001.
7. F. T. Liu, K. M. Ting and Z.-H. Zhou, "Isolation forest," *IEEE
   International Conference on Data Mining (ICDM)*, 2008.
8. S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural
   Computation*, 9(8), 1997.
9. D. P. Kingma and J. Ba, "Adam: a method for stochastic optimization,"
   *International Conference on Learning Representations (ICLR)*, 2015.
10. T. G. Dietterich, "Approximate statistical tests for comparing supervised
    classification learning algorithms," *Neural Computation*, 10(7), 1998.
11. S. Holm, "A simple sequentially rejective multiple test procedure,"
    *Scandinavian Journal of Statistics*, 6(2), 1979.
12. B. E. Strom et al., "MITRE ATT&CK: Design and philosophy," MITRE
    Technical Report, 2018.
13. R. Sommer and V. Paxson, "Outside the closed world: on using machine
    learning for network intrusion detection," *IEEE Symposium on Security
    and Privacy*, 2010.
14. D. Hendrycks and K. Gimpel, "A baseline for detecting misclassified and
    out-of-distribution examples in neural networks," *ICLR*, 2017.
15. N. Moustafa, J. Slay and G. Creech, "Novel geometric area analysis
    technique for anomaly detection using trapezoidal area estimation on
    large-scale networks," *IEEE Transactions on Big Data*, 2017.
16. N. Moustafa, G. Creech and J. Slay, "Big data analytics for intrusion
    detection system: statistical decision-making using finite Dirichlet
    mixture models," in *Data Analytics and Decision Support for
    Cybersecurity*, Springer, 2017.
