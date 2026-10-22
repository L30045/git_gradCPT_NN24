# EEG-informed fNIRS modeling (gradCPT_NN24)

Code for modeling the fNIRS hemodynamic response (HbO) during the gradCPT task with simultaneous EEG,
and for comparing the GLM noise models and estimators used to do it.

---

## 1. Abstract

**Question.** Can scalp EEG explain the fNIRS signal during gradCPT? If so, what does the EEG→HbO
"HRF" look like across cortical parcels and subjects? A second question is which estimator gives a
reliable HRF: the standard Cedalion AR-IRLS, a simpler fixed-whitening IRLS, or a joint low-rank
(iRRR) model across parcels.

**What was tested, in order.**
1. **Event-based, EEG-informed GLM in channel space.** HRF regressors locked to mountain (mnt) trials,
   with single-trial EEG features (ERP area, alpha power) as parametric modulators. Full vs reduced
   models were compared with F-tests.
2. **Move to image space.** The same models were fit on image-reconstructed parcel time series
   (Adot v26, sensitivity mask `log10 Σ A > −2`, 429/601 parcels). These were checked against the
   lab's Cedalion image-space GLM.
3. **Continuous-EEG GLM.** The Cz EEG, delayed 0–15 s and projected on a B-spline basis, is used
   directly as the regressor. Fitting is three-stage: OLS drift → OLS GSR → AR-IRLS. Variants:
   trial-based EEG, EEG + VTC, resting state, and EEG masked to city/mnt trials.
4. **HRF structure.** SVD/ICA of the parcel × delay HRF matrix, per subject and jointly across
   subjects.
5. **Low-rank multi-parcel model (iRRR).** All parcels are fit jointly with a nuclear-norm penalty.
   Per-parcel AR whitening removed the low-frequency EEG content, so this led to
   **spectral (fixed) whitening**, which in turn gave **fixAR-IRLS** and **AR-iRRR**. Y scaling and
   the choice of λ₁ (by CV, or by target rank) were also tested.
6. **Generalization.** Train on run 1 and test on runs 2–3, against training on all runs.
7. **Estimator comparison on an event-based design.** AR-IRLS vs fixAR-IRLS vs AR-iRRR on
   Gaussian-kernel mnt-correct HRFs for all subjects (N = 20). This was also checked against the
   coworker's Cedalion pipeline.
8. **Network-based fits.** One whitening kernel and one iRRR per functional network. This includes a
   four-way comparison of mnt-correct HRFs: EEG-weighted vs Gaussian-kernel design, crossed with
   whole-brain vs per-network pipeline.

A side track in the same folder covers pupil, VTC (attention fluctuation), and behavior.

**Main findings so far.**
- **The EEG→HbO effect is tiny.** In-sample R² is < 2% (Cz EEG design) and < 0.3% (mnt-correct
  design) of the whitened variance. Trained on run 1, neither fixAR-IRLS nor AR-iRRR predicts
  runs 2–3 (test R² ≈ 0 or negative). λ₁ chosen by CV shrinks 3 of 6 subjects to rank 0.
- **The noise model barely changes HRF shape.** AR-IRLS and fixAR-IRLS agree closely (median
  per-parcel r = 0.87; ContA network-mean r = 0.98). fixAR-IRLS amplitudes are ~19% smaller, with
  similar standard errors.
- **AR-iRRR is a shrunken, smoother version.** It has r ≈ 0.70–0.77 against the IRLS models and
  ~4× smaller amplitude at rank 2. Its temporal components differ between subjects, so there is
  no single shared HRF shape.
- **Y must be scaled before fitting.** Without scaling, statsmodels `RLM` stops after 2 iterations,
  and a fixed λ₁ = 1 zeroes out iRRR.
- **Network-based fits:** rank 2 is reached in every network. AR-iRRR vs fixAR-IRLS shape
  agreement has a median r of 0.66, and the AR-iRRR amplitude is ~0.6× that of fixAR-IRLS.
- **Open:** the four-script mnt-correct comparison (`summary_run_model_cont_EEG_city_mnt.ipynb`,
  section 16) still needs its summary. The last commit fixed the EEG→fNIRS time alignment: the
  EEG is lowpassed to the fNIRS Nyquist frequency and sampled at the exact fNIRS time points
  instead of using `mne.resample`. Results saved before that commit should be rerun.

---

## 2. Conventions

### Shared modules
| File | Role |
|---|---|
| [params_setting.py](params_setting.py) | Paths, `cfg_GLM`, `cfg_HRF_basis` (−2 to 18 s Gaussian kernels), `NOISE_MODEL`, `excluded_subj` (low fNIRS quality), and the path helpers `get_betas_path`, `get_stats_path`, `get_Y_all_path`, `get_dm_all_path`, `get_scale_tag` |
| [model.py](model.py) | Design matrices (`create_eeg_dm`, `get_cont_EEG_bspline_regressor`, `sample_cont_EEG_bspline_regressor`, drift/GSR regressors), fitters (`my_ar_irls_GLM`, `irls_fit`, `my_ols_GLM`), HRF reconstruction (`estimate_HRF_from_beta`, `estimate_HRF_cov`), and EEG features (`get_ERP_area`, `get_alpha_power`) |
| `../preproc_pipe/utils.py` | EEG preprocessing (`eeg_preproc_subj_level`) and event handling |
| `/projectnb/stephenlab/cchang1/iRRR_python` | iRRR solver (external) |

### Subject inclusion
- Subjects need gradCPT fNIRS and at least 500 clean EEG epochs (−0.2 to 1.0 s, rejected above
  100 µV). See [check_available_data.py](check_available_data.py).
- Subjects in `excluded_subj` are dropped, as are subjects with Cz removed in any run.
- The continuous-EEG work therefore uses 6 subjects (670, 695, 721, 723, 726, 730). The
  event-based Gaussian-kernel comparison does not need EEG, so it uses 20 subjects.

### Shared `Y_all`
- Pipeline: per run, highpass (0.02 Hz, 4th-order Butterworth, zero-phase), then truncate to
  [first onset, last onset + 15 s], then OLS Legendre drift, then OLS GSR. Runs are concatenated
  as gradcpt1, 2, 3.
- [run_model_cont_EEG_fNIRS.py](run_model_cont_EEG_fNIRS.py) saves it once per subject as
  `Y_all/<subject>_parcel_Y_all_truncated_to_trials_<Hp|noHp>.pkl.gz`.
- All `cont_EEG_cz_3-stage*` models and the network-based scripts reuse it.

### Output layout
All outputs go under `D = /projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/eeg`:
```
D/<subject>/betas/<subject>_<model>_betas.pkl        fitted betas
D/<subject>/stats/<subject>_<model>_stats.pkl        F-tests, λ1, rank, Y_scale, ...
D/<subject>/Y_all/, dm/                              Y and design matrices
D/<subject>/<subject>_<model_name>.pkl               one-file results of the cmp_* scripts
D/HRF_surf/<subject>/<model>/                        per-subject figures
D/HRF_surf/group/<model>/                            group figures (summary_nb/ = from notebooks)
D/EV_surf/, D/fit_vis/, D/components_analysis/       other figure roots
```

### Model glossary
| Name | Noise model | Fit |
|---|---|---|
| **AR-IRLS** | Cedalion `glm.fit(noise_model='ar_irls')`, AR(≤30) re-estimated per parcel | robust IRLS per parcel |
| **fixAR-IRLS** | one spectral-whitening kernel from the OLS residuals, averaged over parcels (or per network), applied to Y and X; Y divided by `Y_scale` | robust IRLS per parcel (Tukey biweight, c = 4.685); betas scaled back |
| **AR-iRRR** (older name `iRRR_AR`) | same whitened, scaled Y and X as fixAR-IRLS | iRRR on all parcels (or one network) jointly; λ₁ = smallest penalty giving rank 2 (earlier: by CV) |
| **iRRR** (no AR) | none | iRRR on unwhitened Y; abandoned |

| Design | Regressors | HRF unit |
|---|---|---|
| **Cz EEG (cont)** | continuous Cz EEG, delays 0–15 s, 15 cubic B-splines | M / V |
| **Cz EEG within trials (`city`, `EEG-net`)** | Cz EEG zeroed outside the trials of one type, delays −2 to 18 s, 20 B-splines | M / V |
| **Gaussian kernels (`GK`)** | 12 kernels (−2 to 10 s), and later 20 kernels (−2 to 18 s), 1 s apart with σ = 1 s, at mnt-correct onsets | M |

---

## 3. Logic flow

Each stage lists its scripts and what it showed. The `summary_*.ipynb` notebooks only read saved
results (they never refit) and contain the full write-ups.

### Stage 0: Data checks
- [check_available_data.py](check_available_data.py): lists the subjects with fNIRS, pupil, and
  EEG, and the epoch count of each.
- [check_image_recon.py](check_image_recon.py): inspects the image-reconstruction outputs.
- [vis_parcel_cluster.py](vis_parcel_cluster.py): hierarchical clustering of parcel centroids by
  spatial distance.

### Stage 1: Event-based EEG-informed GLM (channel space)
- [model_pilot.py](model_pilot.py): first look at the data. Includes the mnt-correct vs
  mnt-incorrect ERP comparison (mean ± 2 SEM).
- [model_EEG_inform_single_subject.py](model_EEG_inform_single_subject.py) and
  [run_model_EEG_inform.py](run_model_EEG_inform.py): fit four design matrices per subject:
  - `full`: HRF + EEG-modulated HRF
  - `onlyStim`: HRF only
  - `onlyEEG`: EEG-modulated HRF + basis
  - `basis`: drift/short-separation regressors only

  The EEG modulators are the single-trial ERP area and alpha power at Cz.
- [run_stat_test.py](run_stat_test.py): F-tests (full vs stim, full vs basis) and contrast t-tests.
  [vis_hrf.py](vis_hrf.py) plots the HRFs of significant channels.
- [run_model_network_based.py](run_model_network_based.py) and
  [vis_hrf_network.py](vis_hrf_network.py): the same models with network-averaged signals.
- Sanity checks on the fitting itself:
  - [cmp_model_hrf.py](cmp_model_hrf.py) and
    [cmp_model_MSE_EV_single_subj_single_channel.py](cmp_model_MSE_EV_single_subj_single_channel.py):
    model fit (MSE, explained variance) per model type.
  - [manual_fit_model.py](manual_fit_model.py): a hand-built fit.
  - [ar-irls_test.py](ar-irls_test.py): AR prewhitening with `lfilter`.
  - [proof_RLM_RSS_full_smaller_than_reduced.py](proof_RLM_RSS_full_smaller_than_reduced.py): shows
    that a robust (IRLS) fit can give a full model with a larger weighted RSS than the reduced model,
    which breaks the usual F-test.
  - [check_yhat_AR-IRLS_OLS.py](check_yhat_AR-IRLS_OLS.py): AR-IRLS vs OLS predictions on one channel.

### Stage 2: Image space (parcels)
- [CORRECT_CODE_runGLM_on_image.py](CORRECT_CODE_runGLM_on_image.py) and
  [hrf_estimate_Laura_copy.py](hrf_estimate_Laura_copy.py): reference copies of the lab's
  image-space HRF GLM.
- [run_model_EEG_inform_parcel_based.py](run_model_EEG_inform_parcel_based.py): the Stage 1 models
  on parcels, with three-stage regression (drift → GSR → AR-IRLS).
- [check_yhat_AR-IRLS_image_space.py](check_yhat_AR-IRLS_image_space.py): compares Ŷ from the
  event-based HRF GLM and from the continuous-EEG GLM on the same parcel.

### Stage 3: Continuous-EEG GLM
- [model_continuous_EEG_fNIRS.py](model_continuous_EEG_fNIRS.py): first version, with all EEG
  channels as delayed regressors.
- [run_model_cont_EEG_fNIRS.py](run_model_cont_EEG_fNIRS.py): the main pipeline. It uses
  `cont_EEG_cz_3-stage_bspline-test` (Cz, 0–15 s delays, B-splines, drift/GSR by OLS, then AR-IRLS)
  and saves the shared `Y_all`.
- Variants:
  - [run_model_cont_EEG_trial_based.py](run_model_cont_EEG_trial_based.py): per-trial EEG epochs.
  - [run_model_cont_EEG_fNIRS_add_VTC.py](run_model_cont_EEG_fNIRS_add_VTC.py): adds the VTC
    regressor.
  - [run_model_cont_EEG_fNIRS_rest.py](run_model_cont_EEG_fNIRS_rest.py): resting-state runs.
  - [run_model_cont_EEG_city_mnt.py](run_model_cont_EEG_city_mnt.py): separate HRFs for EEG within
    city and mnt trials. It now fits mnt-correct and mnt-incorrect with AR-IRLS, fixAR-IRLS, and
    AR-iRRR (Stage 8).
- Visualization:
  - [vis_hrf_cont_EEG_cz.py](vis_hrf_cont_EEG_cz.py): HRF maps and near-Cz HRFs, for AR-IRLS or
    fixAR-IRLS.
  - [vis_hrf_cont_EEG_city_mnt.py](vis_hrf_cont_EEG_city_mnt.py): the same plots for the
    city/mnt models.
  - [vis_fit.py](vis_fit.py): Y vs Ŷ and residual diagnostics.
  - [vis_EV_on_surface.py](vis_EV_on_surface.py): explained variance on the brain surface.
  - [vis_hrf_single_parcel_xSubjects.py](vis_hrf_single_parcel_xSubjects.py): one parcel across
    subjects, event-based vs continuous-EEG.

### Stage 4: Structure of the EEG HRF across parcels and subjects
- [single_subj_HRF_analysis.py](single_subj_HRF_analysis.py): SVD or ICA of each subject's
  parcel × delay HRF matrix.
- [xSubj_shared_HRF_analysis_hungarian_matching.py](xSubj_shared_HRF_analysis_hungarian_matching.py):
  matches components across subjects (an earlier approach) and clusters them by HRF shape and
  surface location.
- [xSubj_shared_HRF_analysis.py](xSubj_shared_HRF_analysis.py): one joint SVD over all subjects'
  parcels, which gives directly comparable shared HRF shapes. These are compared against a
  DorsAttn vs Default template.
- The idea of a few shared HRF shapes led to the low-rank model in Stage 5.

### Stage 5: iRRR and the switch to spectral whitening
- [check_iRRR.py](check_iRRR.py) and
  [run_model_cont_EEG_fNIRS_iRRR.py](run_model_cont_EEG_fNIRS_iRRR.py): iRRR on one subject and on
  all subjects. [run_model_cont_EEG_fNIRS_iRRR_group.py](run_model_cont_EEG_fNIRS_iRRR_group.py)
  fits a group iRRR. [vis_hrf_cont_EEG_cz_iRRR.py](vis_hrf_cont_EEG_cz_iRRR.py) plots shared
  components and their weights.
- **Problem:** applying the AR-IRLS per-parcel AR filter before iRRR removed the low-frequency EEG
  content (commit `33bb9bb`).
- **Fix:** one *spectral* whitening kernel shared by all parcels and applied to both Y and X. This
  gave two models:
  - [run_model_cont_EEG_fNIRS_fixAR-IRLS.py](run_model_cont_EEG_fNIRS_fixAR-IRLS.py) (fixAR-IRLS)
  - [run_model_cont_EEG_fNIRS_AR-iRRR.py](run_model_cont_EEG_fNIRS_AR-iRRR.py) (AR-iRRR, with λ₁
    chosen by CV)
- [cmp_model_hrf_r2.py](cmp_model_hrf_r2.py): RMSE and R² of fixAR-IRLS, AR-iRRR, and group iRRR on
  the whitened Y, plus an event-based comparison within mnt trials.
- **Write-up:** [summary_Hp_fixAR-IRLS_iRRR_AR.ipynb](summary_Hp_fixAR-IRLS_iRRR_AR.ipynb). Results:
  - Y scaling is required (see the abstract).
  - CV selects rank 0 for 3 of 6 subjects.
  - Where the rank is above 0, held-out PMSE drops by only 0.03–0.15%.
  - Mean in-sample R² is 0.0043 for fixAR-IRLS and 0.0014 for AR-iRRR.

### Stage 6: Does it generalize? (single subject at a time, 6 subjects)
- [cmp_fixAR-IRLS_AR-iRRR_single-subj_single-run.py](cmp_fixAR-IRLS_AR-iRRR_single-subj_single-run.py):
  train on gradcpt1 and test on gradcpt2/3. Each run is whitened before truncation.
- [cmp_fixAR-IRLS_AR-iRRR_single-subj_all-runs.py](cmp_fixAR-IRLS_AR-iRRR_single-subj_all-runs.py):
  train on all runs (in-sample). Each run is truncated before whitening.
- Both scripts are run with the Cz EEG design and the mnt-correct Gaussian-kernel design.
- **Write-up:**
  [summary_cmp_fixAR-IRLS_AR-iRRR_single-subj.ipynb](summary_cmp_fixAR-IRLS_AR-iRRR_single-subj.ipynb).
  Results:
  - Test R² is about 0 or below for both models and both designs.
  - In the rank sweep, the best test rank is usually 0.
  - In-sample R² is < 0.02.
  - The run-1 HRFs only partly agree with the all-runs HRFs (median r 0.58–0.94 for fixAR-IRLS).

### Stage 7: AR-IRLS vs fixAR-IRLS vs AR-iRRR, event-based, all subjects
- [cmp_AR-IRLS_fixAR-IRLS_single-subj.py](cmp_AR-IRLS_fixAR-IRLS_single-subj.py): the two IRLS
  models on one subject.
- [cmp_AR-IRLS_fixAR-IRLS_AR-iRRR.py](cmp_AR-IRLS_fixAR-IRLS_AR-iRRR.py): all three models, every
  subject, using Gaussian kernels at mnt-correct onsets.
  - `--train-runs gradcpt1` scores the held-out runs.
  - `--ranks 2 4 6 8` saves one iRRR fit per rank.
- [cmp_Cedalion-pipe_My-pipe.py](cmp_Cedalion-pipe_My-pipe.py): compares this pipeline against the
  coworker's Cedalion pipeline (0.01 Hz highpass, three trial types, per-run GSR regressor) on the
  same parcel time series.
- **Write-up:** [summary_cmp_AR-IRLS_fixAR-IRLS_AR-iRRR.ipynb](summary_cmp_AR-IRLS_fixAR-IRLS_AR-iRRR.ipynb).
  It covers N = 20 subjects and was run with 0.02 Hz highpass and 12 kernels from −2 to 10 s. The
  script now uses 0.01 Hz and 20 kernels from −2 to 18 s. Results:
  - Median per-parcel shape r: AR-IRLS vs fixAR-IRLS 0.87, AR-IRLS vs AR-iRRR 0.70, fixAR-IRLS vs
    AR-iRRR 0.77.
  - AR-iRRR is ~4× smaller.
  - ContA peaks at ~4.5 s and dips at ~8 s.
  - The outlier subjects are sub-751 and sub-719.

### Stage 8: Network-based fits and the four-way mnt-correct comparison
Each network has its own whitening kernel, `Y_scale`, and iRRR. The network is the parcel-name
prefix, for 17 networks.
- [run_model_cont_EEG_fNIRS_AR-iRRR_network-based.py](run_model_cont_EEG_fNIRS_AR-iRRR_network-based.py):
  continuous Cz EEG design. **Write-up:**
  [summary_AR-iRRR_fixAR-IRLS_network-based.ipynb](summary_AR-iRRR_fixAR-IRLS_network-based.ipynb).
  - Median shape r is 0.66 and the RMS ratio is 0.60.
  - 68% of parcels have p < 0.05 in the fixAR-IRLS F-test. This is probably anti-conservative.
- [run_model_gaussian-kernel_mnt-correct_network-based.py](run_model_gaussian-kernel_mnt-correct_network-based.py)
  (`GK-net`): Gaussian kernels, per network.
- [run_model_cont_EEG_mnt-correct_network-based.py](run_model_cont_EEG_mnt-correct_network-based.py)
  (`EEG-net`): Cz EEG within mnt-correct trials, per network.
- [run_model_cont_EEG_city_mnt.py](run_model_cont_EEG_city_mnt.py) (`city`): Cz EEG within trials,
  mnt-correct and mnt-incorrect fit together, whole brain, all three models.
- **Write-up:**
  [summary_run_model_cont_EEG_city_mnt.ipynb](summary_run_model_cont_EEG_city_mnt.ipynb). It
  compares `city`, `EEG-net`, `GK-net`, and `GK-cmp` (the Stage 7 script):
  - `EEG-net` vs `GK-net`: same pipeline, different design.
  - `city` vs `EEG-net` and `GK-cmp` vs `GK-net`: same design, different pipeline.
  - Section 16 (summary) is still to be written.
- sub-670 is skipped by the network scripts because two of its runs hold duplicated fNIRS data in
  the shared `Y_all`.

### Side track: pupil, VTC, behavior
- [cmp_vtc_pupil.py](cmp_vtc_pupil.py): VTC smoothing and its relation to pupil.
- [reproduce_Brink_2016.py](reproduce_Brink_2016.py): reproduces the van den Brink 2016
  pupil/behavior time-on-task analysis.
- [vis_pupil_preproc.py](vis_pupil_preproc.py): pupil preprocessing steps.
- [alpha_pupil_coupling.py](alpha_pupil_coupling.py): posterior alpha vs high-frequency pupil.
- [ar-irls_eeg-power_vtc.py](ar-irls_eeg-power_vtc.py): EEG band power vs VTC, fit with AR-IRLS.
- [find_EEG_ERP_by_GLM.py](find_EEG_ERP_by_GLM.py): ERP by GLM deconvolution of continuous EEG.

---

## 4. Figure index

`D = /projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/eeg`. Placeholders: `<s>` = subject,
`<m>` = model name built in the script, `<net>` = network, `<p>` = parcel.

### 4a. Summary notebooks (recommended starting point)
| Figure | Notebook (section) | Saved to |
|---|---|---|
| Y_all rebuild steps and PSD (sanity check) | summary_Hp_fixAR-IRLS_iRRR_AR (1f) | `D/HRF_surf/group/cont_EEG_cz_3-stage_model_cmp/summary_Hp/sanity_Y_all_{steps,psd}_*.png` |
| iRRR λ₁ vs CV PMSE | summary_Hp_fixAR-IRLS_iRRR_AR (2) | `…/summary_Hp/irrr_lam1_vs_PMSE.png` |
| Example time series Y vs Ŷ | summary_Hp_fixAR-IRLS_iRRR_AR (3) | `…/summary_Hp/ts_example_<s>.png` |
| R² / RMSE bars, surface maps | summary_Hp_fixAR-IRLS_iRRR_AR (4) | `…/summary_Hp/bar_r2_rmse.png`, `surf_r2_rmse.png` |
| HRF at 5 parcels nearest Cz | summary_Hp_fixAR-IRLS_iRRR_AR (5) | `…/summary_Hp/hrf_5_parcels_nearest_Cz.png` |
| iRRR components and weights | summary_Hp_fixAR-IRLS_iRRR_AR (6) | `…/summary_Hp/irrr_components_<tag>.png` |
| Train/test R² and PMSE | summary_cmp_fixAR-IRLS_AR-iRRR_single-subj (3) | `D/HRF_surf/group/cmp_fixAR-IRLS_AR-iRRR_single-subj/train_test_R2_PMSE.png` |
| iRRR rank sweep (test) | same (4) | `…/iRRR_rank_sweep.png` |
| Per-parcel test R² box and surface | same (5) | `…/test_R2_parcel_box.png`, `surf_test_r2.png` |
| HRFs, plot parcel + near Cz | same (6, 14) | `…/hrf_<tag>_<design>.png` |
| Rank-2 iRRR temporal components | same (7, 15) | `…/iRRR_components_<tag>.png` |
| All-runs in-sample R², rank sweep, surface | same (11–13) | `…/allRuns_insample_R2_PMSE.png`, `allRuns_iRRR_rank_sweep.png`, `allRuns_surf_insample_r2.png` |
| HRF agreement gradcpt1 vs all runs | same (14) | `…/hrf_agreement_gradcpt1_vs_allRuns.png` |
| HRF shape r between models | summary_cmp_AR-IRLS_fixAR-IRLS_AR-iRRR (3) | `D/HRF_surf/group/mnt-correct_gauss_cmp_AR-IRLS_fixAR-IRLS_AR-iRRR/summary_nb/hrf_corr_box_hist.png` |
| HRF amplitude, SE | same (4, 5) | `…/summary_nb/hrf_amplitude.png`, `hrf_se.png` |
| Group HRF (ContA) | same (6) | `…/summary_nb/group_hrf.png` |
| Network HRF grids (all 17; 12 main; LH/RH; normalized) | same (7) | `…/summary_nb/group_hrf_{all_networks,12_networks}[_LH\|_RH][_norm].png` |
| HRF r vs sensitivity; thresholded r surface | same (8) | `…/summary_nb/hrf_corr_vs_sensitivity.png`, `group_hrf_corr_surf_with_thresh.png` |
| AR-iRRR components | same (9) | `…/summary_nb/irrr_components.png` |
| λ₁ / Y_scale per network | summary_AR-iRRR_fixAR-IRLS_network-based (2) | `D/HRF_surf/group/cont_EEG_cz_3-stage_AR-iRRR_vs_fixAR-IRLS-network_Hp/fit_summary.png` |
| HRF at plot parcel | same (3) | `…/hrf_DefaultA_PFCd_1_LH.png` |
| Network HRFs (µM, normalized) | same (4) | `…/hrf_networks[_normalized].png` |
| Shape/amplitude agreement per network | same (5) | `…/agreement_by_network.png` |
| F-test fraction significant | same (6) | `…/ftest_frac_sig.png` |
| `city`: F-test, HRF r, amplitude, group HRF | summary_run_model_cont_EEG_city_mnt (4–7) | `D/HRF_surf/group/<city eeg_reg_type>/summary_nb/{ftest_frac_sig,hrf_corr,hrf_amplitude,group_hrf}.png` |
| `city`: 12-network grids per trial type; correct vs incorrect | same (8, 9) | `…/summary_nb/group_hrf_12_networks_<tt>.png`, `group_hrf_12_networks_<model>_correct_vs_incorrect.png` |
| `city`: iRRR components | same (10) | `…/summary_nb/irrr_components.png` |
| `GK-net`: network grids, per-subject HRF | same (12) | `D/HRF_surf/group/mnt-correct_gauss_3-stage/summary_nb/group_hrf_12_networks[_norm].png`, `subject_hrf_<net>.png` |
| `EEG-net`: network grids | same (13) | `D/HRF_surf/group/cont_EEG_cz_mnt-correct_pre2post18_3-stage/summary_nb/group_hrf_12_networks[_norm].png` |
| `GK-cmp`: agreement, network grids | same (14a–b) | `D/HRF_surf/group/mnt-correct_gauss_cmp_AR-IRLS_fixAR-IRLS_AR-iRRR/summary_nb/hrf_corr_amplitude.png`, `group_hrf_12_networks[_norm].png` |
| `GK-cmp` ranks 2/4/6/8: network grids, parcel HRFs, one subject, iRRR basis HRFs, test rank sweep | same (14c–g) | `D/HRF_surf/group/mnt-correct_gauss_cmp_AR-IRLS_fixAR-IRLS_AR-iRRR_rank_sweep/summary_nb/{group_hrf_12_networks_train-<tag>[_norm], parcel_hrf_train-<tag>[_norm], sub-723_parcel_hrf_train-<tag>[_norm], sub-723_irrr_components_train-<tag>_rank<k>, iRRR_rank_sweep_train-gradcpt1}.png` |
| Four-script HRF r matrix, network grids per model | same (15) | `D/HRF_surf/group/mnt-correct_4-script_cmp/summary_nb/hrf_corr_matrix.png`, `group_hrf_12_networks_<model>_norm.png` |

### 4b. Scripts that save figures
| Figure | Script | Saved to |
|---|---|---|
| Per-subject HRF at most sensitive parcel / network; HRF r surface | cmp_AR-IRLS_fixAR-IRLS_AR-iRRR.py | `D/HRF_surf/<s>/<m>/<s>_hrf_<p>.png`, `<s>_hrf_network_<net>.png`, `<s>_hrf_corr_surf[_<pair>].png` |
| Group median HRF r, group network HRF, group r surface | cmp_AR-IRLS_fixAR-IRLS_AR-iRRR.py | `D/HRF_surf/group/<m>/group_median_hrf_corr.png`, `group_hrf_<net>.png`, `group_hrf_corr_surf[_<pair>].png` |
| HRF at parcel / network, HRF r surface | cmp_AR-IRLS_fixAR-IRLS_single-subj.py | `D/HRF_surf/<s>/mnt-correct_gauss_cmp_AR-IRLS_fixAR-IRLS/` |
| Cedalion vs my pipeline: network HRF, HRF r | cmp_Cedalion-pipe_My-pipe.py | `D/HRF_surf/<s>/cmp_Cedalion-pipe_My-pipe/<s>_network_hrf_<hemi>[_norm].png`, `<s>_hrf_corr_vs_cedalion_pipe.png` |
| iRRR rank vs RMSE/R², Y_white fit, HRFs near Cz | cmp_fixAR-IRLS_AR-iRRR_single-subj_{single-run,all-runs}.py | `D/HRF_surf/<s>/<m>/<s>_iRRR_rank_vs_RMSE_R2.png`, `<s>_<p>_Y_white_fit.png`, `<s>_hrf_select_and_3_nearest_Cz_parcels.png` |
| RMSE/R² bars and surfaces, HRF near Cz, mnt-trial comparison | cmp_model_hrf_r2.py | `D/HRF_surf/group/cont_EEG_cz_3-stage_model_cmp/model_cmp_*.png`, `surf_*.png` (+ `.csv`); mnt-trial section in `D/HRF_surf/group/<mnt_reg_type>_model_cmp/` |
| iRRR λ₁ vs PMSE | run_model_cont_EEG_fNIRS_AR-iRRR.py | `D/HRF_surf/<s>/<m>/<s>_iRRR_lam1_vs_PMSE.png` |
| iRRR λ₁ vs rank, per network | run_model_cont_EEG_fNIRS_AR-iRRR_network-based.py, run_model_gaussian-kernel_mnt-correct_network-based.py, run_model_cont_EEG_mnt-correct_network-based.py | `D/HRF_surf/<s>/<irrr model>/<s>_<net>_iRRR_lam1_vs_rank.png` |
| iRRR λ₁ vs rank | run_model_cont_EEG_city_mnt.py | `D/HRF_surf/<s>/<irrr model>/<s>_iRRR_lam1_vs_rank.png` |
| Group iRRR: Y vs Ŷ, R² per subject, components and weights | run_model_cont_EEG_fNIRS_iRRR_group.py | `D/HRF_surf/group/cont_EEG_cz_3-stage_iRRR/` |
| iRRR component weight surfaces | check_iRRR.py | `D/HRF_surf/<s>/iRRR_components/component<k>_weight.png` |
| iRRR per-subject and group HRFs, components, heatmap | vis_hrf_cont_EEG_cz_iRRR.py | `D/HRF_surf/<s>/<m>/`, `D/HRF_surf/group/cont_EEG_cz_3-stage_iRRR/{xSubj_iRRR_network_HRF,group_iRRR_network_HRF,group_iRRR_parcel_HRF_heatmap,group_iRRR_shared_HRF_components}.png` |
| HRF surfaces and near-Cz HRFs (AR-IRLS / fixAR-IRLS) | vis_hrf_cont_EEG_cz.py | `D/HRF_surf/{<s>,group}/<plot_tag>/` (+ `near_Cz/<p>_HRF.png`) |
| HRF surfaces and near-Cz HRFs (city/mnt or event parcel models) | vis_hrf_cont_EEG_city_mnt.py | `D/HRF_surf/{<s>,group}/<eeg_reg_type>/<trial type>/` (+ `near_Cz/`) |
| Explained variance on surface, by network | vis_EV_on_surface.py | `D/EV_surf/group_EV_by_network.png`, `D/EV_surf/<s>/<model>/` |
| Full/nuisance/EEG fit, worst-EV trials, residual diagnostics | vis_fit.py | `D/fit_vis/<s>/<model>/*.png` |
| Per-subject SVD/ICA component weights | single_subj_HRF_analysis.py | `D/HRF_surf/<s>/{SVD,ICA}_components/component<k>_U_weight.png` |
| Joint-SVD component weights | xSubj_shared_HRF_analysis.py | `D/components_analysis/joint_svd_component<k>_avg_U_weight.png` |
| Matched component weights | xSubj_shared_HRF_analysis_hungarian_matching.py | `D/components_analysis/<ref>_component<k>_median_U_weight.png` |
| mnt-correct vs mnt-incorrect ERP | model_pilot.py | `fig_save_path/mntC_vs_mntIC_<ch>_mean_2SEM.png` (set in script) |
| Parcellation legend (paper Fig. 1) | vis_hrf_brain_surface.py | `…/derivatives/cedalion/PAPER_FIGS/FIG1/parcellation_legend.png` |

### 4c. Scripts that only display figures (nothing saved)
Run these interactively (`#%%` cells) to regenerate their plots. To keep a figure, add a `savefig`.

| Script | What it plots |
|---|---|
| model_continuous_EEG_fNIRS.py, run_model_cont_EEG_fNIRS.py, run_model_cont_EEG_trial_based.py, run_model_cont_EEG_fNIRS_add_VTC.py, run_model_cont_EEG_fNIRS_rest.py | design matrices, Y vs Ŷ, HRFs during fitting |
| vis_hrf.py, vis_hrf_network.py, run_stat_test.py | Stage 1 channel/network HRFs and F-test maps |
| cmp_model_hrf.py, cmp_model_MSE_EV_single_subj_single_channel.py, manual_fit_model.py, check_yhat_AR-IRLS_OLS.py, check_yhat_AR-IRLS_image_space.py, ar-irls_test.py | fit sanity checks |
| CORRECT_CODE_runGLM_on_image.py, hrf_estimate_Laura_copy.py, check_image_recon.py | reference image-space GLM |
| vis_hrf_single_parcel_xSubjects.py, vis_parcel_cluster.py | single-parcel HRF across subjects; parcel clustering |
| cmp_vtc_pupil.py, reproduce_Brink_2016.py, alpha_pupil_coupling.py, vis_pupil_preproc.py, ar-irls_eeg-power_vtc.py, find_EEG_ERP_by_GLM.py | pupil / VTC / behavior / ERP side track |
