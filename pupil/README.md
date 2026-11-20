# pupil

Pupil vs behavior research for gradCPT (moved from `../modeling/`). The HRF / EEG-fNIRS models stay in `../modeling/`.

- [cmp_vtc_pupil.py](cmp_vtc_pupil.py): VTC smoothing and its relation to pupil.
- [reproduce_Brink_2016.py](reproduce_Brink_2016.py): reproduces the van den Brink 2016 pupil/behavior time-on-task analysis.
- [vis_pupil_preproc.py](vis_pupil_preproc.py): pupil preprocessing steps.
- [alpha_pupil_coupling.py](alpha_pupil_coupling.py): posterior alpha vs high-frequency pupil.
- [investigating_phasic.py](investigating_phasic.py): city-correct epochs (±8 s) sorted by a per-trial covariate (`covariate = 'vtc'` or `'theta'`, EEG Fz theta from the extra-trial-feature table); pupil power at the 1.25 Hz stimulus rate vs the covariate; phase locking of that rhythm to the stimulus train. Theta outputs go to `derivatives/eye_tracking/phasic/fz_theta_win/`.
- [phasic_single_run.py](phasic_single_run.py): one subject, one run (`subj`, `run_id`): is the 1.25 Hz pupil phase stable across trials? Raster in trial order and by covariate, per-trial phase vs time in run (16 s and short 2-cycle windows), ITPC vs shift surrogate, Rayleigh on non-overlapping trials, phase similarity vs trial separation, and power/ITPC vs covariate. Figures go to `derivatives/eye_tracking/phasic/<subj>/`.
- [check_phase_across_runs.py](check_phase_across_runs.py): one subject, why the stimulus-rate phase differs across runs: per-run spaghetti plots, cycle-folded waveform, phase over time, and a Neon-vs-fNIRS accelerometer sync check.

These scripts import `params_setting` from `../modeling/` and `utils` from `../preproc_pipe/`, plus `utils_eyetracking` from `code/eyetracking`. All figures are shown interactively; nothing is saved, except `investigating_phasic.py` and `phasic_single_run.py`, which also write its figures to `derivatives/eye_tracking/phasic/`.
