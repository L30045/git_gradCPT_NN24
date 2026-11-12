# pupil

Pupil vs behavior research for gradCPT (moved from `../modeling/`). The HRF / EEG-fNIRS models stay in `../modeling/`.

- [cmp_vtc_pupil.py](cmp_vtc_pupil.py): VTC smoothing and its relation to pupil.
- [reproduce_Brink_2016.py](reproduce_Brink_2016.py): reproduces the van den Brink 2016 pupil/behavior time-on-task analysis.
- [vis_pupil_preproc.py](vis_pupil_preproc.py): pupil preprocessing steps.
- [alpha_pupil_coupling.py](alpha_pupil_coupling.py): posterior alpha vs high-frequency pupil.

These scripts import `params_setting` from `../modeling/` and `utils` from `../preproc_pipe/`, plus `utils_eyetracking` from `code/eyetracking`. All figures are shown interactively; nothing is saved.
