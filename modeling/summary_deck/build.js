const pptxgen = require('pptxgenjs');
const { applyTheme } = require('/usr2/collab/cchang1/.claude/skills/synced/6653a02f-c571-443c-b604-8de73257e9f4_9d0cfe53-209d-46d9-babc-2e45d8cbde1c/pptx/scripts/apply_theme.js');
const THEME = { name: 'HRF-EEG', headFontFace: 'Cambria', bodyFontFace: 'Calibri',
  colors: { dk1: '1C2333', lt1: 'FFFFFF', dk2: '24324F', lt2: 'EEF1F6',
    accent1: 'EB6834', accent2: '2A78D6', accent3: '1BAF7A', accent4: '6250D6', accent5: '8A93A6', accent6: 'E34948',
    hlink: '2A78D6', folHlink: '6250D6' } };
const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE';
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = 'HRF estimation with EEG: summary';
const C = pres.SchemeColor;
const INK = '1C2333', MUTED = '52514E', ORANGE = 'EB6834', BLUE = '2A78D6', AQUA = '1BAF7A', GREY = '8A93A6';

pres.defineSlideMaster({ title: 'DARK', background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: 0.8, y: 1.6, w: 11.7, h: 1.4, fontSize: 40, bold: true, color: C.background1, valign: 'bottom', fontFace: 'Cambria' }, text: '' } },
    { placeholder: { options: { name: 'body', type: 'body', x: 0.8, y: 3.25, w: 11.7, h: 3.4, fontSize: 18, color: 'CADCFC', valign: 'top' }, text: '' } },
  ] });
pres.defineSlideMaster({ title: 'CONTENT', background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: 0.6, y: 0.3, w: 12.1, h: 0.8, fontSize: 28, bold: true, color: C.text2, valign: 'middle', fontFace: 'Cambria' }, text: '' } },
    { text: { text: 'gradCPT_NN24 · EEG-informed HRF modeling', options: { x: 0.6, y: 7.05, w: 7, h: 0.3, fontSize: 10, color: GREY } } },
  ],
  slideNumber: { x: 12.2, y: 7.05, w: 0.6, h: 0.3, fontSize: 10, color: GREY, align: 'right' } });

const sub = (s, text) => s.addText(text, { x: 0.6, y: 1.1, w: 12.1, h: 0.45, fontSize: 15, color: MUTED, italic: true, isTextBox: true, margin: 0, objectName: 'subtitle' });
const img = (s, path, x, y, w, aspect, name) => s.addImage({ path, x, y, w, h: w / aspect, objectName: name, altText: name });
function badge(s, n, x, y, color, d = 0.5) {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color }, line: { color }, objectName: `badge ${n}` });
  s.addText(String(n), { x, y, w: d, h: d, fontSize: d > 0.45 ? 16 : 13, bold: true, color: 'FFFFFF', align: 'center', valign: 'middle', margin: 0, isTextBox: true });
}
function stat(s, x, y, w, big, label, color, bigSize = 34) {
  s.addText(big, { x, y, w, h: 0.7, fontSize: bigSize, bold: true, color, margin: 0, isTextBox: true, fontFace: 'Cambria' });
  s.addText(label, { x, y: y + 0.7, w, h: 0.55, fontSize: 13, color: MUTED, margin: 0, valign: 'top', isTextBox: true });
}
function card(s, x, y, w, h, n, color, head, big, body, name) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: C.background2 }, line: { color: 'E0E4EC', width: 0.75 }, objectName: name });
  badge(s, n, x + 0.25, y + 0.25, color);
  s.addText(head, { x: x + 0.9, y: y + 0.2, w: w - 1.1, h: 0.6, fontSize: 17, bold: true, color: C.text2, valign: 'middle', margin: 0, isTextBox: true });
  if (big) s.addText(big, { x: x + 0.25, y: y + 0.95, w: w - 0.5, h: 0.75, fontSize: 32, bold: true, color, margin: 0, isTextBox: true, fontFace: 'Cambria' });
  s.addText(body, { x: x + 0.25, y: y + (big ? 1.75 : 0.95), w: w - 0.5, h: h - (big ? 1.9 : 1.1), fontSize: 14, color: INK, valign: 'top', margin: 0, isTextBox: true, paraSpaceAfter: 6 });
}
const bl = (t, o = {}) => ({ text: t, options: { bullet: true, breakLine: true, ...o } });
const last = (t, o = {}) => ({ text: t, options: { bullet: true, ...o } });
const statRow = (s, items, y = 5.75) => { const w = 12.1 / items.length; items.forEach(([b, l, c], i) => stat(s, 0.6 + i * w, y, w - 0.3, b, l, c)); };
function arrow(s, x, y) { s.addShape(pres.shapes.CHEVRON, { x, y, w: 0.22, h: 0.36, fill: { color: 'C9CFDA' }, line: { color: 'C9CFDA' }, objectName: 'flow arrow' }); }

// ---------------- Intro
pres.addSection({ title: 'Intro' });
let s = pres.addSlide({ masterName: 'DARK', sectionTitle: 'Intro' });
s.addText('Can EEG inform the fNIRS HRF?', { placeholder: 'title' });
s.addText('HRF estimation on gradCPT mnt-correct trials, with the event-based HRF as the benchmark\nN = 6 subjects with simultaneous Cz EEG · image-space parcels · AR-IRLS', { placeholder: 'body' });
s.addNotes('Every design is compared against the event-based Gaussian-kernel (GK) HRF for mnt-correct trials. All HRFs are fit on the same parcel time series with Cedalion AR-IRLS, and all span -2 to 18 s.');

s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Intro' });
s.addText('Three EEG designs, each judged against the benchmark', { placeholder: 'title' });
sub(s, 'AR-IRLS on the same 410 parcels (new Adot); all predict the average mnt-correct response, −2 to 18 s');
const hdr = (t) => ({ text: t, options: { bold: true, color: 'FFFFFF', fill: { color: '24324F' }, fontSize: 14, valign: 'middle' } });
const cell = (t, o = {}) => ({ text: t, options: { fontSize: 13, color: INK, valign: 'middle', ...o } });
const rowLab = (t, sw) => ({ text: t, options: { fontSize: 14, bold: true, color: INK, valign: 'middle', fill: { color: sw } } });
s.addTable([
  [hdr('Design'), hdr('What drives the regressor'), hdr('Uses EEG'), hdr('Section')],
  [rowLab('Benchmark (event GK)', 'EEF1F6'), cell('Mnt-correct onsets, 20 Gaussian kernels (−2 to 18 s, 1 s apart, σ = 1 s), every trial weight = 1'), cell('No'), cell('Reference everywhere')],
  [rowLab('EEG-informed GK', 'FCE6DC'), cell('Same kernels; each trial weighted by its Cz N2–P3 peak-to-peak ÷ run mean'), cell('Trial amplitude'), cell('Part 1')],
  [rowLab('Cont-EEG (mnt-correct)', 'DCE9F8'), cell('Cz EEG kept only inside mnt-correct trials, delayed −2 to 18 s on 20 B-splines'), cell('Waveform in trials'), cell('Part 2')],
  [rowLab('Cont-EEG (entire run)', 'D6F2E7'), cell('All continuous Cz, delayed 0 to 15 s on 15 B-splines'), cell('Whole waveform'), cell('Part 3')],
], { x: 0.6, y: 1.8, w: 12.1, colW: [2.8, 6.0, 1.6, 1.7], rowH: [0.5, 0.72, 0.72, 0.72, 0.72], border: { type: 'solid', color: 'D5DAE3', pt: 0.75 }, objectName: 'design table' });
s.addText([
  bl('Preprocessing shared by all fits: 0.02 Hz highpass (0.01 Hz for Cont-EEG) → OLS drift → OLS GSR, then AR-IRLS'),
  last('Comparison: per-parcel shape r over 0–18 s, RMS amplitude ratio, F-tests, and network means across subjects'),
], { x: 0.6, y: 5.6, w: 12.1, h: 1.2, fontSize: 15, color: INK, margin: 0, paraSpaceAfter: 6, isTextBox: true, objectName: 'notes list' });
s.addNotes('Subjects: 670, 695, 721, 723, 726, 730 (clean Cz). sub-670 is now included: its shared Y_all no longer has duplicated runs after the fNIRS-EEG matching fix. The estimator comparison (AR-IRLS vs fixAR-IRLS vs AR-iRRR) moved to the appendix.');

// ---------------- Part 1: EEG-informed GK
pres.addSection({ title: 'EEG-informed GK' });
s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'EEG-informed GK' });
s.addText('EEG-informed GK: scale each trial’s kernels by its ERP amplitude', { placeholder: 'title' });
sub(s, 'Pipeline for one subject (sub-723); the benchmark is the same pipeline with every weight = 1');
const steps = [['Measure N2–P3 ptp', 'Woody P3 peak minus the N2 trough before it, Cz, per trial'],
  ['Normalize within run', 'aᵢ / ā, so the average trial weighs 1'],
  ['Weight each onset', 'Mnt-correct onsets get height aᵢ / ā instead of 1'],
  ['Convolve with kernels', '20 Gaussian kernels (−2 to 18 s) → 20 design columns']];
const pw = 2.8, gap = 0.3, x0 = 0.6;
steps.forEach(([h, b], i) => {
  const x = x0 + i * (pw + gap);
  badge(s, i + 1, x, 1.75, ORANGE, 0.42);
  s.addText(h, { x: x + 0.52, y: 1.73, w: pw - 0.52, h: 0.46, fontSize: 15, bold: true, color: C.text2, margin: 0, valign: 'middle', isTextBox: true });
  img(s, `pipe_gk_${i + 1}.png`, x, 2.35, pw, 1100 / 792, `pipeline step ${i + 1}`);
  s.addText(b, { x, y: 4.45, w: pw, h: 0.75, fontSize: 13, color: INK, margin: 0, valign: 'top', isTextBox: true });
  if (i < 3) arrow(s, x + pw + 0.04, 3.2);
});
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 5.5, w: 12.1, h: 1.25, rectRadius: 0.06, fill: { color: C.background2 }, line: { color: 'E0E4EC', width: 0.75 }, objectName: 'note box' });
s.addText([
  { text: 'Why it could help: ', options: { bold: true } },
  { text: 'if bigger ERPs mean more neural activity, the HbO response should scale with them, and a weighted design should fit better than the onset-only benchmark. Trials rejected for artifacts (> 100 µV) get the mean weight of 1.' }],
  { x: 0.85, y: 5.6, w: 11.6, h: 1.05, fontSize: 14, color: INK, margin: 0, valign: 'middle', isTextBox: true });
s.addNotes('Built by build_dm_gaussian-kernel_mnt-correct_EEG-informed.py with features from extract_EEG_trial_features.py (cz_N2P3_ptp_woody; Pz when Cz has no clean trials in a run). In sub-723 the weights span 0.28 to 1.94 (SD 0.32). The kernel panel shows kernel 8 of 20 (centred at +5 s).');

s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'EEG-informed GK' });
s.addText('ERP amplitude and latency do not track the zone', { placeholder: 'title' });
sub(s, 'Cz, clean mnt-correct trials; in-zone = VTC below the subject’s median; one line per subject (dark: the 6 modeling subjects)');
img(s, 'fig_zone_erp.png', 0.6, 1.75, 8.3, 1700 / 919, 'ERP zone comparison');
stat(s, 9.4, 1.95, 3.3, 'p = 0.27', 'N2–P3 ptp: 28.2 vs 27.3 µV (in vs out), N = 17', ORANGE);
stat(s, 9.4, 3.45, 3.3, 'p = 0.36', 'P3 latency: 688 vs 692 ms (in vs out), N = 17', ORANGE);
stat(s, 9.4, 4.95, 3.3, '≥ 10', 'Clean trials per zone required per subject (sub-747 has none)', GREY);
s.addNotes('Paired t-tests on per-subject means. Trial weights in the EEG-informed design vary a lot within subject (SD of a_i/a-bar about 0.3-0.4), but that variation is not tied to attention state. Source: eeg_trial_feats.csv built from the features table and VTC_smoothed.');

s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'EEG-informed GK' });
s.addText('Within-trial band amplitude does not differ between zones either', { placeholder: 'title' });
sub(s, 'Cz Hilbert amplitude of the band-passed continuous EEG, summed over each mnt-correct trial (onset to onset + duration), all trials, N = 18');
img(s, 'fig_zone_hilbert.png', 0.6, 1.75, 12.1, 2600 / 880, 'Hilbert zone comparison');
statRow(s, [['p = 0.72', 'Delta, 1–4 Hz', BLUE], ['p = 1.00', 'Theta, 4–8 Hz', BLUE], ['p = 0.88', 'Alpha, 8–13 Hz', BLUE], ['p = 0.51', 'Beta, 13–30 Hz', BLUE]], 5.95);
s.addNotes('Envelope = |hilbert| of a zero-phase band-pass of the preprocessed Cz (0.1-45 Hz, mastoid reference, ICA). Paired t-tests on per-subject means of the per-trial sum (uV*s). The one high subject is the same in every band (overall amplitude, not a zone effect).');

s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'EEG-informed GK' });
s.addText('EEG-informed columns are rescaled benchmark columns', { placeholder: 'title' });
sub(s, 'sub-723, run 1, 60 s; rows = fNIRS samples, columns = the 20 Gaussian kernels');
img(s, 'dm_example.png', 0.6, 1.7, 8.9, 2100 / 1200, 'design matrix example');
s.addText([
  bl('Same onsets, same 20 kernels; only the height of each trial’s diagonal band changes'),
  bl('Matching columns correlate r = 0.92–0.96 (median per subject)'),
  last('So the two designs can only differ through the trial-to-trial spread of aᵢ / ā'),
], { x: 9.8, y: 1.85, w: 2.95, h: 4.9, fontSize: 15, color: INK, margin: 0, paraSpaceAfter: 12, valign: 'top', isTextBox: true, objectName: 'DM bullets' });
s.addNotes('Labels on the right of the EEG-informed matrix are the onset weights in this window. Column correlation computed from the saved full DM (stim vs eeg column, per kernel, median over kernels) for each of the 6 subjects.');

s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'EEG-informed GK' });
s.addText('Two parcels: the EEG-informed HRF overlays the benchmark', { placeholder: 'title' });
sub(s, 'DorsAttnB_PostC_2_RH and SomMotA_25_LH (locations in appendix); bands = 95% CI (mean panel: across subjects)');
img(s, 'fig_2p_eeginf.png', 0.95, 1.6, 11.4, 3200 / 1120, 'two parcels EEG-informed');
statRow(s, [['r = 0.96', 'Median shape r vs benchmark, 12 subject-parcels (0.76–0.99)', ORANGE], ['0.96×', 'Median peak amplitude relative to the benchmark', ORANGE], ['8 vs 8', 'Of 12 F-tests significant: EEG-informed vs benchmark', ORANGE]]);
s.addNotes('Curves are the predicted response to an average mnt-correct trial, baseline (t<0) removed; the EEG-informed curve is multiplied by the mean weight (1.0). Grey thin line: block average of Y. From cmp_hrf_gaussian-kernel_vs_cont-EEG_mnt-correct.py outputs (2026-11-06, new Adot).');

s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'EEG-informed GK' });
s.addText('12 networks: EEG-informed GK matches the benchmark', { placeholder: 'title' });
sub(s, 'Network-mean HRF, AR-IRLS, −2 to 18 s, N = 6 (mean ± 95% CI across subjects), not normalized');
img(s, 'fig_net_eeginf.png', 0.6, 1.65, 9.35, 2720 / 1530, 'networks EEG-informed');
s.addText('vs benchmark', { x: 10.25, y: 1.8, w: 2.5, h: 0.4, fontSize: 14, bold: true, color: C.text2, margin: 0, isTextBox: true });
[['0.99', 'Median network r (12 networks, 0.97–1.00)'], ['0.97', 'Median per-parcel shape r'], ['0.93×', 'Median per-parcel RMS amplitude']].forEach(([v, l], i) => stat(s, 10.25, 2.35 + i * 1.45, 2.5, v, l, ORANGE, 30));
s.addNotes('Per-parcel r and RMS ratio: median over parcels per subject, then median over subjects. Computed from the saved onlyStim and onlyEEG-woodyN2P3ptp AR-IRLS betas (2026-11-06).');

s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'EEG-informed GK' });
s.addText('Conclusion 1: EEG-informed GK adds little beyond onset timing', { placeholder: 'title' });
sub(s, 'The ERP weights vary from trial to trial, but neither carry attention-state information nor change the HRF');
card(s, 0.6, 1.8, 3.85, 3.9, 1, ORANGE, 'Weights carry no state', 'p ≥ 0.27', [bl('N2–P3 ptp, P3 latency and 4-band Hilbert amplitude do not differ in vs out of the zone'), last('Trial-to-trial spread looks like noise, not attention')], 'conclusion card 1');
card(s, 4.73, 1.8, 3.85, 3.9, 2, ORANGE, 'Little design leverage', 'r ≈ 0.93', [bl('Weighted and unweighted columns are nearly collinear'), last('Mean weight = 1, so the average-trial response is the same by construction')], 'conclusion card 2');
card(s, 8.86, 1.8, 3.85, 3.9, 3, ORANGE, 'HRF unchanged', 'r = 0.99', [bl('Network-level shape identical to the benchmark'), bl('Per-parcel r = 0.97; amplitude 0.93×'), last('Same F-test significance at the example parcels')], 'conclusion card 3');
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 5.95, w: 12.1, h: 0.8, rectRadius: 0.06, fill: { color: 'FCE6DC' }, line: { color: 'FCE6DC' }, objectName: 'bottom line' });
s.addText([{ text: 'Bottom line: ', options: { bold: true } }, { text: 'the EEG-informed model’s information comes from the onset times; Cz ERP amplitude does not track the state that shapes the HRF.' }],
  { x: 0.85, y: 5.95, w: 11.6, h: 0.8, fontSize: 15, color: INK, margin: 0, valign: 'middle', isTextBox: true });
s.addNotes('Bottom line: the information in the EEG-informed model comes from the onset times. ERP amplitude as measured at Cz does not track the attentional state that modulates the HRF.');

// ---------------- Part 2: Cont-EEG
pres.addSection({ title: 'Cont-EEG' });
s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Cont-EEG' });
s.addText('Cont-EEG: mask Cz to mnt-correct trials, then convolve', { placeholder: 'title' });
sub(s, 'sub-723, run 2, 24 s; the delay kernel h(d) is fit per parcel and convolved with the masked EEG');
img(s, 'pipe_c1.png', 0.6, 1.7, 6.0, 1364 / 550, 'cont pipeline 1');
img(s, 'pipe_c2.png', 0.6, 4.3, 6.0, 1364 / 550, 'cont pipeline 2');
img(s, 'pipe_c3.png', 6.95, 1.7, 3.0, 704 / 550, 'cont pipeline 3');
s.addText([
  bl('Mask: keep Cz from onset to onset + duration (~0.8 s) of each mnt-correct trial; all city and mnt-incorrect time is zero'),
  last('Delay axis −2 to 18 s projected on 20 B-splines, the same span as the benchmark kernels'),
], { x: 10.15, y: 1.75, w: 2.6, h: 2.35, fontSize: 13, color: INK, margin: 0, paraSpaceAfter: 6, valign: 'top', isTextBox: true, objectName: 'mask bullets' });
img(s, 'pipe_c4.png', 6.95, 4.3, 5.8, 1364 / 550, 'cont pipeline 4');
s.addNotes('Same masking as run_model_cont_EEG_city_mnt.py (and the example trial of diag_contEEG_flat_hrf.py): Cz baseline-corrected per trial over -0.2 to 0 s, zero elsewhere; each B-spline column is the masked EEG convolved with that spline, low-passed at the fNIRS Nyquist and read at the fNIRS samples. The model predicts y = h * EEG, so the response to an average trial is h convolved with the average within-trial Cz.');

s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Cont-EEG' });
s.addText('Two parcels: the Cont-EEG response is nearly flat', { placeholder: 'title' });
sub(s, 'Predicted response to an average mnt-correct trial (kernel ∗ average within-trial Cz); bands = 95% CI');
img(s, 'fig_2p_cont.png', 0.95, 1.6, 11.4, 3200 / 1120, 'two parcels cont-EEG');
statRow(s, [['0.14×', 'Median peak amplitude relative to the benchmark', BLUE], ['r = 0.28', 'Median shape r, 12 subject-parcels (−0.82 to 0.87)', BLUE], ['9 of 12', 'Cont-EEG F-tests not significant (p > 0.05)', BLUE]]);
s.addNotes('Per-subject Cont-EEG curves come from the comparison fit (mnt-correct trials only, 0.02 Hz highpass, as the GK models), which also gives the CIs. The 12-network figure uses the saved run_model_cont_EEG_city_mnt fit (mnt-correct + mnt-incorrect, 0.01 Hz); at these parcels the two agree (median r about 0.93).');

s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Cont-EEG' });
s.addText('12 networks: Cont-EEG stays near zero', { placeholder: 'title' });
sub(s, 'Network-mean response, AR-IRLS, −2 to 18 s, N = 6 (mean ± 95% CI across subjects), not normalized');
img(s, 'fig_net_cont.png', 0.6, 1.65, 9.35, 2720 / 1530, 'networks cont-EEG');
s.addText('vs benchmark', { x: 10.25, y: 1.8, w: 2.5, h: 0.4, fontSize: 14, bold: true, color: C.text2, margin: 0, isTextBox: true });
[['0.20×', 'Median per-parcel RMS amplitude'], ['0.21', 'Median per-parcel shape r (subjects −0.16 to 0.51)'], ['0.30', 'Median network r (−0.86 to 0.85)']].forEach(([v, l], i) => stat(s, 10.25, 2.35 + i * 1.45, 2.5, v, l, BLUE, 30));
s.addNotes('Kernel betas from cont_EEG_cz_mnt_correct_incorrect_pre2post18_bl0.2-0_ar_irls_Hp0.01 (2026-11-06), convolved with each subject’s average within-trial Cz. Shape r varies in sign across networks and subjects, so there is no consistent HRF shape.');

s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Cont-EEG' });
s.addText('Conclusion 2: Cont-EEG finds no HRF, and why it cannot', { placeholder: 'title' });
sub(s, 'Diagnosis on sub-723 (diag_contEEG_flat_hrf.py): the basis is fine; the EEG input cancels itself');
img(s, 'diag.png', 0.6, 1.65, 7.3, 2040 / 1440, 'cont-EEG diagnosis');
const reasons = [
  ['Not the basis', 'The 20 B-splines reproduce the benchmark HRF almost exactly (R² = 0.999).'],
  ['Weights ≈ 0', 'Kernel weights: |t| max 1.4–2.5, median ≈ 0.5; F-test p = 0.8 at DorsAttnB.'],
  ['EEG sums to ≈ 0 per trial', 'Cz summed over a trial: mean/SD = 0.08, negative in 45% of trials, so the average-trial response ≈ mean(w)·h(d) is flat.'],
  ['No consistent shape', 'Amplitude 0.2× the benchmark; shape r changes sign across networks and subjects.']];
reasons.forEach(([h, b], i) => {
  const y = 1.8 + i * 1.25;
  badge(s, i + 1, 8.2, y, BLUE);
  s.addText(h, { x: 8.85, y: y - 0.02, w: 3.9, h: 0.38, fontSize: 16, bold: true, color: C.text2, margin: 0, isTextBox: true });
  s.addText(b, { x: 8.85, y: y + 0.36, w: 3.9, h: 0.85, fontSize: 13, color: INK, margin: 0, valign: 'top', isTextBox: true });
});
s.addNotes('After per-trial baseline correction the within-trial EEG is roughly zero-mean, so convolving a smooth kernel with it gives a response proportional to the trial sum w, which is near zero on average. To reproduce the benchmark, the kernel would need to be 30-50x larger than the fitted one.');

s = pres.addSlide({ masterName: 'DARK', sectionTitle: 'Cont-EEG' });
s.addText('Takeaways', { placeholder: 'title' });
s.addText([
  bl('The event-based mnt-correct HRF is the result to report; EEG designs do not improve on it'),
  bl('EEG-informed GK ≈ benchmark (r = 0.99): ERP amplitude, latency and band amplitude do not track attention state'),
  bl('Cont-EEG gives no HRF: zero-mean within-trial EEG cancels, leaving 0.2× amplitude and no consistent shape'),
  last('Next: EEG features that do not cancel and do vary with state (band-power envelopes, pre-stimulus alpha, frontal theta), with held-out or permutation tests'),
], { placeholder: 'body', fontSize: 18, paraSpaceAfter: 10 });
s.addNotes('Part 3 checks whether using the whole continuous EEG instead of the masked trials changes the Cont-EEG conclusion.');

// ---------------- Part 3: Cont-EEG entire run
pres.addSection({ title: 'Cont-EEG entire run' });
s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Cont-EEG entire run' });
s.addText('Additional model: Cont-EEG over the entire run is also flat', { placeholder: 'title' });
sub(s, 'No trial mask: all continuous Cz, delays 0–15 s on 15 B-splines; prediction = kernel ∗ Cz averaged around mnt-correct onsets');
img(s, 'fig_2p_contrun.png', 0.95, 1.6, 11.4, 3200 / 1120, 'two parcels cont-EEG run');
statRow(s, [['0.14×', 'Median peak amplitude relative to the benchmark', AQUA], ['r = 0.19', 'Median shape r, 12 subject-parcels (−0.57 to 0.56)', AQUA], ['15 vs 20', 'Spline basis size: 0–15 s causal kernel vs −2 to 18 s', GREY]]);
s.addNotes('Model cont_EEG_cz_3-stage_bspline-test_ar_irls_Hp (2026-11-06). The kernel covariance was not saved, so per-subject panels have no band; the mean panel shows the across-subject 95% CI. The event-locked Cz average covers -17 to 18 s around each mnt-correct onset, so it includes the ERPs of the surrounding city trials.');

s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Cont-EEG entire run' });
s.addText('12 networks: entire-run Cont-EEG does not recover the benchmark', { placeholder: 'title' });
sub(s, 'Network-mean response, AR-IRLS, −2 to 18 s, N = 6 (mean ± 95% CI across subjects), not normalized');
img(s, 'fig_net_contrun.png', 0.6, 1.65, 9.35, 2720 / 1530, 'networks cont-EEG run');
s.addText('vs benchmark', { x: 10.25, y: 1.8, w: 2.5, h: 0.4, fontSize: 14, bold: true, color: C.text2, margin: 0, isTextBox: true });
[['0.15×', 'Median per-parcel RMS amplitude'], ['0.19', 'Median per-parcel shape r (subjects −0.14 to 0.29)'], ['0.28', 'Median network r (−0.24 to 0.73)']].forEach(([v, l], i) => stat(s, 10.25, 2.35 + i * 1.45, 2.5, v, l, AQUA, 30));
s.addNotes('Same conclusion as Cont-EEG within trials: tiny amplitude and no consistent shape. Earlier train/test analysis of this design (fixAR-IRLS and AR-iRRR, train run 1) gave in-sample R² < 2% and held-out R² ≈ 0.');

// ---------------- Appendix
pres.addSection({ title: 'Appendix' });
s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Appendix' });
s.addText('Appendix: location of the two example parcels', { placeholder: 'title' });
sub(s, 'ICBM152 brain, superior and right views; green dot = Cz landmark');
img(s, 'parcel_views.png', 0.6, 1.75, 9.3, 1800 / 860, 'parcel locations');
[['SomMotA_25_LH', 'Left somatomotor, next to the midline under Cz', 'E36C55'], ['DorsAttnB_PostC_2_RH', 'Right postcentral, dorsal attention B', '8DB0FE']].forEach(([h, b, col], i) => {
  const y = 2.2 + i * 1.6;
  s.addShape(pres.shapes.OVAL, { x: 10.2, y: y + 0.05, w: 0.3, h: 0.3, fill: { color: col }, line: { color: col }, objectName: `swatch ${h}` });
  s.addText(h, { x: 10.6, y, w: 2.2, h: 0.4, fontSize: 14, bold: true, color: C.text2, margin: 0, isTextBox: true });
  s.addText(b, { x: 10.6, y: y + 0.42, w: 2.2, h: 0.8, fontSize: 13, color: INK, margin: 0, valign: 'top', isTextBox: true });
});
s.addNotes('Rendered with cedalion image_recon on the standard icbm152 head model, as in cmp_fixAR-IRLS_AR-iRRR_single-subj_all-runs.py.');

s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Appendix' });
s.addText('Appendix: EEG and fNIRS preprocessing', { placeholder: 'title' });
sub(s, 'Both streams end on the same fNIRS sample grid before any model is fit');
function flow(y, color, label, items) {
  s.addText(label, { x: 0.6, y, w: 1.5, h: 1.3, fontSize: 18, bold: true, color, margin: 0, valign: 'middle', isTextBox: true });
  const bw = 2.0, g = 0.17;
  items.forEach(([h, b], i) => {
    const x = 2.1 + i * (bw + g);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: bw, h: 1.3, rectRadius: 0.06, fill: { color: C.background2 }, line: { color: 'E0E4EC', width: 0.75 }, objectName: `${label} step ${i + 1}` });
    s.addText(h, { x: x + 0.12, y: y + 0.08, w: bw - 0.24, h: 0.4, fontSize: 13, bold: true, color, margin: 0, isTextBox: true });
    s.addText(b, { x: x + 0.12, y: y + 0.48, w: bw - 0.24, h: 0.78, fontSize: 11, color: INK, margin: 0, valign: 'top', isTextBox: true });
  });
}
flow(1.8, BLUE, 'EEG', [['Band-pass', '0.1–45 Hz'], ['Bad channels', 'Flat, abnormal variance or large amplitude removed'], ['Re-reference', 'Mastoids TP9h / TP10h'], ['ICA', 'Infomax; components matching hEOG / vEOG removed'], ['Trial features', 'Epoch −0.2 to 1.6 s, reject > 100 µV; Woody N2–P3, Hilbert bands']]);
flow(3.5, ORANGE, 'fNIRS', [['Prune channels', 'SNR > 5, 1–40 mm, amplitude range'], ['OD → image', 'Image recon with the updated Adot (vertex space)'], ['Parcels', 'Mean per parcel; keep sensitive parcels (410)'], ['Filter & window', '0.02 Hz highpass; first onset → last onset + 15 s'], ['Nuisance', 'OLS Legendre drift, then OLS GSR; runs concatenated']]);
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 2.1, y: 5.25, w: 10.65, h: 1.45, rectRadius: 0.06, fill: { color: 'FFFFFF' }, line: { color: '24324F', width: 1 }, objectName: 'alignment box' });
s.addText([
  { text: 'Alignment and design: ', options: { bold: true, color: C.text2 } },
  { text: 'fNIRS and EEG runs are matched by the first stimulus onset and first reaction times; EEG regressors are low-passed at the fNIRS Nyquist and read at the exact fNIRS sample times; every design (benchmark, EEG-informed GK, Cont-EEG) is then fit with Cedalion AR-IRLS.' }],
  { x: 2.3, y: 5.3, w: 10.3, h: 1.35, fontSize: 13, color: INK, margin: 0, valign: 'middle', isTextBox: true });
s.addNotes('EEG: eeg_preprocess_pipeline.py and extract_EEG_trial_features.py. fNIRS: cedalion_v26/STEP1_image_recon.py (ar_irls branch: no TDDR, no band-pass), then run_model_cont_EEG_fNIRS.py builds the shared Y_all. Cont-EEG models use a 0.01 Hz highpass.');

s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Appendix' });
s.addText('Appendix: P3 in-zone vs out-of-zone', { placeholder: 'title' });
sub(s, 'Grand-average Cz ERP, clean mnt-correct trials, 10 Hz low-pass, baseline −0.2–0 s, N = 17');
img(s, 'fig_app_erp_zone.png', 0.6, 1.7, 8.6, 1500 / 880, 'ERP zone waveform');
s.addText([
  bl('The two waveforms overlap through the N2 and P3 windows'),
  bl('Matches the single-trial result: no amplitude (p = 0.27) or latency (p = 0.36) difference'),
  last('In-zone = VTC below each subject’s median VTC'),
], { x: 9.6, y: 1.85, w: 3.15, h: 4.8, fontSize: 15, color: INK, margin: 0, paraSpaceAfter: 12, valign: 'top', isTextBox: true, objectName: 'ERP bullets' });
s.addNotes('Per-subject average per zone, then mean and 95% CI across subjects. Shaded: P3 search window 0.25-0.80 s.');

s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Appendix' });
s.addText('Appendix: Hilbert amplitude in four bands, in vs out of the zone', { placeholder: 'title' });
sub(s, 'Cz envelope around mnt-correct onsets, −0.2 to 1.6 s, no baseline correction, N = 17 (mean ± 95% CI)');
img(s, 'fig_app_hilbert_tc.png', 0.6, 1.75, 12.1, 2800 / 800, 'Hilbert time courses');
s.addText('In-zone and out-of-zone envelopes overlap in every band and at every latency; the within-trial sums tested on the main slide are flat for the same reason.', { x: 0.6, y: 5.6, w: 12.1, h: 0.9, fontSize: 15, color: INK, margin: 0, isTextBox: true, objectName: 'Hilbert note' });
s.addNotes('Same envelopes as the BandPowerImage figures in single_Subj_analysis_ERP_ERSP.py, shown as amplitude (uV) rather than power.');

s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Appendix' });
s.addText('Appendix: estimators change amplitude, not shape', { placeholder: 'title' });
sub(s, 'AR-IRLS vs fixAR-IRLS vs AR-iRRR, event GK −2 to 18 s, N = 20; fit before the Adot update');
img(s, 'models_agree.png', 0.6, 1.8, 8.4, 1800 / 675, 'estimator agreement');
stat(s, 9.5, 1.9, 3.2, 'r ≈ 0.86', 'AR-IRLS vs fixAR-IRLS shape (median)', GREY);
stat(s, 9.5, 3.4, 3.2, '≈ 0.9×', 'fixAR-IRLS amplitude vs AR-IRLS', GREY);
stat(s, 9.5, 4.9, 3.2, '≈ 0.12×', 'AR-iRRR amplitude (shrunk; r ≈ 0.47)', GREY);
s.addNotes('From summary_run_model_cont_EEG_city_mnt section 14 (hrf_corr_amplitude.png, 2026-10-21, previous Adot). Values read from the figure. Not rerun with the new Adot.');

(async () => {
  await pres.writeFile({ fileName: 'HRF_EEG_summary.pptx' });
  await applyTheme('HRF_EEG_summary.pptx', THEME);
  console.log('done');
})();
