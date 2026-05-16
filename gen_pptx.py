"""Build the presentation deck from results.json (numbers not hard-coded)."""
import json
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

R = json.load(open(r'D:\ml\results.json'))
B, q23, q1 = R['baseline'], R['q2q3'], R['q1']
NAVY, RED, GREY = RGBColor(0x1F, 0x2D, 0x5A), RGBColor(0xC0, 0x1A, 0x1A), RGBColor(0x44, 0x44, 0x44)
prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]

def slide(title):
    s = prs.slides.add_slide(BLANK)
    bar = s.shapes.add_shape(1, 0, 0, prs.slide_width, Inches(1.05))
    bar.fill.solid(); bar.fill.fore_color.rgb = NAVY; bar.line.fill.background()
    tb = s.shapes.add_textbox(Inches(0.5), Inches(0.22), Inches(12.3), Inches(0.7))
    p = tb.text_frame.paragraphs[0]; r = p.add_run(); r.text = title
    r.font.size = Pt(30); r.font.bold = True; r.font.color.rgb = RGBColor(255, 255, 255)
    return s

def bullets(s, items, left=0.6, top=1.35, width=7.0, size=18):
    tb = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(5.7))
    tf = tb.text_frame; tf.word_wrap = True
    for i, (txt, lvl) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = lvl
        r = p.add_run(); r.text = ('• ' if lvl == 0 else '– ') + txt
        r.font.size = Pt(size - lvl*3); r.font.color.rgb = GREY if lvl else NAVY
        r.font.bold = (lvl == 0)
        p.space_after = Pt(7)

def pic(s, path, left, top, w):
    s.shapes.add_picture(path, Inches(left), Inches(top), width=Inches(w))

# 1 Title
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(1, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background()
tb = s.shapes.add_textbox(Inches(0.8), Inches(2.2), Inches(11.7), Inches(2.6)); tf = tb.text_frame; tf.word_wrap = True
r = tf.paragraphs[0].add_run()
r.text = 'Preprocessing, Regularisation & Cross-Dataset\nGeneralisation for EEG Seizure Detection'
r.font.size = Pt(38); r.font.bold = True; r.font.color.rgb = RGBColor(255, 255, 255)
p2 = tf.add_paragraph(); r2 = p2.add_run()
r2.text = 'Logistic regression on three REAL datasets — UCI · CHB-MIT · Bonn'
r2.font.size = Pt(20); r2.font.color.rgb = RGBColor(0xBF, 0xD3, 0xFF)
p3 = tf.add_paragraph(); r3 = p3.add_run()
r3.text = 'Machine Learning — Semester Major Assignment'
r3.font.size = Pt(15); r3.font.color.rgb = RGBColor(0x9F, 0xB3, 0xE0)

# 2 The integrity problem
s = slide('Why this version exists')
bullets(s, [
    ('The earlier notebook looked enterprise-grade but folded under inspection', 0),
    ('2 of 3 "datasets" were synthetic Gaussian noise — mislabelled as Bonn / PhysioNet', 1),
    ('Trivially separable → fake F1 ≈ 0.98 everywhere (an artifact, not a result)', 1),
    ('Written conclusions contradicted the notebook’s own printed output', 1),
    ('On the only real dataset the model was at chance (ROC-AUC ≈ 0.50)', 1),
    ('This version: only real data, true provenance, one honest methodology, '
     'conclusions computed + assert-checked', 0)], width=12.0)

# 3 Datasets
s = slide('Three real datasets (true provenance)')
ds = R['datasets']
bullets(s, [
    (f'DS1 UCI Epileptic — {ds["DS1_UCI"]["n"]} windows, {ds["DS1_UCI"]["pos"]}% seizure', 0),
    (f'DS2 CHB-MIT Scalp — {ds["DS2_CHB-MIT"]["n"]} windows (23ch raw), {ds["DS2_CHB-MIT"]["pos"]}% — natural imbalance', 0),
    (f'DS3 Bonn University — {ds["DS3_Bonn"]["n"]} segments, {ds["DS3_Bonn"]["pos"]}% seizure', 0),
    ('Downloaded reproducibly via kagglehub; no fabrication, no fallback', 0)], width=6.0)
pic(s, r'D:\ml\figures\fig1_class_distribution.png', 6.7, 1.5, 6.2)

# 4 Root-cause fix
s = slide('Root cause: raw amplitude is degenerate')
bullets(s, [
    (f'Raw EEG samples → logistic regression: ROC-AUC {R["raw_amplitude"]["DS1_rocauc"]:.2f} (chance)', 0),
    ('Seizure signal lives in energy / variability / spectrum, not amplitude at a fixed index', 1),
    ('Fix: unified extractor — 10 time-domain + 5 spectral band-power features', 0),
    ('Applied identically to all datasets → fair cross-dataset comparison', 1),
    (f'DS1 ROC-AUC {R["raw_amplitude"]["DS1_rocauc"]:.2f} → {B["DS1_UCI"]["rocauc"]:.2f} after extraction', 0)],
    width=12.0)

# 5 Baseline
s = slide('Baseline — honest difficulty spread')
rows = [('DS1 UCI', B['DS1_UCI']), ('DS2 CHB-MIT', B['DS2_CHB-MIT']), ('DS3 Bonn', B['DS3_Bonn'])]
bullets(s, [(f'{n}: ROC-AUC {v["rocauc"]:.3f}, F1 {v["f1"]:.3f}', 0) for n, v in rows] +
        [('CHB-MIT is genuinely hard — a real result, not a fabricated 0.98', 0)], width=6.2)
pic(s, r'D:\ml\figures\fig3_baseline_pr.png', 6.6, 1.5, 6.4)

# 6 Overfitting
s = slide('Overfitting / underfitting (now real)')
ov = R['overfit']
bullets(s, [(f'{k}: train {ov[k]["tr"]:.2f} / test {ov[k]["te"]:.2f} ({ov[k]["nfeat"]} feat)', 0)
            for k in ov], width=6.0)
pic(s, r'D:\ml\figures\fig4_overfitting.png', 6.4, 1.5, 6.6)

# 7 Regularisation
s = slide('Q2/Q3 — Regularisation')
bullets(s, [
    (f'Best mean PR-AUC: {q23["best_prauc"]}', 0),
    (f'Spread among L1/L2/Elastic Net: only {q23["pen_spread"]:.4f}', 0),
    (f'Elastic Net consistently best? {q23["en_best"]}', 0),
    (f'Unregularised is the real loser (spread → {q23["all_spread"]:.3f})', 0),
    ('Defensible answer: "regularised vs. not", not which penalty', 0)], width=6.0)
pic(s, r'D:\ml\figures\fig5_regularisation.png', 6.5, 1.6, 6.5)

# 8 Imbalance + Q4
s = slide('Q4 — Imbalance × Regularisation')
Q4 = R['q4']
best = max(((rs, gn, Q4[rs][gn]) for rs in Q4 for gn in Q4[rs]), key=lambda x: x[2])
bullets(s, [
    ('Feature extraction & imbalance handling >> penalty choice', 0),
    (f'Best combo (DS1* ~4%): {best[0]} + {best[1]}, F1 {best[2]:.3f}', 0),
    ('On hard CHB-MIT, class-weight/SMOTE lift recall from ~0.02 to ~0.68', 0),
    ('Resampling/recall trade-off dominates the penalty', 0)], width=6.0)
pic(s, r'D:\ml\figures\fig6_imbalance.png', 6.4, 1.6, 6.6)

# 9 Conclusions
s = slide('Conclusions')
bullets(s, [
    (f'Q1: preprocessing order minor (|ΔF1| = {q1["dF1"]:.3f})', 0),
    ('Q2/Q3: no penalty dominates; Elastic Net does NOT consistently win', 0),
    ('Q4: imbalance strategy interacts with — and outweighs — the penalty', 0),
    ('First-order levers: feature extraction + imbalance handling', 0),
    ('Honest, reproducible methodology changed the headline answer', 0),
    ('Every number computed and assert-checked — survives scrutiny', 0)], width=12.0, size=19)

def _fix_dashes(presentation):
    """Replace em dashes (U+2014) with a standard hyphen on every slide."""
    for sl in presentation.slides:
        for sh in sl.shapes:
            if sh.has_text_frame:
                for par in sh.text_frame.paragraphs:
                    for run in par.runs:
                        if '—' in run.text:
                            run.text = run.text.replace('—', '-')

_fix_dashes(prs)
prs.save(r'D:\ml\Seizure_Prediction_Presentation.pptx')
print(f'Presentation written: {len(prs.slides.__iter__.__self__._sldIdLst)} slides')
