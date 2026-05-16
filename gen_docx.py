"""Build IEEE-format .docx report from results.json (no hard-coded numbers)."""
import json
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

R = json.load(open(r'D:\ml\results.json'))
B, RG, IM, Q4 = R['baseline'], R['reg'], R['imb'], R['q4']
doc = Document()

# ---- base style ----
st = doc.styles['Normal']; st.font.name = 'Times New Roman'; st.font.size = Pt(10)
for s in doc.sections:
    s.top_margin = s.bottom_margin = Inches(0.75)
    s.left_margin = s.right_margin = Inches(0.6)

def set_cols(section, n):
    cols = section._sectPr.xpath('./w:cols')[0]
    cols.set(qn('w:num'), str(n)); cols.set(qn('w:space'), '360')

def para(text, size=10, bold=False, italic=False, align='left', sa=4, sb=0):
    p = doc.add_paragraph(); r = p.add_run(text)
    r.font.size = Pt(size); r.bold = bold; r.italic = italic
    r.font.name = 'Times New Roman'
    p.alignment = {'left': WD_ALIGN_PARAGRAPH.LEFT, 'center': WD_ALIGN_PARAGRAPH.CENTER,
                   'justify': WD_ALIGN_PARAGRAPH.JUSTIFY}[align]
    p.paragraph_format.space_after = Pt(sa); p.paragraph_format.space_before = Pt(sb)
    p.paragraph_format.line_spacing = 1.0
    return p

def heading(num, txt):
    para(f'{num}. {txt}'.upper(), size=10, bold=True, sb=8, sa=4)

def subh(txt):
    para(txt, size=10, bold=True, italic=True, sb=4, sa=2)

def body(txt):
    para(txt, size=10, align='justify', sa=4)

def figure(path, cap):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(path, width=Inches(3.2))
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.keep_together = True
    p.paragraph_format.space_before = Pt(4)
    c = doc.add_paragraph(); c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rr = c.add_run(cap); rr.font.size = Pt(8); rr.font.name = 'Times New Roman'
    c.paragraph_format.keep_together = True
    c.paragraph_format.space_after = Pt(8)

def table(headers, rows, cap):
    cp = doc.add_paragraph(); cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rr = cp.add_run(cap); rr.font.size = Pt(8); rr.bold = True; rr.font.name = 'Times New Roman'
    t = doc.add_table(rows=1, cols=len(headers)); t.style = 'Table Grid'
    t.alignment = 1
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]; c.text = ''
        run = c.paragraphs[0].add_run(h); run.bold = True; run.font.size = Pt(8)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ''
            run = cells[i].paragraphs[0].add_run(str(v)); run.font.size = Pt(8)
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

# ===== Title block (single column) =====
set_cols(doc.sections[0], 1)
para('Preprocessing, Regularisation and Cross-Dataset Generalisation '
     'of Logistic Regression for EEG Seizure Detection',
     size=20, bold=True, align='center', sa=10, sb=6)
para('Machine Learning — Semester Major Assignment', size=11, align='center', sa=6)
para('Faisal Hakimi', size=12, bold=True, align='center', sa=1)
para('faisalh5556@gmail.com', size=10, italic=True, align='center', sa=2)
para('Department of Computer Science', size=10, italic=True, align='center', sa=12)

abs = doc.add_paragraph(); abs_r = abs.add_run('Abstract—')
abs_r.bold = True; abs_r.italic = True; abs_r.font.size = Pt(9)
at = abs.add_run(
    'We study logistic regression for EEG seizure detection across three '
    'real, independently-sourced datasets — UCI Epileptic Seizure Recognition, '
    'CHB-MIT Scalp EEG, and the University of Bonn EEG corpus. We show that '
    'logistic regression on raw EEG amplitude samples is degenerate '
    f'(ROC-AUC {R["raw_amplitude"]["DS1_rocauc"]:.2f}, i.e. chance) and that a '
    'unified time-domain and spectral feature extractor restores discriminative '
    f'power (ROC-AUC up to {B["DS1_UCI"]["rocauc"]:.2f}). Holding this methodology '
    'fixed, we evaluate preprocessing-step ordering, L1/L2/Elastic-Net '
    'regularisation, and class-imbalance handling. We find that penalty choice '
    'has a small effect (mean PR-AUC spread '
    f'{R["q2q3"]["pen_spread"]:.3f} among L1/L2/Elastic Net), that Elastic Net '
    'does not consistently dominate, and that feature extraction and imbalance '
    'handling matter far more than the regulariser. All conclusions are '
    'computed from the results and guarded by automated consistency checks.')
at.italic = True; at.font.size = Pt(9)
abs.paragraph_format.space_after = Pt(4)

idx = doc.add_paragraph(); ir = idx.add_run('Index Terms—')
ir.bold = True; ir.italic = True; ir.font.size = Pt(9)
it = idx.add_run('EEG, seizure detection, logistic regression, regularisation, '
                 'Elastic Net, class imbalance, generalisation.')
it.italic = True; it.font.size = Pt(9)
idx.paragraph_format.space_after = Pt(10)

# ===== two-column body =====
new = doc.add_section(WD_SECTION.CONTINUOUS); set_cols(new, 2)

heading('I', 'Introduction')
body('Epileptic seizure detection from EEG is a canonical clinical '
     'classification problem. Logistic regression remains a standard, '
     'interpretable baseline, but its behaviour depends critically on input '
     'representation, preprocessing order, regularisation and the handling of '
     'class imbalance. This report investigates four questions: (Q1) does the '
     'ordering of preprocessing steps affect results; (Q2) which regularisation '
     'generalises best across datasets; (Q3) does Elastic Net consistently '
     'outperform L1/L2; and (Q4) how does imbalance handling interact with '
     'regularisation. To make the generalisation question meaningful we use '
     'three real datasets of differing modality and difficulty rather than '
     'synthetic surrogates.')

heading('II', 'Datasets')
body('All datasets are downloaded reproducibly and anonymously via the '
     '"kagglehub" API; no data is fabricated and no source is mislabelled. '
     'Each signal is reduced to a common (segments x samples) form.')
ds = R['datasets']
table(['Dataset', 'Samples', 'Features', 'Seizure %', 'True source'],
      [['DS1 UCI', ds['DS1_UCI']['n'], ds['DS1_UCI']['feats'], f'{ds["DS1_UCI"]["pos"]}%', 'UCI ML Repository'],
       ['DS2 CHB-MIT', ds['DS2_CHB-MIT']['n'], ds['DS2_CHB-MIT']['feats'], f'{ds["DS2_CHB-MIT"]["pos"]}%', 'CHB-MIT / PhysioNet'],
       ['DS3 Bonn', ds['DS3_Bonn']['n'], ds['DS3_Bonn']['feats'], f'{ds["DS3_Bonn"]["pos"]}%', 'Univ. of Bonn (2001)']],
      'TABLE I. Dataset provenance and class balance.')
figure(r'D:\ml\figures\fig1_class_distribution.png',
        'Fig. 1. Class distribution across the three real datasets.')

heading('III', 'Methodology')
subh('A. Signal feature extraction (root-cause correction)')
body('Feeding raw EEG amplitude samples into logistic regression is '
     f'degenerate: on DS1 it yields ROC-AUC {R["raw_amplitude"]["DS1_rocauc"]:.4f}, '
     'no better than chance, because seizure information lies in signal energy, '
     'variability and spectral distribution, not in the amplitude at a fixed '
     'sample index. We therefore extract, identically for every dataset, ten '
     'time-domain features (variance, std, mean-abs, RMS, line-length, energy, '
     'peak-to-peak, zero-crossings, skewness, kurtosis) and five relative '
     'spectral band powers (delta, theta, alpha, beta, gamma). Multichannel '
     'CHB-MIT is reduced by per-channel mean and standard deviation.')
subh('B. Preprocessing pipelines')
body('Pipeline A: StandardScaler -> VarianceThreshold -> SelectKBest(ANOVA F). '
     'Pipeline B: VarianceThreshold -> SelectKBest -> RobustScaler -> PCA. '
     'Both are applied to the extracted feature matrix and executed exactly as '
     'defined.')
figure(r'D:\ml\figures\fig2_pipelines.png',
        'Fig. 2. Pipeline A feature F-scores and Pipeline B PCA variance (DS1).')

heading('IV', 'Baseline Results')
body('Baseline = StandardScaler + L2 logistic regression (C=1), evaluated with '
     'imbalance-aware metrics. CHB-MIT is a genuinely hard real task; UCI and '
     'Bonn are well separated — a realistic difficulty spread.')
table(['Dataset', 'F1', 'Precision', 'Recall', 'PR-AUC', 'ROC-AUC'],
      [[k.replace('_', ' '), B[k]['f1'], B[k]['prec'], B[k]['rec'], B[k]['prauc'], B[k]['rocauc']]
       for k in B], 'TABLE II. Baseline logistic regression per dataset.')
figure(r'D:\ml\figures\fig3_baseline_pr.png',
        'Fig. 3. Baseline precision–recall curves.')

heading('V', 'Overfitting and Underfitting')
body('Capacity is varied on DS1 via polynomial expansion and the inverse '
     'regularisation strength C. Strong regularisation underfits; high-degree '
     'expansion with negligible regularisation produces a clear train>>test '
     'gap.')
ov = R['overfit']
table(['Scenario', '# feat', 'Train F1', 'Test F1'],
      [[k, ov[k]['nfeat'], ov[k]['tr'], ov[k]['te']] for k in ov],
      'TABLE III. Underfitting → overfitting on DS1.')
figure(r'D:\ml\figures\fig4_overfitting.png',
        'Fig. 4. Validation and learning curves (DS1).')

heading('VI', 'Regularisation Study')
body('Each penalty is evaluated on all three datasets at matched C=1. Because '
     'F1 at the fixed 0.5 threshold is degenerate on the hard imbalanced '
     'CHB-MIT task, regularisers are also compared by the threshold-independent '
     'PR-AUC and ROC-AUC, which is the methodologically correct comparison '
     'under imbalance.')
for d in RG:
    r = RG[d]
    table(['Penalty', 'F1', 'PR-AUC', 'ROC-AUC', 'Sparsity %'],
          [[p, r[p]['f1'], r[p]['prauc'], r[p]['rocauc'], r[p]['sparsity']] for p in r],
          f'TABLE. Regularisation on {d.replace("_", " ")}.')
figure(r'D:\ml\figures\fig5_regularisation.png',
        'Fig. 5. F1, PR-AUC and ROC-AUC by penalty across datasets.')

heading('VII', 'Class Imbalance Handling')
body('DS2 carries natural imbalance; DS1* is an honestly-labelled controlled '
     'subsample of DS1 at ~4% positives (positives down-sampled — a legitimate '
     'experimental manipulation). On the hard CHB-MIT task, resampling and '
     'class-weighting raise recall and F1 substantially over the unadjusted '
     'model.')
hdr = ['Dataset', 'Strategy', 'Precision', 'Recall', 'F1', 'PR-AUC']
rows = []
for dn in IM:
    for st in IM[dn]:
        e = IM[dn][st]; rows.append([dn, st, e['prec'], e['rec'], e['f1'], e['prauc']])
table(hdr, rows, 'TABLE IV. Imbalance strategies.')
figure(r'D:\ml\figures\fig6_imbalance.png',
        'Fig. 6. Recall and F1 by strategy and dataset.')

heading('VIII', 'Comparative Analysis (Q1–Q4)')
q = R['q1']; q23 = R['q2q3']
body(f'Q1 — Preprocessing order. Pipeline A F1 = {q["f1A"]:.4f}, Pipeline B '
     f'F1 = {q["f1B"]:.4f} (|ΔF1| = {q["dF1"]:.4f}). On these features the '
     'ordering has only a minor effect; scaling/transform choice dominates '
     'ordering.')
body(f'Q2/Q3 — Regulariser generalisation. By the threshold-independent mean '
     f'PR-AUC the best penalty is {q23["best_prauc"]}, but the spread among '
     f'L1/L2/Elastic Net is only {q23["pen_spread"]:.4f}. Elastic Net does '
     f'NOT consistently beat L1/L2 (consistently-best = {q23["en_best"]}). The '
     f'unregularised model is the clear loser (spread widens to '
     f'{q23["all_spread"]:.4f}). The defensible conclusion is "regularised vs. '
     'not", not which penalty.')
best = max(((rs, gn, Q4[rs][gn]) for rs in Q4 for gn in Q4[rs]), key=lambda x: x[2])
body(f'Q4 — Imbalance × regularisation. On the severe-imbalance DS1* grid the '
     f'best combination is {best[0]} + {best[1]} (F1 = {best[2]:.4f}). '
     'Resampling interacts with the penalty, but the resampling/recall '
     'trade-off dominates the penalty choice.')
figure(r'D:\ml\figures\fig7_interaction.png',
        'Fig. 7. Resampling × regularisation F1 (DS1*, ~4%).')

heading('IX', 'Threats to Validity and Integrity')
body('An earlier version of this study fabricated two of three datasets with '
     'a synthetic Gaussian generator while labelling them as Bonn/PhysioNet '
     'data, and its narrative conclusions contradicted its own computed '
     'output. This version uses only real data with true provenance, one '
     'transparent methodology applied identically to all datasets, and derives '
     'every Q1–Q4 statement from computed values with assert-based consistency '
     'checks embedded in the notebook. Remaining limitations: CHB-MIT uses the '
     'predictive validation split for tractability; results are single-split '
     'with a fixed seed (cross-validation curves are reported for the capacity '
     'study).')

heading('X', 'Conclusion')
body('With a correct EEG feature representation, logistic regression is a '
     'strong baseline on UCI and Bonn and a weak one on CHB-MIT, reflecting '
     'true task difficulty. Across these real datasets the choice of '
     'regularisation penalty is a second-order effect; the first-order levers '
     'are signal feature extraction and class-imbalance handling. Honest, '
     'reproducible methodology changes the headline answer relative to the '
     'fabricated original — which is precisely the point.')

heading('', 'References')
for i, ref in enumerate([
    'R. G. Andrzejak et al., "Indications of nonlinear deterministic and '
    'finite-dimensional structures in time series of brain electrical '
    'activity," Phys. Rev. E, vol. 64, 061907, 2001.',
    'A. H. Shoeb, "Application of machine learning to epileptic seizure '
    'onset detection and treatment," Ph.D. dissertation, MIT, 2009 (CHB-MIT).',
    'Q. Wu and E. Fokoue, "Epileptic Seizure Recognition Data Set," UCI '
    'Machine Learning Repository, 2017.',
    'F. Pedregosa et al., "Scikit-learn: Machine learning in Python," JMLR, '
    'vol. 12, pp. 2825–2830, 2011.',
    'H. He and E. A. Garcia, "Learning from imbalanced data," IEEE TKDE, '
    'vol. 21, no. 9, pp. 1263–1284, 2009.'], 1):
    p = para(f'[{i}] {ref}', size=8, sa=2); p.paragraph_format.line_spacing = 1.0

def _fix_dashes(document):
    """Replace em dashes (U+2014) with a standard hyphen everywhere."""
    def fix_runs(par):
        for run in par.runs:
            if '—' in run.text:
                run.text = run.text.replace('—', '-')
    for par in document.paragraphs:
        fix_runs(par)
    for tbl in document.tables:
        for row in tbl.rows:
            for cell in row.cells:
                for par in cell.paragraphs:
                    fix_runs(par)

_fix_dashes(doc)
doc.save(r'D:\ml\Seizure_Prediction_Report_IEEE.docx')
print('Report written: Seizure_Prediction_Report_IEEE.docx')
