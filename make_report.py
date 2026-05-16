"""Generate the IEEE-format Word report from the notebook's real results.

Numbers are recomputed here with the SAME seeded code path as the notebook
(local kaggle cache, ~30 s) so the report cannot drift from the notebook.
"""
import json, os, numpy as np, pandas as pd
import kagglehub
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, precision_score, recall_score,
    average_precision_score, roc_auc_score, accuracy_score)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

RNG = 42
np.random.seed(RNG)

def find(root, name):
    for r, _, fs in os.walk(root):
        if name in fs:
            return os.path.join(r, name)
    raise FileNotFoundError(name)

d1 = find(kagglehub.dataset_download('harunshimanto/epileptic-seizure-recognition'),
          'Epileptic Seizure Recognition.csv')
d2 = find(kagglehub.dataset_download('adibadea/chbmitseizuredataset'), 'eeg-predictive_val.npz')
d3 = find(kagglehub.dataset_download('peimandaii/epilepsy-diagnosis-dataset'), 'EEG_Signal.csv')

_a = pd.read_csv(d1); y1 = (_a['y'] == 1).astype(int).values
S1 = _a.drop(columns=[_a.columns[0], 'y']).select_dtypes('number').values.astype(float)
_z = np.load(d2, allow_pickle=True); RAW2 = _z['val_signals'].astype(float); y2 = _z['val_labels'].astype(int)
_c = pd.read_csv(d3); _g = _c.groupby('id person', sort=False); _sl = int(_g.size().min())
_ids = list(_g.groups)
S3 = np.stack([_g.get_group(i)['Signal'].values[:_sl] for i in _ids]).astype(float)
y3 = (np.array([_g.get_group(i)['Labels'].iloc[0] for i in _ids]) == 'E').astype(int)
FS = dict(DS1_UCI=178.0, DS2_CHB=256.0, DS3_Bonn=173.61)

def sigfeat(S, fs):
    S = np.asarray(S, float); mu = S.mean(1, keepdims=True); sd = S.std(1) + 1e-9
    o = {'var': S.var(1), 'std': S.std(1), 'mav': np.abs(S).mean(1),
         'rms': np.sqrt((S**2).mean(1)), 'll': np.abs(np.diff(S, axis=1)).sum(1),
         'energy': (S**2).sum(1), 'ptp': np.ptp(S, axis=1),
         'zc': (np.diff(np.sign(S - mu), axis=1) != 0).sum(1),
         'skew': ((S - mu)**3).mean(1) / sd**3, 'kurt': ((S - mu)**4).mean(1) / sd**4}
    P = np.abs(np.fft.rfft(S, axis=1))**2; fr = np.fft.rfftfreq(S.shape[1], 1.0/fs); tot = P.sum(1)+1e-9
    for lo, hi, nm in [(0.5,4,'d'),(4,8,'t'),(8,13,'a'),(13,30,'b'),(30,min(fs/2-1,80),'g')]:
        o['bp'+nm] = P[:, (fr >= lo) & (fr < hi)].sum(1)/tot
    return pd.DataFrame(o).values

def mcfeat(raw, fs):
    pc = np.stack([sigfeat(raw[:, c, :], fs) for c in range(raw.shape[1])], 1)
    return np.hstack([pc.mean(1), pc.std(1)])

X1, X2, X3 = sigfeat(S1, FS['DS1_UCI']), mcfeat(RAW2, FS['DS2_CHB']), sigfeat(S3, FS['DS3_Bonn'])
DS = {'DS1_UCI': (X1, y1), 'DS2_CHB-MIT': (X2, y2), 'DS3_Bonn': (X3, y3)}

def ev(m, a, b, c, d):
    m.fit(a, c); p = m.predict(b); pr = m.predict_proba(b)[:, 1]
    return dict(acc=accuracy_score(d, p), f1=f1_score(d, p, zero_division=0),
                prec=precision_score(d, p, zero_division=0), rec=recall_score(d, p, zero_division=0),
                prauc=average_precision_score(d, pr), rocauc=roc_auc_score(d, pr))

R = {'datasets': {}, 'baseline': {}, 'raw_amplitude': {}, 'reg': {}, 'imb': {}, 'q4': {}}
for nm, (X, y) in DS.items():
    R['datasets'][nm] = dict(n=int(X.shape[0]), feats=int(X.shape[1]), pos=float(round(100*y.mean(), 1)))

# raw-amplitude degeneracy (DS1)
Xtr, Xte, ytr, yte = train_test_split(S1, y1, test_size=.2, stratify=y1, random_state=RNG)
sc = StandardScaler().fit(Xtr)
R['raw_amplitude']['DS1_rocauc'] = round(ev(LogisticRegression(C=1, max_iter=2000),
    sc.transform(Xtr), sc.transform(Xte), ytr, yte)['rocauc'], 4)

splits = {}
for nm, (X, y) in DS.items():
    a, b, c, d = train_test_split(X, y, test_size=.2, stratify=y, random_state=RNG)
    s = StandardScaler().fit(a); splits[nm] = (s.transform(a), s.transform(b), c, d)
    R['baseline'][nm] = {k: round(v, 4) for k, v in
        ev(LogisticRegression(C=1, max_iter=2000, random_state=RNG), *splits[nm]).items()}

regs = [('L2 (Ridge)', dict(penalty='l2', solver='lbfgs', max_iter=3000)),
        ('L1 (Lasso)', dict(penalty='l1', solver='saga', max_iter=5000)),
        ('Elastic Net', dict(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=5000)),
        ('No penalty', dict(penalty=None, solver='lbfgs', max_iter=3000))]
for nm in DS:
    a, b, c, d = splits[nm]; R['reg'][nm] = {}
    for rn, kw in regs:
        m = LogisticRegression(C=1.0, random_state=RNG, **kw)
        e = ev(m, a, b, c, d); sp = float(round(100*(np.abs(m.coef_[0]) < 1e-6).mean(), 1))
        R['reg'][nm][rn] = dict(f1=round(e['f1'], 4), prauc=round(e['prauc'], 4),
                                rocauc=round(e['rocauc'], 4), sparsity=sp)

# overfitting demo (DS1)
a, b, c, d = splits['DS1_UCI']; R['overfit'] = {}
for nm, (C, dg) in {'Severe underfit (C=1e-4)': (1e-4, 1), 'Mild underfit (C=1e-2)': (1e-2, 1),
                    'Good fit (C=1)': (1.0, 1), 'Overfit (deg=3,C=1e5)': (1e5, 3)}.items():
    pf = PolynomialFeatures(dg, include_bias=False); A = pf.fit_transform(a); B = pf.transform(b)
    m = LogisticRegression(C=C, max_iter=3000, random_state=RNG).fit(A, c)
    R['overfit'][nm] = dict(nfeat=int(A.shape[1]),
        tr=round(f1_score(c, m.predict(A), zero_division=0), 4),
        te=round(f1_score(d, m.predict(B), zero_division=0), 4))

# imbalance + Q4 on controlled DS1*
Xtr0, Xte0, ytr0, yte0 = train_test_split(X1, y1, test_size=.2, stratify=y1, random_state=RNG)
pos = np.where(ytr0 == 1)[0]; neg = np.where(ytr0 == 0)[0]
kp = np.random.RandomState(RNG).choice(pos, int(0.05*len(pos)), replace=False)
idx = np.concatenate([neg, kp]); np.random.shuffle(idx)
s = StandardScaler().fit(Xtr0[idx])
imb_sets = {'DS1 (20% nat.)': splits['DS1_UCI'], 'DS2 (22% nat.)': splits['DS2_CHB-MIT'],
            'DS1* (~4% controlled)': (s.transform(Xtr0[idx]), s.transform(Xte0), ytr0[idx], yte0)}
for dn, (a, b, c, d) in imb_sets.items():
    R['imb'][dn] = {}
    for st in ['None', 'Class weight', 'SMOTE', 'Undersample']:
        Xr, yr = a, c
        if st == 'SMOTE': Xr, yr = SMOTE(random_state=RNG, k_neighbors=min(5, int(c.sum())-1)).fit_resample(a, c)
        elif st == 'Undersample': Xr, yr = RandomUnderSampler(random_state=RNG).fit_resample(a, c)
        mdl = LogisticRegression(C=1, max_iter=2000, random_state=RNG,
              class_weight='balanced' if st == 'Class weight' else None).fit(Xr, yr)
        p = mdl.predict(b); pr = mdl.predict_proba(b)[:, 1]
        R['imb'][dn][st] = dict(prec=round(precision_score(d, p, zero_division=0), 4),
            rec=round(recall_score(d, p, zero_division=0), 4),
            f1=round(f1_score(d, p, zero_division=0), 4),
            prauc=round(average_precision_score(d, pr), 4))

a, b, c, d = imb_sets['DS1* (~4% controlled)']
for rs, fn in {'None': lambda X, y: (X, y),
               'SMOTE': lambda X, y: SMOTE(random_state=RNG, k_neighbors=min(5, int(y.sum())-1)).fit_resample(X, y),
               'Undersample': lambda X, y: RandomUnderSampler(random_state=RNG).fit_resample(X, y)}.items():
    Xr, yr = fn(a, c); R['q4'][rs] = {}
    for gn, kw in {'L2': dict(penalty='l2', solver='lbfgs'), 'L1': dict(penalty='l1', solver='saga'),
                   'Elastic Net': dict(penalty='elasticnet', solver='saga', l1_ratio=0.5)}.items():
        m = LogisticRegression(C=1, max_iter=5000, random_state=RNG, **kw).fit(Xr, yr)
        R['q4'][rs][gn] = round(f1_score(d, m.predict(b), zero_division=0), 4)

# derived Q1/Q2/Q3
Xtr, Xte, ytr, yte = train_test_split(X1, y1, test_size=.2, stratify=y1, random_state=RNG)
kA = min(12, X1.shape[1])
pA = Pipeline([('s', StandardScaler()), ('v', VarianceThreshold(1e-9)), ('k', SelectKBest(f_classif, k=kA))]).fit(Xtr, ytr)
pB = Pipeline([('v', VarianceThreshold(1e-9)), ('k', SelectKBest(f_classif, k=kA)),
               ('r', RobustScaler()), ('p', PCA(min(8, kA), random_state=RNG))]).fit(Xtr, ytr)
f1A = ev(LogisticRegression(C=1, max_iter=2000, random_state=RNG), pA.transform(Xtr), pA.transform(Xte), ytr, yte)['f1']
f1B = ev(LogisticRegression(C=1, max_iter=2000, random_state=RNG), pB.transform(Xtr), pB.transform(Xte), ytr, yte)['f1']
R['q1'] = dict(f1A=round(f1A, 4), f1B=round(f1B, 4), dF1=round(abs(f1A-f1B), 4))

prauc = pd.DataFrame({d: {r: R['reg'][d][r]['prauc'] for r in R['reg'][d]} for d in R['reg']})
prauc['Mean'] = prauc.mean(1)
pen = ['L2 (Ridge)', 'L1 (Lasso)', 'Elastic Net']
R['q2q3'] = dict(best_prauc=prauc['Mean'].idxmax(),
                 pen_spread=round(prauc.loc[pen, 'Mean'].max()-prauc.loc[pen, 'Mean'].min(), 4),
                 all_spread=round(prauc['Mean'].max()-prauc['Mean'].min(), 4),
                 en_best=bool(np.isclose(prauc.loc['Elastic Net', 'Mean'], prauc['Mean'].max())))

json.dump(R, open(r'D:\ml\results.json', 'w'), indent=1)
print('results.json written. Keys:', list(R.keys()))
print('DS1 raw-amplitude ROC-AUC =', R['raw_amplitude']['DS1_rocauc'],
      '| baseline DS1 =', R['baseline']['DS1_UCI']['rocauc'])
