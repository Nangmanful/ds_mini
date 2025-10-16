# predict_with_preproc.py
# 사용법: python predict_with_preproc.py --train train.csv --input predict_input.csv --model toxicity_model.pkl

import argparse, json, re, joblib
import numpy as np, pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import chi2

def detect_smiles_col(df):
    for c in df.columns:
        if c.lower() in ("smiles","smile","smiles_code","smile_code","smile code","smiles code"):
            return c
    raise ValueError("SMILES column not found")

def detect_label_col(df):
    for c in df.columns:
        if "label" in c.lower():
            return c
    raise ValueError("label column not found")

def smiles_to_ecfp_bits(smi, nBits=1024, radius=2):
    m = Chem.MolFromSmiles(smi)
    arr = np.zeros((nBits,), dtype=np.uint8)
    if not m:
        return arr
    bv = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nBits)
    for i in bv.GetOnBits():
        arr[i] = 1
    return arr

def smiles_to_brics_set(smi):
    m = Chem.MolFromSmiles(smi)
    return set(BRICS.BRICSDecompose(m)) if m else set()

def smiles_to_scaffold_smiles(smi):
    m = Chem.MolFromSmiles(smi)
    if not m: return None
    sc = MurckoScaffold.GetScaffoldForMol(m)
    return Chem.MolToSmiles(sc) if sc else None

def make_brics_maps(df_train, smiles_col, label_col, min_support):
    rec = []
    for y, smi in zip(df_train[label_col].values, df_train[smiles_col].values):
        for f in smiles_to_brics_set(smi):
            rec.append((f, y))
    if not rec:
        return {}, df_train[label_col].mean()
    tmp = pd.DataFrame(rec, columns=["fragment","label"])
    stat = tmp.groupby("fragment").agg(n=("label","size"), tox=("label","mean")).reset_index()
    stat = stat[stat["n"] >= min_support]
    return dict(zip(stat["fragment"], stat["tox"])), df_train[label_col].mean()

def make_scaffold_map(df_train, smiles_col, label_col, min_support):
    sc = df_train[smiles_col].apply(smiles_to_scaffold_smiles)
    stat = pd.DataFrame({"_scaffold": sc, "label": df_train[label_col].values}) \
            .groupby("_scaffold").agg(n=("label","size"), tox=("label","mean")).reset_index()
    stat = stat[stat["n"] >= min_support]
    return dict(zip(stat["_scaffold"], stat["tox"]))

def brics_feat(df, smiles_col, frag2tox, global_tox):
    out = {"brics_known_count":[],"brics_total_count":[],"brics_score_mean":[],"brics_score_max":[]}
    for smi in df[smiles_col].values:
        sset = smiles_to_brics_set(smi)
        if not sset or not frag2tox:
            out["brics_known_count"].append(0); out["brics_total_count"].append(len(sset))
            out["brics_score_mean"].append(global_tox); out["brics_score_max"].append(global_tox); continue
        scores = [frag2tox.get(f, np.nan) for f in sset]
        known = [s for s in scores if not np.isnan(s)]
        out["brics_known_count"].append(len(known))
        out["brics_total_count"].append(len(sset))
        out["brics_score_mean"].append(float(np.mean(known)) if known else global_tox)
        out["brics_score_max"].append(float(np.max(known)) if known else global_tox)
    return pd.DataFrame(out, index=df.index)

def scaffold_feat(df, smiles_col, scaf2tox, global_tox):
    out = {"scaffold_known":[],"scaffold_score":[],"scaffold_freq":[]}
    for smi in df[smiles_col].values:
        s = smiles_to_scaffold_smiles(smi)
        if s in scaf2tox:
            out["scaffold_known"].append(1); out["scaffold_score"].append(float(scaf2tox[s])); out["scaffold_freq"].append(1.0)
        else:
            out["scaffold_known"].append(0); out["scaffold_score"].append(global_tox); out["scaffold_freq"].append(0.0)
    return pd.DataFrame(out, index=df.index)

def parse_ecfp_idx_from_names(names):
    # names like 'ecfp_123' -> 123
    idx = []
    for n in names:
        m = re.match(r'(?i)ecfp_(\d+)', str(n))
        if m: idx.append(int(m.group(1)))
    return np.array(idx, dtype=int)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, default="train.csv")
    ap.add_argument("--input", type=str, default="predict_input.csv")
    ap.add_argument("--model", type=str, default="toxicity_model.pkl")
    ap.add_argument("--artifacts", type=str, default="model_artifacts.json")
    ap.add_argument("--min_frag_support", type=int, default=20)
    ap.add_argument("--min_scaf_support", type=int, default=20)
    ap.add_argument("--fp_lower", type=float, default=0.01)
    ap.add_argument("--fp_upper", type=float, default=0.99)
    ap.add_argument("--fp_topk", type=int, default=256)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--n_bits", type=int, default=1024)
    ap.add_argument("--out", type=str, default="predict_output_김창규.csv")
    args = ap.parse_args()

    # load model + (optional) artifacts
    model = joblib.load(args.model)
    sel_idx = None
    try:
        with open(args.artifacts, "r", encoding="utf-8") as f:
            art = json.load(f)
        if "selected_fp_cols" in art and art["selected_fp_cols"]:
            sel_idx = parse_ecfp_idx_from_names(art["selected_fp_cols"])
    except Exception:
        pass  # artifacts가 없거나 열 수 없으면 아래에서 재선택

    # load data
    tr = pd.read_csv(args.train)
    inp = pd.read_csv(args.input)

    smi_col_tr = detect_smiles_col(tr)
    lab_col_tr = detect_label_col(tr)
    smi_col_in = detect_smiles_col(inp)

    # numeric features (러프)
    num_candidates = ['MolWt','clogp','qed','sa_score']
    num_cols = [c for c in num_candidates if c in tr.columns and c in inp.columns]
    scaler = StandardScaler() if num_cols else None
    if scaler:
        Xnum_tr = scaler.fit_transform(tr[num_cols].fillna(0).values)
        Xnum_in = scaler.transform(inp[num_cols].fillna(0).values)
    else:
        Xnum_tr = np.zeros((len(tr),0)); Xnum_in = np.zeros((len(inp),0))

    # BRICS/Scaffold maps from train
    frag2tox, global_tox = make_brics_maps(tr, smi_col_tr, lab_col_tr, args.min_frag_support)
    scaf2tox = make_scaffold_map(tr, smi_col_tr, lab_col_tr, args.min_scaf_support)

    # BRICS/Scaffold features for input
    in_brics = brics_feat(inp, smi_col_in, frag2tox, global_tox)
    in_scaf  = scaffold_feat(inp, smi_col_in, scaf2tox, global_tox)

    # ECFP matrices (train for selection, input for prediction)
    Xfp_tr = np.vstack([smiles_to_ecfp_bits(s, nBits=args.n_bits, radius=args.radius) for s in tr[smi_col_tr].values]).astype(np.float32)
    Xfp_in = np.vstack([smiles_to_ecfp_bits(s, nBits=args.n_bits, radius=args.radius) for s in inp[smi_col_in].values]).astype(np.float32)
    ytr = tr[lab_col_tr].values.astype(int)

    # Select informative FP indices (if not provided by artifacts)
    if sel_idx is None or len(sel_idx)==0:
        act = Xfp_tr.mean(axis=0)
        mask = (act > args.fp_lower) & (act < args.fp_upper)
        if np.sum(mask) > 0:
            chi_vals, _ = chi2(Xfp_tr[:, mask], ytr)
            K = min(args.fp_topk, int(np.sum(mask)))
            top_local = np.argsort(chi_vals)[::-1][:K]
            sel_idx = np.where(mask)[0][top_local]
        else:
            sel_idx = np.array([], dtype=int)

    # Bit toxicity probs from train (for fp-based scores)
    if len(sel_idx) > 0:
        bit_pos = (Xfp_tr[:, sel_idx].T @ ytr)
        bit_cnt = np.maximum(1, Xfp_tr[:, sel_idx].sum(axis=0))
        bit_tox = bit_pos / bit_cnt
        def fp_scores(X):
            mns, mxs = [], []
            Xsel = X[:, sel_idx]
            for row in Xsel:
                idx = np.where(row>0)[0]
                if len(idx)==0:
                    mns.append(global_tox); mxs.append(global_tox)
                else:
                    s = bit_tox[idx]
                    mns.append(float(np.mean(s))); mxs.append(float(np.max(s)))
            return np.array(mns), np.array(mxs)
        fp_mean_in, fp_max_in = fp_scores(Xfp_in)
        Xfp_sel_in = Xfp_in[:, sel_idx]
    else:
        fp_mean_in = np.full(len(inp), global_tox); fp_max_in = fp_mean_in.copy()
        Xfp_sel_in = np.zeros((len(inp), 0), dtype=np.float32)

    # Assemble input feature matrix (훈련 때와 동일한 블록 순서)
    Xin = np.hstack([
        in_brics[["brics_known_count","brics_total_count","brics_score_mean","brics_score_max"]].values,
        in_scaf [["scaffold_known","scaffold_score","scaffold_freq"]].values,
        fp_mean_in.reshape(-1,1), fp_max_in.reshape(-1,1),
        Xnum_in,
        Xfp_sel_in
    ])

    # Predict
    yhat = (model.predict_proba(Xin)[:,1] >= 0.5).astype(int) if hasattr(model,"predict_proba") else model.predict(Xin).astype(int)

    # Save
    out = pd.DataFrame({"SMILES": inp[smi_col_in], "output": yhat})
    out.to_csv(args.out, index=False)
    print(f"✅ Saved: {args.out}  (rows={len(out)})")

if __name__ == "__main__":
    main()

'''
python predict.py `
  --train train.csv `
  --input predict_input.csv `
  --model toxicity_model.pkl `
  --artifacts model_artifacts.json `
  --out predict_output_김창규2.csv
'''