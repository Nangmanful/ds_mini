import argparse
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.preprocessing import StandardScaler

# --- 전처리 함수 (훈련 코드와 동일하게 유지) ---
def smiles_to_brics(smi: str):
    m = Chem.MolFromSmiles(smi)
    return list(BRICS.BRICSDecompose(m)) if m else []

def smiles_to_scaffold(smi: str):
    m = Chem.MolFromSmiles(smi)
    if not m:
        return None
    sc = MurckoScaffold.GetScaffoldForMol(m)
    return Chem.MolToSmiles(sc) if sc else None

def brics_features(frame, frag2tox, global_tox):
    out = {"brics_known_count":[],"brics_total_count":[],"brics_score_mean":[],"brics_score_max":[]}
    for smi in frame["SMILES"].values:
        sset = set(smiles_to_brics(smi))
        if not sset:
            out["brics_known_count"].append(0)
            out["brics_total_count"].append(0)
            out["brics_score_mean"].append(global_tox)
            out["brics_score_max"].append(global_tox)
            continue
        scores = [frag2tox.get(f, np.nan) for f in sset]
        known = [s for s in scores if not np.isnan(s)]
        out["brics_known_count"].append(len(known))
        out["brics_total_count"].append(len(sset))
        out["brics_score_mean"].append(float(np.mean(known)) if known else global_tox)
        out["brics_score_max"].append(float(np.max(known)) if known else global_tox)
    return pd.DataFrame(out, index=frame.index)

def scaffold_features(frame, scaf2tox, global_tox):
    out = {"scaffold_known":[],"scaffold_score":[],"scaffold_freq":[]}
    for smi in frame["SMILES"].values:
        s = smiles_to_scaffold(smi)
        if s in scaf2tox:
            out["scaffold_known"].append(1)
            out["scaffold_score"].append(float(scaf2tox[s]))
            out["scaffold_freq"].append(1.0)
        else:
            out["scaffold_known"].append(0)
            out["scaffold_score"].append(global_tox)
            out["scaffold_freq"].append(0.0)
    return pd.DataFrame(out, index=frame.index)

def assemble_only_brics_scaf_num(B, S, Xn):
    return np.hstack([
        B[["brics_known_count","brics_total_count","brics_score_mean","brics_score_max"]].values,
        S[["scaffold_known","scaffold_score","scaffold_freq"]].values,
        Xn
    ])

# --- main ---
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--model_a", type=str, default="model_A_brics_scaf_num.pkl")
    ap.add_argument("--model_b", type=str, default="model_B_full.pkl")
    ap.add_argument("--artifacts", type=str, default="ensemble_artifacts.json")
    ap.add_argument("--out", type=str, default="predict_output.csv")
    args = ap.parse_args()

    # load model & artifacts
    clf_A = joblib.load(args.model_a)
    clf_B = joblib.load(args.model_b)
    art = pd.read_json(args.artifacts, typ='series')
    w = float(art["w"])
    global_tox = float(art["global_tox"])
    sel_fp_cols = list(art["selected_fp_cols"])

    frag2tox = art.get("frag2tox", {}) if "frag2tox" in art else {}
    scaf2tox = art.get("scaf2tox", {}) if "scaf2tox" in art else {}

    df = pd.read_csv(args.input)
    # 컬럼명 정규화
    if "SMILES" not in df.columns:
        smiles_col = next(c for c in df.columns if c.lower() in ("smiles","smile","smiles_code","smile_code"))
        df = df.rename(columns={smiles_col: "SMILES"})

    # BRICS & Scaffold
    df_brics = brics_features(df, frag2tox, global_tox)
    df_scaf = scaffold_features(df, scaf2tox, global_tox)

    # Numeric 처리
    num_cols = [c for c in ['MolWt','clogp','qed','sa_score'] if c in df.columns]
    if len(num_cols) > 0:
        scaler = StandardScaler()
        Xn = scaler.fit_transform(df[num_cols].fillna(0).values)
    else:
        Xn = np.zeros((len(df), 0))

    # A 피처 (brics+scaf+num)
    Xin_A = assemble_only_brics_scaf_num(df_brics, df_scaf, Xn)

    # B 피처 (full)
    fp_exists = [c for c in sel_fp_cols if c in df.columns]
    if len(fp_exists) > 0:
        Xfp = df[fp_exists].values.astype(float)
    else:
        Xfp = np.zeros((len(df), 0))
    Xin_B = np.hstack([
        Xin_A,
        Xfp
    ])

    # predict
    pA = clf_A.predict_proba(Xin_A)[:,1]
    pB = clf_B.predict_proba(Xin_B)[:,1]
    p = w*pA + (1-w)*pB
    yhat = (p >= 0.5).astype(int)

    out = pd.DataFrame({
        "SMILES": df["SMILES"],
        "output": yhat
    })
    out.to_csv(args.out, index=False)
    print(f"✅ Saved predictions -> {args.out}")

'''
python predict_ensemble.py '
  --input predict_input.csv '
  --model_a model_A_brics_scaf_num.pkl '
  --model_b model_B_full.pkl '
  --artifacts ensemble_artifacts.json '
  --out predict_output_김창규.csv

'''