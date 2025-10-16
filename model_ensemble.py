# train_ensemble_brics_stack_xgb.py
# - Model A: LightGBM (BRICS + Scaffold + Numeric)
# - Model B: XGBoost (Full: A + FP scores + 선택된 FP bits)
# - Ensemble: best w 탐색 + 50트리 점검
# - Meta: LogisticRegression on [pA, pB, BRICS 요약]
# 저장: clf_A.pkl, clf_B_xgb.pkl, meta.pkl, preproc_artifacts.pkl, ensemble_report.json

import argparse, json, joblib
import numpy as np, pandas as pd
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    accuracy_score, classification_report
)

# ---------------------- Models ----------------------
import lightgbm as lgb
import xgboost as xgb
HAS_LGBM = True

# ---------------------- Feature Utils ----------------------
def detect_columns(df: pd.DataFrame):
    smiles_col = next((c for c in df.columns if c.lower() in ("smiles","smile","smiles_code","smile_code")), None)
    label_col  = next((c for c in df.columns if "label" in c.lower()), None)
    if smiles_col is None or label_col is None:
        raise ValueError("SMILES/label 컬럼을 찾지 못했습니다.")
    num_cols = [c for c in ['MolWt','clogp','qed','sa_score'] if c in df.columns]
    ecfp_cols = [c for c in df.columns if c.lower().startswith('ecfp_')]
    if len(ecfp_cols) > 0:
        fp_cols = ecfp_cols
    else:
        fp_cols = [c for c in df.columns if any(c.lower().startswith(p) for p in ["ecfp_","fcfp_","ptfp_"])]
    return smiles_col, label_col, num_cols, fp_cols

def smiles_to_brics(smi: str):
    m = Chem.MolFromSmiles(smi)
    return list(BRICS.BRICSDecompose(m)) if m else []

def smiles_to_scaffold(smi: str):
    m = Chem.MolFromSmiles(smi)
    if not m: return None
    sc = MurckoScaffold.GetScaffoldForMol(m)
    return Chem.MolToSmiles(sc) if sc else None

def brics_stats(train_df, smiles_col, label_col, min_support):
    records = []
    for y, smi in zip(train_df[label_col].values, train_df[smiles_col].values):
        for f in set(smiles_to_brics(smi)):
            records.append((f, y))
    if not records:
        return pd.DataFrame(columns=["fragment","n","tox_rate"])
    tmp = pd.DataFrame(records, columns=["fragment","label"])
    stat = tmp.groupby("fragment").agg(n=("label","size"), tox_rate=("label","mean")).reset_index()
    stat = stat[stat["n"] >= min_support].reset_index(drop=True)
    return stat

def brics_features(frame: pd.DataFrame, smiles_col: str, frag2tox: dict, global_tox: float):
    out = {"brics_known_count":[],"brics_total_count":[],"brics_score_mean":[],"brics_score_max":[],"brics_known_frac":[]}
    for smi in frame[smiles_col].values:
        sset = set(smiles_to_brics(smi))
        if not sset:
            out["brics_known_count"].append(0)
            out["brics_total_count"].append(0)
            out["brics_score_mean"].append(global_tox)
            out["brics_score_max"].append(global_tox)
            out["brics_known_frac"].append(0.0)
            continue
        scores = [frag2tox.get(f, np.nan) for f in sset]
        known = [s for s in scores if not np.isnan(s)]
        out["brics_known_count"].append(len(known))
        out["brics_total_count"].append(len(sset))
        out["brics_score_mean"].append(float(np.mean(known)) if known else global_tox)
        out["brics_score_max"].append(float(np.max(known)) if known else global_tox)
        out["brics_known_frac"].append(len(known)/max(1,len(sset)))
    return pd.DataFrame(out, index=frame.index)

def scaffold_stats(train_df, smiles_col, label_col, min_support):
    scafs = train_df[smiles_col].apply(smiles_to_scaffold)
    stat = pd.DataFrame({"_scaffold": scafs, label_col: train_df[label_col].values}) \
            .groupby("_scaffold").agg(n=(label_col,"size"), tox_rate=(label_col,"mean")).reset_index()
    stat = stat[stat["n"] >= min_support].reset_index(drop=True)
    return stat

def scaffold_features(frame, smiles_col, scaf2tox: dict, global_tox: float):
    out = {"scaffold_known":[],"scaffold_score":[],"scaffold_freq":[]}
    for smi in frame[smiles_col].values:
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

def fp_select_and_scores(train_df, test_df, fp_cols, label_col, global_tox, lower=0.01, upper=0.99, topk=256):
    if len(fp_cols) == 0:
        X_fp_tr = np.zeros((len(train_df), 0)); X_fp_te = np.zeros((len(test_df), 0))
        ytr = train_df[label_col].values.astype(int); yte = test_df[label_col].values.astype(int)
        sel_idx = np.array([], dtype=int); sel_cols = []
        fp_mean_tr = np.full(len(train_df), global_tox); fp_max_tr = fp_mean_tr.copy()
        fp_mean_te = np.full(len(test_df), global_tox); fp_max_te = fp_mean_te.copy()
        return (X_fp_tr, X_fp_te, ytr, yte, sel_idx, sel_cols, fp_mean_tr, fp_max_tr, fp_mean_te, fp_max_te)

    X_fp_tr = train_df[fp_cols].values.astype(np.float32)
    X_fp_te = test_df[fp_cols].values.astype(np.float32)
    ytr = train_df[label_col].values.astype(int)
    yte = test_df[label_col].values.astype(int)

    act = X_fp_tr.mean(axis=0)
    mask = (act > lower) & (act < upper)
    if np.sum(mask) == 0:
        sel_idx = np.array([], dtype=int); sel_cols = []
        fp_mean_tr = np.full(len(train_df), global_tox); fp_max_tr = fp_mean_tr.copy()
        fp_mean_te = np.full(len(test_df), global_tox); fp_max_te = fp_mean_te.copy()
        return (X_fp_tr, X_fp_te, ytr, yte, sel_idx, sel_cols, fp_mean_tr, fp_max_tr, fp_mean_te, fp_max_te)

    chi_vals, _ = chi2(X_fp_tr[:, mask], ytr)
    K = min(topk, int(np.sum(mask)))
    top_local = np.argsort(chi_vals)[::-1][:K]
    sel_idx = np.where(mask)[0][top_local]
    sel_cols = [fp_cols[i] for i in sel_idx]

    bit_pos = (X_fp_tr[:, sel_idx].T @ ytr)
    bit_cnt = np.maximum(1, X_fp_tr[:, sel_idx].sum(axis=0))
    bit_tox = bit_pos / bit_cnt

    def fp_scores(X):
        mean_s, max_s = [], []
        for row in X[:, sel_idx]:
            idx = np.where(row > 0)[0]
            if len(idx) == 0:
                mean_s.append(global_tox); max_s.append(global_tox)
            else:
                s = bit_tox[idx]
                mean_s.append(float(np.mean(s))); max_s.append(float(np.max(s)))
        return np.array(mean_s), np.array(max_s)

    fp_mean_tr, fp_max_tr = fp_scores(X_fp_tr)
    fp_mean_te, fp_max_te = fp_scores(X_fp_te)
    return (X_fp_tr, X_fp_te, ytr, yte, sel_idx, sel_cols, fp_mean_tr, fp_max_tr, fp_mean_te, fp_max_te)

def assemble_only_brics_scaf_num(B, S, Xn):
    return np.hstack([
        B[["brics_known_count","brics_total_count","brics_score_mean","brics_score_max","brics_known_frac"]].values,
        S[["scaffold_known","scaffold_score","scaffold_freq"]].values,
        Xn
    ])

def assemble_full(train_brics, test_brics, train_scaf, test_scaf,
                  fp_mean_tr, fp_max_tr, fp_mean_te, fp_max_te,
                  X_num_tr, X_num_te, X_fp_tr, X_fp_te, sel_idx):
    def pack(B, S, fpm, fpx, Xn, Xfp):
        blocks = [
            B[["brics_known_count","brics_total_count","brics_score_mean","brics_score_max","brics_known_frac"]].values,
            S[["scaffold_known","scaffold_score","scaffold_freq"]].values,
            fpm.reshape(-1,1), fpx.reshape(-1,1),
            Xn,
            Xfp[:, sel_idx] if Xfp.shape[1] > 0 and len(sel_idx) > 0 else np.zeros((len(B),0))
        ]
        return np.hstack(blocks)
    Xtr = pack(train_brics, train_scaf, fp_mean_tr, fp_max_tr, X_num_tr, X_fp_tr)
    Xte = pack(test_brics,  test_scaf,  fp_mean_te, fp_max_te, X_num_te, X_fp_te)
    return Xtr, Xte

def make_lgb(args):
    return lgb.LGBMClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.lr,
        subsample=0.85,
        colsample_bytree=0.5,
        max_depth=-1,
        random_state=args.rs
    )

def make_xgb(args):
    return xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.lr,
        subsample=0.85,
        colsample_bytree=0.5,
        max_depth=6,
        tree_method="hist",
        random_state=args.rs,
        use_label_encoder=False,
        eval_metric="logloss"
    )

def prob(clf, X):
    return clf.predict_proba(X)[:,1] if hasattr(clf, "predict_proba") else clf.predict(X).astype(float)

# ---------------------- Main ----------------------
def main(args):
    df = pd.read_csv(args.data)
    smiles_col, label_col, num_cols, fp_cols = detect_columns(df)
    train_df, test_df = train_test_split(df, test_size=args.test_size, stratify=df[label_col], random_state=args.rs)
    global_tox = train_df[label_col].mean()

    # --- Feature ---
    brics_stat = brics_stats(train_df, smiles_col, label_col, args.min_frag_support)
    frag2tox = dict(zip(brics_stat["fragment"], brics_stat["tox_rate"]))
    scaf_stat = scaffold_stats(train_df, smiles_col, label_col, args.min_scaf_support)
    scaf2tox = dict(zip(scaf_stat["_scaffold"], scaf_stat["tox_rate"]))

    train_brics = brics_features(train_df, smiles_col, frag2tox, global_tox)
    test_brics  = brics_features(test_df,  smiles_col, frag2tox, global_tox)
    train_scaf  = scaffold_features(train_df, smiles_col, scaf2tox, global_tox)
    test_scaf   = scaffold_features(test_df,  smiles_col, scaf2tox, global_tox)

    # Numeric
    if len(num_cols) > 0:
        scaler_num = StandardScaler()
        X_num_tr = scaler_num.fit_transform(train_df[num_cols].fillna(0).values)
        X_num_te = scaler_num.transform(test_df[num_cols].fillna(0).values)
    else:
        scaler_num = None
        X_num_tr = np.zeros((len(train_df),0)); X_num_te = np.zeros((len(test_df),0))

    # FP
    (X_fp_tr, X_fp_te, ytr, yte, sel_idx, sel_cols,
     fp_mean_tr, fp_max_tr, fp_mean_te, fp_max_te) = fp_select_and_scores(
        train_df, test_df, fp_cols, label_col, global_tox,
        lower=args.fp_lower, upper=args.fp_upper, topk=args.fp_topk
    )

    # Assemble
    Xtr_B, Xte_B = assemble_full(train_brics, test_brics, train_scaf, test_scaf,
                                 fp_mean_tr, fp_max_tr, fp_mean_te, fp_max_te,
                                 X_num_tr, X_num_te, X_fp_tr, X_fp_te, sel_idx)
    Xtr_A = assemble_only_brics_scaf_num(train_brics, train_scaf, X_num_tr)
    Xte_A = assemble_only_brics_scaf_num(test_brics,  test_scaf,  X_num_te)

    # --- Train base models ---
    clf_A = make_lgb(args)
    clf_B = make_xgb(args)
    clf_A.fit(Xtr_A, ytr)
    clf_B.fit(Xtr_B, ytr)

    # ----------------- Ensemble Best w + 50 tree evaluation -----------------
    pA_full = prob(clf_A, Xte_A)
    pB_full = prob(clf_B, Xte_B)
    w_grid = np.linspace(0.0, 1.0, 21)
    best = {"w": 0.0, "f1": -1.0, "acc": -1.0}
    for w in w_grid:
        p_ens = w * pA_full + (1 - w) * pB_full
        yhat = (p_ens >= 0.5).astype(int)
        f1 = f1_score(yte, yhat); acc = accuracy_score(yte, yhat)
        if f1 > best["f1"] or (f1 == best["f1"] and acc > best["acc"]):
            best.update({"w": float(w), "f1": float(f1), "acc": float(acc)})
    print(f"\n✅ Ensemble best w={best['w']:.2f} (ACC={best['acc']:.4f}, F1={best['f1']:.4f})")

    # 50트리 점검 (XGBoost는 best_ntree_limit 사용)
    print("\n--- Ensemble Intermediate Scores ---")
    nA = clf_A.best_iteration_ or clf_A.n_estimators
    nB = clf_B.n_estimators
    n_iters = min(nA, nB)
    best_iter_ens, best_f1_ens, best_acc_ens = 0, -1.0, -1.0

    for i in range(50, n_iters + 1, 50):
        pA_i = clf_A.predict_proba(Xte_A, num_iteration=i)[:, 1]
        pB_i = clf_B.predict_proba(Xte_B, iteration_range=(0, i))[:, 1] 
        p_i = best["w"] * pA_i + (1 - best["w"]) * pB_i
        yhat_i = (p_i >= 0.5).astype(int)
        f1_i = f1_score(yte, yhat_i)
        acc_i = accuracy_score(yte, yhat_i)
        print(f"[{i} trees] F1={f1_i:.4f}, ACC={acc_i:.4f}")
        if f1_i > best_f1_ens:
            best_f1_ens, best_acc_ens, best_iter_ens = f1_i, acc_i, i

    print(f"✅ Best Ensemble (fixed w={best['w']:.2f}) at {best_iter_ens} trees: "
          f"F1={best_f1_ens:.4f}, ACC={best_acc_ens:.4f}")

    # --- Meta Stacking ---
    pA_tr = prob(clf_A, Xtr_A); pB_tr = prob(clf_B, Xtr_B)
    pA_te = prob(clf_A, Xte_A); pB_te = prob(clf_B, Xte_B)
    Z_tr = np.c_[pA_tr, pB_tr,
                 train_brics[["brics_score_mean","brics_score_max","brics_known_frac"]].values]
    Z_te = np.c_[pA_te, pB_te,
                 test_brics[["brics_score_mean","brics_score_max","brics_known_frac"]].values]
    meta = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=300)
    meta.fit(Z_tr, ytr)
    p_stack = meta.predict_proba(Z_te)[:,1]
    yhat_stack = (p_stack >= 0.5).astype(int)

    print("\n=== Stacking (A + XGB B -> meta) Metrics ===")
    print("ROC AUC:", round(roc_auc_score(yte, p_stack), 4))
    print("PR  AUC:", round(average_precision_score(yte, p_stack), 4))
    print("F1     :", round(f1_score(yte, yhat_stack), 4))
    print("ACC    :", round(accuracy_score(yte, yhat_stack), 4))
    print("\nClassification report:\n", classification_report(yte, yhat_stack))

    # --- Save ---
    joblib.dump(clf_A, "clf_A_lgb.pkl")
    joblib.dump(clf_B, "clf_B_xgb.pkl")
    joblib.dump(meta,  "meta_stack.pkl")

    preproc = {
        "global_tox": float(global_tox),
        "num_cols": num_cols,
        "scaler_num": scaler_num,
        "frag2tox": frag2tox,
        "scaf2tox": scaf2tox,
        "selected_fp_cols": [str(c) for c in sel_cols],
        "n_fp_selected": int(len(sel_cols)),
        "brics_min_support": int(args.min_frag_support),
        "scaf_min_support": int(args.min_scaf_support)
    }
    joblib.dump(preproc, "preproc_artifacts.pkl")

    with open("ensemble_report.json","w", encoding="utf-8") as f:
        json.dump({
            "ensemble": best,
            "stack_metrics": {
                "roc_auc": float(roc_auc_score(yte, p_stack)),
                "pr_auc": float(average_precision_score(yte, p_stack)),
                "f1": float(f1_score(yte, yhat_stack)),
                "acc": float(accuracy_score(yte, yhat_stack)),
            },
            "n_features_full": int(Xtr_B.shape[1]),
            "n_features_A": int(Xtr_A.shape[1])
        }, f, ensure_ascii=False, indent=2)

    print("💾 Saved: clf_A_lgb.pkl, clf_B_xgb.pkl, meta_stack.pkl, preproc_artifacts.pkl, ensemble_report.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="train.csv")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--rs", type=int, default=42)
    ap.add_argument("--min_frag_support", type=int, default=40)
    ap.add_argument("--min_scaf_support", type=int, default=40)
    ap.add_argument("--fp_lower", type=float, default=0.1)
    ap.add_argument("--fp_upper", type=float, default=0.9)
    ap.add_argument("--fp_topk", type=int, default=512)
    ap.add_argument("--n_estimators", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.05)
    args = ap.parse_args()
    main(args)
