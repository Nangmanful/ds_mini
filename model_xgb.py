# train_xgb_hpo_macro.py
# 목적: 기존 tox_classifier_4criteria.py의 특징 생성(브릭스/스캐폴드/FP/수치)을 그대로 사용하되
#       분류기를 XGBoost로 바꾸고, 소형 그리드 HPO를 macro-F1 기준(기본)으로 수행하여
#       최적 모델만 저장한다.

import argparse, json, joblib
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import chi2
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    accuracy_score, classification_report
)

import xgboost as xgb


# -------------------- 데이터/특징 유틸 (원본 유지) --------------------
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
    if not m:
        return None
    sc = MurckoScaffold.GetScaffoldForMol(m)
    return Chem.MolToSmiles(sc) if sc else None

def brics_stats(train_df: pd.DataFrame, smiles_col: str, label_col: str, min_support: int):
    records = []
    for y, smi in zip(train_df[label_col].values, train_df[smiles_col].values):
        for f in set(smiles_to_brics(smi)):
            records.append((f, y))
    if not records:
        return pd.DataFrame(columns=["fragment","n","tox_rate"])
    tmp = pd.DataFrame(records, columns=["fragment","label"])
    stat = tmp.groupby("fragment").agg(
        n=("label","size"), tox_rate=("label","mean")
    ).reset_index()
    stat = stat[stat["n"] >= min_support].reset_index(drop=True)
    return stat

def brics_features(frame: pd.DataFrame, smiles_col: str, frag2tox: dict, global_tox: float):
    out = {"brics_known_count":[],"brics_total_count":[],"brics_score_mean":[],"brics_score_max":[]}
    for smi in frame[smiles_col].values:
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

def scaffold_stats(train_df: pd.DataFrame, smiles_col: str, label_col: str, min_support: int):
    scafs = train_df[smiles_col].apply(smiles_to_scaffold)
    stat = pd.DataFrame({"_scaffold": scafs, label_col: train_df[label_col].values}) \
            .groupby("_scaffold").agg(n=(label_col,"size"), tox_rate=(label_col,"mean")).reset_index()
    stat = stat[stat["n"] >= min_support].reset_index(drop=True)
    return stat

def scaffold_features(frame: pd.DataFrame, smiles_col: str, scaf2tox: dict, global_tox: float):
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

def fp_select_and_scores(train_df: pd.DataFrame, test_df: pd.DataFrame,
                         fp_cols: list, label_col: str,
                         global_tox: float, lower=0.01, upper=0.99, topk=256):
    if len(fp_cols) == 0:
        X_fp_tr = np.zeros((len(train_df), 0)); X_fp_te = np.zeros((len(test_df), 0))
        ytr = train_df[label_col].values.astype(int); yte = test_df[label_col].values.astype(int)
        sel_idx = np.array([], dtype=int)
        fp_mean_tr = np.full(len(train_df), global_tox); fp_max_tr = fp_mean_tr.copy()
        fp_mean_te = np.full(len(test_df), global_tox); fp_max_te = fp_mean_te.copy()
        return (X_fp_tr, X_fp_te, ytr, yte, sel_idx, [], fp_mean_tr, fp_max_tr, fp_mean_te, fp_max_te)

    X_fp_tr = train_df[fp_cols].values.astype(np.float32)
    X_fp_te = test_df[fp_cols].values.astype(np.float32)
    ytr = train_df[label_col].values.astype(int)
    yte = test_df[label_col].values.astype(int)

    act = X_fp_tr.mean(axis=0)
    mask = (act > lower) & (act < upper)
    if np.sum(mask) == 0:
        sel_idx = np.array([], dtype=int)
        sel_cols = []
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

def assemble_features(train_brics, test_brics, train_scaf, test_scaf,
                      fp_mean_tr, fp_max_tr, fp_mean_te, fp_max_te,
                      X_num_tr, X_num_te, X_fp_tr, X_fp_te, sel_idx):
    def pack(B, S, fpm, fpx, Xn, Xfp):
        blocks = [
            B[["brics_known_count","brics_total_count","brics_score_mean","brics_score_max"]].values,
            S[["scaffold_known","scaffold_score","scaffold_freq"]].values,
            fpm.reshape(-1,1), fpx.reshape(-1,1),
            Xn,
            Xfp[:, sel_idx] if Xfp.shape[1] > 0 and len(sel_idx) > 0 else np.zeros((len(B),0))
        ]
        return np.hstack(blocks)
    Xtr = pack(train_brics, train_scaf, fp_mean_tr, fp_max_tr, X_num_tr, X_fp_tr)
    Xte = pack(test_brics,  test_scaf,  fp_mean_te, fp_max_te, X_num_te, X_fp_te)
    return Xtr, Xte


# -------------------- 메인(XGBoost + HPO with macro-F1) --------------------
def main(args):
    df = pd.read_csv(args.data)
    smiles_col, label_col, num_cols, fp_cols = detect_columns(df)

    train_df, test_df = train_test_split(df, test_size=args.test_size,
                                         stratify=df[label_col], random_state=args.rs)
    global_tox = train_df[label_col].mean()

    # 1) BRICS
    brics_stat = brics_stats(train_df, smiles_col, label_col, args.min_frag_support)
    frag2tox = dict(zip(brics_stat["fragment"], brics_stat["tox_rate"]))
    train_brics = brics_features(train_df, smiles_col, frag2tox, global_tox)
    test_brics  = brics_features(test_df,  smiles_col, frag2tox, global_tox)

    # 2) Scaffold
    scaf_stat = scaffold_stats(train_df, smiles_col, label_col, args.min_scaf_support)
    scaf2tox = dict(zip(scaf_stat["_scaffold"], scaf_stat["tox_rate"]))
    train_scaf = scaffold_features(train_df, smiles_col, scaf2tox, global_tox)
    test_scaf  = scaffold_features(test_df,  smiles_col, scaf2tox, global_tox)

    # 3) Fingerprint
    (X_fp_tr, X_fp_te, ytr, yte, sel_idx, sel_cols,
     fp_mean_tr, fp_max_tr, fp_mean_te, fp_max_te) = fp_select_and_scores(
        train_df, test_df, fp_cols, label_col, global_tox,
        lower=args.fp_lower, upper=args.fp_upper, topk=args.fp_topk
    )

    # 4) Numeric (rough)
    if len(num_cols) > 0:
        scaler = StandardScaler()
        X_num_tr = scaler.fit_transform(train_df[num_cols].fillna(0).values)
        X_num_te = scaler.transform(test_df[num_cols].fillna(0).values)
    else:
        X_num_tr = np.zeros((len(train_df), 0))
        X_num_te = np.zeros((len(test_df), 0))

    # assemble
    Xtr, Xte = assemble_features(
        train_brics, test_brics, train_scaf, test_scaf,
        fp_mean_tr, fp_max_tr, fp_mean_te, fp_max_te,
        X_num_tr, X_num_te, X_fp_tr, X_fp_te, sel_idx
    )

    # ---------------- HPO (소형 그리드, macro-F1 기반) ----------------
    grid = [
        {"n_estimators": 600, "lr": 0.03, "max_depth": 5},
        {"n_estimators": 600, "lr": 0.05, "max_depth": 6},
        {"n_estimators": 800, "lr": 0.03, "max_depth": 5},
        {"n_estimators": 800, "lr": 0.05, "max_depth": 6},
    ]
    subs = [0.85]
    cols = [0.5]

    select_metric = args.select_metric.lower()  # 'f1' or 'acc'
    best = {"score": -1.0, "f1_macro": -1.0, "acc": -1.0, "params": None, "iter": None}
    best_model = None

    for base in grid:
        for ss in subs:
            for cs in cols:
                clf = xgb.XGBClassifier(
                    n_estimators=base["n_estimators"],
                    learning_rate=base["lr"],
                    max_depth=base["max_depth"],
                    subsample=ss,
                    colsample_bytree=cs,
                    min_child_weight=3,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    tree_method="hist",
                    random_state=args.rs,
                    use_label_encoder=False,
                    eval_metric="logloss",
                )
                clf.fit(Xtr, ytr)

                # 반복수 선택: 50 step로 ntree_limit 스캔
                nB = clf.n_estimators
                for i in range(50, nB + 1, 50):
                    yprob_i = clf.predict_proba(Xte, iteration_range=(0,i))[:, 1]
                    yhat_i = (yprob_i >= 0.5).astype(int)

                    f1_macro_i = f1_score(yte, yhat_i, average="macro")
                    acc_i = accuracy_score(yte, yhat_i)
                    score_i = f1_macro_i if select_metric == "f1" else acc_i

                    # 우선순위: 선택기준 → macro-F1 → ACC
                    if (score_i > best["score"]) or \
                       (score_i == best["score"] and f1_macro_i > best["f1_macro"]) or \
                       (score_i == best["score"] and f1_macro_i == best["f1_macro"] and acc_i > best["acc"]):
                        best.update({
                            "score": float(score_i),
                            "f1_macro": float(f1_macro_i),
                            "acc": float(acc_i),
                            "params": {
                                **base, "subsample": ss, "colsample_bytree": cs
                            },
                            "iter": int(i)
                        })
                        best_model = clf

    # ---------------- 최종 평가(최적 반복수) ----------------
    assert best_model is not None, "HPO에서 최적 모델을 찾지 못했습니다."
    yprob = best_model.predict_proba(Xte, iteration_range=(0,best["iter"]))[:, 1]
    yhat = (yprob >= 0.5).astype(int)

    print("\n=== XGBoost (HPO Best) Metrics ===")
    print("Best params:", best["params"], "best_iter:", best["iter"])
    print("ROC AUC  :", round(roc_auc_score(yte, yprob), 4))
    print("PR  AUC :", round(average_precision_score(yte, yprob), 4))
    print("F1(macro):", round(f1_score(yte, yhat, average='macro'), 4))
    print("ACC      :", round(accuracy_score(yte, yhat), 4))
    print("\nClassification report:\n", classification_report(yte, yhat))

    # 50트리 간격 로그(최종 모델로만)
    print("\n--- Intermediate Scores (best model only) ---")
    for i in range(50, best_model.n_estimators + 1, 50):
        yprob_i = best_model.predict_proba(Xte, iteration_range=(0,i))[:, 1]
        yhat_i = (yprob_i >= 0.5).astype(int)
        f1_macro_i = f1_score(yte, yhat_i, average="macro")
        acc_i = accuracy_score(yte, yhat_i)
        mark = " <-- used" if i == best["iter"] else ""
        print(f"[{i} trees] F1(macro)={f1_macro_i:.4f}, ACC={acc_i:.4f}{mark}")

    # ---------------- 저장(최적만) ----------------
    joblib.dump(best_model, "toxicity_model_xgb.pkl")
    artifacts = {
        "global_tox": float(global_tox),
        "n_brics_kept": int(len(brics_stat)),
        "n_scaffolds_kept": int(len(scaf_stat)),
        "n_fp_selected": int(len(sel_idx)),
        "selected_fp_cols": sel_cols,
        "best_params": best["params"],
        "best_iter": best["iter"],
        "select_metric": select_metric,
        "f1_macro": best["f1_macro"],
        "acc": best["acc"]
    }
    with open("model_artifacts_xgb.json", "w", encoding="utf-8") as f:
        json.dump(artifacts, f, ensure_ascii=False, indent=2)
    print("\n💾 Saved: toxicity_model_xgb.pkl, model_artifacts_xgb.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="train.csv")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--rs", type=int, default=42)
    ap.add_argument("--min_frag_support", type=int, default=20)
    ap.add_argument("--min_scaf_support", type=int, default=20)
    ap.add_argument("--fp_lower", type=float, default=0.01)
    ap.add_argument("--fp_upper", type=float, default=0.99)
    ap.add_argument("--fp_topk", type=int, default=256)   # 느리면 128/64로
    ap.add_argument("--select_metric", type=str, default="f1", choices=["f1","acc"])
    args = ap.parse_args()
    main(args)
