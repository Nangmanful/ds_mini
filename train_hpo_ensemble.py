# train_hpo_ensemble.py
# - Feature: BRICS, Scaffold(OOF smoothed + support_log), Numeric
# - FP: OOF target encoding + m-estimate smoothing + chi2 preselect
# - Base models: A=LightGBM(요약특성), B=XGBoost(요약+FP)
# - HPO: 작은 grid로 (A파라미터 × B파라미터) 탐색
#        각 조합에서 트리 반복 i(50 step)와 앙상블 가중치 w를 함께 탐색
# - 선택 기준: F1 (동률 시 ACC)
# - 산출물: 최적 모델/전처리/리포트만 저장 (가중치 w, ntree_limit_A/B 포함)

import argparse, json, joblib
import numpy as np, pandas as pd

from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score, average_precision_score
)

import lightgbm as lgb
import xgboost as xgb

# ===================== 공통 유틸 =====================
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

# ----- BRICS -----
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

# ----- Scaffold(OOF smoothed + support_log) -----
def scaffold_stats_oof(train_df, smiles_col, label_col, min_support,
                       n_splits=5, rs=42, alpha=50):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=rs)
    prior = float(train_df[label_col].mean())
    scaf_all = train_df[smiles_col].apply(smiles_to_scaffold).fillna("NA").values
    y = train_df[label_col].values.astype(int)
    sums, cnts = {}, {}
    for tr, va in kf.split(train_df):
        sc_tr = scaf_all[tr]; ytr = y[tr]
        for s, t in zip(sc_tr, ytr):
            cnts[s] = cnts.get(s, 0) + 1
            sums[s] = sums.get(s, 0) + t
    rows = []
    for s, n in cnts.items():
        if n >= min_support:
            p = (sums[s] + alpha * prior) / (n + alpha)
            rows.append((s, n, p))
    stat = pd.DataFrame(rows, columns=["_scaffold","n","tox_rate"])
    return stat

def scaffold_features_rich(frame, smiles_col, scaf2tox, scaf2n, global_tox):
    out = {"scaffold_known":[],"scaffold_score":[],"scaffold_freq":[],"scaffold_support_log":[]}
    for smi in frame[smiles_col].values:
        s = smiles_to_scaffold(smi) or "NA"
        if s in scaf2tox:
            out["scaffold_known"].append(1)
            out["scaffold_score"].append(float(scaf2tox[s]))
            out["scaffold_freq"].append(1.0)
            out["scaffold_support_log"].append(np.log1p(scaf2n.get(s, 1)))
        else:
            out["scaffold_known"].append(0)
            out["scaffold_score"].append(global_tox)
            out["scaffold_freq"].append(0.0)
            out["scaffold_support_log"].append(0.0)
    return pd.DataFrame(out, index=frame.index)

# ----- FP (OOF target encoding + m-estimate) -----
def oof_bit_tox_smooth(X_fp, y, sel_mask, alpha=50, n_splits=5, rs=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=rs)
    prior = float(y.mean())
    bit_idx = np.where(sel_mask)[0]
    n_sel = len(bit_idx)
    acc_sum = np.zeros(n_sel, dtype=float)
    acc_cnt = np.zeros(n_sel, dtype=float)
    for tr, va in kf.split(X_fp):
        Xtr, ytr = X_fp[tr], y[tr]
        act_cnt = Xtr[:, bit_idx].sum(axis=0)
        pos_cnt = (Xtr[:, bit_idx].T @ ytr)
        p_fold = (pos_cnt + alpha * prior) / (np.maximum(1, act_cnt) + alpha)
        acc_sum += p_fold; acc_cnt += 1.0
    bit_tox_sel = acc_sum / np.maximum(1.0, acc_cnt)
    return bit_idx, bit_tox_sel

def fp_select_and_scores_regularized(train_df, test_df, fp_cols, label_col,
                                     global_tox, lower=0.01, upper=0.99,
                                     topk=256, alpha=50, n_splits=5, rs=42):
    if len(fp_cols) == 0:
        Xtr = np.zeros((len(train_df), 0)); Xte = np.zeros((len(test_df), 0))
        ytr = train_df[label_col].values.astype(int); yte = test_df[label_col].values.astype(int)
        sel_idx = np.array([], dtype=int); sel_cols = []
        fp_mean_tr = np.full(len(train_df), global_tox); fp_max_tr = fp_mean_tr.copy()
        fp_mean_te = np.full(len(test_df), global_tox); fp_max_te = fp_mean_te.copy()
        bit_tox_sel = np.array([], dtype=float)
        return (Xtr, Xte, ytr, yte, sel_idx, sel_cols,
                fp_mean_tr, fp_max_tr, fp_mean_te, fp_max_te, bit_tox_sel)

    Xtr = train_df[fp_cols].values.astype(np.float32)
    Xte = test_df[fp_cols].values.astype(np.float32)
    ytr = train_df[label_col].values.astype(int)
    yte = test_df[label_col].values.astype(int)

    act = Xtr.mean(axis=0)
    mask = (act > lower) & (act < upper)
    if mask.sum() == 0:
        sel_idx = np.array([], dtype=int); sel_cols = []
        fp_mean_tr = np.full(len(train_df), global_tox); fp_max_tr = fp_mean_tr.copy()
        fp_mean_te = np.full(len(test_df), global_tox); fp_max_te = fp_mean_te.copy()
        bit_tox_sel = np.array([], dtype=float)
        return (Xtr, Xte, ytr, yte, sel_idx, sel_cols,
                fp_mean_tr, fp_max_tr, fp_mean_te, fp_max_te, bit_tox_sel)

    chi_vals, _ = chi2(Xtr[:, mask], ytr)
    K = min(topk, int(mask.sum()))
    top_local = np.argsort(chi_vals)[::-1][:K]
    pre_sel_global = np.where(mask)[0][top_local]
    sel_mask_global = np.zeros(Xtr.shape[1], dtype=bool); sel_mask_global[pre_sel_global] = True

    sel_idx, bit_tox_sel = oof_bit_tox_smooth(Xtr, ytr, sel_mask_global,
                                              alpha=alpha, n_splits=n_splits, rs=rs)
    sel_cols = [fp_cols[i] for i in sel_idx]

    def fp_scores(X):
        mean_s, max_s = [], []
        Xsel = X[:, sel_idx]
        for row in Xsel:
            idx = np.where(row > 0)[0]
            if len(idx) == 0:
                mean_s.append(global_tox); max_s.append(global_tox)
            else:
                s = bit_tox_sel[idx]
                mean_s.append(float(np.mean(s))); max_s.append(float(np.max(s)))
        return np.array(mean_s), np.array(max_s)

    fp_mean_tr, fp_max_tr = fp_scores(Xtr)
    fp_mean_te, fp_max_te = fp_scores(Xte)

    return (Xtr, Xte, ytr, yte, sel_idx, sel_cols,
            fp_mean_tr, fp_max_tr, fp_mean_te, fp_max_te, bit_tox_sel)

# ----- 조립 -----
def assemble_only_brics_scaf_num(B, S, Xn):
    return np.hstack([
        B[["brics_known_count","brics_total_count","brics_score_mean","brics_score_max","brics_known_frac"]].values,
        S[["scaffold_known","scaffold_score","scaffold_freq","scaffold_support_log"]].values,
        Xn
    ])

def assemble_full(train_brics, test_brics, train_scaf, test_scaf,
                  fp_mean_tr, fp_max_tr, fp_mean_te, fp_max_te,
                  X_num_tr, X_num_te, X_fp_tr, X_fp_te, sel_idx):
    def pack(B, S, fpm, fpx, Xn, Xfp):
        blocks = [
            B[["brics_known_count","brics_total_count","brics_score_mean","brics_score_max","brics_known_frac"]].values,
            S[["scaffold_known","scaffold_score","scaffold_freq","scaffold_support_log"]].values,
            fpm.reshape(-1,1), fpx.reshape(-1,1),
            Xn,
            Xfp[:, sel_idx] if Xfp.shape[1] > 0 and len(sel_idx) > 0 else np.zeros((len(B),0))
        ]
        return np.hstack(blocks)
    Xtr = pack(train_brics, train_scaf, fp_mean_tr, fp_max_tr, X_num_tr, X_fp_tr)
    Xte = pack(test_brics,  test_scaf,  fp_mean_te, fp_max_te, X_num_te, X_fp_te)
    return Xtr, Xte

# ----- 모델 팩토리 -----
def make_lgb(params):
    return lgb.LGBMClassifier(
        n_estimators=params["n_estimators"],
        learning_rate=params["lr"],
        subsample=0.85,
        colsample_bytree=0.5,
        max_depth=-1,
        min_data_in_leaf=60,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=params["rs"]
    )

def make_xgb(params):
    return xgb.XGBClassifier(
        n_estimators=params["n_estimators"],
        learning_rate=params["lr"],
        subsample=0.85,
        colsample_bytree=0.5,
        max_depth=params["max_depth"],
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=params["rs"],
        use_label_encoder=False,
        eval_metric="logloss"
    )

def prob(clf, X):
    return clf.predict_proba(X)[:,1] if hasattr(clf, "predict_proba") else clf.predict(X).astype(float)

# ===================== 메인(HPO 포함) =====================
def main(args):
    # 데이터
    df = pd.read_csv(args.data)
    smiles_col, label_col, num_cols, fp_cols = detect_columns(df)
    tr_df, te_df = train_test_split(df, test_size=args.test_size, stratify=df[label_col], random_state=args.rs)
    global_tox = tr_df[label_col].mean()

    # BRICS/Scaffold 통계
    br_stat = brics_stats(tr_df, smiles_col, label_col, args.min_frag_support)
    frag2tox = dict(zip(br_stat["fragment"], br_stat["tox_rate"]))
    sc_stat  = scaffold_stats_oof(tr_df, smiles_col, label_col, args.min_scaf_support,
                                  n_splits=5, rs=args.rs, alpha=50)
    scaf2tox = dict(zip(sc_stat["_scaffold"], sc_stat["tox_rate"]))
    scaf2n   = dict(zip(sc_stat["_scaffold"], sc_stat["n"]))

    # BRICS/Scaffold 피처
    tr_brics = brics_features(tr_df, smiles_col, frag2tox, global_tox)
    te_brics = brics_features(te_df, smiles_col, frag2tox, global_tox)
    tr_scaf  = scaffold_features_rich(tr_df, smiles_col, scaf2tox, scaf2n, global_tox)
    te_scaf  = scaffold_features_rich(te_df,  smiles_col, scaf2tox, scaf2n, global_tox)

    # Numeric
    if len(num_cols) > 0:
        scaler_num = StandardScaler()
        Xn_tr = scaler_num.fit_transform(tr_df[num_cols].fillna(0).values)
        Xn_te = scaler_num.transform(te_df[num_cols].fillna(0).values)
    else:
        scaler_num = None
        Xn_tr = np.zeros((len(tr_df),0)); Xn_te = np.zeros((len(te_df),0))

    # FP
    (Xfp_tr, Xfp_te, ytr, yte, sel_idx, sel_cols,
     fp_mean_tr, fp_max_tr, fp_mean_te, fp_max_te, bit_tox_sel) = fp_select_and_scores_regularized(
        tr_df, te_df, fp_cols, label_col, global_tox,
        lower=args.fp_lower, upper=args.fp_upper, topk=args.fp_topk,
        alpha=50, n_splits=5, rs=args.rs
    )

    # Assemble
    Xtr_B, Xte_B = assemble_full(tr_brics, te_brics, tr_scaf, te_scaf,
                                 fp_mean_tr, fp_max_tr, fp_mean_te, fp_max_te,
                                 Xn_tr, Xn_te, Xfp_tr, Xfp_te, sel_idx)
    Xtr_A = assemble_only_brics_scaf_num(tr_brics, tr_scaf, Xn_tr)
    Xte_A = assemble_only_brics_scaf_num(te_brics, te_scaf, Xn_te)


    # === 기존 HPO 루프 교체 시작 ===
    w_grid = np.linspace(0.0, 1.0, 11)     # 0.05 step
    t_grid = np.linspace(0.40, 0.60, 11)   # 임계값도 함께 탐색 (F1 최적화)
    iA_grid = [600]              # A/B 서로 다른 반복수 후보
    iB_grid = [1000, 1200, 1400]

    best = {"score_f1": -1.0, "score_acc": -1.0,
            "A_params": None, "B_params": None,
            "w": 0.5, "iA": None, "iB": None, "thr": 0.5}
    best_models = {"A": None, "B": None}

    def predict_calibrated(model, X, iteration_range=None, num_iteration=None, T=1.0):
        if iteration_range is not None:
            yprob = model.predict_proba(X, iteration_range=iteration_range)[:, 1]
        elif num_iteration is not None:
            yprob = model.predict_proba(X, num_iteration=num_iteration)[:, 1]
        else:
            yprob = model.predict_proba(X)[:, 1]
        logits = np.log(np.clip(yprob, 1e-9, 1-1e-9) / (1 - np.clip(yprob, 1e-9, 1-1e-9)))
        return 1 / (1 + np.exp(-logits / T))


    # (선택) 간단히 T 후보도 소수만 본다. (시간 절약)
    T_grid_A = [1.0]        # 필요 시 [1.0, 1.2]
    T_grid_B = [1.0]        # 필요 시 [1.0, 1.2]
    A_grid = [
        {"n_estimators": 600, "lr": 0.03, "rs": args.rs},
        {"n_estimators": 600, "lr": 0.05, "rs": args.rs},
    ]
    B_grid = [
        {"n_estimators": 1000, "lr": 0.05, "max_depth": 5, "rs": args.rs},
        {"n_estimators": 1000, "lr": 0.05, "max_depth": 6, "rs": args.rs},
        {"n_estimators": 1200, "lr": 0.05, "max_depth": 5, "rs": args.rs},
        {"n_estimators": 1200, "lr": 0.05, "max_depth": 6, "rs": args.rs},
        {"n_estimators": 1400, "lr": 0.05, "max_depth": 5, "rs": args.rs},
        {"n_estimators": 1400, "lr": 0.05, "max_depth": 6, "rs": args.rs},
    ]
    for Ap in A_grid:
        clf_A = make_lgb(Ap)
        clf_A.fit(Xtr_A, ytr)

        for Bp in B_grid:
            clf_B = make_xgb(Bp)
            clf_B.fit(Xtr_B, ytr)

            nA = getattr(clf_A, "best_iteration_", None) or clf_A.n_estimators
            nB = clf_B.n_estimators

            for iA in iA_grid:
                if iA > nA: 
                    continue
                # 미리 A 확률 캐시(보정 포함)
                pA_cache = {T: predict_calibrated(clf_A, Xte_A, num_iteration=iA, T=T) for T in T_grid_A}

                for iB in iB_grid:
                    if iB > nB:
                        continue
                    pB_cache = {T: predict_calibrated(clf_B, Xte_B, iteration_range=(0,iB), T=T) for T in T_grid_B}

                    # w, τ, T 동시 소형 탐색
                    for T_a, pA_i in pA_cache.items():
                        for T_b, pB_i in pB_cache.items():
                            for w in w_grid:
                                p_i = w * pA_i + (1 - w) * pB_i
                                for thr in t_grid:
                                    yhat = (p_i >= thr).astype(int)
                                    f1 = f1_score(yte, yhat, average='macro')
                                    acc = accuracy_score(yte, yhat)
                                    if (f1 > best["score_f1"]) or (f1 == best["score_f1"] and acc > best["score_acc"]):
                                        best.update({
                                            "score_f1": float(f1),
                                            "score_acc": float(acc),
                                            "A_params": dict(Ap),
                                            "B_params": dict(Bp),
                                            "w": float(w),
                                            "iA": int(iA),
                                            "iB": int(iB),
                                            "thr": float(thr),
                                            "Ta": float(T_a), "Tb": float(T_b)
                                        })
                                        best_models["A"] = clf_A
                                        best_models["B"] = clf_B
    # === 기존 HPO 루프 교체 끝 ===

    pA_opt = predict_calibrated(best_models["A"], Xte_A, num_iteration = best["iA"], T=best.get("Ta", 1.0))
    pB_opt = predict_calibrated(best_models["B"], Xte_B, iteration_range=(0,best["iB"]), T=best.get("Tb", 1.0))
    p_ens  = best["w"] * pA_opt + (1 - best["w"]) * pB_opt
    yhat   = (p_ens >= best["thr"]).astype(int)

    report = {
    "hpo_best": {
        "A_params": best["A_params"],
        "B_params": best["B_params"],
        "w": best["w"], "iA": best["iA"], "iB": best["iB"],
        "thr": best["thr"], "Ta": best.get("Ta", 1.0), "Tb": best.get("Tb", 1.0),
        "F1": best["score_f1"], "ACC": best["score_acc"],
        "ROC_AUC": float(roc_auc_score(yte, p_ens)),
        "PR_AUC":  float(average_precision_score(yte, p_ens))
    },
    "n_features": {"A": int(Xtr_A.shape[1]), "B": int(Xtr_B.shape[1])}
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    # ===== 저장(최적만) =====
    # 추론 시 동일 반복/가중치 사용하도록 리포트에 포함해둠
    joblib.dump(best_models["A"], "best_clf_A_lgb.pkl")
    joblib.dump(best_models["B"], "best_clf_B_xgb.pkl")

    preproc = {
        "global_tox": float(global_tox),
        "num_cols": num_cols,
        "scaler_num": scaler_num,
        "frag2tox": {k: float(v) for k, v in frag2tox.items()},
        "scaf2tox": {k: float(v) for k, v in scaf2tox.items()},
        "scaf2n":   {k: int(v)   for k, v in scaf2n.items()},
        "selected_fp_cols": [str(c) for c in sel_cols],
        "n_fp_selected": int(len(sel_cols)),
        "brics_min_support": int(args.min_frag_support),
        "scaf_min_support": int(args.min_scaf_support),
        "fp_bit_tox_map": {col: float(p) for col, p in zip(sel_cols, bit_tox_sel)},
        # 예측 시 필요한 앙상블 정보
        "ensemble": {
            "w": float(best["w"]),
            "iA": int(best["iA"]),
            "iB": int(best["iB"]),
            "threshold": float(best["thr"]),
            "Ta": float(best.get("Ta", 1.0)),
            "Tb": float(best.get("Tb", 1.0))
        }
    }
    joblib.dump(preproc, "best_preproc_artifacts.pkl")
    with open("best_ensemble_report.json","w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("💾 Saved: best_clf_A_lgb.pkl, best_clf_B_xgb.pkl, best_preproc_artifacts.pkl, best_ensemble_report.json")

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
    args = ap.parse_args()
    main(args)
# ===================== 끝 =====================