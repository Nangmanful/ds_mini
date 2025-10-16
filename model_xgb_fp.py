# train_xgb_fp_only_macro.py
# 목적: BRICS/Scaffold 제거. FP(ECFP/FCFP/PTFP) + Numeric으로만 XGBoost 학습/HPO(macro-F1) 후 최적 모델 저장.

import argparse, json, joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import chi2
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    accuracy_score, classification_report
)

import xgboost as xgb


# -------------------- 유틸 --------------------
def detect_columns(df: pd.DataFrame):
    # SMILES/label은 여기선 쓰지 않지만, 컬럼 검증용으로 간단히 유지할 수 있음(없어도 모델에는 영향 없음).
    smiles_col = next((c for c in df.columns if c.lower() in ("smiles","smile","smiles_code","smile_code")), None)
    label_col  = next((c for c in df.columns if "label" in c.lower()), None)
    if label_col is None:
        raise ValueError("label 컬럼을 찾지 못했습니다.")
    # numeric
    num_cols = [c for c in ['MolWt','clogp','qed','sa_score'] if c in df.columns]
    # fingerprints: ecfp_* 우선, 없으면 fcfp_/ptfp_ 포함
    ecfp_cols = [c for c in df.columns if c.lower().startswith('ecfp_')]
    if len(ecfp_cols) > 0:
        fp_cols = ecfp_cols
    else:
        fp_cols = [c for c in df.columns if any(c.lower().startswith(p) for p in ["ecfp_","fcfp_","ptfp_"])]
    return smiles_col, label_col, num_cols, fp_cols


def fp_select_and_scores(train_df: pd.DataFrame, test_df: pd.DataFrame,
                         fp_cols: list, label_col: str,
                         lower=0.01, upper=0.99, topk=256):
    """
    1) 활성화율(lower, upper)로 극단 비트 제거
    2) χ²로 상위 topk 선택
    3) 선택 비트에 대해 bit_tox(= 양성 비율)로 각 샘플의 mean/max FP 점수 산출
    4) 반환: 선택 비트 행렬 + FP mean/max + y
    """
    ytr = train_df[label_col].values.astype(int)
    yte = test_df[label_col].values.astype(int)

    if len(fp_cols) == 0:
        X_fp_tr = np.zeros((len(train_df), 0), dtype=np.float32)
        X_fp_te = np.zeros((len(test_df), 0), dtype=np.float32)
        sel_idx, sel_cols = np.array([], dtype=int), []
        # FP 점수는 정보가 없으니 0.5로 중립값
        fp_mean_tr = np.full(len(train_df), 0.5, dtype=np.float32)
        fp_max_tr  = np.full(len(train_df), 0.5, dtype=np.float32)
        fp_mean_te = np.full(len(test_df),  0.5, dtype=np.float32)
        fp_max_te  = np.full(len(test_df),  0.5, dtype=np.float32)
        return (X_fp_tr, X_fp_te, ytr, yte, sel_idx, sel_cols,
                fp_mean_tr, fp_max_tr, fp_mean_te, fp_max_te)

    X_fp_tr = train_df[fp_cols].values.astype(np.float32)
    X_fp_te = test_df[fp_cols].values.astype(np.float32)

    # 1) 활성화율 필터
    act = X_fp_tr.mean(axis=0)
    mask = (act > lower) & (act < upper)
    if np.sum(mask) == 0:
        sel_idx, sel_cols = np.array([], dtype=int), []
        fp_mean_tr = np.full(len(train_df), 0.5, dtype=np.float32)
        fp_max_tr  = np.full(len(train_df), 0.5, dtype=np.float32)
        fp_mean_te = np.full(len(test_df),  0.5, dtype=np.float32)
        fp_max_te  = np.full(len(test_df),  0.5, dtype=np.float32)
        return (X_fp_tr, X_fp_te, ytr, yte, sel_idx, sel_cols,
                fp_mean_tr, fp_max_tr, fp_mean_te, fp_max_te)

    # 2) χ² 상위 topk
    chi_vals, _ = chi2(X_fp_tr[:, mask], ytr)
    K = min(topk, int(np.sum(mask)))
    top_local = np.argsort(chi_vals)[::-1][:K]
    sel_idx = np.where(mask)[0][top_local]
    sel_cols = [fp_cols[i] for i in sel_idx]

    # 3) bit 독성률 기반 mean/max 점수
    bit_pos = (X_fp_tr[:, sel_idx].T @ ytr)              # 해당 비트가 1인 샘플 중 양성 개수
    bit_cnt = np.maximum(1, X_fp_tr[:, sel_idx].sum(axis=0))
    bit_tox = bit_pos / bit_cnt                          # P(toxic | bit=1)

    def fp_scores(X):
        mean_s, max_s = [], []
        sel = X[:, sel_idx]
        for row in sel:
            idx = np.where(row > 0)[0]
            if len(idx) == 0:
                mean_s.append(0.5); max_s.append(0.5)
            else:
                s = bit_tox[idx]
                mean_s.append(float(np.mean(s))); max_s.append(float(np.max(s)))
        return np.array(mean_s, dtype=np.float32), np.array(max_s, dtype=np.float32)

    fp_mean_tr, fp_max_tr = fp_scores(X_fp_tr)
    fp_mean_te, fp_max_te = fp_scores(X_fp_te)
    return (X_fp_tr, X_fp_te, ytr, yte, sel_idx, sel_cols,
            fp_mean_tr, fp_max_tr, fp_mean_te, fp_max_te)


def assemble_features(X_num_tr, X_num_te,
                      X_fp_tr, X_fp_te, sel_idx,
                      fp_mean_tr, fp_max_tr, fp_mean_te, fp_max_te):
    """
    최종 특성: [FP_mean, FP_max, Numeric, Selected FP bits]
    """
    def pack(Xn, Xfp, fpm, fpx):
        blocks = [
            fpm.reshape(-1,1), fpx.reshape(-1,1), Xn,
            Xfp[:, sel_idx] if Xfp.shape[1] > 0 and len(sel_idx) > 0 else np.zeros((len(Xn),0), dtype=np.float32)
        ]
        return np.hstack(blocks).astype(np.float32)

    Xtr = pack(X_num_tr, X_fp_tr, fp_mean_tr, fp_max_tr)
    Xte = pack(X_num_te, X_fp_te, fp_mean_te, fp_max_te)
    return Xtr, Xte


# -------------------- 메인(XGBoost + HPO with macro-F1) --------------------
def main(args):
    df = pd.read_csv(args.data)
    smiles_col, label_col, num_cols, fp_cols = detect_columns(df)

    train_df, test_df = train_test_split(df, test_size=args.test_size,
                                         stratify=df[label_col], random_state=args.rs)

    # Numeric
    if len(num_cols) > 0:
        scaler = StandardScaler()
        X_num_tr = scaler.fit_transform(train_df[num_cols].fillna(0).values)
        X_num_te = scaler.transform(test_df[num_cols].fillna(0).values)
    else:
        scaler = None
        X_num_tr = np.zeros((len(train_df), 0), dtype=np.float32)
        X_num_te = np.zeros((len(test_df), 0), dtype=np.float32)

    # Fingerprints 선택 및 점수
    (X_fp_tr, X_fp_te, ytr, yte, sel_idx, sel_cols,
     fp_mean_tr, fp_max_tr, fp_mean_te, fp_max_te) = fp_select_and_scores(
        train_df, test_df, fp_cols, label_col,
        lower=args.fp_lower, upper=args.fp_upper, topk=args.fp_topk
    )

    # 최종 특성 조립 (BRICS/Scaffold 없음)
    Xtr, Xte = assemble_features(
        X_num_tr, X_num_te, X_fp_tr, X_fp_te, sel_idx,
        fp_mean_tr, fp_max_tr, fp_mean_te, fp_max_te
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

                # 반복수 선택: 50 step로 iteration_range 스캔
                nB = clf.n_estimators
                for i in range(50, nB + 1, 50):
                    yprob_i = clf.predict_proba(Xte, iteration_range=(0, i))[:, 1]
                    yhat_i  = (yprob_i >= 0.5).astype(int)

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
                            "params": {**base, "subsample": ss, "colsample_bytree": cs},
                            "iter": int(i)
                        })
                        best_model = clf

    # ---------------- 최종 평가(최적 반복수) ----------------
    assert best_model is not None, "HPO에서 최적 모델을 찾지 못했습니다."
    yprob = best_model.predict_proba(Xte, iteration_range=(0, best["iter"]))[:, 1]
    yhat  = (yprob >= 0.5).astype(int)

    print("\n=== XGBoost FP-only (HPO Best) Metrics ===")
    print("Best params:", best["params"], "best_iter:", best["iter"])
    print("ROC AUC  :", round(roc_auc_score(yte, yprob), 4))
    print("PR  AUC :", round(average_precision_score(yte, yprob), 4))
    print("F1(macro):", round(f1_score(yte, yhat, average='macro'), 4))
    print("ACC      :", round(accuracy_score(yte, yhat), 4))
    print("\nClassification report:\n", classification_report(yte, yhat))

    # 50트리 간격 로그(최종 모델로만)
    print("\n--- Intermediate Scores (best model only) ---")
    for i in range(50, best_model.n_estimators + 1, 50):
        yprob_i = best_model.predict_proba(Xte, iteration_range=(0, i))[:, 1]
        yhat_i  = (yprob_i >= 0.5).astype(int)
        f1_macro_i = f1_score(yte, yhat_i, average="macro")
        acc_i = accuracy_score(yte, yhat_i)
        mark = " <-- used" if i == best["iter"] else ""
        print(f"[{i} trees] F1(macro)={f1_macro_i:.4f}, ACC={acc_i:.4f}{mark}")

    # ---------------- 저장(최적만) ----------------
    joblib.dump(best_model, "toxicity_model_xgb_fp_only.pkl")
    artifacts = {
        "n_fp_selected": int(len(sel_idx)),
        "selected_fp_cols": [str(c) for c in sel_cols],
        "numeric_cols": num_cols,
        "scaler_numeric": bool(len(num_cols) > 0),
        "best_params": best["params"],
        "best_iter": best["iter"],
        "select_metric": select_metric,
        "f1_macro": best["f1_macro"],
        "acc": best["acc"]
    }
    with open("model_artifacts_xgb_fp_only.json", "w", encoding="utf-8") as f:
        json.dump(artifacts, f, ensure_ascii=False, indent=2)
    print("\n💾 Saved: toxicity_model_xgb_fp_only.pkl, model_artifacts_xgb_fp_only.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="train.csv")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--rs", type=int, default=42)
    ap.add_argument("--fp_lower", type=float, default=0.01)
    ap.add_argument("--fp_upper", type=float, default=0.99)
    ap.add_argument("--fp_topk", type=int, default=512)   # 느리면 128/64로
    ap.add_argument("--select_metric", type=str, default="f1", choices=["f1","acc"])
    args = ap.parse_args()
    main(args)
