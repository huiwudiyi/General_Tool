from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import jieba
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


def jieba_tokenizer(text: str) -> List[str]:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return list(jieba.cut(text))


def build_models(random_state: int, use_class_balance: bool) -> Dict[str, object]:
    class_weight = "balanced" if use_class_balance else None
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=random_state, class_weight=class_weight),
        "linear_svm": LinearSVC(random_state=random_state, class_weight=class_weight),
        "multinomial_nb": MultinomialNB(),
        "decision_tree": DecisionTreeClassifier(random_state=random_state, class_weight=class_weight),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1, class_weight=class_weight),
    }


def top_features_from_weights(weights: np.ndarray, feature_names: np.ndarray, topk: int) -> List[Dict[str, float]]:
    idx = np.argsort(weights)[-topk:][::-1]
    return [{"feature": str(feature_names[i]), "weight": float(weights[i])} for i in idx]


def extract_features(pipe: Pipeline, topk: int) -> Dict[str, List[Dict[str, float]]]:
    tfidf: TfidfVectorizer = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    feature_names = tfidf.get_feature_names_out()

    out: Dict[str, List[Dict[str, float]]] = {}
    if hasattr(clf, "coef_"):
        coef = clf.coef_
        if coef.ndim == 1 or coef.shape[0] == 1:
            out["positive"] = top_features_from_weights(coef.ravel(), feature_names, topk)
            neg = top_features_from_weights(-coef.ravel(), feature_names, topk)
            out["negative"] = [{"feature": d["feature"], "weight": -d["weight"]} for d in neg]
        else:
            for i, cls in enumerate(clf.classes_):
                out[str(cls)] = top_features_from_weights(coef[i], feature_names, topk)
    elif hasattr(clf, "feature_importances_"):
        out["importance"] = top_features_from_weights(clf.feature_importances_, feature_names, topk)
    else:
        out["note"] = [{"feature": "N/A", "weight": 0.0}]
    return out


def evaluate(y_true: List[str], y_pred: List[str]) -> Dict[str, object]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "report": classification_report(y_true, y_pred, digits=4, output_dict=True),
    }


def ensemble_predict(pred_map: Dict[str, np.ndarray]) -> np.ndarray:
    pred_matrix = np.vstack(list(pred_map.values())).T
    final = []
    for row in pred_matrix:
        values, counts = np.unique(row, return_counts=True)
        final.append(values[np.argmax(counts)])
    return np.array(final)


def get_model_scores(pipe: Pipeline, x_test: List[str]) -> np.ndarray:
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        return pipe.predict_proba(x_test).max(axis=1)
    if hasattr(clf, "decision_function"):
        df = pipe.decision_function(x_test)
        if isinstance(df, list):
            df = np.array(df)
        if getattr(df, "ndim", 1) == 1:
            return np.abs(df)
        return np.max(df, axis=1)
    return np.full(len(x_test), np.nan)




def get_class0_score(pipe: Pipeline, x_test: List[str], class_zero_label: str = "0") -> np.ndarray:
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        proba = pipe.predict_proba(x_test)
        classes = [str(c) for c in clf.classes_]
        if class_zero_label in classes:
            return proba[:, classes.index(class_zero_label)]
    if hasattr(clf, "decision_function"):
        raw = pipe.decision_function(x_test)
        classes = [str(c) for c in getattr(clf, "classes_", [])]
        if np.ndim(raw) == 1:
            # 二分类，假设 raw 对应 classes_[1]
            if len(classes) == 2 and class_zero_label == classes[0]:
                return 1.0 / (1.0 + np.exp(raw))
            return 1.0 / (1.0 + np.exp(-raw))
        if class_zero_label in classes:
            idx = classes.index(class_zero_label)
            ex = np.exp(raw - np.max(raw, axis=1, keepdims=True))
            sm = ex / np.sum(ex, axis=1, keepdims=True)
            return sm[:, idx]
    return np.full(len(x_test), np.nan)


def tune_threshold_for_class0(y_true: List[str], class0_score: np.ndarray, target_precision: float = 0.9) -> float:
    y = np.array([str(v) for v in y_true])
    candidates = np.unique(class0_score[~np.isnan(class0_score)])
    if len(candidates) == 0:
        return float("nan")
    best_t = 1.0
    best_recall = -1.0
    for t in candidates:
        pred0 = class0_score >= t
        tp = np.sum((pred0) & (y == "0"))
        fp = np.sum((pred0) & (y != "0"))
        fn = np.sum((~pred0) & (y == "0"))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision >= target_precision and recall > best_recall:
            best_recall = recall
            best_t = float(t)
    return best_t


def apply_class0_threshold(base_pred: np.ndarray, class0_score: np.ndarray, threshold: float, non0_label: str = 1) -> np.ndarray:
    out = np.array(base_pred, dtype=object)
    if np.isnan(threshold):
        return out
    out[class0_score >= threshold] = 0
    out[class0_score < threshold] = np.where(out[class0_score < threshold] == 0, non0_label, out[class0_score < threshold])
    return out

def get_local_evidence(pipe: Pipeline, x_test: List[str], preds: np.ndarray, topk: int) -> List[List[Dict[str, float]]]:
    tfidf: TfidfVectorizer = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    x_vec = tfidf.transform(x_test)
    feature_names = tfidf.get_feature_names_out()

    if not hasattr(clf, "coef_"):
        return [[{"feature": "N/A", "contribution": 0.0}] for _ in x_test]

    classes = list(clf.classes_)
    coefs = clf.coef_
    evidences: List[List[Dict[str, float]]] = []

    for i in range(x_vec.shape[0]):
        row = x_vec.getrow(i)
        pred = preds[i]
        class_idx = classes.index(pred)
        weights = coefs[0] if coefs.ndim == 1 or coefs.shape[0] == 1 else coefs[class_idx]
        contrib = row.multiply(weights).toarray().ravel()
        nonzero = np.where(contrib != 0)[0]
        if len(nonzero) == 0:
            evidences.append([{"feature": "N/A", "contribution": 0.0}])
            continue
        local_idx = nonzero[np.argsort(contrib[nonzero])[-topk:][::-1]]
        evidences.append([
            {"feature": str(feature_names[j]), "contribution": float(contrib[j])} for j in local_idx
        ])

    return evidences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="jieba + 多模型中文文本分类")
    parser.add_argument("-f", default="pass", help="pass")
    parser.add_argument("--train", default="data/工作写作标注结果.xlsx", help="输入 xlsx 文件路径")
    #output/验收数据/文创线上数据_清洗数据_vote.xlsx
    parser.add_argument("--test", default="output/验收数据/文创线上数据_清洗数据_vote.xlsx", help="输入 xlsx 文件路径")
    parser.add_argument("--train_col", default="随机Q", help="文本列名")
    parser.add_argument("--train_label", default="评分（0、1、2）", help="标签列名")
    parser.add_argument("--test_col", default="Query", help="文本列名")
    parser.add_argument("--test_label", default="评分", help="标签列名")
    parser.add_argument("--labels", default=[0,1,2], help="标签列名")
    parser.add_argument("--test-size", type=float, default=0.2, help="测试集比例")
    parser.add_argument("--random-state", type=int, default=42, help="随机种子")
    parser.add_argument("--topk-features", type=int, default=15, help="展示TopK重要特征")
    parser.add_argument("--output-dir", default="outputs/jieba_cls", help="结果输出目录")
    parser.add_argument(
        "--use-class-balance",
        default=True,
        help="启用 class_weight='balanced' 自动缓解类别不均衡（NB除外）",
    )
    return parser.parse_args()


# def main() -> None:


args = parse_args()
out_dir = Path(args.output_dir)
out_dir.mkdir(parents=True, exist_ok=True)

train_df = pd.read_excel(args.train)
train_df = train_df[train_df[args.train_label].isin(args.labels)]
train_df[args.train_label] = train_df[args.train_label].replace(2, 1)
x_train = train_df[args.train_col].tolist()
y_train = train_df[args.train_label].tolist()

test_df = pd.read_excel(args.test)
test_df[args.test_label] = 2
test_df = test_df[test_df[args.test_label].isin(args.labels)]
test_df[args.test_label] = test_df[args.test_label].replace(2, 1)
x_test = test_df[args.test_col].tolist()
y_test = test_df[args.test_label].tolist()
if args.train_col not in train_df.columns or args.train_label not in train_df.columns or args.test_col not in test_df.columns:
    raise ValueError(f"列名不存在，当前列为: {list(df.columns)}")

print("=" * 80)
print("trian 标签分布:")
print(train_df[args.train_label].value_counts(dropna=False))

print("=" * 80)
print("test 标签分布:")
print(test_df[args.test_label].value_counts(dropna=False))

models = build_models(random_state=args.random_state, use_class_balance=args.use_class_balance)
rows = []
pred_map: Dict[str, np.ndarray] = {}
score_map: Dict[str, np.ndarray] = {}
feature_map: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
local_evidence_map: Dict[str, List[List[Dict[str, float]]]] = {}

for name, estimator in models.items():
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(tokenizer=jieba_tokenizer, token_pattern=None, ngram_range=(1, 2), min_df=2)),
        ("clf", estimator),
    ])
    pipe.fit(x_train, y_train)
    pred = pipe.predict(x_test)
    score = get_model_scores(pipe, x_test)
    pred_map[name] = pred
    score_map[name] = score

    metric = evaluate(y_test, pred)
    rows.append({"model": name, "accuracy": metric["accuracy"], "macro_f1": metric["macro_f1"]})
    feature_map[name] = extract_features(pipe, args.topk_features)
    local_evidence_map[name] = get_local_evidence(pipe, x_test, pred, args.topk_features)

    with (out_dir / f"{name}_report.json").open("w", encoding="utf-8") as f:
        json.dump(metric["report"], f, ensure_ascii=False, indent=2)

    class0_score = get_class0_score(pipe, x_test, class_zero_label="0")
    threshold = tune_threshold_for_class0(y_test, class0_score, target_precision=0.9)
    adj_pred = apply_class0_threshold(pred, class0_score, threshold, non0_label=1)
    adj_metric = evaluate(y_test, adj_pred.tolist())
    rows.append({"model": f"{name}_threshold_for_class0", "accuracy": adj_metric["accuracy"], "macro_f1": adj_metric["macro_f1"]})

ens_pred = ensemble_predict(pred_map)
vote_count = np.vstack([(p == ens_pred).astype(int) for p in pred_map.values()]).sum(axis=0)
ens_score = vote_count / len(pred_map)

ens_metric = evaluate(y_test, ens_pred.tolist())
rows.append({"model": "ensemble_majority_vote", "accuracy": ens_metric["accuracy"], "macro_f1": ens_metric["macro_f1"]})
pred_map["ensemble_majority_vote"] = ens_pred
score_map["ensemble_majority_vote"] = ens_score

pd.DataFrame(rows).sort_values("macro_f1", ascending=False).to_csv(out_dir / "metrics_summary.csv", index=False)

with (out_dir / "key_features.json").open("w", encoding="utf-8") as f:
    json.dump(feature_map, f, ensure_ascii=False, indent=2)

with (out_dir / "local_evidence.json").open("w", encoding="utf-8") as f:
    json.dump(local_evidence_map, f, ensure_ascii=False, indent=2)

with (out_dir / "ensemble_report.json").open("w", encoding="utf-8") as f:
    json.dump(ens_metric["report"], f, ensure_ascii=False, indent=2)

pred_df = test_df.copy()
for name in pred_map:
    pred_df[f"pred_{name}"] = pred_map[name]
    pred_df[f"score_{name}"] = score_map[name]
pred_df.to_excel(args.test.replace(".xlsx", "_predictions.xlsx"), index=False)

print(f"完成。结果已保存到: {out_dir.resolve()}")
print(pd.DataFrame(rows).sort_values("macro_f1", ascending=False).to_string(index=False))


