"""Huấn luyện mô hình TF-IDF + Logistic Regression và lưu lại cho web.

Cách dùng:
    python train_model.py                      # đọc data/train.csv
    python train_model.py duong/dan/file.csv   # hoặc chỉ định file khác
    python train_model.py --all                # huấn luyện lại trên TOÀN BỘ dữ liệu sau khi đánh giá

Quy trình giống hệt notebook: loại trùng -> tiền xử lý -> chia 80/20 (stratify, random_state=42)
-> fit TF-IDF chỉ trên tập train -> Logistic Regression (class_weight="balanced").
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from preprocessing import heavy_clean

BASE = Path(__file__).parent
args = [a for a in sys.argv[1:] if not a.startswith("--")]
USE_ALL = "--all" in sys.argv
DATA_PATH = Path(args[0]) if args else BASE / "data" / "train.csv"

TEXT_CANDIDATES = ["sms", "message", "text", "v2", "Message", "SMS"]
LABEL_CANDIDATES = ["label", "Label", "category", "Category", "v1", "class"]


def pick(columns, candidates, what):
    for c in candidates:
        if c in columns:
            return c
    raise SystemExit(f"Không tìm thấy cột {what}. Các cột hiện có: {list(columns)}")


# 1. Đọc dữ liệu
try:
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(DATA_PATH, encoding="latin-1")
text_col = pick(df.columns, TEXT_CANDIDATES, "nội dung tin nhắn")
label_col = pick(df.columns, LABEL_CANDIDATES, "nhãn")
df = df[[label_col, text_col]].rename(columns={label_col: "label", text_col: "message"})
print(f"Đã đọc {len(df)} dòng từ {DATA_PATH}")

# 2. Làm sạch cấu trúc + mã hóa nhãn (1 = spam, 0 = ham)
df = df.dropna(subset=["message", "label"]).copy()
df["message"] = df["message"].astype(str).str.strip()
df = df.drop_duplicates(subset=["message"]).reset_index(drop=True)
if df["label"].dtype == object:
    df["y"] = df["label"].astype(str).str.strip().str.lower().map({"spam": 1, "ham": 0, "1": 1, "0": 0})
else:
    df["y"] = df["label"].astype(int)
df = df.dropna(subset=["y"]).reset_index(drop=True)
df["y"] = df["y"].astype(int)

# 3. Tiền xử lý, bỏ các dòng rỗng sau khi làm sạch
df["text_clean"] = df["message"].apply(heavy_clean)
df = df[df["text_clean"].str.len() > 0].reset_index(drop=True)
print(f"Số mẫu sau tiền xử lý: {len(df)} (spam {df['y'].mean()*100:.2f}%)")

# 4. Chia train/test
idx = np.arange(len(df))
idx_train, idx_test = train_test_split(idx, test_size=0.2, stratify=df["y"], random_state=42)
X_tr_txt, X_te_txt = df["text_clean"].iloc[idx_train], df["text_clean"].iloc[idx_test]
y_tr, y_te = df["y"].iloc[idx_train].values, df["y"].iloc[idx_test].values

# 5. TF-IDF (fit chỉ trên train) + Logistic Regression
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
Xtr = vec.fit_transform(X_tr_txt)
Xte = vec.transform(X_te_txt)
clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
clf.fit(Xtr, y_tr)

# 6. Đánh giá trên tập test
y_pred = clf.predict(Xte)
y_prob = clf.predict_proba(Xte)[:, 1]
cm = confusion_matrix(y_te, y_pred)
metrics = {
    "accuracy": accuracy_score(y_te, y_pred),
    "precision": precision_score(y_te, y_pred),
    "recall": recall_score(y_te, y_pred),
    "f1": f1_score(y_te, y_pred),
    "roc_auc": roc_auc_score(y_te, y_prob),
}
print("\n=== Kết quả trên tập test ===")
for k, v in metrics.items():
    print(f"{k:10s}: {v:.4f}")
print("Confusion matrix [[TN, FP], [FN, TP]]:", cm.tolist())
print("(Kết quả trong báo cáo: F1 = 0.9690, Precision = 0.9843, Recall = 0.9542, FP = 2, FN = 6)")

# 7. (Tùy chọn) huấn luyện lại trên toàn bộ dữ liệu để dùng cho web
if USE_ALL:
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_all = vec.fit_transform(df["text_clean"])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42).fit(X_all, df["y"].values)
    print("\nĐã huấn luyện lại trên toàn bộ dữ liệu (--all). Số liệu ở trên vẫn là của mô hình huấn luyện trên tập train.")

# 8. Lưu mô hình và thông tin mô tả
out = BASE / "model"
out.mkdir(exist_ok=True)
joblib.dump({"vectorizer": vec, "model": clf}, out / "spam_model.joblib")
meta = {
    "model": "TF-IDF (1-2 gram) + Logistic Regression",
    "n_samples": int(len(df)), "n_train": int(len(idx_train)), "n_test": int(len(idx_test)),
    "n_features": int(len(vec.get_feature_names_out())),
    "spam_ratio": float(df["y"].mean()),
    "metrics_on_test": {k: float(v) for k, v in metrics.items()},
    "confusion_matrix": cm.tolist(),
    "trained_on_all_data": USE_ALL,
    "sklearn_version": sklearn.__version__,
}
(out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nĐã lưu mô hình vào {out / 'spam_model.joblib'}")
