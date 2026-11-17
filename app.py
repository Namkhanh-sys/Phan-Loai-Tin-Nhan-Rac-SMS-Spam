"""Web demo: phát hiện tin nhắn rác (SMS spam) bằng TF-IDF + Logistic Regression.

Chạy:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from preprocessing import heavy_clean

BASE = Path(__file__).parent
MODEL_PATH = BASE / "model" / "spam_model.joblib"
META_PATH = BASE / "model" / "meta.json"

st.set_page_config(page_title="Phát hiện tin nhắn rác", page_icon="📩", layout="wide")

# Streamlit mặc định làm mờ chữ phụ (caption, nhãn số liệu, tab chưa chọn) nên khó đọc ở giao diện sáng.
# Các quy tắc dưới đây cho chúng dùng đúng màu chữ của giao diện hiện tại (sáng hay tối đều dùng được).
st.markdown(
    """
    <style>
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {
        color: inherit !important; opacity: 1 !important; font-size: 0.95rem !important;
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
        color: inherit !important; opacity: 1 !important; font-size: 1rem !important;
    }
    [data-testid="stTabs"] button { opacity: 1 !important; }
    [data-testid="stTabs"] button * { color: inherit !important; font-size: 1.05rem !important; }
    [data-testid="stTabs"] button[aria-selected="true"] * { font-weight: 700 !important; }
    textarea::placeholder { opacity: 0.75 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

EXAMPLES = {
    "Ví dụ spam": "URGENT! You have won a £1000 cash prize! Call 09061701461 now to claim. Claim code KL341.",
    "Ví dụ ham": "Ok lor... Joking wif u oni. Are you coming to dinner tonight?",
    "Ví dụ khó": "Your bill at 3 is £33.65 so thats not bad",
}


@st.cache_resource(show_spinner=False)
def load_artifacts():
    bundle = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}
    return bundle["vectorizer"], bundle["model"], meta


if not MODEL_PATH.exists():
    st.error("Chưa có mô hình. Hãy chạy `python train_model.py` trước (xem README.md), rồi tải lại trang.")
    st.stop()

vec, clf, meta = load_artifacts()
FEATURE_NAMES = vec.get_feature_names_out()


def spam_proba(texts):
    """Trả về (danh sách văn bản đã làm sạch, ma trận TF-IDF, xác suất spam)."""
    cleaned = [heavy_clean(t) for t in texts]
    X = vec.transform(cleaned)
    return cleaned, X, clf.predict_proba(X)[:, 1]


def contributions(X_row):
    """Đóng góp của từng n-gram vào điểm spam (log-odds) = trọng số TF-IDF x hệ số của mô hình."""
    row = X_row.tocoo()
    if row.nnz == 0:
        return pd.DataFrame(columns=["n-gram", "Đóng góp"])
    return pd.DataFrame({"n-gram": FEATURE_NAMES[row.col], "Đóng góp": row.data * clf.coef_[0][row.col]})


# ------------------------------ Thanh bên ------------------------------
with st.sidebar:
    st.header("⚙️ Cài đặt")
    threshold = st.slider("Ngưỡng gán nhãn SPAM", 0.05, 0.95, 0.50, 0.05,
                          help="Tin có xác suất spam ≥ ngưỡng sẽ bị gán là spam. "
                               "Hạ ngưỡng: bắt được nhiều spam hơn nhưng dễ báo nhầm tin thường.")
    st.caption(f"Mô hình: {meta.get('model', 'TF-IDF + Logistic Regression')}")
    if "metrics_on_test" in meta:
        m_side = meta["metrics_on_test"]
        st.metric("Độ chính xác (Accuracy)", f"{m_side['accuracy']:.2%}", help="Tính trên tập kiểm tra. Dữ liệu mất cân bằng (spam ~12.6%) nên cần xem thêm F1, Precision, Recall.")
        st.caption(f"F1 (spam): {m_side['f1']:.4f} | Precision: {m_side['precision']:.4f} | Recall: {m_side['recall']:.4f}")

st.title("📩 Phát hiện tin nhắn rác (SMS Spam)")
st.caption("Bài tập lớn Nhập môn học máy – Đề tài 3. Mô hình: TF-IDF (1-2 gram) + Logistic Regression.")

tab1, tab2, tab3 = st.tabs(["🔍 Kiểm tra một tin", "📋 Nhiều tin cùng lúc", "📊 Về mô hình"])

# ------------------------------ Tab 1 ------------------------------
with tab1:
    if "msg_input" not in st.session_state:
        st.session_state["msg_input"] = ""

    def set_example(text):
        st.session_state["msg_input"] = text

    cols = st.columns(len(EXAMPLES))
    for col, (name, text) in zip(cols, EXAMPLES.items()):
        col.button(name, on_click=set_example, args=(text,), width="stretch")

    msg = st.text_area("Nhập nội dung tin nhắn (tiếng Anh):", key="msg_input", height=130,
                       placeholder="Ví dụ: Congratulations! You have won a free prize, call now...")

    if st.button("Phân loại", type="primary"):
        if not msg.strip():
            st.warning("Hãy nhập nội dung tin nhắn.")
        else:
            cleaned, X, probs = spam_proba([msg])
            p = float(probs[0])
            is_spam = p >= threshold

            if is_spam:
                st.error(f"🚨 **SPAM** – xác suất spam: **{p:.1%}**")
            else:
                st.success(f"✅ **HAM (tin bình thường)** – xác suất spam: **{p:.1%}**")
            st.progress(min(max(p, 0.0), 1.0))
            conf = p if is_spam else 1 - p
            st.caption(f"Ngưỡng hiện tại: {threshold:.2f}. Xác suất spam = {p:.4f}. Độ tin cậy của nhãn dự đoán: {conf:.1%}")

            with st.expander("Văn bản sau tiền xử lý", expanded=False):
                st.code(cleaned or "(rỗng)")
                st.caption("Số điện thoại, đường link, mã dịch vụ, số và ký hiệu tiền được thay bằng token "
                           "(phonetoken, urltoken, shortcodetoken, numtoken, moneysymbol).")

            contrib = contributions(X)
            if contrib.empty:
                st.info("Không có từ nào của tin nhắn nằm trong từ vựng của mô hình nên không có gì để giải thích.")
            else:
                st.subheader("Vì sao mô hình quyết định như vậy?")
                left, right = st.columns(2)
                spam_df = contrib[contrib["Đóng góp"] > 0].nlargest(8, "Đóng góp")
                ham_df = contrib[contrib["Đóng góp"] < 0].nsmallest(8, "Đóng góp")
                with left:
                    st.markdown("**Đẩy về phía SPAM**")
                    if spam_df.empty:
                        st.caption("Không có n-gram nào.")
                    else:
                        st.bar_chart(spam_df.set_index("n-gram"), color="#e4572e")
                with right:
                    st.markdown("**Đẩy về phía HAM**")
                    if ham_df.empty:
                        st.caption("Không có n-gram nào.")
                    else:
                        st.bar_chart(ham_df.assign(**{"Đóng góp": ham_df["Đóng góp"].abs()}).set_index("n-gram"),
                                     color="#2e9e6f")
                st.caption("Đóng góp = trọng số TF-IDF × hệ số Logistic Regression (đơn vị log-odds, chưa tính hệ số chặn).")

# ------------------------------ Tab 2 ------------------------------
with tab2:
    st.write("Dán nhiều tin nhắn, **mỗi dòng một tin**:")
    batch_text = st.text_area("Danh sách tin nhắn", height=200, key="batch_input",
                              placeholder="Free entry in 2 a wkly comp to win FA Cup final tkts...\nOk lor... Joking wif u oni...")
    if st.button("Phân loại tất cả"):
        lines = [l.strip() for l in batch_text.splitlines() if l.strip()]
        if not lines:
            st.warning("Hãy nhập ít nhất một tin nhắn.")
        else:
            _, _, probs = spam_proba(lines)
            result = pd.DataFrame({"Tin nhắn": lines, "Xác suất spam": probs.round(4),
                                   "Nhãn": ["SPAM" if p >= threshold else "HAM" for p in probs]})
            st.dataframe(result, width="stretch", hide_index=True)
            st.caption(f"Tổng: {len(result)} tin, trong đó {int((result['Nhãn'] == 'SPAM').sum())} tin bị gán là SPAM.")
            st.download_button("Tải kết quả (CSV)", result.to_csv(index=False).encode("utf-8-sig"),
                               "ket_qua_phan_loai.csv", "text/csv")

# ------------------------------ Tab 3 ------------------------------
with tab3:
    st.subheader("Hiệu suất trên tập kiểm tra")
    if "metrics_on_test" in meta:
        m = meta["metrics_on_test"]
        c = st.columns(5)
        for col, (label, key) in zip(c, [("Accuracy", "accuracy"), ("Precision (spam)", "precision"),
                                         ("Recall (spam)", "recall"), ("F1 (spam)", "f1"), ("ROC-AUC", "roc_auc")]):
            col.metric(label, f"{m[key]:.4f}")
        tn, fp = meta["confusion_matrix"][0]
        fn, tp = meta["confusion_matrix"][1]
        st.dataframe(pd.DataFrame([[tn, fp], [fn, tp]], index=["Thực tế: ham", "Thực tế: spam"],
                                  columns=["Dự đoán: ham", "Dự đoán: spam"]), width="stretch")
        st.caption(f"Dữ liệu: {meta['n_samples']} tin ({meta['spam_ratio']:.1%} spam); train {meta['n_train']}, "
                   f"test {meta['n_test']}; {meta['n_features']} đặc trưng TF-IDF.")
        if meta.get("trained_on_all_data"):
            st.caption("Lưu ý: mô hình đang chạy được huấn luyện lại trên toàn bộ dữ liệu; số liệu ở trên là của "
                       "mô hình huấn luyện trên tập train.")

    st.subheader("N-gram quan trọng nhất của mô hình")
    coef = pd.Series(clf.coef_[0], index=FEATURE_NAMES)
    a, b = st.columns(2)
    a.markdown("**Thiên về SPAM**")
    a.dataframe(coef.nlargest(15).rename("Hệ số").round(3).to_frame(), width="stretch")
    b.markdown("**Thiên về HAM**")
    b.dataframe(coef.nsmallest(15).rename("Hệ số").round(3).to_frame(), width="stretch")

    st.subheader("Quy trình")
    st.markdown(
        "1. **Tiền xử lý**: chữ thường, thay URL/số điện thoại/mã dịch vụ/số/tiền bằng token, bỏ dấu câu, chuẩn hóa từ lóng.\n"
        "2. **TF-IDF** với unigram + bigram, `min_df=2`, `sublinear_tf=True`.\n"
        "3. **Logistic Regression** với `class_weight=\"balanced\"` để xử lý mất cân bằng lớp (spam chỉ ~12.6%)."
    )
