"""Tiền xử lý văn bản cho tin nhắn SMS tiếng Anh.

File này là bản sao của hàm heavy_clean trong notebook (Cell 2.2). Huấn luyện (train_model.py)
và web (app.py) đều dùng chung file này để tin nhắn được xử lý GIỐNG HỆT lúc huấn luyện.
"""
import html
import re
import unicodedata

# Từ điển từ lóng SMS -> dạng đầy đủ
SLANG_EN = {
    "u": "you", "ur": "your", "r": "are", "pls": "please", "plz": "please",
    "thx": "thanks", "tks": "thanks", "2day": "today", "2moro": "tomorrow",
    "2nite": "tonight", "b4": "before", "gr8": "great", "l8r": "later",
    "msg": "message", "txt": "text", "im": "i am", "dont": "do not",
    "cant": "can not", "wont": "will not", "n": "and", "luv": "love",
    "ok": "okay", "k": "okay", "ya": "you", "wat": "what", "wot": "what",
    "ive": "i have", "youre": "you are", "thats": "that is",
    "didnt": "did not", "doesnt": "does not", "isnt": "is not",
    "wasnt": "was not", "id": "i would",
}

# Các mẫu nhận diện "thực thể đặc biệt" thường gặp trong tin rác
URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b\S+\.(?:com|net|org|co\.uk|info|biz)\b\S*)", re.I)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s-]?)?\d{9,11}(?!\d)")
SHORT_RE = re.compile(r"(?<!\d)(?!(?:19|20)\d{2}(?!\d))\d{4,6}(?!\d)")  # mã dịch vụ 4-6 số, bỏ qua năm
MONEY_RE = re.compile(r"[£$€]")
REPEAT_RE = re.compile(r"(.)\1{2,}")


def heavy_clean(text) -> str:
    """Làm sạch mạnh cho TF-IDF: chữ thường, thay thực thể đặc biệt bằng token, bỏ dấu câu, thay từ lóng."""
    text = unicodedata.normalize("NFC", html.unescape(str(text))).lower()
    text = URL_RE.sub(" urltoken ", text)
    text = EMAIL_RE.sub(" emailtoken ", text)
    text = PHONE_RE.sub(" phonetoken ", text)
    text = SHORT_RE.sub(" shortcodetoken ", text)
    text = MONEY_RE.sub(" moneysymbol ", text)
    text = REPEAT_RE.sub(r"\1\1", text)
    text = text.replace("'", "").replace("\u2019", "")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = " ".join(SLANG_EN.get(t, t) for t in text.split())
    text = re.sub(r"\d+", " numtoken ", text)
    return re.sub(r"\s+", " ", text).strip()
