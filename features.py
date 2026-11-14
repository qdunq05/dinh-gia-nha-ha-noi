"""
features.py — Xử lý dữ liệu & tạo đặc trưng, dùng CHUNG cho lúc train (Colab) và lúc chạy app (HF Space).
Giữ một file duy nhất để đảm bảo đặc trưng lúc train và lúc dự đoán giống hệt nhau.
"""
import re
import unicodedata

import numpy as np
import pandas as pd

RAW_COLS = [
    "name", "description", "property_type_name", "province_name", "district_name",
    "ward_name", "street_name", "project_name", "price", "area", "floor_count",
    "frontage_width", "house_depth", "road_width", "bedroom_count", "bathroom_count",
    "house_direction", "published_at",
]

LOAI_NHA = "Nhà riêng"
LOAI_CC = "Chung cư"
START_MONTH = pd.Timestamp("2025-06-01")


# ----------------------------------------------------------------------------- văn bản
def norm_text(s) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = unicodedata.normalize("NFC", str(s)).lower()
    return re.sub(r"\s+", " ", s).strip()


# Che giá trong văn bản để mô hình KHÔNG "đọc trộm" đáp án (tránh rò rỉ dữ liệu)
_NUM = r"\d+(?:[.,]\d+)*"
_NL = r"(?![^\W\d_])"  # không được nối tiếp bằng chữ cái (tránh "5 tiện ích", "1 trệt")
_PRICE_PATTERNS = [
    rf"{_NUM}\s*(?:tỷ|tỉ|ty|ti){_NL}(?:\s*{_NUM})?",                            # 8 tỷ, 8,2 tỷ, 8 tỷ 2, 8ty2
    rf"{_NUM}\s*(?:triệu|trieu|tr|củ|cu){_NL}(?:\s*/\s*(?:m2|m²|m))?",          # 950 triệu, 120tr/m2
    rf"{_NUM}\s*(?:nghìn|ngàn|k)\s*/\s*(?:m2|m²|tháng)",                         # 500k/m2
    rf"{_NUM}\s*(?:vnđ|vnd|đồng|đ){_NL}",                                       # 7.400.000.000 đ
    r"giá\s*(?:chỉ|chào|bán|tốt|rẻ|:)*\s*\[giá\]",
]
_PRICE_RE = [re.compile(p) for p in _PRICE_PATTERNS]


def mask_price(text: str) -> str:
    t = norm_text(text)
    for rx in _PRICE_RE:
        t = rx.sub("[giá]", t)
    return t


def make_text(name, description, max_chars: int = 700) -> str:
    """Văn bản đưa vào PhoBERT: tiêu đề + mô tả, đã che giá."""
    t = f"{norm_text(name)} . {norm_text(description)}"
    return mask_price(t)[:max_chars]


# ----------------------------------------------------------------------------- từ khoá
# (tên cột, nhãn tiếng Việt, regex, cụm từ chèn vào khi người dùng tick ô trên giao diện)
KEYWORDS = [
    ("kw_mat_pho", "Mặt phố / mặt đường", r"mặt phố|mặt đường|mặt tiền phố", "mặt phố"),
    ("kw_oto", "Ô tô đỗ cửa / ngõ ô tô", r"ô ?tô|oto|xe hơi", "ô tô đỗ cửa"),
    ("kw_kinh_doanh", "Kinh doanh được", r"kinh doanh|buôn bán", "kinh doanh tốt"),
    ("kw_lo_goc", "Lô góc / căn góc", r"lô góc|căn góc|góc 2 mặt|hai mặt thoáng|2 mặt thoáng|3 mặt thoáng", "lô góc"),
    ("kw_thang_may", "Có thang máy", r"thang máy", "có thang máy"),
    ("kw_noi_that", "Full nội thất", r"full nội thất|đầy đủ nội thất|nội thất đầy đủ|nội thất cao cấp|nội thất xịn", "full nội thất"),
    ("kw_view", "View hồ / sông / công viên", r"view hồ|view sông|view công viên|gần hồ|hướng hồ", "view hồ"),
    ("kw_phan_lo", "Phân lô", r"phân lô", "phân lô"),
    ("kw_ngo_nong", "Ngõ nông, gần phố", r"ngõ nông|gần phố|cách phố|ra phố", "ngõ nông gần phố"),
    ("kw_ban_gap", "Cần bán gấp / cắt lỗ", r"bán gấp|cần bán gấp|cắt lỗ|ngộp|giảm sâu|hạ giá", "cần bán gấp"),
    ("kw_dong_tien", "Có dòng tiền cho thuê", r"dòng tiền|đang cho thuê|cho thuê \d", "đang cho thuê có dòng tiền"),
    ("kw_xay_moi", "Nhà mới xây", r"mới xây|xây mới|nhà mới|mới tinh", "nhà mới xây"),
]
KW_COLS = [k[0] for k in KEYWORDS]
KW_LABEL = {k[0]: k[1] for k in KEYWORDS}
KW_PHRASE = {k[1]: k[3] for k in KEYWORDS}
_KW_RE = {k[0]: re.compile(k[2]) for k in KEYWORDS}

PHAP_LY = ["Sổ đỏ/sổ hồng", "HĐMB/đang chờ sổ", "Giấy tờ khác", "Không rõ"]
PHAP_LY_PHRASE = {
    "Sổ đỏ/sổ hồng": "sổ đỏ chính chủ",
    "HĐMB/đang chờ sổ": "hợp đồng mua bán đang chờ sổ",
    "Giấy tờ khác": "giấy tờ viết tay",
    "Không rõ": "",
}
_RE_HDMB = re.compile(r"chưa có sổ|chờ sổ|đang làm sổ|hđmb|hợp đồng mua bán")
_RE_KHAC = re.compile(r"viết tay|sổ chung|vi bằng|giấy tờ tay|sổ chung")
_RE_SO = re.compile(r"sổ đỏ|sổ hồng|sđcc|sổ riêng|có sổ|sổ vuông|sổ đẹp|sổ nở hậu|sẵn sổ")


def phap_ly_of(text: str) -> str:
    if _RE_HDMB.search(text):
        return "HĐMB/đang chờ sổ"
    if _RE_KHAC.search(text):
        return "Giấy tờ khác"
    if _RE_SO.search(text):
        return "Sổ đỏ/sổ hồng"
    return "Không rõ"


# ----------------------------------------------------------------------------- chuẩn hoá cột
def map_loai(t) -> str | None:
    t = norm_text(t)
    if "chung cư" in t or "căn hộ" in t:
        return LOAI_CC
    if t in {"nhà", "nhà riêng", "nhà mặt phố", "nhà trong ngõ", "nhà phố", "nhà mặt tiền", "nhà hẻm"}:
        return LOAI_NHA
    return None  # đất, biệt thự/liền kề, shophouse... -> bỏ


HUONG = ["Đông", "Tây", "Nam", "Bắc", "Đông Bắc", "Đông Nam", "Tây Bắc", "Tây Nam"]


def norm_huong(h) -> str:
    t = norm_text(h).replace("-", " ").replace("_", " ")
    t = re.sub(r"\s+", " ", t).strip()
    for v in HUONG:
        if t == v.lower():
            return v
    return "Không rõ"


def clean_place(s) -> str:
    t = unicodedata.normalize("NFC", str(s)).strip() if s is not None and not (isinstance(s, float) and np.isnan(s)) else ""
    t = re.sub(r"^(quận|huyện|thị xã|phường|xã|thị trấn)\s+", "", t, flags=re.I).strip()
    return t if t else "Không rõ"


# ----------------------------------------------------------------------------- làm sạch (chỉ dùng lúc train)
def clean_listings(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    log = []
    n0 = len(df)
    df = df.copy()
    df["loai"] = df["property_type_name"].map(map_loai)
    df = df[df["loai"].notna()]
    log.append(("Giữ nhà riêng + chung cư", len(df)))

    for c in ["price", "area", "floor_count", "frontage_width", "house_depth", "road_width", "bedroom_count", "bathroom_count"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["price"].notna() & df["area"].notna() & (df["price"] > 0)]
    log.append(("Có giá & diện tích", len(df)))

    a_ok = np.where(df["loai"] == LOAI_CC, df["area"].between(20, 300), df["area"].between(12, 500))
    df = df[a_ok]
    df["ppm2"] = df["price"] / df["area"]
    df = df[df["ppm2"].between(15e6, 1.5e9)]  # 15 triệu – 1,5 tỷ / m²
    log.append(("Diện tích & đơn giá hợp lý", len(df)))

    # loại trùng lặp (tin đăng lại nhiều lần) — giữ tin đăng sớm nhất
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df = df.sort_values("published_at")
    df["_k1"] = df["loai"] + "|" + df["district_name"].astype(str) + "|" + df["area"].round(0).astype(str) + "|" + df["price"].round(-7).astype(str)
    df["_k2"] = df["description"].map(lambda s: norm_text(s)[:300])
    df = df.drop_duplicates("_k1")
    df = df[(df["_k2"].str.len() < 30) | ~df["_k2"].duplicated()]
    log.append(("Bỏ tin trùng lặp", len(df)))

    # bỏ ngoại lai theo từng (loại, quận): ngoài khoảng 1%–99% log đơn giá
    df["y"] = np.log(df["ppm2"])
    g = df.groupby(["loai", "district_name"], dropna=False)["y"]
    lo, hi = g.transform(lambda s: s.quantile(0.01)), g.transform(lambda s: s.quantile(0.99))
    df = df[(df["y"] >= lo) & (df["y"] <= hi)]
    log.append(("Bỏ ngoại lai theo quận", len(df)))

    df = df.drop(columns=["_k1", "_k2"]).reset_index(drop=True)
    if verbose:
        print(f"Ban đầu: {n0:,}")
        for k, v in log:
            print(f"  → {k}: {v:,}")
    return df


# ----------------------------------------------------------------------------- đặc trưng
NUM_COLS = [
    "area", "log_area", "floor_count", "san_xd", "frontage_width", "house_depth", "road_width",
    "bedroom_count", "bathroom_count", "month_idx",
]
CAT_COLS = ["loai", "district", "ward", "project", "huong", "phap_ly"]


def base_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo các cột đặc trưng 'thô' (chưa gắn danh mục)."""
    out = pd.DataFrame(index=df.index)
    loai = df["loai"] if "loai" in df else df["property_type_name"].map(map_loai)
    out["loai"] = loai.fillna(LOAI_NHA)
    is_cc = out["loai"] == LOAI_CC
    num = lambda c: pd.to_numeric(df.get(c), errors="coerce") if c in df else pd.Series(np.nan, index=df.index)

    out["area"] = num("area")
    out["log_area"] = np.log(out["area"].clip(lower=1))
    fl = num("floor_count").where(lambda s: s.between(1, 15))
    out["floor_count"] = fl.where(~is_cc)
    out["san_xd"] = (out["area"] * out["floor_count"]).where(~is_cc)
    out["frontage_width"] = num("frontage_width").where(lambda s: s.between(1, 30)).where(~is_cc)
    out["house_depth"] = num("house_depth").where(lambda s: s.between(2, 100)).where(~is_cc)
    out["road_width"] = num("road_width").where(lambda s: s.between(0.5, 60)).where(~is_cc)
    out["bedroom_count"] = num("bedroom_count").where(lambda s: s.between(0, 15))
    out["bathroom_count"] = num("bathroom_count").where(lambda s: s.between(0, 15))

    pub = pd.to_datetime(df["published_at"], errors="coerce") if "published_at" in df else pd.Series(pd.NaT, index=df.index)
    out["month_idx"] = ((pub.dt.year - START_MONTH.year) * 12 + (pub.dt.month - START_MONTH.month)).astype(float)

    text = (df.get("name", "").fillna("").astype(str) + " . " + df.get("description", "").fillna("").astype(str)).map(norm_text)
    for col, rx in _KW_RE.items():
        out[col] = text.str.contains(rx).astype(np.int8)
    out["phap_ly"] = text.map(phap_ly_of)

    out["district"] = df["district_name"].map(clean_place)
    out["ward"] = out["district"] + " | " + df["ward_name"].map(clean_place)
    proj = df["project_name"].map(lambda s: unicodedata.normalize("NFC", str(s)).strip() if isinstance(s, str) and s.strip() else "Không rõ")
    out["project"] = proj.where(is_cc, "Không áp dụng")
    out["huong"] = df["house_direction"].map(norm_huong) if "house_direction" in df else "Không rõ"
    return out


def fit_categories(F: pd.DataFrame, min_count: dict | None = None) -> dict:
    """Lưu danh mục hợp lệ (đủ số mẫu); nhóm hiếm gộp thành 'Khác'."""
    min_count = min_count or {"district": 30, "ward": 30, "project": 30}
    cats = {}
    for c in CAT_COLS:
        vc = F[c].value_counts()
        keep = vc[vc >= min_count.get(c, 1)].index.tolist()
        cats[c] = sorted(set(keep) | {"Khác"})
    return cats


def apply_categories(F: pd.DataFrame, cats: dict) -> pd.DataFrame:
    F = F.copy()
    for c in CAT_COLS:
        v = F[c].where(F[c].isin(cats[c]), "Khác")
        F[c] = pd.Categorical(v, categories=cats[c])
    return F


def feature_list(use_text_kw: bool = True, use_phobert: bool = False) -> list:
    cols = NUM_COLS + CAT_COLS
    if use_text_kw:
        cols = cols + KW_COLS
    else:
        cols = [c for c in cols if c != "phap_ly"]
    if use_phobert:
        cols = cols + ["phobert"]
    return cols


# Nhóm đặc trưng để giải thích kết quả trên giao diện
def group_of(col: str) -> str:
    if col in ("district", "ward"):
        return "Vị trí (quận, phường)"
    if col == "project":
        return "Dự án chung cư"
    if col in ("area", "log_area", "san_xd"):
        return "Diện tích"
    if col == "floor_count":
        return "Số tầng"
    if col in ("frontage_width", "house_depth"):
        return "Mặt tiền & chiều sâu"
    if col == "road_width":
        return "Độ rộng ngõ/đường"
    if col in ("bedroom_count", "bathroom_count"):
        return "Số phòng"
    if col == "huong":
        return "Hướng nhà"
    if col == "phap_ly":
        return "Pháp lý"
    if col == "loai":
        return "Loại bất động sản"
    if col == "month_idx":
        return "Thời điểm"
    if col == "phobert":
        return "Nội dung mô tả (PhoBERT)"
    if col in KW_LABEL:
        return KW_LABEL[col]
    return col
