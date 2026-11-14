"""
streamlit_app.py — Web demo định giá nhà đất Hà Nội (Streamlit Community Cloud).
Mô hình (LightGBM + PhoBERT) được tải từ kho mô hình Hugging Face ghi trong file hf_repo.txt.
"""
import json
import math
import os
import re
import unicodedata

import lightgbm as lgb
import numpy as np
import pandas as pd
import streamlit as st

from features import (
    HUONG, KEYWORDS, KW_LABEL, KW_PHRASE, LOAI_CC, LOAI_NHA, PHAP_LY, PHAP_LY_PHRASE, _KW_RE,
    apply_categories, base_frame, feature_list, group_of, make_text, mask_price, norm_text, phap_ly_of,
)

st.set_page_config(page_title="Định giá nhà đất Hà Nội", page_icon="🏠", layout="wide")
HERE = os.path.dirname(os.path.abspath(__file__))


def _light_mode() -> bool:
    try:
        return str(st.secrets.get("LIGHT_MODE", "")).lower() in ("1", "true", "yes")
    except Exception:
        return os.environ.get("LIGHT_MODE", "") == "1"


# ----------------------------------------------------------------------------- tải mô hình
@st.cache_resource(show_spinner="Đang tải mô hình (lần đầu mất khoảng 1–2 phút)...")
def load_models():
    path = os.environ.get("LOCAL_ARTIFACTS")
    if not path:
        from huggingface_hub import snapshot_download
        with open(os.path.join(HERE, "hf_repo.txt"), encoding="utf-8") as f:
            repo = f.read().strip()
        path = snapshot_download(repo, repo_type="model")
    with open(os.path.join(path, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    light = _light_mode() and os.path.exists(os.path.join(path, "model_lgb_light.txt"))
    use_phobert = bool(meta.get("use_phobert")) and not light
    booster = lgb.Booster(model_file=os.path.join(path, "model_lgb_light.txt" if light else "model_lgb.txt"))
    feats = feature_list(use_text_kw=True) if light else meta["features"]
    comps = pd.read_parquet(os.path.join(path, "comps.parquet"))
    phobert = None
    if use_phobert:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(2)
        p = os.path.join(path, "phobert")
        tok = AutoTokenizer.from_pretrained(p)
        model = AutoModelForSequenceClassification.from_pretrained(p).to(torch.bfloat16).eval()
        phobert = (tok, model)
    return dict(meta=meta, booster=booster, feats=feats, comps=comps, phobert=phobert)


M = load_models()
META, BOOSTER, FEATS, COMPS, PHOBERT = M["meta"], M["booster"], M["feats"], M["comps"], M["phobert"]


def phobert_score(text: str) -> float:
    import torch

    tok, model = PHOBERT
    with torch.no_grad():
        enc = tok([text], truncation=True, max_length=META["max_len"], return_tensors="pt")
        return float(model(**enc).logits.float().reshape(-1)[0])


# ----------------------------------------------------------------------------- định dạng
def vn(x: float, d: int = 1) -> str:
    s = f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return s[:-2] if s.endswith(",0") else s


def fmt_price(v: float) -> str:
    return f"{vn(v / 1e9, 2)} tỷ" if v >= 1e9 else f"{vn(v / 1e6, 0)} triệu"


def fmt_ppm2(v: float) -> str:
    return f"{vn(v / 1e6, 1)} triệu/m²"


# ----------------------------------------------------------------------------- dự đoán
def build_input(loai, quan, phuong, du_an, dien_tich, so_tang, mat_tien, duong, ngu, wc, huong, phap_ly, dac_diem, mo_ta):
    is_cc = loai == LOAI_CC
    phrases = [KW_PHRASE[d] for d in (dac_diem or [])]
    if PHAP_LY_PHRASE.get(phap_ly):
        phrases.append(PHAP_LY_PHRASE[phap_ly])
    if is_cc:
        title = f"Bán căn hộ chung cư {du_an + ' ' if du_an and du_an != 'Không rõ' else ''}{vn(dien_tich, 0)}m2 {int(ngu or 0)} ngủ {quan}"
    else:
        title = f"Bán nhà {quan} {vn(dien_tich, 0)}m2 {int(so_tang or 0)} tầng mặt tiền {vn(mat_tien or 0)}m"
    if phrases:
        title += ", " + ", ".join(phrases)
    none_if = lambda v: None if v in ("Không rõ", "", None) else v
    return pd.DataFrame([{
        "name": title, "description": (mo_ta or "").strip(),
        "property_type_name": "Căn hộ chung cư" if is_cc else "Nhà", "loai": loai,
        "district_name": quan, "ward_name": none_if(phuong),
        "project_name": none_if(du_an) if is_cc else None, "area": dien_tich,
        "floor_count": None if is_cc else so_tang, "frontage_width": None if is_cc else (mat_tien or None),
        "house_depth": None, "road_width": None if is_cc else (duong or None),
        "bedroom_count": ngu, "bathroom_count": wc, "house_direction": none_if(huong),
        "published_at": META["latest_date"],
    }])


def predict(loai, quan, phuong, du_an, dien_tich, so_tang, mat_tien, duong, ngu, wc, huong, phap_ly, dac_diem, mo_ta):
    df1 = build_input(loai, quan, phuong, du_an, dien_tich, so_tang, mat_tien, duong, ngu, wc, huong, phap_ly, dac_diem, mo_ta)
    F = apply_categories(base_frame(df1), META["cats"])
    if "phobert" in FEATS:
        F["phobert"] = phobert_score(make_text(df1.at[0, "name"], df1.at[0, "description"]))
    X = F[FEATS]
    y = float(BOOSTER.predict(X)[0])
    contrib = BOOSTER.predict(X, pred_contrib=True)[0]
    ppm2 = math.exp(y)
    q = META["interval"][loai]
    return dict(price=ppm2 * dien_tich, ppm2=ppm2, contrib=contrib,
                lo=math.exp(y + q["q10"]) * dien_tich, hi=math.exp(y + q["q90"]) * dien_tich)


BORDER = "border:1px solid rgba(128,128,128,.35);border-radius:14px;padding:18px 22px;margin-bottom:14px"


def card_html(loai, quan, phuong, r):
    med = META["median_ppm2"]
    ref = []
    k_w = f"{loai}|{quan} | {phuong}"
    if phuong and phuong != "Không rõ" and k_w in med["ward"]:
        ref.append((f"Trung vị phường {phuong}", med["ward"][k_w]))
    if f"{loai}|{quan}" in med["district"]:
        ref.append((f"Trung vị {quan}", med["district"][f"{loai}|{quan}"]))
    ref.append((f"Trung vị {loai.lower()} toàn Hà Nội", med["all"][loai]))
    rows = ""
    for lbl, v in ref:
        d = r["ppm2"] / v - 1
        color = "#16a34a" if d < 0 else "#dc2626"
        rows += (f"<tr><td style='padding:3px 14px 3px 0;border:none'>{lbl}</td>"
                 f"<td style='padding:3px 14px;border:none'>{fmt_ppm2(v)}</td>"
                 f"<td style='color:{color};font-weight:600;border:none'>{'+' if d >= 0 else ''}{vn(d * 100, 0)}%</td></tr>")
    return f"""<div style="{BORDER}">
  <div style="opacity:.7;font-size:14px">Giá ước tính</div>
  <div style="font-size:42px;font-weight:700;line-height:1.15;margin:4px 0">{fmt_price(r['price'])}</div>
  <div style="font-size:15px">Khoảng giá hợp lý (80%): <b>{fmt_price(r['lo'])} – {fmt_price(r['hi'])}</b></div>
  <div style="font-size:15px;margin-top:2px">Đơn giá: <b>{fmt_ppm2(r['ppm2'])}</b></div>
  <div style="margin-top:12px;font-size:14px;opacity:.85">So sánh đơn giá với thị trường:</div>
  <table style="font-size:14px;margin-top:4px;border:none">{rows}</table>
</div>"""


def factors_html(contrib, loai):
    groups = {}
    for f, c in zip(FEATS, contrib[:-1]):
        g = group_of(f)
        if f in ("loai", "project"):
            g = "Dự án chung cư" if (loai == LOAI_CC and f == "project") else "Loại bất động sản"
        groups[g] = groups.get(g, 0.0) + float(c)
    items = sorted([(g, c) for g, c in groups.items() if abs(c) >= 0.01], key=lambda t: -abs(t[1]))[:9]
    base = math.exp(float(contrib[-1]))
    mx = max([abs(math.exp(c) - 1) for _, c in items] + [0.05])
    bars = ""
    for g, c in items:
        pct = math.exp(c) - 1
        w = min(100, abs(pct) / mx * 100)
        color = "#16a34a" if pct >= 0 else "#dc2626"
        bars += f"""<div style="display:flex;align-items:center;gap:10px;margin:6px 0;font-size:14px">
  <div style="width:210px;flex-shrink:0">{g}</div>
  <div style="flex:1;background:rgba(128,128,128,.15);border-radius:6px;height:18px">
    <div style="width:{w:.0f}%;background:{color};height:18px;border-radius:6px;opacity:.85"></div></div>
  <div style="width:70px;text-align:right;font-weight:600;color:{color}">{'+' if pct >= 0 else ''}{vn(pct * 100, 0)}%</div>
</div>"""
    return f"""<div style="{BORDER}">
  <div style="font-weight:600;margin-bottom:4px">Yếu tố làm giá cao / thấp hơn</div>
  <div style="font-size:13px;opacity:.7;margin-bottom:8px">So với đơn giá tham chiếu {fmt_ppm2(base)} (mức trung bình của mô hình).
  Các tác động tính theo phần trăm và nhân dồn với nhau.</div>
  {bars or '<i>Không có yếu tố nổi bật.</i>'}
</div>"""


def comps_table(loai, quan, phuong, du_an, dien_tich):
    c = COMPS[(COMPS["loai"] == loai) & (COMPS["district"] == quan)]
    if loai == LOAI_CC and du_an and du_an != "Không rõ" and (c["project"] == du_an).sum() >= 3:
        c = c[c["project"] == du_an]
    elif phuong and phuong != "Không rõ" and (c["ward"] == phuong).sum() >= 5:
        c = c[c["ward"] == phuong]
    c = c[(c["area"] >= dien_tich * 0.7) & (c["area"] <= dien_tich * 1.3)]
    if c.empty:
        return None
    c = c.assign(_d=np.log(c["area"] / dien_tich).abs()).sort_values("_d").head(6)
    return pd.DataFrame({
        "Tiêu đề tin rao": c["title"], "Phường": c["ward"],
        "Diện tích (m²)": c["area"].map(lambda v: vn(v, 0)), "Giá rao": c["price"].map(fmt_price),
        "Đơn giá": c["ppm2"].map(fmt_ppm2), "Ngày đăng": c["published"],
    })


def metrics_md():
    lines = ["| Mô hình | Sai số trung vị | MAPE | Lệch ≤ 10% | Lệch ≤ 20% |", "|---|---|---|---|---|"]
    for r in META["metrics"]:
        lines.append(f"| {r['model']} | {vn(r['MdAPE'])}% | {vn(r['MAPE'])}% | {vn(r['within10'])}% | {vn(r['within20'])}% |")
    i = META["data_info"]
    return (f"Huấn luyện trên **{vn(i['n_train'], 0)}** tin rao bán nhà riêng và chung cư tại Hà Nội "
            f"({i['date_from']} – {i['date_to']}) sau khi làm sạch và loại tin trùng. Kiểm tra trên **{vn(i['n_test'], 0)}** "
            f"tin đăng **sau ngày {i['test_from']}**, là những tin mô hình chưa từng thấy.\n\n" + "\n".join(lines) +
            "\n\n*Sai số tính trên giá rao bán, không phải giá giao dịch thực tế.*")


# ----------------------------------------------------------------------------- giao diện
st.title("🏠 Định giá nhà đất Hà Nội bằng AI")
st.markdown(
    "Dán một đoạn tin rao hoặc nhập thông tin căn nhà/căn hộ, mô hình sẽ ước tính giá dựa trên hàng trăm nghìn tin rao bán thực tế. "
    "Mô hình kết hợp **LightGBM** (đọc các thông số) và **PhoBERT** (đọc hiểu phần mô tả tiếng Việt)."
)

DISTRICTS = META["districts"] or ["Không rõ"]


# ----------------------------------------------------------------------------- đọc tin rao tự do
def _strip(s: str) -> str:
    s = unicodedata.normalize("NFD", norm_text(s).replace("đ", "d"))
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


def _num(x: str) -> float:
    return float(x.replace(",", "."))


def _find(rx, text, cast=_num):
    m = re.search(rx, text)
    return cast(m.group(1)) if m else None


def _match_name(names, text_n, text_s, min_len=3):
    """Tìm tên (quận/phường/dự án) xuất hiện trong văn bản, có dấu hoặc không dấu; ưu tiên tên dài."""
    for n in sorted(names, key=len, reverse=True):
        if len(n) < min_len or n in ("Khác", "Không rõ"):
            continue
        if re.search(rf"(?<!\w){re.escape(norm_text(n))}(?!\w)", text_n) or \
           re.search(rf"(?<!\w){re.escape(_strip(n))}(?!\w)", text_s):
            return n
    return None


def parse_listing(raw: str) -> dict:
    t = mask_price(raw)                     # bỏ phần giá để không đọc nhầm số
    ts = _strip(t)
    N = r"(\d+(?:[.,]\d+)?)"
    out = {}
    out["loai"] = LOAI_CC if re.search(r"chung cư|căn hộ|\bcc\b|chcc|apartment", t) else LOAI_NHA
    out["dien_tich"] = (_find(rf"(?:diện tích|dt|dtsd|dtđ)\s*[:=]?\s*{N}\s*m", t)
                        or _find(rf"{N}\s*(?:m2|m²|m vuông|mét vuông|met vuong)", t))
    out["so_tang"] = _find(r"(\d+)\s*(?:tầng|lầu)(?!\w)", t, int)
    out["mat_tien"] = _find(rf"(?:mặt tiền|mt)\s*[:=]?\s*(?:rộng\s*)?{N}\s*m", t)
    out["duong"] = _find(rf"(?:ngõ|ngách|hẻm|đường|lối vào)\s*(?:trước nhà\s*)?(?:rộng\s*)?{N}\s*m(?!2|²)", t)
    out["ngu"] = _find(r"(\d+)\s*(?:phòng ngủ|pn|ngủ)(?!\w)", t, int)
    out["wc"] = _find(r"(\d+)\s*(?:wc|vệ sinh|vs|toilet|phòng tắm)(?!\w)", t, int)
    hm = re.search(r"hướng\s*:?\s*(đông nam|đông bắc|tây nam|tây bắc|đông|tây|nam|bắc)", t)
    out["huong"] = hm.group(1).title() if hm else "Không rõ"
    q = _match_name(DISTRICTS, t, ts)
    out["quan"] = q
    out["phuong"] = _match_name(META["wards_by_district"].get(q, []), t, ts) if q else None
    out["du_an"] = _match_name(META["projects_by_district"].get(q, []), t, ts, 4) if q and out["loai"] == LOAI_CC else None
    out["phap_ly"] = phap_ly_of(t)
    out["dac_diem"] = [lbl for col, lbl in KW_LABEL.items() if re.search(_KW_RE[col], t)]
    return out


def show_result(res):
    loai, quan, phuong, du_an, dien_tich, r = res
    st.markdown(card_html(loai, quan, phuong, r), unsafe_allow_html=True)
    st.markdown(factors_html(r["contrib"], loai), unsafe_allow_html=True)
    st.markdown("#### Tin rao tương tự trong khu vực")
    t = comps_table(loai, quan, phuong, du_an, dien_tich)
    if t is None:
        st.info("Chưa có tin rao tương tự trong dữ liệu.")
    else:
        st.dataframe(t, hide_index=True, width="stretch")


VI_DU = ("Bán nhà Đống Đa, ngõ ô tô đỗ cửa, cách phố 30m. Diện tích 42m2, 5 tầng, mặt tiền 4,2m, "
         "4 phòng ngủ 4 wc, nhà mới xây full nội thất, hướng Đông Nam. Sổ đỏ chính chủ, sẵn sàng giao dịch.")

left, right = st.columns([5, 6], gap="large")
with left:
    tab_free, tab_form = st.tabs(["✍️ Dán tin rao (tự do)", "📋 Nhập theo từng mục"])

    with tab_free:
        st.caption("Gõ hoặc dán nguyên một đoạn mô tả như tin rao trên mạng. AI tự đọc diện tích, số tầng, "
                   "vị trí, pháp lý, đặc điểm... rồi định giá.")
        txt = st.text_area("Mô tả căn nhà / căn hộ", value=VI_DU, height=170, key="free_text")
        go_free = st.button("Đọc tin & định giá", type="primary", width="stretch", key="btn_free")
        if go_free:
            p = parse_listing(txt)
            st.session_state["parsed"] = p
            miss = [n for n, k in [("diện tích", "dien_tich"), ("quận/huyện", "quan")] if not p[k]]
            if miss:
                st.session_state.pop("res", None)
                st.warning("Chưa đọc được **" + ", ".join(miss) + "**. Bạn ghi thêm vào mô tả (ví dụ: "
                           "\"45m2\", \"quận Cầu Giấy\") hoặc dùng tab *Nhập theo từng mục*.")
            else:
                with st.spinner("Đang định giá..."):
                    r = predict(p["loai"], p["quan"], p["phuong"] or "Không rõ", p["du_an"] or "Không rõ",
                                p["dien_tich"], p["so_tang"], p["mat_tien"], p["duong"], p["ngu"], p["wc"],
                                p["huong"], "Không rõ", [], txt)
                st.session_state["res"] = (p["loai"], p["quan"], p["phuong"] or "Không rõ",
                                           p["du_an"] or "Không rõ", p["dien_tich"], r)
        if st.session_state.get("parsed") and go_free:
            p = st.session_state["parsed"]
            show = lambda v, u="": "—" if v in (None, "Không rõ", []) else f"{vn(v, 1) if isinstance(v, float) else v}{u}"
            st.markdown("**AI đã đọc được:**")
            st.dataframe(pd.DataFrame({
                "Thông tin": ["Loại", "Quận/huyện", "Phường", "Dự án", "Diện tích", "Số tầng", "Mặt tiền",
                              "Ngõ/đường", "Phòng ngủ", "WC", "Hướng", "Pháp lý", "Đặc điểm"],
                "Giá trị": [p["loai"], show(p["quan"]), show(p["phuong"]), show(p["du_an"]), show(p["dien_tich"], " m²"),
                            show(p["so_tang"]), show(p["mat_tien"], " m"), show(p["duong"], " m"), show(p["ngu"]),
                            show(p["wc"]), show(p["huong"]), p["phap_ly"], ", ".join(p["dac_diem"]) or "—"],
            }), hide_index=True, width="stretch")

    with tab_form:
        loai = st.radio("Loại bất động sản", [LOAI_NHA, LOAI_CC], horizontal=True)
        is_cc = loai == LOAI_CC
        c1, c2 = st.columns(2)
        quan = c1.selectbox("Quận / huyện (gõ để tìm)", DISTRICTS)
        phuong = c2.selectbox("Phường / xã", ["Không rõ"] + META["wards_by_district"].get(quan, []))
        du_an = "Không rõ"
        if is_cc:
            du_an = st.selectbox("Dự án chung cư", ["Không rõ"] + META["projects_by_district"].get(quan, []))
        c1, c2, c3 = st.columns(3)
        dien_tich = c1.number_input("Diện tích (m²)", min_value=10.0, max_value=500.0, value=75.0 if is_cc else 45.0, step=1.0)
        so_tang = mat_tien = duong = None
        if not is_cc:
            so_tang = c2.number_input("Số tầng", min_value=1, max_value=15, value=4)
            mat_tien = c3.number_input("Mặt tiền (m)", min_value=0.0, max_value=30.0, value=4.0, step=0.5)
            duong = c1.number_input("Ngõ / đường trước nhà (m)", min_value=0.0, max_value=60.0, value=3.0, step=0.5)
            ngu = c2.number_input("Số phòng ngủ", min_value=0, max_value=15, value=3)
            wc = c3.number_input("Số WC", min_value=0, max_value=15, value=3)
        else:
            ngu = c2.number_input("Số phòng ngủ", min_value=0, max_value=10, value=2)
            wc = c3.number_input("Số WC", min_value=0, max_value=10, value=2)
        c1, c2 = st.columns(2)
        huong = c1.selectbox("Hướng nhà", ["Không rõ"] + HUONG)
        phap_ly = c2.selectbox("Pháp lý", PHAP_LY)
        dac_diem = st.multiselect("Đặc điểm nổi bật", [k[1] for k in KEYWORDS])
        mo_ta = st.text_area("Mô tả thêm (không bắt buộc)", key="form_text",
                             placeholder="Ví dụ: Nhà ngõ thông, cách phố 30m, gần trường học và chợ...")
        if st.button("Định giá", type="primary", width="stretch", key="btn_form"):
            st.session_state.pop("parsed", None)
            with st.spinner("Đang định giá..."):
                r = predict(loai, quan, phuong, du_an, dien_tich, so_tang, mat_tien, duong, ngu, wc, huong, phap_ly, dac_diem, mo_ta)
            st.session_state["res"] = (loai, quan, phuong, du_an, dien_tich, r)

with right:
    if st.session_state.get("res"):
        show_result(st.session_state["res"])
    else:
        st.info("👈 Dán một đoạn tin rao hoặc nhập thông tin bên trái, rồi bấm định giá.")

with st.expander("Mô hình chính xác đến đâu?"):
    st.markdown(metrics_md())
st.caption(
    "⚠️ Kết quả chỉ mang tính tham khảo, dựa trên giá rao bán (thường cao hơn giá giao dịch thực tế). "
    "Dữ liệu: tinixai/vietnam-real-estates (CC BY-NC 4.0). Dữ liệu còn dùng địa giới quận/huyện cũ (trước 01/07/2025)."
)
