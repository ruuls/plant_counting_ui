import os, re, unicodedata, urllib.parse, io, base64, tempfile, pathlib, shutil, hashlib, json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

import requests
import httpx
from requests_toolbelt.multipart.encoder import MultipartEncoder, MultipartEncoderMonitor
import streamlit as st

st.set_page_config(page_title="Weed Detection & Geolocation", layout="wide")

BASE_FONT = int(os.getenv("BASE_FONT_PT", "18"))
SHOW_DEBUG = bool(int(os.getenv("UI_DEBUG", "0")))
RIGHT_PAD_PX = int(os.getenv("RIGHT_PAD_PX", "220"))
OVERLAY_LABEL_MULT = float(os.getenv("OVERLAY_LABEL_MULT", "1.8"))
BOLD_FAMILY = os.getenv("BOLD_FONT_FAMILY", "Arial Black, Arial, sans-serif")
BAR_TEXT_MULT = float(os.getenv("BAR_TEXT_MULT", "1.2"))  # 1.6× base font; tweak as you like

def bump_bar_label_fonts(fig, base=BASE_FONT):
    # Make bar labels larger (both inside/outside)
    fig.update_traces(
        selector=dict(type="bar"),
        textfont=dict(size=int(base*BAR_TEXT_MULT), family=BOLD_FAMILY),
        insidetextfont=dict(size=int(base*BAR_TEXT_MULT), family=BOLD_FAMILY),
        outsidetextfont=dict(size=int(base*BAR_TEXT_MULT), family=BOLD_FAMILY),
    )
    # Force Plotly to render larger text instead of hiding it
    fig.update_layout(uniformtext_minsize=int(base*BAR_TEXT_MULT), uniformtext_mode="show")
    return fig

def bump_streamlit_fonts(base=BASE_FONT):
    st.markdown(f"""
    <style>
      html, body, [class*="css"] {{ font-size: {base}px; }}
      h1 {{ font-size: {int(base*2.0)}px !important; }}
      h2 {{ font-size: {int(base*1.6)}px !important; }}
      h3 {{ font-size: {int(base*1.3)}px !important; }}
      div[data-testid="stMetricValue"] {{ font-size: {int(base*1.4)}px; }}
      div[data-testid="stMetricLabel"] {{ font-size: {int(base*0.95)}px; }}
      div[data-baseweb="slider"] span {{ font-size: {int(base*0.95)}px; }}
      .stButton>button {{ font-size: {int(base*1.0)}px; }}
      .js-plotly-plot .hovertext text {{ font-size: {int(base*0.95)}px; }}
      .ctl-label {{ font-size: {int(base*1.15)}px; font-weight: 600; margin: 0.2rem 0 0.4rem; }}
      .ctl-label-xl {{ font-size: {int(base*OVERLAY_LABEL_MULT)}px; font-weight: 800; margin: 0.15rem 0 0.4rem; letter-spacing: 0.3px; }}
      .big-sub {{ font-size: {int(base*2.6)}px !important; font-weight: 800 !important; line-height: 1.05; margin: 0.2rem 0 0.35rem; letter-spacing: 0.2px; }}
      .big-cap {{ font-size: {int(base*1.25)}px; line-height: 1.35; margin: 0.1rem 0 0.8rem; }}
    </style>
    """, unsafe_allow_html=True)

def apply_plotly_fonts(fig, base=BASE_FONT):
    fig.update_layout(
        font=dict(size=base, color="black"),
        legend=dict(font=dict(size=int(base*0.95), color="black")),
        hoverlabel=dict(font_size=int(base*0.95), font_family=BOLD_FAMILY, font_color="black"),
    )
    fig.update_xaxes(
        title_font=dict(size=int(base*1.05), color="black", family=BOLD_FAMILY),
        tickfont=dict(size=int(base*0.95), color="black", family=BOLD_FAMILY),
    )
    fig.update_yaxes(
        title_font=dict(size=int(base*1.05), color="black", family=BOLD_FAMILY),
        tickfont=dict(size=int(base*0.95), color="black", family=BOLD_FAMILY),
    )
    if getattr(fig.layout, "yaxis2", None) is not None:
        fig.update_layout(
            yaxis2=dict(
                tickfont=dict(size=int(base*0.95), color="black", family=BOLD_FAMILY),
                title_font=dict(size=int(base*1.05), color="black", family=BOLD_FAMILY),
            )
        )
    title_text = getattr(getattr(fig.layout, "title", None), "text", None)
    if title_text not in (None, ""):
        fig.update_layout(title_font=dict(size=int(base*1.25), color="black", family=BOLD_FAMILY))
    for tr in getattr(fig, "data", []):
        cb = getattr(tr, "colorbar", None)
        if cb:
            cb.title.font.size = int(base*0.95)
            cb.title.font.color = "black"
            cb.title.font.family = BOLD_FAMILY
            cb.tickfont.size = int(base*0.9)
            cb.tickfont.color = "black"
            cb.tickfont.family = BOLD_FAMILY
    return fig

bump_streamlit_fonts()

# ---------- helpers ----------
def _clean(s: str) -> str:
    import unicodedata
    s = "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")
    return s.strip().rstrip("/")

def resolve_api_base() -> str:
    import urllib.parse, re
    candidate = os.getenv("API_BASE") or "http://127.0.0.1:8000"
    candidate = _clean(candidate)
    if not re.match(r"^https?://", candidate):
        candidate = "http://" + candidate
    candidate = candidate.replace("0.0.0.0", "127.0.0.1")
    parsed = urllib.parse.urlparse(candidate)
    if not parsed.scheme or not parsed.netloc or " " in candidate:
        st.error("Backend URL is invalid. Set API_BASE like http://127.0.0.1:8000")
        st.stop()
    return candidate

API_BASE = resolve_api_base()

def sha1_bytes(b: bytes) -> str:
    h = hashlib.sha1(); h.update(b); return h.hexdigest()

def sanitize_filename(name: str | None) -> str:
    if not name:
        return "image.tif"
    return re.sub(r"[^A-Za-z0-9._\-]+", "_", name) or "image.tif"

# ---------- HTTP clients (reused) ----------
requests_sess = requests.Session()
requests_sess.trust_env = False

httpx_client = httpx.Client(
    timeout=httpx.Timeout(connect=15.0, read=600.0, write=600.0, pool=15.0),
    follow_redirects=False,
    transport=httpx.HTTPTransport(retries=0),
    trust_env=False,
    http2=False,
)

# ---------- health check ----------
try:
    hr = requests_sess.get(
        f"{API_BASE}/health",
        timeout=(5, 5),
        allow_redirects=False,
        proxies={"http": None, "https": None},
        headers={"Connection": "close"},
    )
    hr.raise_for_status()
except Exception:
    st.error("Cannot reach the backend service. Make sure the API is running at API_BASE.")
    st.stop()

# ---------- SERVER CALL: cached ----------
@st.cache_data(show_spinner=True, ttl=3600, max_entries=64)
def infer_cached(file_bytes: bytes, safe_name: str, api_base: str) -> dict:
    """
    Cache the backend /infer response keyed by (sha1(file_bytes), safe_name, api_base).
    """
    _ = sha1_bytes(file_bytes)  # used implicitly by cache key
    files = {"file": (safe_name, io.BytesIO(file_bytes), "image/tiff")}
    # Try simple requests first
    try:
        r = requests_sess.post(
            f"{api_base}/infer", files=files, timeout=(15, 600),
            allow_redirects=False, proxies={"http": None, "https": None},
            headers={"Connection": "close"},
        )
        r.raise_for_status()
        return r.json()
    except Exception as e1:
        if SHOW_DEBUG: st.info(f"requests upload failed, retrying with toolbelt: {repr(e1)}")

    # Fallback: toolbelt multipart
    try:
        bio = io.BytesIO(file_bytes)
        encoder = MultipartEncoder(fields={"file": (safe_name, bio, "image/tiff")})
        monitor = MultipartEncoderMonitor(encoder, lambda m: None)
        headers = {"Content-Type": monitor.content_type, "Connection": "close"}
        r = requests_sess.post(
            f"{api_base}/infer", data=monitor, timeout=(15, 600),
            allow_redirects=False, proxies={"http": None, "https": None},
            headers=headers,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e2:
        if SHOW_DEBUG: st.info(f"toolbelt upload failed, retrying with httpx: {repr(e2)}")

    # Last fallback: httpx
    files = {"file": (safe_name, file_bytes, "image/tiff")}
    r = httpx_client.post(f"{api_base}/infer", files=files, headers={"Connection": "close"})
    r.raise_for_status()
    return r.json()

# ---------- arbitrary-meter binning (supports 1.5× etc.) ----------
@st.cache_data(show_spinner=False)
def build_counts_any(utm_x, utm_y, utm_bounds, target_cell_m: float):
    """
    Bin UTM points into cells of (target_cell_m) meters.
    Returns (counts, x0_m, y0_m, cell_m_actual).
    """
    left, bottom, right, top = utm_bounds
    x0 = float(np.floor(left))
    y0 = float(np.floor(bottom))
    cm = max(float(target_cell_m), 1e-6)
    nx = int(np.ceil((right - x0) / cm))
    ny = int(np.ceil((top   - y0) / cm))
    nx = max(nx, 1); ny = max(ny, 1)
    counts = np.zeros((ny, nx), dtype=np.uint32)
    if len(utm_x) == 0:
        return counts, x0, y0, cm
    utm_x = np.asarray(utm_x, dtype=np.float64)
    utm_y = np.asarray(utm_y, dtype=np.float64)
    ii = np.floor((utm_x - x0) / cm).astype(np.int64)
    jj = np.floor((utm_y - y0) / cm).astype(np.int64)
    ii = np.clip(ii, 0, nx - 1); jj = np.clip(jj, 0, ny - 1)
    np.add.at(counts, (jj, ii), 1)
    return counts, x0, y0, cm

# ---------- cached decoders ----------
@st.cache_data(show_spinner=False)
def decode_thumb_b64_to_np(b64_str: str) -> np.ndarray:
    img = Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")
    return np.asarray(img, dtype=np.uint8)

# ---------- UI ----------
st.markdown("""
### 📍 Weed Detection and Geolocation App
Upload a georeferenced **orthomosaic TIFF**, detect **weeds** via YOLO, and download a **shapefile** of GPS detections.

**Key Features**
- Upload large `.tif` orthomosaics
- Tile-wise YOLO inference
- GPS (WGS84) conversion of detections
- Downloadable **.shp** (all weeds or per weed)
- Interactive maps, heatmaps, KPIs, histograms

Code and demo: https://github.com/rutvij-25/weedgeolocator
""")

uploaded = st.file_uploader("Upload Orthomosaic GeoTIFF", type=["tif", "tiff"])
if uploaded is None:
    st.info("Upload a georeferenced TIFF to begin.")
    st.stop()

# Read once → bytes (enables hashing + cache)
safe_name = sanitize_filename(getattr(uploaded, "name", None))
file_bytes = uploaded.getbuffer().tobytes()  # avoids extra copies vs .read()

with st.spinner("Processing…"):
    try:
        data = infer_cached(file_bytes, safe_name, API_BASE)
    except Exception as e:
        st.error(str(e))
        if SHOW_DEBUG: st.exception(e)
        st.stop()

# ---------- Decode payload (cached) ----------
thumb_np = decode_thumb_b64_to_np(data["thumb_png_b64"])

disp_w, disp_h = data["disp_w"], data["disp_h"]
scale_x, scale_y = data["scale_x"], data["scale_y"]

counts_base = np.array(data["counts_base"], dtype=np.int64)
x0_m, y0_m = data["x0_m"], data["y0_m"]
cell_m = float(int(data["cell_m"]))  # base cell (meters)
left_m, bottom_m, right_m, top_m = data["meta"]["utm_bounds"]
px_w, px_h = data["meta"]["px_w"], data["meta"]["px_h"]

# ---------- Class Names (from backend) ----------
class_names = data.get("class_names", ["ragweed","grass","palmer","soybean"])
NUM_CLASSES = len(class_names)
totals_by_class = data.get("totals_by_class", [0]*NUM_CLASSES)
pix_x_by_class = data.get("pix_x_by_class", [[] for _ in range(NUM_CLASSES)])
pix_y_by_class = data.get("pix_y_by_class", [[] for _ in range(NUM_CLASSES)])
counts_base_by_class = data.get("counts_base_by_class", None)

# NEW: UTM per-class points for custom-meter binning
utm_x_by_class = data.get("utm_x_by_class", [[] for _ in range(NUM_CLASSES)])
utm_y_by_class = data.get("utm_y_by_class", [[] for _ in range(NUM_CLASSES)])

# ---------- DEFENSIVE FILTER: remove soybean everywhere ----------
weed_idx = [i for i, n in enumerate(class_names) if str(n).strip().lower() != "soybean"]
if len(weed_idx) != len(class_names):
    class_names     = [class_names[i] for i in weed_idx]
    totals_by_class = [totals_by_class[i] for i in weed_idx]
    pix_x_by_class  = [pix_x_by_class[i]  for i in weed_idx]
    pix_y_by_class  = [pix_y_by_class[i]  for i in weed_idx]
    utm_x_by_class  = [utm_x_by_class[i]  for i in weed_idx]
    utm_y_by_class  = [utm_y_by_class[i]  for i in weed_idx]
    if counts_base_by_class:
        counts_base_by_class = [counts_base_by_class[i] for i in weed_idx]
NUM_CLASSES = len(class_names)

# ---------- Display Names (capitalized for UI) ----------

def _to_display_name(s: str) -> str:
    s_norm = str(s).strip().lower()
    if s_norm == "ragweed":
        return "Common Ragweed"
    elif s_norm == "palmer":
        return "Palmer Amaranth"
    elif s_norm == "grass":
        return "Grass Weeds"
    else:
        # fallback: replace underscores → spaces, Title Case
        return re.sub(r"\s+", " ", re.sub(r"_+", " ", str(s))).strip().title()

display_names = [_to_display_name(n) for n in class_names]


# Shapefile zips (no need to cache; small)
shp_zip_all = base64.b64decode(data["shp_zip_b64_all"])
shp_zip_by_class = [base64.b64decode(z) for z in data["shp_zip_b64_by_class"]]

st.success("Completed.")

# ---------- Downloads ----------
st.markdown("<div class='big-sub'>Download detections as shape file</div>", unsafe_allow_html=True)
c_all, *c_classes = st.columns(1 + NUM_CLASSES)
with c_all:
    st.download_button(
        label="Doanload all classes detections",
        data=shp_zip_all,
        file_name="detections_all.zip",
        mime="application/zip",
        use_container_width=True,
    )
for i, cname in enumerate(class_names):
    with c_classes[i]:
        st.download_button(
            label=f"Download {display_names[i]} detections",
            data=shp_zip_by_class[i] if i < len(shp_zip_by_class) else b"",
            file_name=f"detections_{cname}.zip",
            mime="application/zip",
            use_container_width=True,
            disabled=not (i < len(shp_zip_by_class) and len(shp_zip_by_class[i]) > 0),
        )

# ---------- KPIs ----------
st.markdown("---")
st.markdown(
    f"""
    <style>
      .kpi {{
        padding: 0.9rem 1.1rem;
        border-radius: 12px;
        background: #f7f9fc;
        border: 1px solid rgba(0,0,0,.07);
      }}
      .kpi .kpi-label {{
        font-size: {int(BASE_FONT*1.15)}px;
        color: #444;
        font-weight: 600;
        margin-bottom: 0.1rem;
      }}
      .kpi .kpi-value {{
        font-size: {int(BASE_FONT*2.2)}px;
        font-weight: 800;
        line-height: 1.15;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)
kcols = st.columns(2 + NUM_CLASSES)
with kcols[0]:
    st.markdown(
        f'<div class="kpi"><div class="kpi-label">Field size (m²)</div>'
        f'<div class="kpi-value">{data["field_area_m2"]:,.0f}</div></div>',
        unsafe_allow_html=True,
    )
with kcols[1]:
    st.markdown(
        f'<div class="kpi"><div class="kpi-label">Total detections</div>'
        f'<div class="kpi-value">{data.get("total_weeds", 0):,}</div></div>',
        unsafe_allow_html=True,
    )
for i, cname in enumerate(class_names):
    val = totals_by_class[i] if i < len(totals_by_class) else 0
    with kcols[2 + i]:
        st.markdown(
            f'<div class="kpi"><div class="kpi-label">{display_names[i]} (count)</div>'
            f'<div class="kpi-value">{val:,}</div></div>',
            unsafe_allow_html=True,
        )

# ---------- Points overlay ----------
st.markdown("<div class='big-sub'>Orthomosaic + Detections (points)</div>", unsafe_allow_html=True)
selected_classes = st.multiselect(
    "Classes to show (points)",
    options=list(range(NUM_CLASSES)),
    default=list(range(NUM_CLASSES)),
    format_func=lambda i: display_names[i],
)

# Thumbnail image as background
fig_points = px.imshow(thumb_np)
fig_points.update_xaxes(range=[0, disp_w], showgrid=False, visible=False)
fig_points.update_yaxes(range=[disp_h, 0], showgrid=False, visible=False)
fig_points.update_yaxes(scaleanchor="x", scaleratio=1)  # lock aspect so dots align
fig_points.update_layout(margin=dict(l=0, r=0, t=0, b=0), dragmode=False)

# High-contrast palette for visibility on imagery
palette = ["#ff3b30", "#34c759", "#007aff", "#af52de", "#ff9f0a", "#ff2d55", "#5856d6", "#30d158"]

# Adaptive, clearly visible marker size
BASE_MARK_SIZE = max(5, int(min(disp_w, disp_h) / 200))

for ci in selected_classes:
    if ci >= len(pix_x_by_class) or ci >= len(pix_y_by_class):
        continue
    if not pix_x_by_class[ci]:
        continue

    # Scale original pixel coords to display coords
    sx = np.asarray(pix_x_by_class[ci], dtype=np.float64) * scale_x
    sy = np.asarray(pix_y_by_class[ci], dtype=np.float64) * scale_y

    # Keep only points that fall on the displayed image
    in_bounds = (sx >= 0) & (sx <= disp_w) & (sy >= 0) & (sy <= disp_h)
    if not np.any(in_bounds):
        continue
    sx, sy = sx[in_bounds], sy[in_bounds]

    fig_points.add_trace(
        go.Scattergl(
            x=sx, y=sy, mode="markers",
            name=display_names[ci],
            marker=dict(
                size=BASE_MARK_SIZE,
                color=palette[ci % len(palette)],
                opacity=1.0,
                line=dict(width=1.2, color="white"),  # outline for contrast
            ),
            hoverinfo="skip",
            showlegend=True,
        )
    )

# Legend: bigger text, horizontal, symbols scale with marker size
fig_points.update_layout(
    height=1000,
    legend=dict(
        title_text="Classes",
        title_font=dict(size=int(BASE_FONT*1.4), family=BOLD_FAMILY),
        font=dict(size=int(BASE_FONT*1.2), family=BOLD_FAMILY),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="rgba(0,0,0,0.12)",
        borderwidth=1,
        orientation="v",              # vertical legend
        yanchor="top", y=1.0,         # align to top
        xanchor="left", x=1.02,       # push to right of plot
        itemsizing="trace",
    ),
    margin=dict(r=180)                # add right margin space for legend
)




# Keep overall figure fonts consistent with your helper
fig_points = apply_plotly_fonts(fig_points)
st.plotly_chart(fig_points, use_container_width=True)

# ---------- Two columns: (left) Heatmap, (right) Histogram ----------
left_col, sep_col, right_col = st.columns([1, 0.05, 1], gap="large")

with left_col:
    st.markdown("<div class='big-sub'>Orthomosaic + Weed Density Heatmap</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='big-cap'>Base grid cell = <b>{int(cell_m)} m</b>. Choose your heatmap cell size below (meters).</div>",
        unsafe_allow_html=True
    )

    # Heatmap mode
    st.markdown("<div class='ctl-label-xl'>Heatmap mode</div>", unsafe_allow_html=True)
    heatmap_mode = st.radio(
        "", options=["All classes (combined)", "Per class"],
        index=0, horizontal=True, key="heatmap_mode", label_visibility="collapsed",
    )
    if heatmap_mode == "Per class":
        heatmap_class = st.selectbox(
            "Select class for heatmap",
            options=list(range(NUM_CLASSES)),
            format_func=lambda i: display_names[i],
        )
        selected_heatmap_classes = [heatmap_class]
    else:
        selected_heatmap_classes = list(range(NUM_CLASSES))

    # SINGLE SLIDER: custom meters (supports 1.5× etc.)
    st.markdown("<div class='ctl-label-xl'>Overlay cell size (meters)</div>", unsafe_allow_html=True)
    requested_m = st.slider(
        "", min_value=float(cell_m), max_value=float(20*cell_m),
        value=float(1*cell_m), step=0.5, key="agg_cell_m", label_visibility="collapsed"
    )

    # Combine selected classes' UTM points, rebin at requested meters
    utm_x_sel = []
    utm_y_sel = []
    for ci in selected_heatmap_classes:
        utm_x_sel.extend(utm_x_by_class[ci])
        utm_y_sel.extend(utm_y_by_class[ci])

    overlay_grid, x0_m_any, y0_m_any, overlay_cell_m = build_counts_any(
        utm_x_sel, utm_y_sel, (left_m, bottom_m, right_m, top_m), requested_m
    )
    x0_m = x0_m_any
    y0_m = y0_m_any

    st.markdown("<div class='ctl-label-xl'>Heatmap opacity</div>", unsafe_allow_html=True)
    alpha = st.slider("", 0.0, 1.0, 0.45, 0.05, key="opacity_slider", label_visibility="collapsed")

    # Centers for overlay grid in display coords
    ny_o, nx_o = overlay_grid.shape
    x_centers_m = x0_m + (overlay_cell_m * (np.arange(nx_o) + 0.5))
    y_centers_m = y0_m + (overlay_cell_m * (np.arange(ny_o) + 0.5))
    x_disp = (x_centers_m - left_m) / (right_m - left_m) * disp_w
    y_disp = (top_m - y_centers_m) / (top_m - bottom_m) * disp_h

    fig_overlay = px.imshow(thumb_np)
    fig_overlay.update_xaxes(range=[0, disp_w], showgrid=False, visible=False)
    fig_overlay.update_yaxes(range=[disp_h, 0], showgrid=False, visible=False)
    fig_overlay.update_layout(margin=dict(l=0, r=0, t=0, b=0), dragmode=False, height=560)

    zmax_val = float(np.nanmax(overlay_grid)) if overlay_grid.size else 1.0
    fig_overlay.add_trace(
        go.Heatmap(
            z=overlay_grid, x=x_disp, y=y_disp,
            colorscale="Reds", zmin=0, zmax=zmax_val,
            colorbar=dict(title="weed count (per cell)"),
            opacity=float(alpha),
            hovertemplate=f"{overlay_cell_m:g}×{overlay_cell_m:g} m cell<br>count: %{{z}}<extra></extra>",
            name="Density",
        )
    )
    fig_overlay = apply_plotly_fonts(fig_overlay)
    st.plotly_chart(fig_overlay, use_container_width=True)

with sep_col:
    approx_left = 560
    approx_controls = RIGHT_PAD_PX
    overshoot = 300
    line_h = approx_left + approx_controls + overshoot
    st.markdown(
        f"""<div style="
                height:{line_h}px;
                border-left: 2px solid rgba(0,0,0,0.35);
                margin: 0 auto;
            "></div>""",
        unsafe_allow_html=True,
    )

with right_col:
    st.markdown("<div class='big-sub'>Histogram</div>", unsafe_allow_html=True)
    st.markdown("<div class='ctl-label-xl'>Histogram mode</div>", unsafe_allow_html=True)
    hist_mode = st.radio(
        "", options=["All classes (combined)", "All classes (split)", "Per class"],
        index=0, horizontal=True, key="hist_mode", label_visibility="collapsed",
    )

    palette = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

    # ---------- PER CLASS ----------
    if hist_mode == "Per class":
        hist_class = st.selectbox(
            "Select class for histogram",
            options=list(range(NUM_CLASSES)),
            format_func=lambda i: display_names[i],
            key="hist_class_select",
        )

        base_hist = (
            np.array(counts_base_by_class[hist_class], dtype=np.int64)
            if counts_base_by_class else
            np.array(counts_base, dtype=np.int64)
        )

        vals = base_hist.ravel().astype(int)
        fig_hist = go.Figure()
        if vals.size == 0:
            fig_hist.add_annotation(text="No tiles to display.", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
            fig_hist.update_layout(template="plotly_white", height=560,
                                   margin=dict(l=12, r=12, t=36, b=8),
                                   xaxis=dict(visible=False), yaxis=dict(visible=False))
        else:
            bins = np.bincount(vals)
            x = np.arange(bins.size)
            total = int(bins.sum()) if bins.sum() > 0 else 1
            cum_pct = (bins.cumsum() / total) * 100.0

            ymax = int(bins.max()) if bins.size else 1
            ypad = max(1, int(round(max(1, ymax) * 0.12)))

            # bars at TRUE height (no EPS)
            bar_text = [f"{int(v):,}" for v in bins]
            fig_hist.add_bar(
                x=x, y=bins, name=display_names[hist_class],
                marker=dict(color=palette[hist_class % len(palette)]),
                text=bar_text, textposition="outside",
                textfont=dict(color="#1f77b4", family=BOLD_FAMILY, size=int(BASE_FONT*0.95)),
                hovertemplate="Weeds/tile: %{x}<br>Tiles: %{y}<extra></extra>",
                cliponaxis=False,
            )

            # "0" labels for zero-height bars (text only, small y offset)
            zero_idx = np.where(bins == 0)[0]
            if zero_idx.size:
                y0 = 0.2  # fixed tiny offset (<1) so it never looks like a real bar
                fig_hist.add_trace(go.Scatter(
                    x=x[zero_idx],
                    y=np.full(zero_idx.size, y0),
                    text=["0"] * zero_idx.size,
                    mode="text",
                    textposition="top center",
                    showlegend=False, hoverinfo="skip",
                    textfont=dict(family=BOLD_FAMILY, size=int(BASE_FONT*0.9)),
                    cliponaxis=False,
                ))

            # cumulative line
            n = len(x); step = max(1, n // 15)
            cum_text = [f"{p:.3f}%"
                        if (i % step == 0 or i == n - 1) else ""
                        for i, p in enumerate(cum_pct)]
            fig_hist.add_trace(
                go.Scatter(
                    x=x, y=cum_pct, name="Cumulative %",
                    mode="lines+markers+text", yaxis="y2",
                    line=dict(color="#E45756"),
                    marker=dict(color="#E45756"),
                    text=cum_text, textposition="top center",
                    textfont=dict(color="#d62728", family=BOLD_FAMILY, size=int(BASE_FONT*0.9)),
                    hovertemplate="≤ %{x} weeds: %{y:.3f}%<extra></extra>",
                    cliponaxis=False,
                )
            )

            fig_hist.update_layout(
                template="plotly_white",
                height=560,
                bargap=0.15,
                uniformtext_minsize=8, uniformtext_mode="hide",
                legend=dict(orientation="h", yanchor="bottom", y=1.18, xanchor="left", x=0.0,
                            bgcolor="rgba(255,255,255,0.9)"),
                xaxis=dict(title=f"Weeds per {int(cell_m)} m² tile", showgrid=True, zeroline=False),
                yaxis=dict(title="Number of tiles", showgrid=True, zeroline=False, range=[0, max(1, ymax) + ypad]),
                yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 100],
                            tickformat=".3f", ticksuffix="%"),
                margin=dict(l=12, r=12, t=90, b=12),
            )
        fig_hist = apply_plotly_fonts(fig_hist)
        st.plotly_chart(fig_hist, use_container_width=True)

    # ---------- SPLIT (GROUPED BY CLASS) ----------
    elif hist_mode == "All classes (split)":
        fig_split = go.Figure()

        if not counts_base_by_class or len(counts_base_by_class) < NUM_CLASSES:
            # fallback to combined if per-class grids missing
            vals = np.array(counts_base, dtype=np.int64).ravel().astype(int)
            bins = np.bincount(vals)
            x = np.arange(bins.size)

            ymax = int(bins.max()) if bins.size else 1
            ypad = max(1, int(round(max(1, ymax) * 0.12)))

            # TRUE bars, no EPS
            bar_text = [f"{int(v):,}" for v in bins]
            fig_split.add_bar(
                x=x, y=bins, name="All classes",
                marker=dict(color="#636EFA"),
                text=bar_text, textposition="outside",
                textfont=dict(family=BOLD_FAMILY, size=int(BASE_FONT*0.95)),
                hovertemplate="Weeds/tile: %{x}<br>Tiles: %{y}<extra></extra>",
                cliponaxis=False,
            )

            # "0" labels
            zero_idx = np.where(bins == 0)[0]
            if zero_idx.size:
                y0 = 0.2
                fig_split.add_trace(go.Scatter(
                    x=x[zero_idx], y=np.full(zero_idx.size, y0),
                    text=["0"] * zero_idx.size, mode="text",
                    textposition="top center", showlegend=False, hoverinfo="skip",
                    textfont=dict(family=BOLD_FAMILY, size=int(BASE_FONT*0.9)),
                    cliponaxis=False,
                ))
        else:
            # one bar per class at each bucket
            per_class_bins = []
            max_bin = 0
            for ci in range(NUM_CLASSES):
                arr = np.array(counts_base_by_class[ci], dtype=np.int64).ravel().astype(int)
                b = np.bincount(arr) if arr.size else np.array([0], dtype=int)
                per_class_bins.append(b)
                if b.size - 1 > max_bin:
                    max_bin = b.size - 1

            x = np.arange(max_bin + 1)

            # pad all to equal length, then draw TRUE bars (no EPS)
            B = []
            for ci in range(NUM_CLASSES):
                b = np.asarray(per_class_bins[ci], dtype=int).ravel()
                if b.size < x.size:
                    b = np.pad(b, (0, x.size - b.size), constant_values=0)
                B.append(b)
            B = np.vstack(B)  # (NUM_CLASSES, len(x))

            ymax = max(1, int(B.max())) if B.size else 1
            ypad = max(1, int(round(max(1, ymax) * 0.12)))

            for ci in range(NUM_CLASSES):
                b = B[ci]
                bar_text = [f"{int(v):,}" for v in b]
                fig_split.add_bar(
                    x=x, y=b, name=display_names[ci],
                    marker=dict(color=palette[ci % len(palette)]),
                    text=bar_text, textposition="outside",
                    textfont=dict(family=BOLD_FAMILY, size=int(BASE_FONT*0.95)),
                    cliponaxis=False,
                    hovertemplate=(f"{display_names[ci]}<br>"
                                   "Weeds/tile: %{x}<br>Tiles: %{y}<extra></extra>")
                )

            # single centered "0" only when ALL classes are zero at that bucket
            zero_cols = np.where(B.sum(axis=0) == 0)[0]
            if zero_cols.size:
                y0 = 0.2
                fig_split.add_trace(go.Scatter(
                    x=x[zero_cols], y=np.full(zero_cols.size, y0),
                    text=["0"] * zero_cols.size, mode="text",
                    textposition="top center", showlegend=False, hoverinfo="skip",
                    textfont=dict(family=BOLD_FAMILY, size=int(BASE_FONT*0.9)),
                    cliponaxis=False,
                ))

            fig_split.update_layout(barmode="group", bargroupgap=0.15)
            fig_split.update_traces(selector=dict(type='bar'), textangle=-60, textposition="outside")
            fig_split.update_layout(uniformtext_minsize=8, uniformtext_mode="hide")

        fig_split.update_layout(
            template="plotly_white",
            height=560,
            bargap=0.15,
            legend=dict(orientation="h", yanchor="bottom", y=1.18, xanchor="left", x=0.0,
                        bgcolor="rgba(255,255,255,0.9)"),
            xaxis=dict(title=f"Weeds per {int(cell_m)} m² tile",
                       showgrid=True, zeroline=False),
            yaxis=dict(title="Number of tiles", showgrid=True, zeroline=False,
                       range=[0, max(1, ymax) + ypad]),
            margin=dict(l=12, r=12, t=90, b=12),
        )
        fig_split = bump_bar_label_fonts(apply_plotly_fonts(fig_split))
        st.plotly_chart(fig_split, use_container_width=True)

    # ---------- COMBINED ----------
    else:
        base_hist = (
            sum((np.array(counts_base_by_class[ci], dtype=np.int64) for ci in range(NUM_CLASSES)))
            if counts_base_by_class else
            np.array(counts_base, dtype=np.int64)
        )

        vals = base_hist.ravel().astype(int)
        fig_hist = go.Figure()
        if vals.size == 0:
            fig_hist.add_annotation(text="No tiles to display.", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
            fig_hist.update_layout(template="plotly_white", height=560,
                                   margin=dict(l=12, r=12, t=36, b=8),
                                   xaxis=dict(visible=False), yaxis=dict(visible=False))
        else:
            bins = np.bincount(vals)
            x = np.arange(bins.size)
            total = int(bins.sum()) if bins.sum() > 0 else 1
            cum_pct = (bins.cumsum() / total) * 100.0

            ymax = int(bins.max()) if bins.size else 1
            ypad = max(1, int(round(max(1, ymax) * 0.12)))

            # TRUE bars (no EPS)
            bar_text = [f"{int(v):,}" for v in bins]
            fig_hist.add_bar(
                x=x, y=bins, name="Tiles",
                marker=dict(color="#636EFA"),
                text=bar_text, textposition="outside",
                textfont=dict(color="#1f77b4", family=BOLD_FAMILY, size=int(BASE_FONT*0.95)),
                hovertemplate="Weeds/tile: %{x}<br>Tiles: %{y}<extra></extra>",
                cliponaxis=False,
            )

            # zero labels
            zero_idx = np.where(bins == 0)[0]
            if zero_idx.size:
                y0 = 0.2
                fig_hist.add_trace(go.Scatter(
                    x=x[zero_idx],
                    y=np.full(zero_idx.size, y0),
                    text=["0"] * zero_idx.size,
                    mode="text",
                    textposition="top center",
                    showlegend=False, hoverinfo="skip",
                    textfont=dict(family=BOLD_FAMILY, size=int(BASE_FONT*0.9)),
                    cliponaxis=False,
                ))

            # cumulative line
            n = len(x); step = max(1, n // 15)
            cum_text = [f"{p:.3f}%"
                        if (i % step == 0 or i == n - 1) else ""
                        for i, p in enumerate(cum_pct)]
            fig_hist.add_trace(
                go.Scatter(
                    x=x, y=cum_pct, name="Cumulative %",
                    mode="lines+markers+text", yaxis="y2",
                    line=dict(color="#E45756"),
                    marker=dict(color="#E45756"),
                    text=cum_text, textposition="top center",
                    textfont=dict(color="#d62728", family=BOLD_FAMILY, size=int(BASE_FONT*0.9)),
                    hovertemplate="≤ %{x} weeds: %{y:.3f}%<extra></extra>",
                    cliponaxis=False,
                )
            )

            fig_hist.update_layout(
                template="plotly_white",
                height=560,
                bargap=0.15,
                uniformtext_minsize=8, uniformtext_mode="hide",
                legend=dict(orientation="h", yanchor="bottom", y=1.18, xanchor="left", x=0.0,
                            bgcolor="rgba(255,255,255,0.9)"),
                xaxis=dict(title=f"Weeds per {int(cell_m)} m² tile", showgrid=True, zeroline=False),
                yaxis=dict(title="Number of tiles", showgrid=True, zeroline=False, range=[0, max(1, ymax) + ypad]),
                yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 100],
                            tickformat=".3f", ticksuffix="%"),
                margin=dict(l=12, r=12, t=90, b=12),
            )
        fig_hist = apply_plotly_fonts(fig_hist)
        st.plotly_chart(fig_hist, use_container_width=True)
