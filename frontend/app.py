import os
import re
import urllib.parse
import io
import base64
import hashlib

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

import requests
import httpx
from requests_toolbelt.multipart.encoder import MultipartEncoder, MultipartEncoderMonitor
import streamlit as st

st.set_page_config(page_title="Plant Counting", layout="wide")

BASE_FONT = int(os.getenv("BASE_FONT_PT", "16"))
SHOW_DEBUG = bool(int(os.getenv("UI_DEBUG", "0")))
BOLD_FAMILY = os.getenv("BOLD_FONT_FAMILY", "Inter, Arial, sans-serif")
BAR_TEXT_MULT = float(os.getenv("BAR_TEXT_MULT", "1.1"))


def bump_bar_label_fonts(fig, base=BASE_FONT):
    fig.update_traces(
        selector=dict(type="bar"),
        textfont=dict(size=int(base * BAR_TEXT_MULT), family=BOLD_FAMILY),
        insidetextfont=dict(size=int(base * BAR_TEXT_MULT), family=BOLD_FAMILY),
        outsidetextfont=dict(size=int(base * BAR_TEXT_MULT), family=BOLD_FAMILY),
    )
    fig.update_layout(uniformtext_minsize=int(base * BAR_TEXT_MULT), uniformtext_mode="show")
    return fig


def bump_streamlit_fonts(base=BASE_FONT):
    st.markdown(
        f"""
    <style>
      html, body, [class*="css"] {{ font-size: {base}px; }}
      h1 {{ font-size: {int(base*2.0)}px !important; }}
      h2 {{ font-size: {int(base*1.45)}px !important; }}
      h3 {{ font-size: {int(base*1.2)}px !important; }}
      .stButton>button {{ border-radius: 12px; font-weight: 600; }}
      .stFileUploader {{ border-radius: 12px; }}
      .hero {{
        background: linear-gradient(135deg, #0b1220 0%, #13315c 60%, #0f766e 100%);
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        color: white;
        margin-bottom: 1rem;
      }}
      .hero h2 {{ margin: 0 0 .35rem; color: white; }}
      .hero p {{ margin: 0; opacity: 0.92; }}
      .panel {{
        border: 1px solid rgba(15,23,42,.08);
        border-radius: 14px;
        padding: .8rem 1rem;
        background: #ffffff;
      }}
      .chip {{
        display:inline-block;
        border-radius:999px;
        padding: 0.12rem 0.55rem;
        font-size: {int(base*0.85)}px;
        border:1px solid rgba(2,6,23,.12);
        margin-right:0.35rem;
      }}
    </style>
    """,
        unsafe_allow_html=True,
    )


def apply_plotly_fonts(fig, base=BASE_FONT):
    fig.update_layout(
        font=dict(size=base, color="#0f172a"),
        legend=dict(font=dict(size=int(base * 0.9), color="#0f172a")),
        hoverlabel=dict(font_size=int(base * 0.9), font_family=BOLD_FAMILY, font_color="#0f172a"),
    )
    fig.update_xaxes(
        title_font=dict(size=int(base * 1.0), color="#0f172a", family=BOLD_FAMILY),
        tickfont=dict(size=int(base * 0.9), color="#0f172a", family=BOLD_FAMILY),
    )
    fig.update_yaxes(
        title_font=dict(size=int(base * 1.0), color="#0f172a", family=BOLD_FAMILY),
        tickfont=dict(size=int(base * 0.9), color="#0f172a", family=BOLD_FAMILY),
    )
    return fig


bump_streamlit_fonts()


# ---------- helpers ----------
def _clean(s: str) -> str:
    s = "".join(ch for ch in s if ord(ch) >= 32)
    return s.strip().rstrip("/")


def resolve_api_base() -> str:
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
    h = hashlib.sha1()
    h.update(b)
    return h.hexdigest()


def sanitize_filename(name: str | None) -> str:
    if not name:
        return "image.tif"
    return re.sub(r"[^A-Za-z0-9._\-]+", "_", name) or "image.tif"


# ---------- HTTP clients ----------
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
    _ = sha1_bytes(file_bytes)
    files = {"file": (safe_name, io.BytesIO(file_bytes), "image/tiff")}
    try:
        r = requests_sess.post(
            f"{api_base}/infer",
            files=files,
            timeout=(15, 600),
            allow_redirects=False,
            proxies={"http": None, "https": None},
            headers={"Connection": "close"},
        )
        r.raise_for_status()
        return r.json()
    except Exception as e1:
        if SHOW_DEBUG:
            st.info(f"requests upload failed, retrying with toolbelt: {repr(e1)}")

    try:
        bio = io.BytesIO(file_bytes)
        encoder = MultipartEncoder(fields={"file": (safe_name, bio, "image/tiff")})
        monitor = MultipartEncoderMonitor(encoder, lambda m: None)
        headers = {"Content-Type": monitor.content_type, "Connection": "close"}
        r = requests_sess.post(
            f"{api_base}/infer",
            data=monitor,
            timeout=(15, 600),
            allow_redirects=False,
            proxies={"http": None, "https": None},
            headers=headers,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e2:
        if SHOW_DEBUG:
            st.info(f"toolbelt upload failed, retrying with httpx: {repr(e2)}")

    files = {"file": (safe_name, file_bytes, "image/tiff")}
    r = httpx_client.post(f"{api_base}/infer", files=files, headers={"Connection": "close"})
    r.raise_for_status()
    return r.json()


@st.cache_data(show_spinner=False)
def build_counts_any(utm_x, utm_y, utm_bounds, target_cell_m: float):
    left, bottom, right, top = utm_bounds
    x0 = float(np.floor(left))
    y0 = float(np.floor(bottom))
    cm = max(float(target_cell_m), 1e-6)
    nx = int(np.ceil((right - x0) / cm))
    ny = int(np.ceil((top - y0) / cm))
    nx = max(nx, 1)
    ny = max(ny, 1)
    counts = np.zeros((ny, nx), dtype=np.uint32)
    if len(utm_x) == 0:
        return counts, x0, y0, cm
    utm_x = np.asarray(utm_x, dtype=np.float64)
    utm_y = np.asarray(utm_y, dtype=np.float64)
    ii = np.floor((utm_x - x0) / cm).astype(np.int64)
    jj = np.floor((utm_y - y0) / cm).astype(np.int64)
    ii = np.clip(ii, 0, nx - 1)
    jj = np.clip(jj, 0, ny - 1)
    np.add.at(counts, (jj, ii), 1)
    return counts, x0, y0, cm


@st.cache_data(show_spinner=False)
def decode_thumb_b64_to_np(b64_str: str) -> np.ndarray:
    img = Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _to_display_name(s: str) -> str:
    s_norm = str(s).strip().lower()
    if s_norm == "ragweed":
        return "Common Ragweed"
    if s_norm == "palmer":
        return "Palmer Amaranth"
    if s_norm == "grass":
        return "Grass Weeds"
    return re.sub(r"\s+", " ", re.sub(r"_+", " ", str(s))).strip().title()


# ---------- UI ----------
st.markdown(
    """
    <div class='hero'>
      <h2>Plant Counting Intelligence</h2>
      <p>Upload a georeferenced orthomosaic, run AI detection, and inspect count hotspots in one flow.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Workflow")
    st.markdown("1. Upload GeoTIFF\n2. Run detection\n3. Explore map + heatmap\n4. Export GIS files")
    st.markdown("---")
    st.caption(f"API endpoint: {API_BASE}")

uploaded = st.file_uploader("GeoTIFF orthomosaic", type=["tif", "tiff"])
run = st.button("Run detection", type="primary", use_container_width=True)

if uploaded is None:
    st.info("Upload a georeferenced TIFF to begin.")
    st.stop()

safe_name = sanitize_filename(getattr(uploaded, "name", None))
file_bytes = uploaded.getbuffer().tobytes()
file_sha = sha1_bytes(file_bytes)

st.session_state.setdefault("infer_data", None)
st.session_state.setdefault("infer_sha", None)
st.session_state.setdefault("infer_name", None)

if run:
    with st.spinner("Running inference on orthomosaic tiles..."):
        try:
            data = infer_cached(file_bytes, safe_name, API_BASE)
            st.session_state.infer_data = data
            st.session_state.infer_sha = file_sha
            st.session_state.infer_name = safe_name
        except Exception as e:
            st.error(str(e))
            if SHOW_DEBUG:
                st.exception(e)
            st.stop()

if st.session_state.infer_data is None:
    st.info("Click **Run detection** after uploading the TIFF.")
    st.stop()

if st.session_state.infer_sha != file_sha:
    st.warning("You changed the file. Click **Run detection** to refresh results.")
    st.stop()

data = st.session_state.infer_data
thumb_np = decode_thumb_b64_to_np(data["thumb_png_b64"])

disp_w, disp_h = data["disp_w"], data["disp_h"]
scale_x, scale_y = data["scale_x"], data["scale_y"]
cell_m = float(int(data["cell_m"]))
left_m, bottom_m, right_m, top_m = data["meta"]["utm_bounds"]

class_names = data.get("class_names", ["ragweed", "grass", "palmer", "soybean"])
totals_by_class = data.get("totals_by_class", [0] * len(class_names))
pix_x_by_class = data.get("pix_x_by_class", [[] for _ in range(len(class_names))])
pix_y_by_class = data.get("pix_y_by_class", [[] for _ in range(len(class_names))])
utm_x_by_class = data.get("utm_x_by_class", [[] for _ in range(len(class_names))])
utm_y_by_class = data.get("utm_y_by_class", [[] for _ in range(len(class_names))])

weed_idx = [i for i, n in enumerate(class_names) if str(n).strip().lower() != "soybean"]
class_names = [class_names[i] for i in weed_idx]
totals_by_class = [totals_by_class[i] for i in weed_idx]
pix_x_by_class = [pix_x_by_class[i] for i in weed_idx]
pix_y_by_class = [pix_y_by_class[i] for i in weed_idx]
utm_x_by_class = [utm_x_by_class[i] for i in weed_idx]
utm_y_by_class = [utm_y_by_class[i] for i in weed_idx]
NUM_CLASSES = len(class_names)
display_names = [_to_display_name(n) for n in class_names]

palette = ["#ef4444", "#22c55e", "#3b82f6", "#a855f7", "#f59e0b", "#06b6d4", "#ec4899"]

st.success("Detection complete.")

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Total detections", int(data.get("total_weeds", 0)))
with m2:
    st.metric("Field area (m²)", f"{float(data.get('field_area_m2', 0.0)):,.0f}")
with m3:
    st.metric("Average density", f"{float(data.get('avg_density', 0.0)):.4f} /m²")
with m4:
    dominant = display_names[int(np.argmax(totals_by_class))] if NUM_CLASSES else "N/A"
    st.metric("Dominant class", dominant)

chips = []
for i, name in enumerate(display_names):
    chips.append(f"<span class='chip' style='background:{palette[i % len(palette)]}22'>{name}</span>")
st.markdown("".join(chips), unsafe_allow_html=True)

with st.expander("Run metadata"):
    st.write(f"**Model classes:** {', '.join(display_names) if display_names else 'N/A'}")
    st.write(f"**Base grid cell:** {int(cell_m)} m")
    st.write(f"**Filename:** {safe_name}")

if NUM_CLASSES:
    chart_col, table_col = st.columns([1.2, 1])
    with chart_col:
        fig_bar = px.bar(
            x=display_names,
            y=[int(v) for v in totals_by_class],
            color=display_names,
            color_discrete_sequence=palette,
            labels={"x": "Class", "y": "Detections"},
            text=[int(v) for v in totals_by_class],
            title="Per-class counts",
        )
        fig_bar.update_layout(showlegend=False, height=360)
        fig_bar = bump_bar_label_fonts(fig_bar)
        fig_bar = apply_plotly_fonts(fig_bar)
        st.plotly_chart(fig_bar, use_container_width=True)
    with table_col:
        st.dataframe(
            {"Class": display_names, "Count": [int(v) for v in totals_by_class]},
            hide_index=True,
            use_container_width=True,
        )

viz_tab, heat_tab, export_tab = st.tabs(["Detections map", "Density heatmap", "Export"])

with viz_tab:
    chosen = st.multiselect(
        "Visible classes",
        options=list(range(NUM_CLASSES)),
        default=list(range(NUM_CLASSES)),
        format_func=lambda i: display_names[i],
    )
    marker_size = st.slider("Marker size", 4, 16, 7)

    fig_points = px.imshow(thumb_np)
    fig_points.update_xaxes(range=[0, disp_w], showgrid=False, visible=False)
    fig_points.update_yaxes(range=[disp_h, 0], showgrid=False, visible=False)
    fig_points.update_yaxes(scaleanchor="x", scaleratio=1)
    fig_points.update_layout(margin=dict(l=0, r=0, t=0, b=0), dragmode=False, height=850)

    for ci in chosen:
        if not pix_x_by_class[ci]:
            continue
        sx = np.asarray(pix_x_by_class[ci], dtype=np.float64) * scale_x
        sy = np.asarray(pix_y_by_class[ci], dtype=np.float64) * scale_y
        in_bounds = (sx >= 0) & (sx <= disp_w) & (sy >= 0) & (sy <= disp_h)
        if not np.any(in_bounds):
            continue
        fig_points.add_trace(
            go.Scattergl(
                x=sx[in_bounds],
                y=sy[in_bounds],
                mode="markers",
                name=display_names[ci],
                marker=dict(size=marker_size, color=palette[ci % len(palette)], opacity=0.95, line=dict(width=1, color="white")),
                hoverinfo="skip",
                showlegend=True,
            )
        )

    fig_points = apply_plotly_fonts(fig_points)
    st.plotly_chart(fig_points, use_container_width=True)

with heat_tab:
    heat_classes_mode = st.radio("Heatmap scope", ["All classes", "Single class"], horizontal=True)
    if heat_classes_mode == "Single class" and NUM_CLASSES:
        single_class = st.selectbox("Class", options=list(range(NUM_CLASSES)), format_func=lambda i: display_names[i])
        target_classes = [single_class]
    else:
        target_classes = list(range(NUM_CLASSES))

    overlay_cell_m = st.slider("Heatmap cell size (meters)", min_value=float(cell_m), max_value=float(20 * cell_m), value=float(2 * cell_m), step=0.5)
    heat_alpha = st.slider("Heatmap opacity", 0.0, 1.0, 0.45, 0.05)

    all_utm_x = []
    all_utm_y = []
    for ci in target_classes:
        all_utm_x.extend(utm_x_by_class[ci])
        all_utm_y.extend(utm_y_by_class[ci])

    heat_grid, x0_m_any, y0_m_any, heat_cell_m = build_counts_any(
        all_utm_x, all_utm_y, (left_m, bottom_m, right_m, top_m), overlay_cell_m
    )

    ny_h, nx_h = heat_grid.shape
    x_centers_m = x0_m_any + (heat_cell_m * (np.arange(nx_h) + 0.5))
    y_centers_m = y0_m_any + (heat_cell_m * (np.arange(ny_h) + 0.5))
    x_disp = (x_centers_m - left_m) / (right_m - left_m) * disp_w
    y_disp = (top_m - y_centers_m) / (top_m - bottom_m) * disp_h

    fig_heat = px.imshow(thumb_np)
    fig_heat.update_xaxes(range=[0, disp_w], showgrid=False, visible=False)
    fig_heat.update_yaxes(range=[disp_h, 0], showgrid=False, visible=False)
    fig_heat.update_layout(margin=dict(l=0, r=0, t=0, b=0), dragmode=False, height=850)
    zmax_val = float(np.nanmax(heat_grid)) if heat_grid.size else 1.0

    fig_heat.add_trace(
        go.Heatmap(
            z=heat_grid,
            x=x_disp,
            y=y_disp,
            colorscale="YlOrRd",
            zmin=0,
            zmax=zmax_val,
            opacity=float(heat_alpha),
            colorbar=dict(title="detections/cell"),
            hovertemplate=f"{heat_cell_m:g}×{heat_cell_m:g} m cell<br>count: %{{z}}<extra></extra>",
            name="Density",
        )
    )
    fig_heat = apply_plotly_fonts(fig_heat)
    st.plotly_chart(fig_heat, use_container_width=True)

with export_tab:
    st.markdown("Download detections as GIS shapefile zip.")
    shp_zip_all = data.get("shp_zip_b64_all")
    shp_zip_by_class = data.get("shp_zip_b64_by_class", [])

    if shp_zip_all:
        st.download_button(
            label="Download all detections",
            data=base64.b64decode(shp_zip_all),
            file_name="detections_all.zip",
            mime="application/zip",
            use_container_width=True,
        )

    if NUM_CLASSES:
        cols = st.columns(max(1, min(NUM_CLASSES, 4)))
        for i in range(NUM_CLASSES):
            with cols[i % len(cols)]:
                has_blob = i < len(shp_zip_by_class) and shp_zip_by_class[i]
                st.download_button(
                    label=f"{display_names[i]}",
                    data=base64.b64decode(shp_zip_by_class[i]) if has_blob else b"",
                    file_name=f"detections_{class_names[i]}.zip",
                    mime="application/zip",
                    use_container_width=True,
                    disabled=not has_blob,
                )
