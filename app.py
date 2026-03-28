import streamlit as st
import numpy as np
import cv2
import tempfile
import os
from PIL import Image

from detector import detect_objects, is_low_quality_image
from query_parser import parse_query
from processor import filter_detections, calculate_density_stats, calculate_confidence_score, get_highest_density_region
from visualizer import draw_boxes, generate_heatmap
from insights import generate_insights

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="🛰️ Satellite Analyzer", layout="wide")
st.title("🛰️ AI Satellite Image Analyzer")
st.caption("YOLOv8 + Gemini Flash | Urban Planning · Disaster Monitoring · Traffic Analysis")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    mode = st.selectbox(
        "Use-case Mode",
        ["Urban Planning", "Disaster Monitoring", "Traffic Analysis"]
    )
    st.markdown("---")
    st.markdown("**Detectable classes:**")
    st.code("car · truck · bus · person\nairplane · boat · motorcycle")
    st.caption("YOLO COCO model — no training required")

# ── Image Upload ──────────────────────────────────────────────────────────────
uploaded = st.file_uploader("📤 Upload Satellite Image", type=["jpg", "jpeg", "png"])

if uploaded:
    # Write to temp file (YOLO requires a file path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    if is_low_quality_image(tmp_path):
        st.warning("⚠️ Image appears blurry or low quality — detection accuracy may be reduced.")

    # ── Cache detections per image (run YOLO ONCE) ────────────────────────────
    file_key = uploaded.name + str(uploaded.size)
    if st.session_state.get("last_file") != file_key:
        with st.spinner("🔍 Running YOLO detection (once per image)..."):
            detections = detect_objects(tmp_path)
        st.session_state["detections"] = detections
        st.session_state["last_file"] = file_key
        st.session_state["tmp_path"] = tmp_path
    else:
        detections = st.session_state["detections"]

    pil_img = Image.open(tmp_path).convert("RGB")
    img_np = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_h, img_w = img_bgr.shape[:2]

    st.success(f"✅ {len(detections)} objects detected")

    # ── Raw detection stats ───────────────────────────────────────────────────
    with st.expander("📋 Detection Details"):
        stats = calculate_density_stats(detections, img_w, img_h)
        st.json(stats)

    st.divider()

    # ── Query Input ───────────────────────────────────────────────────────────
    query = st.text_input(
        "🔍 Natural Language Query",
        placeholder="e.g. Show areas with many cars · Find dense vehicle clusters"
    )

    if query:
        with st.spinner("🤖 Parsing query with Gemini..."):
            parsed, parse_source = parse_query(query)

        if parse_source == "gemini":
            st.caption("✅ Parsed by Gemini")
        else:
            st.caption("⚙️ Parsed by fallback (offline)")

        with st.expander("🧠 Parsed Query JSON"):
            st.json(parsed)

        filtered = filter_detections(detections, parsed, img_w, img_h)
        display_detections = filtered if filtered else detections

        confidence_ratio, confidence_label = calculate_confidence_score(filtered, detections)
        hotspot = get_highest_density_region(filtered if filtered else detections, img_w, img_h)

        # ── Metrics row ───────────────────────────────────────────────────────
        m1, m2, m3 = st.columns(3)
        m1.metric("Matched Detections", len(filtered))
        m2.metric("Query Confidence", confidence_label, delta=f"{confidence_ratio:.0%} match ratio")
        m3.metric("Total Detections", len(detections))

        st.info(f"🎯 Matched **{len(filtered)}** detections for query")

        # ── Visual Outputs ────────────────────────────────────────────────────
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📦 Bounding Boxes")
            st.caption("Highlighted = matched query results")
            boxed = draw_boxes(img_bgr, detections, filtered, hotspot=hotspot)
            st.image(cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB), use_column_width=True)

        with col2:
            st.subheader("🌡️ Density Heatmap")
            st.caption("Based on matched detections")
            heatmap = generate_heatmap(img_bgr, display_detections)
            st.image(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), use_column_width=True)

        # ── Hotspot info banner ───────────────────────────────────────────────
        if hotspot is not None:
            st.info(
                f"⚡ Highest activity region: {hotspot['count']} detections at pixels "
                f"({hotspot['x1']}, {hotspot['y1']}) → ({hotspot['x2']}, {hotspot['y2']})"
            )

        # ── Insights ──────────────────────────────────────────────────────────
        st.divider()
        st.subheader("💡 Insights")
        insight_text = generate_insights(detections, filtered, mode, parsed, confidence_label=confidence_label)
        st.markdown(insight_text)

    else:
        # Show raw image before any query
        st.image(pil_img, caption="Uploaded image — enter a query to analyze",
                 use_column_width=True)

    os.unlink(tmp_path)
