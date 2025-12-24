import streamlit as st
import pandas as pd
import tempfile
import json
import re

from pipeline import run_pipeline
from plot_graphs import render_plots_streamlit


# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="Logistics Delay Analyzer",
    layout="wide"
)

st.title("🚚 Logistics Delay Analyzer")
st.write("Analyze shipment delays and get actionable insights using AI.")

# ---- FILE UPLOAD ----
uploaded_file = st.file_uploader(
    "📂 Upload Logistics CSV Dataset",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # Save uploaded CSV to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getvalue())
        csv_path = tmp.name

    # ---- RUN ANALYSIS ----
    if st.button("🚀 Run Delay Analysis"):
        with st.spinner("Analyzing delays... Please wait ⏳"):
            result = run_pipeline(csv_path)

        st.success("✅ Analysis Completed!")

        # -------------------------------------------------
        # AI INSIGHTS & RECOMMENDATIONS (SAFE)
        # -------------------------------------------------
        st.subheader("🧠 AI Insights & Recommendations")
        st.write(result.tasks_output[2].raw)   

        # -------------------------------------------------
        # TASK-WISE OUTPUTS (SKIP VIS TASK)
        # -------------------------------------------------
        st.subheader("🧩 Task-wise Outputs")

        for i, task in enumerate(result.tasks_output, start=1):
            # Task index:
            # 1 → Data Understanding
            # 2 → Delay Analysis
            # 3 → Recommendations
            # 4 → Visualization
            if i in (3,4,5):
                continue

            with st.expander(f"Task {i}: {task.description[:50]}"):
                st.write(task.raw)   # ✅ SAFE

        # -------------------------------------------------
        # VISUALIZATION JSON EXTRACTION
        # -------------------------------------------------
        raw_viz_output = str(result.tasks_output[3].raw).strip()

        # Remove ```json fences if present
        match = re.search(r"```json\s*(.*?)\s*```", raw_viz_output, re.DOTALL)
        if match:
            raw_viz_output = match.group(1)

        try:
            viz_json = json.loads(raw_viz_output)
        except json.JSONDecodeError as e:
            st.error("❌ Failed to parse visualization JSON")
            st.code(raw_viz_output)
            st.stop()

        # -------------------------------------------------
        # RENDER PLOTS
        # -------------------------------------------------
        st.subheader("📈 Visual Insights")
        render_plots_streamlit(viz_json, df)
