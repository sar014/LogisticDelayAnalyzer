# app2.py

import tempfile
import json
import re

import pandas as pd
import streamlit as st

from pipeline2 import run_pipeline
from plot_graphs2 import render_plots_streamlit  # new helper

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="Logistics Delay Analyzer",
    layout="wide"
)

st.title("🚚 Logistics Delay Analyzer (v2)")
st.write("Analyze shipment delays and get actionable insights using AI.")

# ---- FILE UPLOAD ----
uploaded_file = st.file_uploader(
    "📂 Upload Logistics CSV Dataset",
    type=["csv"]
)

if uploaded_file:
    # Preview dataset
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
        # AI INSIGHTS & RECOMMENDATIONS
        # -------------------------------------------------
        if hasattr(result, "tasks_output") and result.tasks_output:
            outputs = result.tasks_output

            # Recommendations (task 3)
            if len(outputs) > 2:
                st.subheader("🧠 AI Insights & Recommendations")
                st.write(outputs[2].raw)

            # -------------------------------------------------
            # TASK-WISE OUTPUTS (summary + delay causes)
            # -------------------------------------------------
            st.subheader("🧩 Task-wise Outputs")
            task_description = ["Understanding Dataset", "Delay Causes"]

            for i, task in enumerate(outputs, start=1):
                # 1 → Data Understanding
                # 2 → Delay Analysis
                # 3 → Recommendations
                # 4 → Visualization plan
                # 5 → Visualization interpretation
                if i in (3, 4, 5):
                    continue
                if i - 1 < len(task_description):
                    with st.expander(f"Task {i}: {task_description[i-1]}"):
                        st.write(task.raw)

            # -------------------------------------------------
            # VISUALIZATION JSON EXTRACTION (tasks 4 & 5)
            # -------------------------------------------------
            viz_json = None
            interpreter_json = None

            # Task 4: viz plan
            if len(outputs) > 3:
                raw_viz_output = str(outputs[3].raw).strip()
                # Remove ```json fences if present
                match = re.search(r"```json\s*(.*?)\s*```", raw_viz_output, re.DOTALL)
                if match:
                    raw_viz_output = match.group(1)
                try:
                    viz_json = json.loads(raw_viz_output)
                except json.JSONDecodeError:
                    # If output_pydantic=VizPlan, raw may already be dict-like
                    if isinstance(outputs.raw, dict):
                        viz_json = outputs.raw[6]
                    else:
                        st.error("❌ Failed to parse visualization JSON (viz plan).")
                        st.code(raw_viz_output)

            # Task 5: viz interpretation
            if len(outputs) > 4:
                raw_interp_output = str(outputs[4].raw).strip()
                match = re.search(r"```json\s*(.*?)\s*```", raw_interp_output, re.DOTALL)
                if match:
                    raw_interp_output = match.group(1)
                try:
                    interpreter_json = json.loads(raw_interp_output)
                except json.JSONDecodeError:
                    if isinstance(outputs[4].raw, dict):
                        interpreter_json = outputs[4].raw
                    else:
                        st.error("❌ Failed to parse visualization JSON (interpreter).")
                        st.code(raw_interp_output)

            # -------------------------------------------------
            # RENDER PLOTS VIA HELPER
            # -------------------------------------------------
            if viz_json:
                st.subheader("📈 Visual Insights")
                render_plots_streamlit(viz_json, df, interpreter_json)
            else:
                st.info("No visualization plan available to render.")
