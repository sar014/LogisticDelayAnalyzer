import streamlit as st
import pandas as pd
import tempfile
from pipeline import run_pipeline
from plot_graphs import render_plots_streamlit
import json
import re


# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="Logistics Delay Analyzer",
    layout="wide"
)

st.title("🚚 Logistics Delay Analyzer")
st.markdown("Analyze shipment delays and get actionable insights using AI.")

# ---- FILE UPLOAD ----
uploaded_file = st.file_uploader(
    "📂 Upload Logistics CSV Dataset",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getvalue())
        csv_path = tmp.name

    # ---- RUN ANALYSIS ----
    if st.button("🚀 Run Delay Analysis"):
        with st.spinner("Analyzing delays... Please wait ⏳"):
            result = run_pipeline(csv_path)

        st.success("✅ Analysis Completed!")

        st.subheader("🧠 AI Insights & Recommendations")
        st.markdown(result.tasks_output[2].raw)

        st.subheader("🧩 Task-wise Outputs")
        for i, task in enumerate(result.tasks_output, 1):
            if i in (3,4):  # skip visualization task
                continue
            with st.expander(f"Task {i}: {task.description[:50]}"):
                st.markdown(task.raw)
    

        raw_output = result.tasks_output[3].raw
        match = re.search(r"```json(.*?)```", raw_output, re.DOTALL)
        if match:
            raw_output = match.group(1).strip()
        else:
            raw_output = raw_output.strip()
            
        # Parse JSON safely
        viz_json = json.loads(raw_output)
        render_plots_streamlit(viz_json, df)

        # Optional download
        # st.download_button(
        #     label="📥 Download Analysis Result",
        #     data=str(result),
        #     file_name="delay_analysis.txt"
        # )
