import streamlit as st
import pandas as pd
import tempfile
from pipeline import run_pipeline


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
        st.markdown(result.raw)

        st.subheader("🧩 Task-wise Outputs")
        for i, task in enumerate(result.tasks_output, 1):
            with st.expander(f"Task {i}: {task.description[:50]}"):
                st.markdown(task.raw)

        # Optional download
        # st.download_button(
        #     label="📥 Download Analysis Result",
        #     data=str(result),
        #     file_name="delay_analysis.txt"
        # )
