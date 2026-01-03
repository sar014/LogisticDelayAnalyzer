# app.py

import os
import tempfile

import streamlit as st
from pipeline2 import run_pipeline  # import your existing function

st.set_page_config(page_title="Logistics Delay Analyzer", layout="wide")
st.title("📦 Logistics Delay Analyzer")

st.write("Upload a logistics CSV file to get delay insights and recommendations.")

uploaded_file = st.file_uploader(
    "Choose a logistics CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    # Show basic file info
    st.info(f"Uploaded file: **{uploaded_file.name}**")

    # Save uploaded file to a temporary path so run_pipeline can use a real path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    if st.button("Run Analysis"):
        with st.spinner("Analyzing delays and generating recommendations..."):
            try:
                result = run_pipeline(tmp_path)

                # If result is a Crew object, it may have .tasks_output; adjust to your actual shape.
                # Example 1: if run_pipeline returns a plain string:
                # st.subheader("Recommendations")
                # st.write(result)

                # Example 2: if run_pipeline returns CrewOutput with multiple tasks:
                #   task 0: data understanding
                #   task 1: delay causes
                #   task 2: recommendations
                # change this slice to whatever you see in print(result)
                if hasattr(result, "tasks_output"):
                    # crude example: assume recommendations are in third task
                    rec_task = result.tasks_output[2]
                    st.subheader("Recommendations")
                    st.write(rec_task.raw)
                else:
                    # fallback: just show whatever was returned
                    st.subheader("Recommendations")
                    st.write(result)

            except Exception as e:
                st.error(f"Error during analysis: {e}")
            finally:
                # clean up temp file
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
