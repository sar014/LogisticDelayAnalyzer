import matplotlib.pyplot as plt
import streamlit as st

MAX_TOP_K = 10

def render_plots_streamlit(json_result, df, explanation_json):

    def pretty_label(col):
        return col.replace("_", " ").title() if col else ""

    # ---------------- BUILD EXPLANATION LOOKUP ----------------
    explanation_map = {
        item["metric"]: item["explanation"]
        for item in explanation_json.get("chart_explanation", [])
    }

    plots = json_result["plots"]

    # ---------------- 2-COLUMN GRID ----------------
    cols = st.columns(2)

    for idx, plot in enumerate(plots):

        col_container = cols[idx % 2]  # alternate columns

        with col_container:

            fig, ax = plt.subplots(figsize=(5.5, 3.8)) 
            chart_type = plot["chart_type"]

            # ---------------- BAR ----------------
            if chart_type == "bar":
                x = plot["x"]
                y = plot["y"]
                agg = plot["aggregation"]
                top_k = plot.get("top_k")

                if agg == "count":
                    data = df[x].value_counts()
                else:
                    grouped = df.groupby(x)[y]
                    if agg == "mean":
                        data = grouped.mean()
                    elif agg == "sum":
                        data = grouped.sum()
                    else:
                        plt.close(fig)
                        continue

                if df[x].nunique() > MAX_TOP_K: # Returns the number of unique values in the column (an integer).
                    top_k = min(top_k or MAX_TOP_K, MAX_TOP_K)
                    data = data.sort_values(ascending=False).head(top_k)
                else:
                    data = data.sort_values(ascending=False)

                data.plot(kind="bar", ax=ax)
                ax.set_xlabel(pretty_label(x))
                ax.set_ylabel("Count" if agg == "count" else f"{agg.title()} of {pretty_label(y)}")

            # ---------------- HISTOGRAM ----------------
            elif chart_type == "histogram":
                col = plot["x"]
                df[col].plot(kind="hist", bins=20, ax=ax, edgecolor="white", linewidth=1)
                ax.set_xlabel(pretty_label(col))
                ax.set_ylabel("Frequency")

            # ---------------- PIE ----------------
            elif chart_type == "pie":
                col = plot.get("column")
                data = df[col].value_counts().head(MAX_TOP_K)
                data.plot(kind="pie", autopct="%1.1f%%", ax=ax)
                ax.set_ylabel("")

            # ---------------- SCATTER ----------------
            elif chart_type == "scatter":
                x, y = plot.get("x"), plot.get("y")
                ax.scatter(df[x], df[y], alpha=0.6)
                ax.set_xlabel(pretty_label(x))
                ax.set_ylabel(pretty_label(y))

            # ---------------- LINE ----------------
            elif chart_type == "line":
                x, y, agg = plot.get("x"), plot.get("y"), plot.get("aggregation")
                grouped = df.groupby(x)[y]

                if agg == "mean":
                    data = grouped.mean()
                elif agg == "sum":
                    data = grouped.sum()
                elif agg == "count":
                    data = grouped.count()
                else:
                    plt.close(fig)
                    continue

                data.plot(ax=ax)
                ax.set_xlabel(pretty_label(x))
                ax.set_ylabel(f"{agg.title()} of {pretty_label(y)}")

            else:
                plt.close(fig)
                continue

            # ---------------- TITLE & STYLE ----------------
            metric = plot["metric"]
            ax.set_title(metric, fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()

            # ---------------- RENDER ----------------
            st.pyplot(fig)
            plt.close(fig)

            # ---------------- EXPLANATION ----------------
            explanation = explanation_map.get(metric)
            if explanation:
                with st.expander("📌 Explanation"):
                    for point in explanation:
                        st.write("•", point)
