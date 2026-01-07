import matplotlib.pyplot as plt
import streamlit as st

MAX_TOPK = 10

def render_plots_streamlit(json_result, df, explanation_json):
    def pretty_label(col):
        return col.replace("_", " ").title() if col else col

    # ---------------- BUILD EXPLANATION LOOKUP ----------------
    explanation_map = {
        item["metric"]: item["explanation"]
        for item in explanation_json.get("chart_explanation", [])
    } if explanation_json else {}

    plots = json_result.get("plots", [])

    cols = st.columns(2)

    for idx, plot in enumerate(plots):
        col_container = cols[idx % 2]  # alternate columns

        with col_container:
            fig, ax = plt.subplots(figsize=(5.5, 3.8))
            chart_type = plot.get("chart_type")

            try:
                # ---------------- BAR ----------------
                if chart_type == "bar":
                    x = plot.get("x")
                    y = plot.get("y")
                    agg = plot.get("aggregation")
                    topk = plot.get("top_k")

                    if agg == "count":
                        data = df[x].value_counts()
                    else:
                        grouped = df.groupby(x)[y]
                        if agg == "mean":
                            data = grouped.mean()
                        elif agg == "sum":
                            data = grouped.sum()
                        else:
                            raise ValueError(f"Unsupported aggregation: {agg}")

                    # top-k trimming
                    if df[x].nunique() > MAX_TOPK:
                        topk = min(topk or MAX_TOPK, MAX_TOPK)
                        data = data.sort_values(ascending=False).head(topk)
                    else:
                        data = data.sort_values(ascending=False)

                    data.plot(kind="bar", ax=ax)
                    ax.set_xlabel(pretty_label(x))
                    ax.set_ylabel(
                        "Count" if agg == "count"
                        else f"{agg.title()} Of {pretty_label(y)}"
                    )

                # ---------------- HISTOGRAM ----------------
                elif chart_type == "histogram":
                    col = plot.get("x")
                    df[col].plot(
                        kind="hist",
                        bins=20,
                        ax=ax,
                        edgecolor="white",
                        linewidth=1,
                    )
                    ax.set_xlabel(pretty_label(col))
                    ax.set_ylabel("Frequency")

                # ---------------- PIE ----------------
                elif chart_type == "pie":
                    col = plot.get("column")
                    data = df[col].value_counts().head(MAX_TOPK)
                    data.plot(kind="pie", autopct="%1.1f%%", ax=ax)
                    ax.set_ylabel("")

                # ---------------- SCATTER ----------------
                elif chart_type == "scatter":
                    x = plot.get("x")
                    y = plot.get("y")
                    ax.scatter(df[x], df[y], alpha=0.6)
                    ax.set_xlabel(pretty_label(x))
                    ax.set_ylabel(pretty_label(y))

                # ---------------- LINE ----------------
                elif chart_type == "line":
                    x = plot.get("x")
                    y = plot.get("y")
                    agg = plot.get("aggregation")
                    grouped = df.groupby(x)[y]
                    if agg == "mean":
                        data = grouped.mean()
                    elif agg == "sum":
                        data = grouped.sum()
                    elif agg == "count":
                        data = grouped.count()
                    else:
                        raise ValueError(f"Unsupported aggregation: {agg}")

                    data.plot(ax=ax)
                    ax.set_xlabel(pretty_label(x))
                    ax.set_ylabel(f"{agg.title()} Of {pretty_label(y)}")

                else:
                    # Unknown chart type -> skip
                    plt.close(fig)
                    continue

                # ---------------- TITLE & RENDER ----------------
                metric = plot.get("metric", "")
                ax.set_title(metric, fontsize=11)
                ax.grid(True, linestyle="--", alpha=0.4)
                plt.tight_layout()
                st.pyplot(fig)

                # ---------------- EXPLANATION ----------------
                explanation = explanation_map.get(metric)
                if explanation:
                    with st.expander("Explanation"):
                        for point in explanation:
                            st.write("- " + point)

            except Exception:
                # Any error -> do NOT display this chart
                plt.close(fig)
                continue
            finally:
                plt.close(fig)
