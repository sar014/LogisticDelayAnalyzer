import matplotlib.pyplot as plt
import streamlit as st

MAX_TOP_K = 10

def render_plots_streamlit(json_result, df):

    def pretty_label(col):
        return col.replace("_", " ").title() if col else ""

    for plot in json_result["plots"]:

        fig, ax = plt.subplots(figsize=(8, 5))
        chart_type = plot["chart_type"]

        # ---------------- BAR CHART ----------------
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

            if df[x].nunique() > MAX_TOP_K:
                top_k = min(top_k or MAX_TOP_K, MAX_TOP_K)
                data = data.sort_values(ascending=False).head(top_k)
            else:
                data = data.sort_values(ascending=False)

            data.plot(kind="bar", ax=ax)
            ax.set_xlabel(pretty_label(x))

            if agg == "count":
                ax.set_ylabel("Count")
            else:
                ax.set_ylabel(f"{agg.title()} of {pretty_label(y)}")

        # ---------------- HISTOGRAM ----------------
        elif chart_type == "histogram":
            col = plot["x"]
            if not col:
                plt.close(fig)
                continue

            df[col].plot(
                kind="hist",
                bins=20,
                ax=ax,
                edgecolor="white",
                linewidth=1.2
            )
            ax.set_xlabel(pretty_label(col))
            ax.set_ylabel("Frequency")

        # ---------------- PIE CHART ----------------
        elif chart_type == "pie":
            col = plot.get("column")
            if not col:
                plt.close(fig)
                continue

            data = df[col].value_counts()
            if len(data) > MAX_TOP_K:
                data = data.head(MAX_TOP_K)

            data.plot(kind="pie", autopct="%1.1f%%", ax=ax)
            ax.set_ylabel("")

        # ---------------- SCATTER ----------------
        elif chart_type == "scatter":
            x, y = plot.get("x"), plot.get("y")
            if not x or not y:
                plt.close(fig)
                continue

            ax.scatter(df[x], df[y], alpha=0.6)
            ax.set_xlabel(pretty_label(x))
            ax.set_ylabel(pretty_label(y))

        # ---------------- LINE ----------------
        elif chart_type == "line":
            x, y, agg = plot.get("x"), plot.get("y"), plot.get("aggregation")
            if not x or not y or not agg:
                plt.close(fig)
                continue

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

        ax.set_title(plot["metric"])
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        # 🔥 STREAMLIT RENDER
        st.pyplot(fig)
        plt.close(fig)