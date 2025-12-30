import pandas as pd
import time
from crewai import Agent, Task, Crew
from CSV_Loaded import CSVLoaderTool
from Stats_Generator import StatsTool
from crewai import LLM
from pydantic import BaseModel
from typing import List, Optional

from dotenv import load_dotenv
import os


load_dotenv()
# os.environ["OPENAI_API_KEY"] = "dummy" 

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["DEEPSEEK_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")
os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY")

# Create ONE LLM instance
groq_llm = LLM(
    model="groq/meta-llama/llama-guard-4-12b", 
    api_key=os.getenv("GROQ_API_KEY"),
    max_tokens=500,
    temperature=0.1,
    max_retries=3
)

openrouter_llm = LLM(
    model="openrouter/mistralai/devstral-2512:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.1,
    max_tokens=850
)


### IN CASE GROQ FAILS ###
# ollama_llm = LLM(
#     model="ollama/llama3",          # Format: ollama/[model_name]
#     base_url="http://localhost:11434", # Default local Ollama endpoint
#     max_retries=3
# )

# --- For the Visualization Agent ---
class PlotConfig(BaseModel):
    metric: str
    chart_type: str
    x: Optional[str] = None
    y: Optional[str] = None
    column: Optional[str] = None
    aggregation: Optional[str] = None
    top_k: Optional[int] = None
    insight: str

class VizPlan(BaseModel):
    plots: List[PlotConfig]

# --- For the Interpreter Agent ---
class ExplanationItem(BaseModel):
    metric: str
    explanation: List[str]

class InterpretationPlan(BaseModel):
    chart_explanation: List[ExplanationItem]


# ---- AGENTS ----
csv_tool = CSVLoaderTool()
stats_tool = StatsTool()

# 1️⃣ Data Understanding Agent
data_agent = Agent(
    role="Logistics Data Understanding Agent",
    goal="""
    Understand unknown logistics CSV data and extract
    information relevant for delay analysis.
    """,
    backstory="""
    You are an expert in exploratory data analysis who can
    interpret messy enterprise logistics datasets without
    prior knowledge of their schema.
    """,
    allow_delegation=False,
    llm=openrouter_llm
)

# 2️⃣ Delay Cause Agent
delay_agent = Agent(
    role="Delay Cause Intelligence Agent",
    goal="""
    Identify why shipment delays occurred by analyzing
    patterns in the structured logistics data.
    """,
    backstory="""
    You specialize in uncovering operational, environmental,
    and systemic causes of logistics delays.
    """,
    allow_delegation=False,
     llm=openrouter_llm
)

# 3️⃣ Recommendation Agent
recommendation_agent = Agent(
    role="Logistics Optimization Advisor",
    goal="""
    Generate actionable insights and recommendations
    to reduce future logistics delays.
    """,
    backstory="""
    You translate analytical findings into practical
    business decisions for logistics teams.
    """,
    allow_delegation=False,
    llm=openrouter_llm
)

viz_agent = Agent(
      role="Logistics Analytics & Visualization Agent",
      goal="""
      Decide which charts and metrics best explain
      shipment delays and operational issues.
      """,
      backstory="""
      You are a data analyst specializing in logistics KPIs.
      You interpret computed statistics and recommend
      meaningful visualizations.
      """,
      allow_delegation=False,
      max_iter = 5,
      llm=openrouter_llm
    )

viz_interpreter_agent = Agent(
    role="Data Visualization Interpreter",
    goal="Explain what each visualization reveals in clear business language",
    backstory="""
    You are an expert data analyst who explains charts to non-technical stakeholders.
    You do not create plots or choose columns.
    You only interpret what the chart shows and what it means for the business.
    """,
    verbose=True,
    allow_delegation=False,
    llm=openrouter_llm
)


# ---- TASKS ----
def create_tasks(csv_file):
    df = pd.read_csv(csv_file)
    total_rows = len(df) 
    columns = list(df.columns)
    sample = df.head(5).to_dict(orient="records")

    numeric_summary = df.describe(include="all").fillna("").to_dict()

    # Optional delay inference helpers
    unique_values = {
        col: df[col].dropna().unique()[:10].tolist()
        for col in df.columns
    }

    task_data_understanding = Task(
        description=f"""
        You are given a logistics dataset that has ALREADY been loaded and analyzed.
        Dataset metadata:
          - Total rows: {total_rows}
          - Columns: {columns}
          Sample rows:{sample}
          Column statistics (precomputed):{numeric_summary}
          Unique sample values per column:{unique_values}
        Your responsibilities:

        1. Load and inspect the dataset using actual rows (not just schema).
        2. Identify which columns indicate shipment delay.
          - This may be an explicit binary column (e.g., is_delayed, Logistics_Delay),
            shipment status values (Delayed / Delivered),
            or inferred from timestamps (expected vs actual dates).
        3. Identify columns that may contribute to delays.
        4. Compute factual summaries from the data, including:
          - Total number of records
          - Number and percentage of delayed shipments
          - Aggregated statistics for delayed vs non-delayed shipments
          - Frequency of each observed delay-related factor
        5. If explicit delay causes exist, summarize their frequencies.
          If not, infer contributing factors using correlations and patterns
          observed in the dataset.
        6. Clearly state any assumptions made due to missing or ambiguous data.
        7. Output only the summary. No need for other explanations or elements being calculated above

        IMPORTANT RULES:
        - Do NOT invent columns
        - Do NOT assume access to raw CSV
        - Use ONLY the provided information
        - Do NOT perform new calculations
        """,
        expected_output="""
        Output a bullet point, fact-based summary derived strictly from the dataset.
        Do not rely on column names alone.
        """,
        agent=data_agent
    )

    task_delay_analysis = Task(
        description="""
        You are given the structured dataset summary produced
        by the Data Understanding Agent.

        Your responsibilities:
        1. Identify recurring delay patterns present in the dataset.
        2. Infer all possible delay causes that are supported by the data.
          - Causes may include traffic, weather, customs, operational congestion,
            demand spikes, asset utilization, or others.
        3. For each inferred cause:
          - Explain how the dataset supports it. For example, if many delayed shipments show
          'heavy traffic', then 'Heavy Traffic' should be
          identified as a delay cause.
        4. Do not use external assumptions or historical knowledge.
          Base all reasoning strictly on the provided dataset summary.

        Output a ranked bullet points of inferred delay causes with evidence-based explanations.

        Use the previous agent’s output strictly as context.
        """,
        expected_output="""
        Output a ranked bullet points of inferred delay causes with evidence-based explanations.
        """,
        agent=delay_agent
    )

    task_recommendation = Task(
        description="""
        You are given the identified delay causes from the
        Delay Cause Intelligence Agent.

        Your responsibilities:

        1. Generate 5 actionable recommendations to reduce or prevent delays.
        2. Each recommendation must directly address an inferred delay cause.
        3. Avoid generic advice; tailor suggestions to the observed data patterns.
        4. If multiple datasets signals exist, prioritize high-impact improvements.
        5. Generate output in bullet points ONLY.

        Use the previous agent’s output as context.
        """,
        expected_output="""
        Clear insights and practical recommendations for
        logistics stakeholders in bullet points.
        """,
        agent=recommendation_agent
    )

    viz_task = Task(
        description=f"""
        You are a data visualization planner.

        You are given a logistics dataset with the following information:

        📌 Column names:
        {list(df.columns)}

        📌 Sample rows (first 5):
        {df.head(5).to_dict(orient="records")}

        Your job is to generate visualization plan that can be
        directly executed in Python.

       Rules:
        - Generate at most 5 plots (fewer is allowed)
        - Use ONLY column names that actually exist in the dataset
        - Do NOT invent data, values, or columns
        - Set unused fields (x, y, column, aggregation, top_k) to null

        Chart-specific rules:
        - Pie chart:
          - Provide ONLY: chart_type, column, metric, insight, top_k
          - Do NOT provide x or y
          - Include `top_k` only if category limiting is clearly required
          - If uncertain, set `top_k` to null
          - The executor will determine an appropriate limit based on data cardinality

        - Histogram:
          - You must provide exact column name from data
          - Set y = null
          - Do NOT provide top_k

        - Bar chart:
          - Always provide x
          - If aggregation is mean or sum, provide y
          - If aggregation is count, set y = null
          - Always provide aggregation
          - Include `top_k` only if category limiting is clearly required
          - If uncertain, set `top_k` to null
          - The executor will determine an appropriate limit based on data cardinality

        - Line chart:
          - Always provide x and y
          - Always provide aggregation

        - Scatter plot:
          - Always provide x and y
          - Do NOT provide aggregation or top_k
        """,
        expected_output="""
        YOUR OUTPUT MUST BE ONLY THE JSON FOLLOWING VizPlan Schema. 
        EXAMPLE FORMAT:
        {
          "plots": [
            {
              "metric": "Metric name",
              "chart_type": "bar | line | scatter | histogram | pie",
              "x": "column_name or null",
              "y": "column_name or null",
              "column": "column_name or null",
              "aggregation": "count | mean | sum | null",
              "top_k": "integer or null",
              "insight": "Business insight"
            }
          ]
        }
        """,
        agent=viz_agent,
        output_pydantic=VizPlan
        # return_json=True
      )

    viz_interpreter_task = Task(
        description="""
        You are given: 
        - The visualization plan from the Visualization Planner.
        - The chart configuration (chart_type, x, y, column, aggregation)

        Your job is to interpret what each chart reveals in clear business language.

        Rules:
        - Do not create plots or choose columns.
        - Only explain what the chart shows and what it means for the business in 5 or less bullet points.
        - Use simple, jargon-free language.
        - Do NOT invent trends
        - Focus on patterns, skew, concentration, outliers, and implications
        """,
        expected_output=""" 
        {
          "chart_explanation": "Clear explanation of what the chart shows"
        }
        """,
        agent=viz_interpreter_agent,
        context=[viz_task],
        output_pydantic=InterpretationPlan
        # return_json=True
    )

    return [
        task_data_understanding,
        task_delay_analysis,
        task_recommendation,
        viz_task,
        viz_interpreter_task
    ]


# ---- PIPELINE RUNNER ----
def run_pipeline(csv_path):
   
    tasks = create_tasks(csv_path)

    tasks[0].agent = data_agent
    tasks[1].agent = delay_agent
    tasks[2].agent = recommendation_agent
    tasks[3].agent = viz_agent
    tasks[4].agent = viz_interpreter_agent

    crew = Crew(
        agents=[data_agent, delay_agent, recommendation_agent,viz_agent, viz_interpreter_agent],
        tasks=tasks,
        verbose=True
    )

    result = crew.kickoff()
    return result
# result = run_pipeline("C:\\Users\\hp\\Desktop\\AIDTM\\GenAI\\End-TermProject\\Final_Code\\logistics-delivery-delay-causes.csv")