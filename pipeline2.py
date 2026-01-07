import pandas as pd
from crewai import Agent, Task, Crew, LLM
from pydantic import BaseModel
from typing import List, Optional
from Tools.logistics_tools import (
    set_dataframe,
    get_dataset_profile,
    find_delay_columns,
    compute_delay_stats,
    list_delay_factors,
)
from dotenv import load_dotenv
import os


load_dotenv()

os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY")

openrouter_llm = LLM(
    model="openrouter/mistralai/devstral-2512:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.1,
    max_tokens=850
)

# --- For the Visualization Agent ---
# Template
class PlotConfig(BaseModel):
    metric: str
    chart_type: str
    x: Optional[str] = None
    y: Optional[str] = None
    column: Optional[str] = None
    aggregation: Optional[str] = None
    top_k: Optional[int] = None
    insight: str

# list of plotconfig objects
class VizPlan(BaseModel):
    plots: List[PlotConfig]

# --- For the Interpreter Agent ---
class ExplanationItem(BaseModel):
    metric: str
    explanation: List[str]

class InterpretationPlan(BaseModel):
    chart_explanation: List[ExplanationItem]



# Attach tools to the agents that actually need them
data_agent = Agent(
    role="Logistics Data Understanding Agent",
    goal="""
    Understand unknown logistics CSV data and extract
    information relevant for delay analysis.
    """,
    backstory="""You are an expert in exploratory data analysis.""",
    allow_delegation=False,
    llm=openrouter_llm,
    tools=[get_dataset_profile, find_delay_columns, compute_delay_stats]
)

delay_agent = Agent(
    role="Delay Cause Intelligence Agent",
    goal="""Identify why shipment delays occurred...""",
    backstory="""You specialize in uncovering causes of delays...""",
    allow_delegation=False,
    llm=openrouter_llm,
    tools=[list_delay_factors]  
)

recommendation_agent = Agent(
    role="Logistics Optimization Advisor",
    goal="""Generate actionable insights.""",
    backstory="""You translate analytical findings into decisions...""",
    allow_delegation=False,
    llm=openrouter_llm,
    tools=[]  
)

viz_agent = Agent(
    role="Logistics Analytics & Visualization Agent",
    goal="""Decide which charts and metrics best explain shipment delays.""",
    backstory="""You are a data analyst specializing in logistics KPIs.""",
    allow_delegation=False,
    max_iter=5,
    llm=openrouter_llm,
    tools=[compute_delay_stats]  
)

viz_interpreter_agent = Agent(
    role="Data Visualization Interpreter",
    goal="Explain what each visualization reveals in clear business language",
    backstory="""You explain charts to non-technical stakeholders.""",
    verbose=True,
    allow_delegation=False,
    llm=openrouter_llm,
    tools=[]
)

def create_tasks(csv_file):
    df = pd.read_csv(csv_file)
    set_dataframe(df)  # make df available to tools

    task_data_understanding = Task(
        description="""
        Use your tools to:
        - Call get_dataset_profile to understand the dataset.
        - Call find_delay_columns to identify likely delay indicator columns.
        - If a clear delay column exists, call compute_delay_stats on it.
        Then produce a bullet-point factual summary of:
        - Dataset size and structure
        - Candidate delay columns
        - Basic delay rates, if computed
        Do NOT invent columns or perform your own arithmetic.
        """,
        expected_output="""
        Bullet-point factual summary strictly based on tool outputs.
        """,
        agent=data_agent,
    )

    task_delay_analysis = Task(
        description="""
        You receive the dataset summary from the previous task.
        Using that summary and, if needed, the list_delay_factors tool:
        - Identify recurring delay patterns.
        - Infer possible delay causes that are supported by the data.
        - For each cause, explain briefly how the data supports it.
        Do not use external knowledge; only rely on the summary and tool outputs.
        Output ranked bullet points of delay causes with evidence.
        """,
        expected_output="""
        Ranked bullet points of delay causes with evidence-based explanations.
        """,
        agent=delay_agent,
    )

    task_recommendation = Task(
        description="""
        You receive the inferred delay causes.
        Generate 5 actionable recommendations, each tied to a specific cause.
        Avoid generic advice; tailor suggestions to the patterns in the causes.
        Output bullet points only.
        """,
        expected_output="""
        Five bullet-point recommendations for logistics stakeholders.
        """,
        agent=recommendation_agent,
    )

    viz_task = Task(
        description=f"""
        You are a data visualization planner.

        You are given a logistics dataset with the following information:

         Column names:
        {list(df.columns)}

         Sample rows (first 5):
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


def run_pipeline(csv_path):
    tasks = create_tasks(csv_path)

    crew = Crew(
        agents=[data_agent, delay_agent, recommendation_agent, viz_agent, viz_interpreter_agent],
        tasks=tasks,
        verbose=True,
    )
    result = crew.kickoff()
    return result

# result = run_pipeline("C:\\Users\\hp\\Desktop\\AIDTM\\GenAI\\End-TermProject\\Final_Code\\logistics-delivery-delay-causes.csv")