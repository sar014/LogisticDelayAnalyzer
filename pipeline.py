import pandas as pd
import time
from crewai import Agent, Task, Crew
from CSV_Loaded import CSVLoaderTool
from crewai import LLM

from dotenv import load_dotenv
import os

load_dotenv()
# os.environ["OPENAI_API_KEY"] = "dummy" 

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Create ONE LLM instance
groq_llm = LLM(
    model="groq/llama-3.3-70b-versatile", 
    api_key=os.getenv("GROQ_API_KEY"),
    max_retries=3
)

### IN CASE GROQ FAILS ###
# ollama_llm = LLM(
#     model="ollama/llama3",          # Format: ollama/[model_name]
#     base_url="http://localhost:11434", # Default local Ollama endpoint
#     max_retries=3
# )


# ---- AGENTS ----
csv_tool = CSVLoaderTool()

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
    tools=[csv_tool],
    allow_delegation=False,
    llm=groq_llm
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
     llm=groq_llm
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
    llm=groq_llm
)


# ---- TASKS ----
def create_tasks(csv_file):
    task_data_understanding = Task(
        description=f"""
        A user has provided a logistics CSV file.

        CSV file path: {csv_file}

        Your responsibilities:

        1. Load and inspect the dataset using actual rows (not just schema).
        2. Identify which columns indicate shipment delay.
          - This may be an explicit binary column (e.g., is_delayed, Logistics_Delay),
            shipment status values (Delayed / Delivered),
            or inferred from timestamps (expected vs actual dates).
        3. Identify columns that may contribute to delays, such as:
          - Traffic, weather, customs, operational metrics, utilization,
            waiting time, demand pressure, or any other relevant factors.
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

        1. Generate actionable recommendations to reduce or prevent delays.
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

    return [
        task_data_understanding,
        task_delay_analysis,
        task_recommendation
    ]


# ---- PIPELINE RUNNER ----
def run_pipeline(csv_path):
   
    tasks = create_tasks(csv_path)

    tasks[0].agent = data_agent
    tasks[1].agent = delay_agent
    tasks[2].agent = recommendation_agent

    crew = Crew(
        agents=[data_agent, delay_agent, recommendation_agent],
        tasks=tasks,
        verbose=True
    )

    result = crew.kickoff()
    return result
