import os
import pandas as pd
from crewai import Agent, Task, Crew, LLM
from tools.tools import inspect_csv  # Your cached tools
from dotenv import load_dotenv

load_dotenv()

# ---------------------------
# Environment & LLM Setup
# ---------------------------
os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY")

openrouter_llm = LLM(
    model="openrouter/mistralai/devstral-2512:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.1,
    max_tokens=850
)

# ---------------------------
# Agent Setup
# ---------------------------
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
    tools=[inspect_csv],
    allow_delegation=False,
    llm=openrouter_llm
)

# ---------------------------
# Task Setup
# ---------------------------
def create_tasks(csv_file):
    task_data_understanding = Task(
        description=f"""
        You are given a logistics CSV file.

        CSV file path: {csv_file}

        Use the provided tool `inspect_csv` to:
        - Inspect the dataset
        - Identify delay-related columns
        - Identify potential contributing factors
        - Summarize factual observations

        Important:
        - Use the field `total_rows` from tool output for the exact row count.
        - Use `sample_rows` only to understand column types and example data.
        - Do NOT invent columns or rows.
        - Clearly state assumptions if data is ambiguous.
        """,
        expected_output="""
        Bullet-point factual summary including:
        - Dataset structure
        - Exact number of rows
        - Delay indicators
        - Potential contributing factors
        """,
        agent=data_agent
    )

    return [task_data_understanding]

# ---------------------------
# Pipeline Runner
# ---------------------------
def run_pipeline(csv_path: str):
    with open(csv_path, "r", encoding="utf-8") as f:
        csv_content = f.read()

    tasks = create_tasks(csv_content)

    crew = Crew(
        agents=[data_agent],
        tasks=tasks,
        verbose=True
    )

    result = crew.kickoff()
    return result

# ---------------------------
# Run
# ---------------------------
csv_file = r"C:\\Users\\hp\\Desktop\\AIDTM\\GenAI\\End-TermProject\\Final_Code\\logistics-delivery-delay-causes.csv"
result = run_pipeline(csv_file)
print(result)
