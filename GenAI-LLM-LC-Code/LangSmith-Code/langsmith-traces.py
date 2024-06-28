# This is an example demo of extracting runs from LangSmith programatically using filters

from langsmith import Client
import pandas as pd
from datetime import datetime, timedelta

client = Client()
project_name = "default"
num_days = 7

project_runs = client.list_runs(project_name=project_name, 
                                filter='and(eq(name, "OpenAI Call Decorator 1"), has(tags, "simplechain"))', 
                                run_type = "llm",   # Can be llm, chain, tool, retriever 
                                start_time=datetime.now() - timedelta(days=num_days),   # extract runs of last 7 days
                                select=["id", "name", "run_type", "inputs", "outputs", "feedback_stats", "tags"]  # extract these fields of the runs
                                )

data = []   # build a List of Dictionary Objects
for root_run in project_runs:
    data.append(
        {
            "run_id": root_run.id,
            "run_name": root_run.name,
            "run_type": root_run.run_type,
            "inputs": root_run.inputs,
            "outputs": root_run.outputs, 
            "feedbacks": root_run.feedback_stats, 
            "tags": root_run.tags
        }
    )

# Convert the List to a pandas DataFrame
df = pd.DataFrame(data)
df.to_csv('ls-runs.csv', index=False)
print(df) 