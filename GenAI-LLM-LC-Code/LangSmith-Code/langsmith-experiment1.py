from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith.evaluation import evaluate, LangChainStringEvaluator

# Target task definition
prompt = ChatPromptTemplate.from_messages([
  ("system", "Please review the content below and rewrite and elaborate it pointwise"),
  ("user", "{postfix}")
])
chat_model = ChatOpenAI()
output_parser = StrOutputParser()

chain = prompt | chat_model | output_parser

# The name or UUID of the LangSmith dataset to evaluate on.
# Alternatively, you can pass an iterator of examples
data = "Sample Dataset X"

# A string to prefix the experiment name with.
# If not provided, a random string will be generated.
experiment_prefix = "Sample Dataset X"

# List of evaluators to score the outputs of target task
evaluators = [
  LangChainStringEvaluator("cot_qa"),
  LangChainStringEvaluator("labeled_criteria", config={"criteria": "controversiality"}),
  LangChainStringEvaluator("labeled_criteria", config={"criteria": "misogyny"}),
  LangChainStringEvaluator("labeled_criteria", config={"criteria": "maliciousness"}),
  LangChainStringEvaluator("labeled_criteria", config={"criteria": "conciseness"})
]

# Evaluate the target task
results = evaluate(
  chain.invoke,
  data=data,
  evaluators=evaluators,
  experiment_prefix=experiment_prefix,
)