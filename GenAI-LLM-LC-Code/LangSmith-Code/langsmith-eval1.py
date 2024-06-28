# This experiment demonstrates the following:
# - We Create a Dataset containing a Set of two pairs of Inputs and Outputs (assuming they came from a LangChain LLM Experiment)
# - These Datasets are Evaluated using 4 Langsmith Evaluators including a Custom Evaluator
# - Evaluations are viewed using the LangSmith UI
# The purpose of this Test is to evaluate Input Output pairs from your LLM experiments and assessing whether they are good results

from langsmith import Client
from langsmith.evaluation import evaluate, LangChainStringEvaluator

# Create your LangSmith Client
client = Client()

# Create Evaluator Objects (LangSmith off-the-shelf Evaluators)
cot_qa_evaluator = LangChainStringEvaluator("cot_qa")   # Chain of thought evaluator
context_evaluator = LangChainStringEvaluator("context_qa")    # Context Evaluator
qa_evaluator = LangChainStringEvaluator("qa")   # QA

# Create your Dataset
dataset_name = "Sample Dataset X"
dataset = client.create_dataset(dataset_name, description="A sample dataset about the benefits of Yoga.")
client.create_examples(
    inputs=[
        {"postfix": '''Yoga offers numerous benefits for both the body and mind. Physically, it enhances flexibility, 
         strength, and balance, contributing to overall fitness and reducing the risk of injury. Regular practice can alleviate chronic pain, 
         improve cardiovascular health, and boost the immune system. Mentally, yoga promotes relaxation and reduces stress by encouraging mindfulness and deep breathing. 
         It also helps in managing anxiety and depression, leading to a more positive outlook on life. 
         The combination of physical exercise and mental focus creates a holistic approach to health that supports well-being and longevity.'''},        
    ],
    outputs=[
        {"output": '''The advantages of practicing yoga are extensive, encompassing both physical and mental health. On a physical level, yoga increases flexibility, 
         strengthens muscles, and improves balance, which can prevent injuries and enhance physical performance. 
         It also aids in relieving chronic pain, supports heart health, and strengthens the immune system. 
         Mentally, yoga fosters relaxation and reduces stress through mindfulness and controlled breathing techniques. 
         It is also effective in managing symptoms of anxiety and depression, promoting a more optimistic and balanced mindset. 
         This integrative practice nurtures overall wellness, contributing to a healthier, more balanced life.'''},
    ],
    dataset_id=dataset.id,
)

# Create a Custom Evaluator for Checking exact match
def exact_match(run, example):
    return {"score": run.outputs["output"] == example.outputs["output"]}

# Run Evaluation Experiment
experiment_results = evaluate(
    lambda input: input['postfix'], # Your AI system goes here
    data=dataset_name, # The data to predict and grade over
    evaluators=[cot_qa_evaluator, exact_match, context_evaluator, qa_evaluator], # The evaluators to score the results
    experiment_prefix="sample-experiment", # The name of the experiment
    metadata={
      "version": "1.0.0",
      "revision_id": "beta"
    },
)
# Print Results
print(experiment_results.experiment_name, " - ", experiment_results._results)