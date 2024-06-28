# Feb 2024....
# This code is the demonstrate two simple ways of forming a Prompt and using it to Chain with a Model  
# 
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

from langchain.prompts  import PromptTemplate 
from dotenv import load_dotenv

load_dotenv()

def main():

    """
    print(demosimple1.__doc__)
    demosimple1() 
    """
    print(demosimple2.__doc__)
    demosimple2()
     
def demosimple1():
    """
    This Function Demonstrates the use of the off the shelf LLMChain to combine the Prompt and an LLM Call to get the desired response
    """
    
    # Create a Prompt Template with a embedded variable
    template = """Question: {question}

    Answer: """
    prompt = PromptTemplate(
            template=template,
        input_variables=['question']
    )

    # User question
    question = "Which is the most popular game in India?"

    # create the Language Model object
    llm = ChatOpenAI()

    # use the LLMChain to stich the prompt and llm - LLMChain is used to run queries against LLMs
    # The LLMChain consists of a PromptTemplate, a language model, and an optional output parser.

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # Invoke (run) the LLM Chain - the Chain returns Dictionary of Named Outputs
    print(llm_chain.invoke(question)['text'])


def demosimple2():
    """
    This Function Demonstrates a simple use of LCEL (LangChain Expression Language) to create a custom Chain with the Prompt and Model
    """

    # Create the Prompt Template
    prompt = ChatPromptTemplate.from_template("Tell me a few key achievements of {name}")
    # Create the LLM Object
    model = ChatOpenAI()
    # Create the Chain
    chain = prompt | model     # LCEL - LangChain Expression Language
    # Invoke (run) the Chain - The Chat Model returns a Message
    print(chain.invoke({"name": "Mahatma Gandhi"}).content)


if __name__ == "__main__":
    main()
