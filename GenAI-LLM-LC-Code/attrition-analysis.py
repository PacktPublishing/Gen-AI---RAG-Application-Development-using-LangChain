import pandas as pd
from dotenv import load_dotenv
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import OpenAI
from dotenv import load_dotenv
import streamlit as st

def main():
    load_dotenv()
    df = pd.read_csv('HR-Employee-Attrition.csv')

    st.set_page_config(
        page_title="Documentation Chatbot",
        page_icon=":books:",
    )

    st.title("Attrition Analysis Chatbot")
    st.subheader("Uncover Insights from Attrition Data!")
    st.markdown(
        """
        This chatbot was created to answer questions from a set of Attrition data from your organisation.
        Ask a question and the chatbot will respond with appropriete Analysis.
        """
    )
    st.write(df.head())
    user_question = st.text_input("Ask your question about the data..")

    agent = create_csv_agent(
        OpenAI(temperature=0),
        "HR-Employee-Attrition.csv",
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    print(agent.agent.llm_chain.prompt.template)

    answer = agent.run(user_question)
    st.write(answer)

    """
    """

if __name__ == "__main__":
    main()