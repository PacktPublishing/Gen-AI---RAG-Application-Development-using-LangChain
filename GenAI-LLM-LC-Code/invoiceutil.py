from langchain.chat_models import ChatOpenAI
from pypdf import PdfReader
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.llm import LLMChain

#load GPT 3.5 model
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    max_tokens=2000
)

# import neccessary packages from korr
from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number

# iterate over files in
# that user uploaded PDF files, one by one
def create_docs(user_pdf_list):
    """This function is used to extract the invoice data from the given PDF files. 
    It uses the LangChain agent to extract the data from the given PDF files."""
    df = pd.DataFrame({'Invoice no.': pd.Series(dtype='str'),
                   'Description': pd.Series(dtype='str'),
                   'Quantity': pd.Series(dtype='str'),
                   'Date': pd.Series(dtype='str'),
	                'Unit price': pd.Series(dtype='str'),
                   'Amount': pd.Series(dtype='int'),
                   'Total': pd.Series(dtype='str'),
                   'Email': pd.Series(dtype='str'),
	                'Phone number': pd.Series(dtype='str'),
                   'Address': pd.Series(dtype='str')
                    })

    for filename in user_pdf_list:

        # Extract PDF Data
        texts = ""
        print("Processing -", filename)
        pdf_reader = PdfReader(filename)
        for page in pdf_reader.pages:
            texts += page.extract_text()

        template = """Extract all the following values : invoice no., Description, Quantity, date, 
            Unit price , Amount, Total, email, phone number and address from the following Invoice content: 
            {texts}
            The fields and values in the above content may be jumbled up as they are extracted from a PDF. Please use your judgement to align
            the fields and values correctly based on the fields asked for in the question abiove.
            Expected output format: 
            {{'Invoice no.': xxxxxxxx','Description': 'xxxxxx','Quantity': 'x','Date': 'dd/mm/yyyy',
            'Unit price': xxx.xx','Amount': 'xxx.xx,'Total': xxx,xx,'Email': 'xxx@xxx.xxx','Phone number': 'xxxxxxxxxx','Address': 'xxxxxxxxx'}}
            Remove any dollar symbols or currency symbols from the extracted values.
            """
        prompt = PromptTemplate.from_template(template)

        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k-0613")

        chain = LLMChain(llm=llm, prompt=prompt)

        data_dict = chain.run(texts)

        print("Dict:...", data_dict)
        new_row_df = pd.DataFrame([eval(data_dict)], columns=df.columns)
        df = pd.concat([df, new_row_df], ignore_index=True)  

        print("********************DONE***************")

    print(df) 
    return df

