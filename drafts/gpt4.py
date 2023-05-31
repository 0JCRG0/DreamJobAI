import os
import openai
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA
import pandas as pd
import pretty_errors

load_dotenv('.env')
openai.api_key = os.getenv("OPENAI_API_KEY")

df = pd.read_csv("/Users/juanreyesgarcia/Library/CloudStorage/OneDrive-FundacionUniversidaddelasAmericasPuebla/DEVELOPER/PROJECTS/LLM_SANDBOX/data/jobs_test.csv")


exp = 'I HAVE 2 YEARS OF EXPERIENCE AS A DATA ANALYST. I HAVE PLENTY OF KNOWLEDGE IN PYTHON AND SQL. I HAVE ALSO WON ML COMPETITIONS'

query = f"""USE THE "Work Experience" provided below to find *THE MOST SUITABLE JOB MATCHES* in "Jobs Text File". YOUR OUTPUTS MUST ONLY BE THE JOB'S ID (eg. vcRRzC7K):

Jobs Text File:
\"\"\"
{df}
\"\"\"

Work Experience:
\"\"\"
{exp}
\"\"\"


"""

assistant_query=""" YOUR OUTPUTS MUST ONLY BE THE JOBS IDs. See the following example:

ID:vcRRzC7K, JOB TITLE:Full Stack Engineer,	DESCRIPTION:Buscamos un candidato proactivo y con una mentalidad orientada a resultados.	LOCATION:Remoto in Ciudad de México

"""
def JobMatchingAI():
    response = openai.ChatCompletion.create(
        messages=[
            {'role': 'system', 'content': 'YOU ARE A JOB HUNTER THAT FINDS THE MOST SUITABLE JOBS ACCORDING TO THE WORK EXPERIENCE OF THE CANDIDATE'},
            {'role': 'user', 'content': query},
            {'role': 'assistant', 'content':assistant_query}
        ],
        model="gpt-3.5-turbo",
        temperature=0,
        max_tokens=100
    )
    print(response['choices'][0]['message']['content'], '\n')
    print("TOTAL TOKENS:", response['usage']['total_tokens'])

JobMatchingAI()

"""
id: vcRRzC7K
- id: BdZOsC30
- id: ls2DBVNE
- id: NFNUQouK
- id: 5JqzZpXK
"""
