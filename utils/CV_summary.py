import os
import openai
from dotenv import load_dotenv
import pandas as pd
import psycopg2
import pretty_errors

load_dotenv('.env')
openai.api_key = os.getenv("OPENAI_API_KEY")


MODEL= "gpt-3.5-turbo-16k"
#MODEL = "gpt-3.5-turbo"

delimiters = "####"

system_query = f""" 

Your task is to extract relevant information from a candidate's CV, to find suitable job openings./

Extract relevant information from the candidate's CV, delimited by {delimiters} characters, in at most 200 words. /

Extract any relevant information about qualifications, previous job titles, /
responsibilities/key duties, skills and other relevant achivements /
(such as publications, conferences, projects, awards, etc.). /

Use the following format:

Qualifications:
Previous job titles:
Responsibilities/Key Duties:
Skills: 
Other Achivements:

"""

def summarise_job(job_description: str) -> str:
    response = openai.ChatCompletion.create(
        messages=[
            {'role': 'user', 'content': system_query},
            {'role': 'user', 'content': f"CV: {delimiters}{job_description}{delimiters}"},
        ],
        model=MODEL,
        temperature=0,
        max_tokens=4000
    )
    response_message = response['choices'][0]['message']['content']
    total_cost = 0
    prompt_tokens = response['usage']['prompt_tokens']
    completion_tokens = response['usage']['completion_tokens']
    print(f"\nSUMMARISE JOBS FUNCTION\n", f"\nPROMPT TOKENS USED:{prompt_tokens}\n", f"COMPLETION TOKENS USED:{completion_tokens}\n" )
    #Approximate cost
    if MODEL == "gpt-3.5-turbo":
        prompt_cost = round((prompt_tokens / 1000) * 0.0015, 3)
        completion_cost = round((completion_tokens / 1000) * 0.002, 3)
        total_cost = prompt_cost + completion_cost
        print(f"COST FOR SUMMARISING: ${total_cost} USD")
    elif MODEL == "gpt-3.5-turbo-16k":
        prompt_cost = round((prompt_tokens / 1000) * 0.003, 3)
        completion_cost = round((completion_tokens / 1000) * 0.004, 3)
        total_cost = prompt_cost + completion_cost
        print(f"COST FOR SUMMARISING: ${total_cost} USD")
    return response_message