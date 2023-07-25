import os
import openai
from dotenv import load_dotenv
import pandas as pd
import pretty_errors
from aiohttp import ClientSession
from typing import Tuple

load_dotenv('.env')
openai.api_key = os.getenv("OPENAI_API_KEY")

#MODEL= "gpt-3.5-turbo-16k"
MODEL= "gpt-3.5-turbo"


delimiters = "####"

system_query = f""" 

Your task is to extract "relevant information" of a job opening/
posted by a company, with the aim of effectively matching /
potential candidates for the position./

From the job opening below, delimited by {delimiters} characters, extract 
"relevant information" in at most 200 words. /
Your main focus is the category of Job Information./

Relevant information to extract: 

Job Information (150 words max):  extract any essential details about the specific position being advertised. Focus on job title, /
job objective, responsibilities/key duties, qualifications/requirements and preferred skills/experience. /

About the company and Compensation and Benefits (50 words max): extract any information about the hiring company,/
such as company name, location, culture, values, or mission. /
Also, extract any financial aspects and additional perks associated with the job./
/ 

"""

assistant_query=""" Remember to think step by step.
"""


async def async_summarise_job_gpt(session, job_description: str) -> Tuple[str, float]:
    openai.aiosession.set(session)
    response = await openai.ChatCompletion.acreate(
        messages=[
            {'role': 'user', 'content': system_query},
            {'role': 'user', 'content': f"Job Opening: {delimiters}{job_description}{delimiters}"},
        ],
        model=MODEL,
        temperature=0,
        max_tokens = 350
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
    return response_message, total_cost

# At the end of your program, close the http session
"""async def close_session():
    # At the end of your program, close the http session
    await openai.aiosession.get().close()"""