import os
import openai
from dotenv import load_dotenv
import pandas as pd
import pretty_errors
from aiohttp import ClientSession
import asyncio
from typing import Tuple

load_dotenv('.env')
openai.api_key = os.getenv("OPENAI_API_KEY")

#MODEL= "gpt-3.5-turbo-16k"
MODEL= "gpt-3.5-turbo"


delimiters = "----"
delimiters_job_info = '####'

system_query = f""" 

Your task is to extract the specified information from a job opening/
posted by a company, with the aim of effectively matching /
potential candidates for the position./

The job opening below is delimited by {delimiters} characters./
Within each job opening there are three sections delimited by {delimiters_job_info} characters: title, location and description./

Extract the following information from its respective section and output your response in the following format:/

Title: found in the "title" section.
Location: found in the "location" section or in the "description" section.
Job Objective: found in the "description" section.
Responsibilities/Key duties: found in the "description" section.
Qualifications/Requirements/Experience: found in the "description" section.
Preferred Skills/Nice to Have: found in the "description" section.
About the company: found in the "description" section.
Compensation and Benefits: found in the "description" section.

"""


async def async_summarise_job_gpt(session, job_description: str) -> Tuple[str, float]:
    await asyncio.sleep(.5)
    openai.aiosession.set(session)
    response = await openai.ChatCompletion.acreate(
        messages=[
            {'role': 'user', 'content': system_query},
            {'role': 'user', 'content': f"Job Opening: {delimiters}{job_description}{delimiters}"},
        ],
        model=MODEL,
        temperature=0,
        max_tokens = 400
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
        print(f"COST FOR SUMMARISING: ${total_cost:.2f} USD")
    elif MODEL == "gpt-3.5-turbo-16k":
        prompt_cost = round((prompt_tokens / 1000) * 0.003, 3)
        completion_cost = round((completion_tokens / 1000) * 0.004, 3)
        total_cost = prompt_cost + completion_cost
        print(f"COST FOR SUMMARISING: ${total_cost:.2f} USD")
    return response_message, total_cost

