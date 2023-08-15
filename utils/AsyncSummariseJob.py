import os
import openai
from dotenv import load_dotenv
import pandas as pd
import pretty_errors
from openai.error import ServiceUnavailableError
import logging
from aiohttp import ClientSession
import asyncio
from typing import Tuple
from utils.handy import count_words, LoggingMain

load_dotenv('.env')
openai.api_key = os.getenv("OPENAI_API_KEY")

#MODEL= "gpt-3.5-turbo-16k"
MODEL= "gpt-3.5-turbo"

logging.basicConfig(filename='/Users/juanreyesgarcia/Library/CloudStorage/OneDrive-FundacionUniversidaddelasAmericasPuebla/DEVELOPER/PROJECTS/DreamedJobAI/logs/LoggingGPT4.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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
    #print(f"\nSUMMARISE JOBS FUNCTION\n", f"\nPROMPT TOKENS USED:{prompt_tokens}\n", f"COMPLETION TOKENS USED:{completion_tokens}\n" )
    #Approximate cost
    if MODEL == "gpt-3.5-turbo":
        prompt_cost = round((prompt_tokens / 1000) * 0.0015, 3)
        completion_cost = round((completion_tokens / 1000) * 0.002, 3)
        total_cost = prompt_cost + completion_cost
        #print(f"COST FOR SUMMARISING: ${total_cost:.2f} USD")
    elif MODEL == "gpt-3.5-turbo-16k":
        prompt_cost = round((prompt_tokens / 1000) * 0.003, 3)
        completion_cost = round((completion_tokens / 1000) * 0.004, 3)
        total_cost = prompt_cost + completion_cost
        #print(f"COST FOR SUMMARISING: ${total_cost:.2f} USD")
    return response_message, total_cost

"""
async def summarise_descriptions(descriptions: list) -> list:
	#start timer
	start_time = asyncio.get_event_loop().time()
	total_cost = 0

	async def process_description(session, i, text):
		attempts = 0
		while attempts < 5:
			try:
				words_per_text = count_words(text)
				if words_per_text > 50:
					description_summary, cost = await async_summarise_job_gpt(session, text)
					print(f"Description with index {i} just added.")
					logging.info(f"Description's index {i} just added.")
					return i, description_summary, cost
				else:
					logging.warning(f"Description with index {i} is too short for being summarised. Number of words: {words_per_text}")
					print(f"Description with index {i} is too short for being summarised. Number of words: {words_per_text}")
					return i, text, 0
			except (Exception, ServiceUnavailableError) as e:
				attempts += 1
				print(f"{e}. Retrying attempt {attempts}...")
				logging.warning(f"{e}. Retrying attempt {attempts}...")
				await asyncio.sleep(5**attempts)  # exponential backoff
		else:
			print(f"Description with index {i} could not be summarised after 5 attempts.")
			return i, text, 0

	async with ClientSession() as session:
		tasks = [process_description(session, i, text) for i, text in enumerate(descriptions)]
		results = await asyncio.gather(*tasks)

	# Sort the results by the index and extract the summaries and costs
	results.sort()
	descriptions_summarised = [result[1] for result in results]
	costs = [result[2] for result in results]
	total_cost = sum(costs)

	#await close_session()
	#processed_time = timeit.default_timer() - start_time
	elapsed_time = asyncio.get_event_loop().time() - start_time

	return descriptions_summarised, total_cost, elapsed_time
"""

async def async_summarise_description(description: str) -> tuple:
    #start timer
    start_time = asyncio.get_event_loop().time()
    total_cost = 0

    async def process_description(session, text):
        attempts = 0
        while attempts < 5:
            try:
                words_per_text = count_words(text)
                if words_per_text > 50:
                    description_summary, cost = await async_summarise_job_gpt(session, text)
                    return description_summary, cost
                else:
                    logging.warning(f"Description is too short for being summarised. Number of words: {words_per_text}")
                    return text, 0
            except (Exception, ServiceUnavailableError) as e:
                attempts += 1
                print(f"{e}. Retrying attempt {attempts}...")
                logging.warning(f"{e}. Retrying attempt {attempts}...")
                await asyncio.sleep(5**attempts)  # exponential backoff
        else:
            print(f"Description could not be summarised after 5 attempts.")
            return text, 0

    async with ClientSession() as session:
        result = await process_description(session, description)

    total_cost = result[1]

    #await close_session()
    #processed_time = timeit.default_timer() - start_time
    elapsed_time = asyncio.get_event_loop().time() - start_time

    return result[0], total_cost, elapsed_time