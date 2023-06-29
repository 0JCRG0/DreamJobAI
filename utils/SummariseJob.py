import os
import openai
from dotenv import load_dotenv
import pandas as pd
import pretty_errors

load_dotenv('.env')
openai.api_key = os.getenv("OPENAI_API_KEY")

MODEL= "gpt-3.5-turbo-16k"
#MODEL="gpt-3.5-turbo"

system_query = """  Write an abstract summary of the following job in a single paragraph. 

Let's think step by step to summarise it into a single paragraph:

1. Focus on the main skills, software required and responsibilities of each role. 
2. One of the bullet points is the total years of experience.
2. Omit the employer names. 
4. Double-check that the summary is in four bullet points.

JOB DESCRIPTION IS BELOW: """

assistant_query=""" Remember to think step by step.
"""
def summarise_job(job_to_summarise):
    response = openai.ChatCompletion.create(
        messages=[
            {'role': 'system', 'content': system_query},
            {'role': 'user', 'content': job_to_summarise},
            {'role': 'assistant', 'content':assistant_query}
        ],
        model=MODEL,
        temperature=0,
        max_tokens=4000
    )
    response_message = response['choices'][0]['message']['content']
    total_cost = 0
    prompt_tokens = response['usage']['prompt_tokens']
    completion_tokens = response['usage']['completion_tokens']
    #print(f"\nSUMMARISE JOBS FUNCTION\n", f"\nPROMPT TOKENS USED:{prompt_tokens}\n", f"COMPLETION TOKENS USED:{completion_tokens}\n", f"\nTOTAL TOKENS USED:{total_tokens}\n", )
    #Approximate cost
    if MODEL == "gpt-3.5-turbo":
        prompt_cost = round((prompt_tokens / 1000) * 0.0015, 3)
        completion_cost = round((completion_tokens / 1000) * 0.002, 3)
        total_cost = prompt_cost + completion_cost
        #print(f"COST FOR SUMMARISING: ${total_cost} USD")
    elif MODEL == "gpt-3.5-turbo-16k":
        prompt_cost = round((prompt_tokens / 1000) * 0.003, 3)
        completion_cost = round((completion_tokens / 1000) * 0.004, 3)
        total_cost = prompt_cost + completion_cost
        #print(f"COST FOR SUMMARISING: ${total_cost} USD")
    return response_message, total_cost

if __name__ == "__main__":
    summarise_job()


"""
id: vcRRzC7K
- id: BdZOsC30
- id: ls2DBVNE
- id: NFNUQouK
- id: 5JqzZpXK
"""
