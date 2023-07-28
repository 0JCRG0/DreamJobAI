import os
import openai
from dotenv import load_dotenv
import pandas as pd
import pretty_errors

load_dotenv('.env')
openai.api_key = os.getenv("OPENAI_API_KEY")

#MODEL= "gpt-3.5-turbo-16k"
MODEL= "gpt-3.5-turbo"


delimiters = "----"
delimiters_job_info = '####'

system_query = f""" 

Your task is to extract the specified information of a job opening/
posted by a company, with the aim of effectively matching /
potential candidates for the position./

The job opening below is delimited by {delimiters} characters./
Within each job opening there are three sections delimited by {delimiters_job_info} characters: title, location and description./

Extract the following information from its respective section and output your response in the following format:/

Title: found in the "title" section.
Location: found in the "location" section or in the "description" section.
Job Objective: found in the "description" section.
Responsibilities/Key duties: found in the "description" section.
Qualifications/Requirements: found in the "description" section.
Preferred Skills/Experience: found in the "description" section.
About the company: found in the "description" section.
Compensation and Benefits: found in the "description" section.

"""

def summarise_job_gpt(job_description: str) -> str:
    response = openai.ChatCompletion.create(
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

if __name__ == "__main__":
    x = "###title: Software Engineer II Compliance at Chainalysis### ###location: Remote### ###description: Job DescriptionThe Compliance organization is focused on growing the crypto ecosystem by simplifying the work needed for compliance and risk management at a massive scale. Our goal is to ensure developers can send data for any crypto asset and network scale our systems to make sure we can handle the increasing volumes of data and provide meaningful insights to our customers. Backend engineers will be critical to that mission by building and scaling the APIs and data layers our customers rely on every day to stop crime understand risk and strategize about their business. Working alongside infrastructure and security-focused engineers they obsess over making our services highly available and safe for our customers to use for their most sensitive and real-time blockchain workflows. They deeply understand what is possible with cloud-native technologies and use those insights to enable our customers to push the boundaries of the cryptocurrency landscape.&nbsp; In one year you’ll know you were successful if…  You have built a high-availability scalable API leveraging the most relevant services from AWS You’ve added features to our product suite that detect activities for market manipulation fraud behavioral patterning and more. You have built cloud-native data ingestion and aggregation processes that intake gigabytes of data per day. You have helped modernize our stack to a streaming architecture. Your team’s services are easy to set up locally and their health in production is simple to understand.&nbsp; You have debugged production issues and participated in a blameless post-mortem process to make our systems stronger.&nbsp;  A background like this helps:&nbsp;  Designed and implemented microservices-based systems in a major cloud provider like AWS or GCP. Experience with object-oriented programming languages. We mostly use Java but appreciate a variety of languages!&nbsp; A bias to ship and iterate alongside product management and design partners Exposure to or interest in the cryptocurrency technology ecosystem Listed in: Cryptocurrency Jobs Remote Crypto Jobs Security Crypto Jobs Developer Web3 Jobs Compliance Web3 Jobs Data Crypto Jobs Full Time Web3 Jobs.###     "
    print(summarise_job_gpt(x))
