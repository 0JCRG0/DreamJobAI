import psycopg2
from utils.handy import *
import pretty_errors
import os
import tiktoken
import openai
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from chromadb.config import Settings
from preprocess import individual_preprocess

"""
Env variables
"""

load_dotenv('.env')
user = os.getenv("user")
password = os.getenv("password")
host = os.getenv("host")
port = os.getenv("port")
database = os.getenv("database")
SAVE_PATH = os.getenv("SAVE_PATH")
#Setting API key
openai.api_key = os.getenv("OPENAI_API_KEY")

"""
Model
"""
model = "gpt-3.5-turbo"  # only matters insofar as it selects which tokenizer to use


"""

TODO: REMEMBER THE NEW PLAN IS AS FOLLOWS:
    TODO: 1. Connect e5 & openai embeddings to LoadRecentJobs 
    TODO: 2. Use function in loader to concatenate main embeddings with recent
    TODO: 3. Recuerda hacer un trigger para una tabla basada en no_usa,
            la cual siempre filtre trabajos que esten a una semana de distancia del día presente.
            Por ejemplo, hoy es 10 de enero, el filtro aceptaría a los trabajos encontrados desde el 3 de enero,
            pero el día de mañana (11 de enero) el filtro se le suma +1 por lo que el corte sería el 4 de enero,
            así sucesivamente.
            3.1. This table will be loaded for subsequent embedding (all jobs & recent jobs). NOT no_usa


"""

def fetch_data_from_table(table_name:str) -> list :
    conn = psycopg2.connect(user=user, password=password, host=host, port=port, database=database)

    # Create a cursor object
    cur = conn.cursor()
    cur.execute(f"SELECT id FROM {table_name}")

    # Fetch all rows from the table
    IDS = cur.fetchall()
    cur.execute(f"SELECT title FROM {table_name}")

    # Fetch all rows from the table
    TITLES = cur.fetchall()
    cur.execute(f"SELECT description FROM {table_name}")

        # Fetch all rows from the table
    DESCRIPTIONS = cur.fetchall()
    cur.execute(f"SELECT location FROM {table_name}")

        # Fetch all rows from the table
    LOCATIONS = cur.fetchall()

    # Close the database connection
    cur.close()
    conn.close()

    return IDS, TITLES, DESCRIPTIONS, LOCATIONS

# Fetch data from the PostgreSQL table

def rows_to_nested_list(all_rows: list =  fetch_data_from_table("no_usa")) -> list:
    #get all the rows
    ids, titles, descriptions, locations = all_rows
    #Ids
    formatted_ids = ["{}".format(id) for id in ids]
    cleaned_ids = [clean_rows(id) for id in formatted_ids]
    #Titles
    formatted_titles = ["passage: {}".format(title) for title in titles]
    cleaned_titles = [clean_rows(title) for title in formatted_titles]
    #Descriptions
    formatted_descriptions = ["{}".format(description) for description in descriptions]
    cleaned_descriptions = [clean_rows(description) for description in formatted_descriptions]
    #Locations
    formatted_locations = ["{}".format(location) for location in locations]
    cleaned_locations = [clean_rows(location) for location in formatted_locations]

    #NEST THE LISTS
    jobs_info = [[title, location, description] for title, location, description in zip(cleaned_titles, cleaned_locations, cleaned_descriptions)]
    jobs_ids = cleaned_ids
    return jobs_ids, jobs_info

all_jobs_ids, jobs_info= rows_to_nested_list()
#print(jobs_ids[0], type(jobs_ids), len(jobs_ids))
#print(jobs_info[0], type(jobs_info), len(jobs_info))


""" THIS ONE CONTAINS *NO* PREPROCESSED JOB INFOs"""
def all_jobs_for_GPT(max_tokens: int, embedding_model:str, print_warning: bool = True) -> list:
    batches = []
    total_tokens = 0
    truncation_counter = 0  # Counter for truncations

    for i in jobs_info:
        job = " ".join(i)  # Join the elements of the list into a single string
        tokens_job_info = num_tokens(job)
        if tokens_job_info <= max_tokens:
            batches.append(job)
        else:
            #TRUNCATE IF STRING MORE THAN 1000 TOKENS
            job_truncated = truncated_string(job, model=model, max_tokens=max_tokens)
            batches.append(job_truncated)
            truncation_counter += 1

    
        total_tokens += num_tokens(job)  # Update the total tokens by adding the tokens of the current job
    
    #Get approximate cost for embeddings
    if embedding_model == "openai":
        approximate_cost = round((total_tokens / 1000) * 0.0004, 4)
    elif embedding_model == "e5":
        approximate_cost = 0

    if print_warning:
        for i, batch in enumerate(batches, start=1):
            print(f"Batch {i}:")
            print("".join(batch))
            print(f"Tokens per batch:", num_tokens(batch))
            print("\n")
        
        print(f"TOTAL NUMBER OF BATCHES:", len(batches))
        print(f"TOTAL NUMBER OF TOKENS:", total_tokens)  # Print the total number of tokens
        print(f"APPROXIMATE COST OF EMBEDDING:", f"${approximate_cost} USD")
        print(f"NUMBER OF TRUNCATIONS:", truncation_counter)  # Print the number of truncations

    return batches

""" This one is PREPROCESSED JOB INFO -> USED FOR EMBEDDINGS """
def all_jobs_to_batches(max_tokens: int, embedding_model: str, print_warning: bool = True) -> list:
    batches = []
    total_tokens = 0
    truncation_counter = 0  # Counter for truncations

    for i in jobs_info:
        job = " ".join(i)  # Join the elements of the list into a single string
        job_clean = individual_preprocess(job) #cleans each job
        tokens_job_info = num_tokens(job_clean)
        if tokens_job_info <= max_tokens:
            batches.append(job_clean)
        else:
            #TRUNCATE IF STRING MORE THAN 1000 TOKENS
            job_truncated = truncated_string(job_clean, model=model, max_tokens=max_tokens)
            batches.append(job_truncated)
            truncation_counter += 1  # Counter for truncations


        # Update the total tokens by adding the tokens of the current job
        total_tokens += num_tokens(job_clean)

    #Get approximate cost for embeddings
    if embedding_model == "openai":
        approximate_cost = round((total_tokens / 1000) * 0.0004, 4)
    elif embedding_model == "e5":
        approximate_cost = 0
    
    if print_warning:
        for i, batch in enumerate(batches, start=1):
            print(f"Batch {i}:")
            print("".join(batch))
            print(f"Tokens per batch:", num_tokens(batch))
            print("\n")
        
        print(f"TOTAL NUMBER OF BATCHES:", len(batches))
        print(f"TOTAL NUMBER OF TOKENS:", total_tokens)  # Print the total number of tokens
        print(f"APPROXIMATE COST OF EMBEDDING:", f"${approximate_cost} USD")
        print(f"NUMBER OF TRUNCATIONS:", truncation_counter)  # Print the number of truncations

    return batches


    
if __name__ == "__main__":
    """ 1st argument: token limit, 2nd arg: embedding model"""
    #all_jobs_for_GPT(512, "e5-large")
    all_jobs_to_batches(512, "openai")
