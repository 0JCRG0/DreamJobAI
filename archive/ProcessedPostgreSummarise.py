import psycopg2
import os
from dotenv import load_dotenv
import pretty_errors
import openai
import chromadb
import timeit
import time
from openai.error import ServiceUnavailableError
import pandas as pd
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from datetime import datetime, timedelta
from utils.handy import *
from utils.preprocess import *
from utils.SummariseJob import summarise_job_gpt


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

def fetch_data_from_table(table_name:str) -> list :
    conn = psycopg2.connect(user=user, password=password, host=host, port=port, database=database)

    # Create a cursor object
    cur = conn.cursor()

    # Calculate the timestamp for 3 hours ago
    two_hours_ago = datetime.now() - timedelta(hours=2)
    
    # Fetch rows from the table with the specified conditions
    cur.execute(f"SELECT id, title, description, location FROM {table_name} WHERE timestamp >= %s", (two_hours_ago,))
    
    #cur.execute(f"SELECT id, title, description, location FROM {table_name}")

    # Fetch all rows from the table
    rows = cur.fetchall()

    # Close the database connection
    cur.close()
    conn.close()

    # Separate the columns into individual lists
    ids = [row[0] for row in rows]
    titles = [row[1] for row in rows]
    descriptions = [row[2] for row in rows]
    locations = [row[3] for row in rows]

    return ids, titles, descriptions, locations

ids, titles, descriptions, locations = fetch_data_from_table("no_usa")


def processed_descriptions_to_batches(max_tokens: int, embedding_model:str, print_messages: bool = True) -> list:
    batches = []
    total_tokens = 0
    truncation_counter = 0  # Counter for truncations

    for i in descriptions:
        processed_descriptions = individual_preprocess(i)
        tokens_description = num_tokens(processed_descriptions)
        if tokens_description <= max_tokens:
            batches.append(processed_descriptions)
        else:
            #TRUNCATE IF STRING MORE THAN 1000 TOKENS
            job_truncated = truncated_string(processed_descriptions, model=model, max_tokens=max_tokens)
            batches.append(job_truncated)
            truncation_counter += 1

        total_tokens += num_tokens(processed_descriptions)  # Update the total tokens by adding the tokens of the current job
    
    #Get approximate cost for embeddings
    if embedding_model == "openai":
        approximate_cost = round((total_tokens / 1000) * 0.0004, 4)
    elif embedding_model == "e5":
        approximate_cost = 0
    
    average_tokens_per_batch = total_tokens / len(batches)
    
    if print_messages:
        for i, batch in enumerate(batches, start=1):
            print(f"Batch {i}:")
            print("".join(batch))
            print(f"Tokens per batch:", num_tokens(batch))
            print("\n")
        
        print(f"TOTAL NUMBER OF BATCHES:", len(batches))
        print(f"TOTAL NUMBER OF TOKENS:", total_tokens)  # Print the total number of tokens
        print(f"NUMBER OF TRUNCATIONS:", truncation_counter)  # Print the number of truncations
        print(f"AVERAGE NUMBER OF TOKENS PER BATCH:",average_tokens_per_batch )
        print(f"APPROXIMATE COST OF EMBEDDING:", f"${approximate_cost} USD")

    return batches

processed_batches = processed_descriptions_to_batches(max_tokens=512, embedding_model="e5", print_messages = True)

df_processed_batches = pd.DataFrame({
    "ids": ids,
    "descriptions":processed_batches})


df_processed_batches.to_csv(SAVE_PATH + "/processed_batches.csv", index=False)


def raw_descriptions_to_batches(max_tokens: int, embedding_model: str, print_messages: bool = True) -> list:
    batches = []
    total_tokens = 0
    truncation_counter = 0  # Counter for truncations

    for i in descriptions:
        tokens_description = num_tokens(i)
        if tokens_description <= max_tokens:
            batches.append(i)
        else:
            #TRUNCATE IF STRING MORE THAN x TOKENS
            job_truncated = truncated_string(i, model=model, max_tokens=max_tokens)
            batches.append(job_truncated)
            truncation_counter += 1

        total_tokens += num_tokens(i)  # Update the total tokens by adding the tokens of the current job
    
    #Get approximate cost for embeddings
    if embedding_model == "openai":
        approximate_cost = round((total_tokens / 1000) * 0.0004, 4)
    elif embedding_model == "e5":
        approximate_cost = 0
    
    average_tokens_per_batch = total_tokens / len(batches)
    
    if print_messages:
        for i, batch in enumerate(batches, start=1):
            print(f"Batch {i}:")
            print("".join(batch))
            print(f"Tokens per batch:", num_tokens(batch))
            print("\n")
        
        print(f"TOTAL NUMBER OF BATCHES:", len(batches))
        print(f"TOTAL NUMBER OF TOKENS:", total_tokens)  # Print the total number of tokens
        print(f"NUMBER OF TRUNCATIONS:", truncation_counter)  # Print the number of truncations
        print(f"AVERAGE NUMBER OF TOKENS PER BATCH:",average_tokens_per_batch )
        print(f"APPROXIMATE COST OF EMBEDDING:", f"${approximate_cost} USD")

    return batches

raw_batches = raw_descriptions_to_batches(max_tokens=512, embedding_model="e5", print_messages = True)

df_raw_batches = pd.DataFrame({
    "ids": ids,
    "descriptions":raw_batches})

df_raw_batches.to_csv(SAVE_PATH + "/raw_batches.csv", index=False)


def summarise_descriptions(descriptions: list) -> list:
    #Start the timer
    start_time = timeit.default_timer()
    descriptions_summarised = []
    total_cost = 0
    for i, text in enumerate(descriptions):
        x = i + 1
        attempts = 0
        while attempts < 3:
            try:
                description_summary, cost = summarise_job_gpt(text)
                descriptions_summarised.append(description_summary)
                total_cost += cost
                print(f"{x} description added.")
                time.sleep(.5)
                break
            except ServiceUnavailableError as e:
                attempts += 1
                print(f"{e}. Retrying attempt {attempts}...")
                time.sleep(2**attempts)  # exponential backoff
        else:
            print(f"Description number {x} could not be summarised after 3 attempts.")
            descriptions_summarised.append(text)
    
    processed_time = timeit.default_timer() - start_time

    return descriptions_summarised, total_cost, processed_time

processed_summarised_batches, processed_total_cost, processed_time_spent = summarise_descriptions(processed_batches)

df_processed_summarised_batches = pd.DataFrame({
    "ids": ids,
    "batches": processed_batches,
    "summaries": processed_summarised_batches})

def save_to_text_file(total_cost, processed_time, filename):
    with open(SAVE_PATH + f'{filename}.txt', 'w') as file:
        file.write(f"Total Cost: {total_cost}\n")
        file.write(f"Processed Time: {processed_time} seconds\n")

save_to_text_file(processed_total_cost, processed_time_spent, "/processed_summarised_batches")

df_processed_summarised_batches.to_csv(SAVE_PATH + "/processed_summarised_batches.csv", index=False)

#Raw...

raw_summarised_batches, raw_total_cost, raw_processed_time = summarise_descriptions(raw_batches)

df_raw_summarised_batches = pd.DataFrame({
    "ids": ids,
    "batches": raw_batches,
    "summaries": raw_summarised_batches})

save_to_text_file(raw_total_cost, raw_processed_time, "/raw_summarised_batches")

df_raw_summarised_batches.to_csv(SAVE_PATH + "/raw_summarised_batches.csv", index=False)

