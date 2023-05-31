import psycopg2
from utils.handy import *
import pretty_errors
import os
import tiktoken
import openai
import pandas as pd
import chromadb
from utils.handy import openai_ef
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from chromadb.config import Settings

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

YOU NEED TO ADD THE ID COLUMN SO THAT WORKS FOR EVERY TABLE.

"""

def fetch_data_from_table():
    conn = psycopg2.connect(user=user, password=password, host=host, port=port, database=database)

    # Create a cursor object
    cur = conn.cursor()
    cur.execute("SELECT id FROM personal")

    # Fetch all rows from the table
    IDS = cur.fetchall()
    cur.execute("SELECT title FROM personal")

    # Fetch all rows from the table
    TITLES = cur.fetchall()
    cur.execute("SELECT description FROM personal")

        # Fetch all rows from the table
    DESCRIPTIONS = cur.fetchall()
    cur.execute("SELECT location FROM personal")

        # Fetch all rows from the table
    LOCATIONS = cur.fetchall()

    # Close the database connection
    cur.close()
    conn.close()

    return IDS, TITLES, DESCRIPTIONS, LOCATIONS

# Fetch data from the PostgreSQL table
all_rows = fetch_data_from_table()

def rows_to_nested_list(all_rows: list) -> list:
    #get all the rows
    ids, titles, descriptions, locations = all_rows
    #Ids
    formatted_ids = ["{}".format(id) for id in ids]
    cleaned_ids = [clean_rows(id) for id in formatted_ids]
    #Titles
    formatted_titles = ["{}".format(title) for title in titles]
    cleaned_titles = [clean_rows(title) for title in formatted_titles]
    #Descriptions
    formatted_descriptions = ["{}".format(description) for description in descriptions]
    cleaned_descriptions = [clean_rows(description) for description in formatted_descriptions]
    #Locations
    formatted_locations = ["{}".format(location) for location in locations]
    cleaned_locations = [clean_rows(location) for location in formatted_locations]

    #NEST THE LISTS
    jobs_info = [[title, description, location] for title, description, location in zip(cleaned_titles, cleaned_descriptions, cleaned_locations)]
    jobs_ids = cleaned_ids
    return jobs_ids, jobs_info

jobs_ids, jobs_info= rows_to_nested_list(all_rows)
#print(jobs_ids[0], type(jobs_ids), len(jobs_ids))
#print(jobs_info[0], type(jobs_info), len(jobs_info))

""" Function to count tokens """

def num_tokens(text: str, model: str = model) -> int:
    #Return the number of tokens in a string.
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

#print(num_tokens("tiktoken is great!"))


def jobs_to_batches(max_tokens: int) -> list:
    batches = []
    for i in jobs_info:
        job = " ".join(i)  # Join the elements of the list into a single string
        tokens_job_info = num_tokens(job)
        if tokens_job_info <= max_tokens:
            batches.append(job)
        else:
            #TRUNCATE IF STRING MORE THAN 1000 TOKENS
            job_truncated = truncated_string(job, model=model, max_tokens=max_tokens)
            batches.append(job_truncated)
    
    """ """
    for i, batch in enumerate(batches, start=1):
        print(f"Batch {i}:")
        print("".join(batch))
        print(f"Tokens per batch:", num_tokens(batch))
        print("\n")
    
    print(f"TOTAL NUMBER OF BATCHES:", len(batches))
    
    return batches

if __name__ == "__main__":
    #The argument is the token limit
    jobs_to_batches(500)
