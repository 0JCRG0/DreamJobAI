import psycopg2
from utils.handy import *
import pretty_errors
import os
import tiktoken
from dotenv import load_dotenv

load_dotenv('.env')
user = os.getenv("user")
password = os.getenv("password")
host = os.getenv("host")
port = os.getenv("port")
database = os.getenv("database")

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

def rows_to_nested_list(all_rows: list) -> list[list]:
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
    nested_list = [[id, [title, description, location]] for id, title, description, location in zip(cleaned_ids, cleaned_titles, cleaned_descriptions, cleaned_locations)]
    return nested_list

nested_jobs = rows_to_nested_list(all_rows)
print(nested_jobs[0])

"""

Function to count tokens

"""

GPT_MODEL = "gpt-3.5-turbo"  # only matters insofar as it selects which tokenizer to use


def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

#print(num_tokens("tiktoken is great!"))


def tokens_per_job(max_tokens: int):
    counter = 0
    batches = [[]]
    job_counts = [0]  # Initialize a list to store the number of jobs in each batch
    for job in nested_jobs:
        job_info = " ".join(job[1])  # Join the elements of the list into a single string
        job_id = job[0]
        print(job_id, job_info)
        tokens_job_info = num_tokens(job_info)

        # Check if adding the current job's tokens to the counter would exceed the limit
        if counter + tokens_job_info > max_tokens:
            counter = 0  # Reset the counter
            batches.append([])  # Start a new batch
            job_counts.append(0)  # Add a new counter for the new batch

        if counter != 0:  # Only add a delimiter if this is not the first job_info in the batch
            batches[-1].append("\n")

        counter += tokens_job_info
        batches[-1].append(job_info)
        job_counts[-1] += 1  # Increment the job count for the current batch

    for i, (batch, job_count) in enumerate(zip(batches, job_counts)):
        print(f"Batch {i + 1}:")
        print("".join(batch))
        batch_tokens = "".join(batch)
        print(f"TOKENS PER BATCH:", num_tokens(batch_tokens))
        print(f"NUMBER OF JOBS IN BATCH: {job_count}")
        print("\n")
    
    print(f"TOTAL NUMBER OF BATCHER:", len(batches))

tokens_per_job(200)
#if __name__ == "__main__":
    