import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
import os
from archive.LoadAllJobs import *
from archive.LoadRecentJobs import *
from datetime import datetime

load_dotenv('.env')
OPENAI_PREEXISTING_JOBS = os.getenv("OPENAI_PREEXISTING_JOBS")
E5_BASE_PREEXISTING_JOBS = os.getenv("E5_BASE_PREEXISTING_JOBS")
OPENAI_RECENT_JOBS = os.getenv("OPENAI_RECENT_JOBS")
E5_BASE_RECENT_JOBS = os.getenv("E5_BASE_RECENT_JOBS")
OPENAI_TOTAL_JOBS = os.getenv("OPENAI_TOTAL_JOBS")
E5_BASE_TOTAL_JOBS = os.getenv("E5_BASE_TOTAL_JOBS")

LoggingMain()

def append_parquet_files(embedding_model:str):
    if embedding_model == "openai":
        recent_path = OPENAI_RECENT_JOBS
        preexisting_path = OPENAI_PREEXISTING_JOBS
        total_path = OPENAI_TOTAL_JOBS
    elif embedding_model == "e5-base":
        recent_path = E5_BASE_RECENT_JOBS
        preexisting_path = E5_BASE_PREEXISTING_JOBS
        total_path = E5_BASE_TOTAL_JOBS
    # Read the Parquet files
    recent_jobs = pq.read_table(recent_path)
    preexisting_jobs = pq.read_table(preexisting_path)
    
    # Append the tables
    appended_jobs = pa.concat_tables([recent_jobs, preexisting_jobs])
    
    # Write the appended table to a new Parquet file
    pq.write_table(appended_jobs, total_path)
    now = datetime.now()
    logging.info(f"{embedding_model} embeddings have been appended at {now}")

if __name__ == "__main__":
    append_parquet_files()