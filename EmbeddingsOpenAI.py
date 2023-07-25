import openai
import pandas as pd
import chromadb
from datetime import datetime
from utils.handy import *
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from chromadb.config import Settings
import os
import pretty_errors
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from LoadAllJobs import all_jobs_to_batches, all_jobs_ids, all_jobs_for_GPT
from LoadRecentJobs import recent_jobs_for_GPT, recent_jobs_to_batches, recent_jobs_ids
from PostgreSummarise import ids, raw_descriptions_to_batches, summarise_descriptions
""" Env variables """


load_dotenv('.env')
SAVE_PATH = os.getenv("SAVE_PATH")
OPENAI_PREEXISTING_JOBS = os.getenv("OPENAI_PREEXISTING_JOBS")
OPENAI_TODAY_JOBS = os.getenv("OPENAI_RECENT_JOBS")
OPENAI_TOTAL_JOBS = os.getenv("OPENAI_TOTAL_JOBS")

#Setting API key
openai.api_key = os.getenv("OPENAI_API_KEY")

#CALL IT
def embeddings_openai(jobs_to_embed: str, max_tokens: int, model_for_cost: str, db: str):
    if jobs_to_embed == "LoadAllJobs":
        jobs_in_batches = all_jobs_to_batches(max_tokens, model_for_cost, False)
        jobs_text = all_jobs_for_GPT(max_tokens, model_for_cost, False)
        jobs_ids = all_jobs_ids
        file_name = "openai_preexisting"
    elif jobs_to_embed == "LoadRecentJobs":
        jobs_in_batches = recent_jobs_to_batches(max_tokens,model_for_cost, False)
        jobs_text = recent_jobs_for_GPT(max_tokens, model_for_cost, False)
        jobs_ids = recent_jobs_ids
        file_name = "openai_today"
    elif jobs_to_embed == "PostgreSummarise":
        jobs_in_batches = raw_descriptions_to_batches(max_tokens=750, embedding_model="e5", print_messages=False)
        jobs_text = summarise_descriptions(jobs_in_batches)
        jobs_ids = ids
        file_name = "openai_summarised"
    # calculate embeddings
    EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI's embedding model
    BATCH_SIZE = 50  # you can submit up to 2048 embedding inputs per request

    def embedded_batches_ada()-> list: 
        embeddings = []
        for batch_start in range(0, len(jobs_in_batches), BATCH_SIZE):
            batch_end = batch_start + BATCH_SIZE
            batch = jobs_in_batches[batch_start:batch_end]
            print(f"Batch {batch_start} to {batch_end-1}")
            response = openai.Embedding.create(model=EMBEDDING_MODEL, input=batch)
            for i, be in enumerate(response["data"]):
                assert i == be["index"]  # double check embeddings are in same order as input
            batch_embeddings = [e["embedding"] for e in response["data"]]
            embeddings.extend(batch_embeddings)
        return embeddings

    def saving_openai_embeddings()-> list:
        if db == "chromadb":
            client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=SAVE_PATH # Optional, defaults to .chromadb/ in the current directory
            ))

            collection = client.get_or_create_collection(name=file_name, embedding_function=openai_ef)

            collection.add(
                ids=jobs_ids,
                documents=jobs_text,
                embeddings=embedded_batches_ada()
            )
            print(collection.peek())
            print(collection.count())
            print(collection.get(include=["documents"]))
        elif db == "parquet":
            df_data = {
            'ids': jobs_ids,
            'text_data': jobs_text,
            'embeddings': embedded_batches_ada()
        }
            df = pd.DataFrame(df_data)
            df.to_parquet(SAVE_PATH+ f"/{file_name}.parquet", engine='pyarrow')
            print(f"Saved embeddings to {file_name}.parquet")
        elif db == "csv":
            df = pd.DataFrame({"id": jobs_ids, "text": jobs_text, "embedding": embedded_batches_ada()})
            df.to_csv(SAVE_PATH+ f"/{file_name}.csv", index=False)
            print(df.head())
    saving_openai_embeddings()
    
    #If you are embedding jobs from today concat it with all_jobs
    if jobs_to_embed == "LoadRecentJobs":
        def concat_parquet_openai():
            # Read the Parquet files
            recent_jobs = pq.read_table(OPENAI_TODAY_JOBS)
            preexisting_jobs = pq.read_table(OPENAI_PREEXISTING_JOBS)
            
            # Append the tables
            appended_jobs = pa.concat_tables([recent_jobs, preexisting_jobs])
            
            # Write the appended table to a new Parquet file
            pq.write_table(appended_jobs, OPENAI_TOTAL_JOBS)
            now = datetime.now()
            logging.info(f"OpenAI embeddings have been appended at {now}")
        concat_parquet_openai()

if __name__ == "__main__":
    embeddings_openai(jobs_to_embed="PostgreSummarise", max_tokens=512, model_for_cost="openai", db="parquet")