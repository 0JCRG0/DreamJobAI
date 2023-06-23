import openai
import pandas as pd
import chromadb
from utils.handy import *
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from chromadb.config import Settings
import os
import pretty_errors
from LoadAllJobs import all_jobs_to_batches, all_jobs_ids, all_jobs_for_GPT
from LoadRecentJobs import recent_jobs_for_GPT, recent_jobs_to_batches, recent_jobs_ids
""" Env variables """


load_dotenv('.env')
SAVE_PATH = os.getenv("SAVE_PATH")
#Setting API key
openai.api_key = os.getenv("OPENAI_API_KEY")

#CALL IT
def embeddings_openai(all: bool, max_tokens: int, model_for_cost: str, db: str, file_name: str):
    if all:
        jobs_in_batches = all_jobs_to_batches(max_tokens, model_for_cost, False)
        jobs_text = all_jobs_for_GPT(max_tokens, model_for_cost, False)
        jobs_ids = all_jobs_ids
    else:
        jobs_in_batches = recent_jobs_to_batches(max_tokens,model_for_cost, False)
        jobs_text = recent_jobs_for_GPT(max_tokens, model_for_cost, False)
        jobs_ids = recent_jobs_ids
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


    #CHROMA -- Below


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

if __name__ == "__main__":
    embeddings_openai(False, 512, "openai", "parquet", "test2")