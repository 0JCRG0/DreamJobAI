import openai
import pandas as pd
import chromadb
from utils.handy import *
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from chromadb.config import Settings
import os
import pretty_errors
from postgre2embedding import jobs_to_batches, jobs_ids, jobs_for_GPT

""" Env variables """


load_dotenv('.env')
SAVE_PATH = os.getenv("SAVE_PATH")
#Setting API key
openai.api_key = os.getenv("OPENAI_API_KEY")

#CALL IT
jobs_in_batches = jobs_to_batches(500, "openai")
jobs_text = jobs_for_GPT(500, "openai")

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


def saving_openai_embeddings(db: str, file_name:str )-> list:
    if db == "chromadb":
        client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=SAVE_PATH # Optional, defaults to .chromadb/ in the current directory
        ))

        collection = client.get_or_create_collection(name=file_name, embedding_function=openai_ef)

        collection.add(
            documents=jobs_text,
            embeddings=embedded_batches_ada(),
            ids=jobs_ids
        )
        print(collection.peek())
        print(collection.count())
        print(collection.get(include=["documents"]))
    elif db == "parquet":
        df_data = {
        'ids': jobs_ids,
        'text_data': jobs_for_GPT(700, "e5-large", False),
        'embeddings': embedded_batches_ada()
    }
        df = pd.DataFrame(df_data)
        df.to_parquet(SAVE_PATH+ f"/{file_name}.parquet", engine='pyarrow')
        print(f"Saved embeddings to {file_name}.parquet")
    elif db == "csv":
        df = pd.DataFrame({"id": jobs_ids, "embedding": embedded_batches_ada(), "text": jobs_text})
        df.to_csv(SAVE_PATH+ f"/{file_name}.csv", index=False)
        print(df.head())

if __name__ == "__main__":
    saving_openai_embeddings("parquet", "openai_embedding")