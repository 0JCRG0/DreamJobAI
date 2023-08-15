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
""" Env variables """


load_dotenv('.env')
SAVE_PATH = os.getenv("SAVE_PATH")
OPENAI_PREEXISTING_JOBS = os.getenv("OPENAI_PREEXISTING_JOBS")
OPENAI_TODAY_JOBS = os.getenv("OPENAI_RECENT_JOBS")
OPENAI_TOTAL_JOBS = os.getenv("OPENAI_TOTAL_JOBS")

#Setting API key
openai.api_key = os.getenv("OPENAI_API_KEY")

#CALL IT
def embeddings_openai(batches_to_embed: list[str], batches_ids: list[str], original_timestamps: list[str], db: str, filename: str):
    # calculate embeddings
    EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI's embedding model
    BATCH_SIZE = 50  # you can submit up to 2048 embedding inputs per request

    def embedded_batches_ada()-> list: 
        embeddings = []
        for batch_start in range(0, len(batches_to_embed), BATCH_SIZE):
            batch_end = batch_start + BATCH_SIZE
            batch = batches_to_embed[batch_start:batch_end]
            print(f"Batch {batch_start} to {batch_end-1}")
            response = openai.Embedding.create(model=EMBEDDING_MODEL, input=batch)
            for i, be in enumerate(response["data"]):
                assert i == be["index"]  # double check embeddings are in same order as input
            batch_embeddings = [e["embedding"] for e in response["data"]]
            embeddings.extend(batch_embeddings)
        return embeddings

    embeddings = embedded_batches_ada() 

    def saving_openai_embeddings()-> list:
        if db == "chromadb":
            client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=SAVE_PATH # Optional, defaults to .chromadb/ in the current directory
            ))

            collection = client.get_or_create_collection(name=filename, embedding_function=openai_ef)

            collection.add(
                id=batches_ids,
                documents=batches_to_embed,
                embeddings=embeddings
            )
            print(collection.peek())
            print(collection.count())
            print(collection.get(include=["documents"]))
        elif db == "parquet":
            df_data = {
            'id': batches_ids,
            'original': batches_to_embed,
            'embedding': embeddings,
            'timestamp': original_timestamps
        }
            df = pd.DataFrame(df_data)
            df.to_parquet(SAVE_PATH+ f"/{filename}.parquet", engine='pyarrow')
            print(f"Saved embeddings to {filename}.parquet")
        elif db == "csv":
            df = pd.DataFrame({"id": batches_ids, "original": batches_to_embed, "embedding": embeddings})
            df.to_csv(SAVE_PATH+ f"/{filename}.csv", index=False)
            print(df.head())
    saving_openai_embeddings()
    
